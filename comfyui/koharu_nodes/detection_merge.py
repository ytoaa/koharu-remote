"""Preserve segmentation pixels when the installed client suppresses boxes."""
import logging
import numpy as np
from .client_mask_audit import client_nms
from .detection_masks import page_size, decode_mask, MAX_PIXELS, MAX_RUNS
from .text_masks import LABEL_CHOICES
from dgx.vision.model import rle

LOG = logging.getLogger(__name__)


def union_instance_masks(items, page, expand_bbox):
    parts = []
    allocated = 0
    for item in items:
        raw = item['mask']
        if any(type(raw.get(k)) is not int for k in ('width','height')):
            raise ValueError('Invalid mask to merge')
        allocated += raw['width']*raw['height']
        if allocated > MAX_PIXELS:
            raise ValueError('Merged source masks exceed pixel limit')
        local, bounds = decode_mask(raw, page)
        parts.append((local > 0.5, bounds))
    left = min(bounds[0] for _,bounds in parts)
    top = min(bounds[1] for _,bounds in parts)
    right = max(bounds[0]+bounds[2] for _,bounds in parts)
    bottom = max(bounds[1]+bounds[3] for _,bounds in parts)
    if (right-left)*(bottom-top) > MAX_PIXELS:
        raise ValueError('Merged mask extent exceeds pixel limit')
    pixels = np.zeros((bottom-top,right-left),bool)
    for local,(x,y,w,h) in parts:
        pixels[y-top:y-top+h,x-left:x-left+w] |= local
    result = dict(items[0])  # Keep winner label, confidence and other metadata.
    rows = np.flatnonzero(pixels.any(axis=1)); cols = np.flatnonzero(pixels.any(axis=0))
    if not len(rows):
        return result
    x0,y0,x1,y1 = int(cols[0]),int(rows[0]),int(cols[-1])+1,int(rows[-1])+1
    local = pixels[y0:y1,x0:x1]
    result['mask'] = {'x':left+x0,'y':top+y0,'width':x1-x0,'height':y1-y0,'runs':rle(local)}
    if expand_bbox:
        box = result['bbox']
        result['bbox'] = [min(float(box[0]),float(left+x0)),min(float(box[1]),float(top+y0)),
                          max(float(box[2]),float(left+x1)),max(float(box[3]),float(top+y1))]
    return result


def merge_overlaps(result, labels='text', expand_bbox=True):
    page = page_size(result)
    if labels not in LABEL_CHOICES or type(expand_bbox) is not bool:
        raise ValueError('Invalid overlap merge options')
    selected = LABEL_CHOICES[labels]
    items = result['detections'][:]
    merged = rounds = 0
    # Box expansion can create additional overlaps. Each nonempty pass removes
    # at least one selected instance, so convergence is bounded by input count.
    for _ in range(len(items)):
        _, dropped = client_nms({**result,'detections':items})
        dropped = [item for item in dropped if item['label'] in selected]
        if not dropped:
            break
        groups = {}
        removed = set()
        for item in dropped:
            groups.setdefault(item['kept_index'],[]).append(item['index'])
            removed.add(item['index'])
        replacements = {owner:union_instance_masks([items[owner]]+[items[i] for i in members],page,expand_bbox)
                        for owner,members in groups.items()}
        items = [replacements.get(i,item) for i,item in enumerate(items) if i not in removed]
        merged += len(removed); rounds += 1
    if sum(item['mask']['width']*item['mask']['height'] for item in items if item['label'] in selected) > MAX_PIXELS:
        raise ValueError('Merged detection masks exceed pixel limit')
    if sum(len(item['mask']['runs']) for item in items) > MAX_RUNS:
        raise ValueError('Merged detection RLE exceeds client response limit')
    output = {**result,'detections':items,
              'mask_preserving_nms':{'labels':labels,'iou':0.5,'merged':merged,'rounds':rounds,
                                     'expand_bbox':expand_bbox}}
    LOG.info('Koharu mask-preserving NMS: input=%s output=%s merged=%s rounds=%s labels=%s',
             len(result['detections']),len(items),merged,rounds,labels)
    return output


class KoharuDetectionMergeOverlaps:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'result':('KOHARU_RESULT',),
            'enabled':('BOOLEAN',{'default':True}),
            'labels':(['text','text + onomatopoeia','onomatopoeia'],),
            'expand_bbox':('BOOLEAN',{'default':True,
                'tooltip':'Expand the retained OCR box to include merged mask pixels; recheck overlaps until stable.'})}}
    RETURN_TYPES = ('KOHARU_RESULT',)
    FUNCTION = 'merge'
    CATEGORY = 'Koharu/Detection'
    def merge(self,result,enabled=True,labels='text',expand_bbox=True):
        if type(enabled) is not bool:
            raise ValueError('Invalid overlap merge switch')
        return (merge_overlaps(result,labels,expand_bbox) if enabled else result,)


NODE_CLASS_MAPPINGS = {'KoharuDetectionMergeOverlaps':KoharuDetectionMergeOverlaps}
NODE_DISPLAY_NAME_MAPPINGS = {'KoharuDetectionMergeOverlaps':'Koharu 겹친 영역 병합 · 마스크 보존'}
