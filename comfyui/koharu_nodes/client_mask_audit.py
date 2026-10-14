"""Read-only projection of the installed client's box NMS, before morphology.

Source: koharu@52bb4f55e2266f72c1523562a583c827ec191ceb,
crates/koharu-pipeline/src/stages/detection.rs. A union preview before
client NMS can hide a suppressed instance's unique text pixels.
This module never rewrites the result sent to Koharu.
"""
import numpy as np
from .detection_masks import page_size, decode_mask, MAX_PIXELS


def client_nms(result):
    page_size(result)
    items = result['detections']
    if len(items) > 160:
        raise ValueError('Client audit exceeds 160 detections')
    boxes, scores = [], []
    for item in items:
        box = np.asarray(item['bbox'], dtype=np.float32)
        score = np.float32(item['score'])
        if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(score):
            raise ValueError('Invalid client audit box/score')
        if box[2] <= box[0] or box[3] <= box[1] or not isinstance(item['label'], str):
            raise ValueError('Invalid client audit bounds/label')
        boxes.append(box); scores.append(score)
    # Match stable score descending, top ascending, left descending, label.
    def total_key(value):
        bits = int(value.view(np.uint32))
        return (~bits & 0xffffffff) if bits & 0x80000000 else bits ^ 0x80000000
    order = sorted(range(len(items)), key=lambda i: (-total_key(scores[i]),
                   total_key(boxes[i][1]), -total_key(boxes[i][0]), items[i]['label']))
    kept, dropped = [], []
    def area(box):
        return np.maximum(box[2]-box[0], np.float32(0)) * np.maximum(box[3]-box[1], np.float32(0))
    for index in order:
        box = boxes[index]
        for owner in kept:
            if items[index]['label'] != items[owner]['label']:
                continue
            other = boxes[owner]
            intersection = (np.maximum(np.minimum(box[2],other[2])-np.maximum(box[0],other[0]),np.float32(0))
                          * np.maximum(np.minimum(box[3],other[3])-np.maximum(box[1],other[1]),np.float32(0)))
            union = area(box) + area(other) - intersection
            iou = intersection / union if union > 0 else np.float32(0)
            if iou >= np.float32(0.5):
                dropped.append({'index':index, 'kept_index':owner, 'label':items[index]['label'],
                                'score':float(scores[index]), 'bbox':box.astype(float).tolist(),
                                'iou':float(iou)})
                break
        else:
            kept.append(index)
    return kept, dropped


def nms_result(result):
    kept, _ = client_nms(result)
    return {**result, 'detections':[result['detections'][i] for i in kept]}


def audit(result):
    width, height = page_size(result)
    kept, dropped = client_nms(result)
    report = {'stage':'before_client_morphology', 'regions':len(result['detections']),
              'kept':len(kept), 'dropped':dropped, 'lost_text_pixels':0, 'lost_text_bounds':None}
    if not any(item['label'] == 'text' for item in dropped):
        return report
    if width*height > MAX_PIXELS:
        report['lost_text_pixels'] = None
        report['pixel_audit'] = 'skipped_page_pixel_limit'
        return report
    # Bounded temporary boolean pages; no retained images or file writes.
    before = np.zeros((height,width),bool)
    after = np.zeros_like(before)
    survivors = set(kept)
    for index, item in enumerate(result['detections']):
        if item['label'] != 'text':
            continue
        local,(x,y,w,h) = decode_mask(item['mask'],(width,height))
        ink = local > 0.5
        before[y:y+h,x:x+w] |= ink
        if index in survivors:
            after[y:y+h,x:x+w] |= ink
    lost = before & ~after
    report['lost_text_pixels'] = int(np.count_nonzero(lost))
    if report['lost_text_pixels']:
        rows = np.flatnonzero(lost.any(axis=1)); cols = np.flatnonzero(lost.any(axis=0))
        report['lost_text_bounds'] = [int(cols[0]),int(rows[0]),int(cols[-1])+1,int(rows[-1])+1]
    return report
