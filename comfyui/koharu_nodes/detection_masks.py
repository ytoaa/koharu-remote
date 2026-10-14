"""Bridge detection-local RLE masks to ordinary ComfyUI mask operations."""
import numpy as np
from dgx.vision.model import rle

# Use local instance crops, not a page-sized float mask per detection.
MAX_PIXELS = 16 * 1024 * 1024
MAX_RUNS = 4_000_000


def page_size(result):
    if not isinstance(result, dict) or result.get('protocol') != 1 or not isinstance(result.get('detections'), list):
        raise ValueError('Expected protocol-1 detection result')
    width, height = result.get('image_width'), result.get('image_height')
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError('Invalid detection page dimensions')
    return width, height


def decode_mask(mask, page):
    if not isinstance(mask, dict):
        raise ValueError('Missing detection mask')
    x, y, width, height = (mask.get(k) for k in ('x','y','width','height'))
    if any(type(v) is not int for v in (x,y,width,height)) or min(x,y) < 0 or min(width,height) <= 0:
        raise ValueError('Invalid local mask coordinates')
    if x + width > page[0] or y + height > page[1] or width * height > MAX_PIXELS:
        raise ValueError('Local mask exceeds page or memory limit')
    runs = mask.get('runs')
    if not isinstance(runs,list) or not runs or len(runs) > MAX_RUNS or any(type(v) is not int or v < 0 for v in runs) or sum(runs) != width * height:
        raise ValueError('Invalid detection mask RLE')
    values = np.repeat(np.arange(len(runs), dtype=np.uint32) % 2, runs).astype(np.float32)
    return values.reshape(height,width), (x,y,width,height)


class KoharuDetectionMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'result': ('KOHARU_RESULT',),
                'labels': (['text + onomatopoeia','text','onomatopoeia','bubble','panel','all'],),
                'padding': ('INT', {'default':8,'min':0,'max':256})}}
    RETURN_TYPES = ('MASK','KOHARU_MASK_CONTEXT')
    RETURN_NAMES = ('masks','context')
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = 'extract'
    CATEGORY = 'Koharu/Detection'
    def extract(self, result, labels='text + onomatopoeia', padding=8):
        import torch
        page = page_size(result)
        choices = {'text + onomatopoeia': {'text','onomatopoeia'}, 'all': {'text','onomatopoeia','bubble','panel'},
                   **{label:{label} for label in ('text','onomatopoeia','bubble','panel')}}
        if labels not in choices or type(padding) is not int or not 0 <= padding <= 256:
            raise ValueError('Invalid mask extraction options')
        masks, entries = [], []
        total = 0
        for index, item in enumerate(result['detections']):
            if item.get('label') not in choices[labels]:
                continue
            # Check padded allocation before decoding the original mask.
            raw = item.get('mask', {})
            if not isinstance(raw, dict) or any(type(raw.get(k)) is not int for k in ('x','y','width','height')):
                raise ValueError('Invalid detection mask')
            x,y,w,h = (raw[k] for k in ('x','y','width','height'))
            left,top = max(0,x-padding),max(0,y-padding)
            right,bottom = min(page[0],x+w+padding),min(page[1],y+h+padding)
            area = (right-left)*(bottom-top)
            total += max(0,area)
            if area <= 0 or total > MAX_PIXELS:
                raise ValueError('Extracted masks exceed memory limit')
            local, _ = decode_mask(raw, page)
            pixels = np.zeros((bottom-top,right-left), dtype=np.float32)
            pixels[y-top:y-top+h,x-left:x-left+w] = local
            masks.append(torch.from_numpy(pixels)[None])
            entries.append({'index':index,'x':left,'y':top,'width':right-left,'height':bottom-top})
        # Core mask nodes have scalar widgets: an empty MASK list would be
        # sliced alongside those widgets. A no-op sentinel keeps warmup valid.
        if not masks:
            masks = [torch.from_numpy(np.zeros((1,1), dtype=np.float32))[None]]
        return (masks, {'result':result,'entries':entries})


class KoharuDetectionMasksApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'masks': ('MASK',), 'context': ('KOHARU_MASK_CONTEXT',),
                'threshold': ('FLOAT', {'default':0.5,'min':0.,'max':1.,'step':0.01}),
                'expand_bbox': ('BOOLEAN', {'default':True})}}
    RETURN_TYPES = ('KOHARU_RESULT',)
    INPUT_IS_LIST = True
    FUNCTION = 'apply'
    CATEGORY = 'Koharu/Detection'
    def apply(self, masks, context, threshold, expand_bbox):
        if len(context) != 1 or len(threshold) != 1 or len(expand_bbox) != 1:
            raise ValueError('Expected one extraction context and option set')
        context, threshold, expand_bbox = context[0], threshold[0], expand_bbox[0]
        page = page_size(context['result'])
        if not isinstance(threshold,(int,float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1 or type(expand_bbox) is not bool:
            raise ValueError('Invalid mask application options')
        if not context['entries']:
            if len(masks) != 1:
                raise ValueError('Expected one empty-selection sentinel mask')
            return (context['result'],)
        if len(masks) != len(context['entries']):
            raise ValueError('Processed mask count differs from detection context')
        original = context['result']
        result = {**original, 'detections': [dict(item) for item in original['detections']]}
        total, run_count = 0, 0
        for tensor, entry in zip(masks,context['entries']):
            values = tensor.detach().cpu().numpy()
            expected = (1,entry['height'],entry['width'])
            if values.shape != expected or not np.isfinite(values).all():
                raise ValueError('Processed masks must preserve crop dimensions and instance order')
            pixels = values[0] > threshold
            ys,xs = np.nonzero(pixels)
            if len(xs):
                left,top,right,bottom = int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
            else:
                left,top,right,bottom = 0,0,1,1
            x,y = entry['x']+left,entry['y']+top
            local = pixels[top:bottom,left:right]
            total += local.size
            if total > MAX_PIXELS or x+local.shape[1] > page[0] or y+local.shape[0] > page[1]:
                raise ValueError('Processed mask exceeds page or memory limit')
            item = result['detections'][entry['index']]
            item['mask'] = {'x':x,'y':y,'width':local.shape[1],'height':local.shape[0],'runs':rle(local)}
            if len(xs) and expand_bbox:
                bbox = item['bbox']
                item['bbox'] = [min(bbox[0],float(x)),min(bbox[1],float(y)),
                                max(bbox[2],float(x+local.shape[1])),max(bbox[3],float(y+local.shape[0]))]
        for item in result['detections']:
            run_count += len(item['mask']['runs'])
            if run_count > MAX_RUNS:
                raise ValueError('Processed detection RLE exceeds response limit')
        return (result,)


NODE_CLASS_MAPPINGS = {'KoharuDetectionMasks':KoharuDetectionMasks,
                       'KoharuDetectionMasksApply':KoharuDetectionMasksApply}
