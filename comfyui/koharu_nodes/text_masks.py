"""SAM text pixels to existing protocol-1 region masks; no client changes."""
import logging,math
import numpy as np
from .detection_masks import page_size,decode_mask,MAX_PIXELS,MAX_RUNS
from dgx.vision.model import rle

LOG=logging.getLogger(__name__)
LABEL_CHOICES={'text + onomatopoeia':{'text','onomatopoeia'},'text':{'text'},'onomatopoeia':{'onomatopoeia'}}


def page_mask(result,labels='text + onomatopoeia'):
    width,height=page_size(result)
    if width*height>MAX_PIXELS:raise ValueError('Text mask exceeds page pixel limit')
    if labels not in LABEL_CHOICES:raise ValueError('Invalid text-mask label selection')
    output=np.zeros((height,width),np.float32)
    for item in result['detections']:
        if item['label'] not in LABEL_CHOICES[labels]:continue
        local,(x,y,w,h)=decode_mask(item['mask'],(width,height))
        np.maximum(output[y:y+h,x:x+w],local,out=output[y:y+h,x:x+w])
    return output


def apply_text_mask(result,mask,mode='replace',labels='text + onomatopoeia',region_padding=8,
                    empty_fallback=True,recover_missed=False,grouping_radius=6,min_component_area=8,max_new_regions=64):
    width,height=page_size(result)
    if width*height>MAX_PIXELS:raise ValueError('Text mask exceeds page pixel limit')
    values=np.asarray(mask,dtype=np.float32)
    if values.shape!=(height,width) or not np.isfinite(values).all():raise ValueError('Text mask must match original page dimensions')
    if mode not in ('replace','union') or labels not in LABEL_CHOICES:raise ValueError('Invalid text-mask apply mode')
    if type(empty_fallback) is not bool or type(recover_missed) is not bool:raise ValueError('Invalid text-mask apply switches')
    for value,low,high in [(region_padding,0,128),(grouping_radius,0,64),(min_component_area,1,4096),(max_new_regions,1,160)]:
        if type(value) is not int or not low<=value<=high:raise ValueError('Invalid text-mask apply option')
    pixels=values>0.5
    output={**result,'detections':[dict(item) for item in result['detections']]}
    covered=np.zeros((height,width),bool)
    replacements=fallbacks=0;allocated=0
    def encode(item,local,left,top):
        ys,xs=np.nonzero(local)
        if len(xs):
            x0,y0,x1,y1=int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
        else:x0=y0=0;x1=y1=1
        trimmed=local[y0:y1,x0:x1]
        item['mask']={'x':left+x0,'y':top+y0,'width':trimmed.shape[1],'height':trimmed.shape[0],'runs':rle(trimmed)}
        if len(xs):
            bbox=item['bbox']
            item['bbox']=[min(bbox[0],float(left+x0)),min(bbox[1],float(top+y0)),
                          max(bbox[2],float(left+x1)),max(bbox[3],float(top+y1))]
    for index,item in enumerate(result['detections']):
        if item['label'] not in LABEL_CHOICES[labels]:continue
        original,(mx,my,mw,mh)=decode_mask(item['mask'],(width,height))
        bbox=item['bbox']
        if len(bbox)!=4 or not np.isfinite(bbox).all():raise ValueError('Invalid detection box')
        left=max(0,min(mx,math.floor(bbox[0]))-region_padding)
        top=max(0,min(my,math.floor(bbox[1]))-region_padding)
        right=min(width,max(mx+mw,math.ceil(bbox[2]))+region_padding)
        bottom=min(height,max(my+mh,math.ceil(bbox[3]))+region_padding)
        covered[top:bottom,left:right]=True
        local=pixels[top:bottom,left:right].copy()
        allocated+=local.size
        if allocated>MAX_PIXELS:raise ValueError('Refined region masks exceed pixel limit')
        if not local.any() and empty_fallback:
            fallbacks+=1;continue
        if mode=='union':local[my-top:my-top+mh,mx-left:mx-left+mw]|=original>0.5
        encode(output['detections'][index],local,left,top);replacements+=1
    recovered=0
    if recover_missed:
        from scipy import ndimage
        remaining=pixels&~covered
        groups=ndimage.maximum_filter(remaining,size=grouping_radius*2+1,mode='constant') if grouping_radius else remaining
        components,count=ndimage.label(groups)
        sizes=np.bincount(components.ravel(),weights=remaining.ravel(),minlength=count+1)
        selected=np.flatnonzero(sizes>=min_component_area);selected=selected[selected!=0]
        if len(selected)>max_new_regions or len(output['detections'])+len(selected)>160:
            raise ValueError('Too many recovered text regions; adjust grouping/area controls (client limit 160)')
        boxes=ndimage.find_objects(components)
        for label_id in selected:
            ys,xs=boxes[label_id-1]
            local=(components[ys,xs]==label_id)&remaining[ys,xs]
            # SAM-TS has no calibrated instance confidence. A zero score marks
            # an unscored pixel-derived region; the client accepts 0..1 scores.
            item={'label_id':0,'label':'text','score':0.0,'score_kind':'unscored_mask_component',
                  'bbox':[float(xs.start),float(ys.start),float(xs.stop),float(ys.stop)]}
            encode(item,local,xs.start,ys.start);output['detections'].append(item);recovered+=1
    if sum(len(item['mask']['runs']) for item in output['detections'])>MAX_RUNS:
        raise ValueError('Refined mask RLE exceeds client response limit')
    output['text_mask_refinement']={'mode':mode,'replaced':replacements,'empty_fallbacks':fallbacks,'recovered':recovered}
    LOG.info('SAM mask apply: mode=%s, replaced=%s, empty_fallbacks=%s, recovered=%s',mode,replacements,fallbacks,recovered)
    return output


class KoharuTextMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'images':('IMAGE',),'model':('KOHARU_TEXT_MASK_MODEL',),
            'mode':(['full_page','tiles'],),
            'threshold':('FLOAT',{'default':0.0,'min':-5.,'max':5.,'step':0.1}),
            'tile_size':([1024,768,512],),
            'overlap':('INT',{'default':128,'min':0,'max':384,'step':16})}}
    RETURN_TYPES=('MASK',)
    FUNCTION='segment'
    CATEGORY='Koharu/Detection'
    def segment(self,images,model,**options):
        import torch
        from comfy.utils import ProgressBar
        from .nodes import pil_image
        image=pil_image(images)
        bar=None
        def progress(done,total):
            nonlocal bar
            if bar is None:bar=ProgressBar(total)
            bar.update(1)
        result=model.segment(image,progress=progress,**options)
        if result.shape!=(image.height,image.width) or not np.isfinite(result).all():raise ValueError('Invalid text-mask backend result')
        return (torch.from_numpy(result[None]),)


class KoharuTextMaskApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'result':('KOHARU_RESULT',),'mask':('MASK',),
            'enabled':('BOOLEAN',{'default':True}),
            'mode':(['replace','union'],),'labels':(list(LABEL_CHOICES),),
            'region_padding':('INT',{'default':8,'min':0,'max':128}),
            'empty_fallback':('BOOLEAN',{'default':True}),
            'recover_missed':('BOOLEAN',{'default':False}),
            'grouping_radius':('INT',{'default':6,'min':0,'max':64}),
            'min_component_area':('INT',{'default':8,'min':1,'max':4096}),
            'max_new_regions':('INT',{'default':64,'min':1,'max':160})}}
    RETURN_TYPES=('KOHARU_RESULT',)
    FUNCTION='apply'
    CATEGORY='Koharu/Detection'
    def apply(self,result,mask,enabled=True,**options):
        if type(enabled) is not bool:raise ValueError('Invalid mask enable switch')
        if not enabled:return (result,)
        values=mask.detach().cpu().numpy()
        if values.ndim!=3 or values.shape[0]!=1:raise ValueError('Expected one page text mask')
        return (apply_text_mask(result,values[0],**options),)


class KoharuDetectionMaskPage:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'result':('KOHARU_RESULT',),'labels':(list(LABEL_CHOICES),)},
                'optional':{'client_nms':('BOOLEAN',{'default':False,
                    'tooltip':'Preview Koharu box NMS (IoU 0.5) before client dilation/closing. Does not alter the detection response.'})}}
    RETURN_TYPES=('MASK',)
    FUNCTION='assemble'
    CATEGORY='Koharu/Detection'
    def assemble(self,result,labels='text + onomatopoeia',client_nms=False):
        import torch
        if type(client_nms) is not bool:raise ValueError('Invalid client NMS preview switch')
        if client_nms:
            from .client_mask_audit import nms_result
            result=nms_result(result)
        return (torch.from_numpy(page_mask(result,labels)[None]),)


NODE_CLASS_MAPPINGS={'KoharuTextMask':KoharuTextMask,'KoharuTextMaskApply':KoharuTextMaskApply,
                     'KoharuDetectionMaskPage':KoharuDetectionMaskPage}
