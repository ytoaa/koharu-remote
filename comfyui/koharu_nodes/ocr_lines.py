"""Optional line segmentation and result joining around ordinary OCR nodes."""
import logging, math

LOG = logging.getLogger(__name__)
import numpy as np


def bands(profile, min_gap, min_extent):
    indices = np.flatnonzero(profile)
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = []
    for values in np.split(indices,cuts):
        start,end = int(values[0]),int(values[-1])+1
        if groups and start-groups[-1][1] < min_gap:
            groups[-1][1] = end
        else:
            groups.append([start,end])
    return [group for group in groups if group[1]-group[0] >= min_extent]


class KoharuOCRLineCrops:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images': ('IMAGE',), 'enabled': ('BOOLEAN', {'default':True}),
            'direction': (['auto','horizontal','vertical_rtl','vertical_ltr'],),
            'polarity': (['auto','bright','dark','both'],),
            'background_window': ('INT', {'default':31,'min':3,'max':255,'step':2}),
            'contrast': ('FLOAT', {'default':16.,'min':1.,'max':128.,'step':1.}),
            'ink_fraction': ('FLOAT', {'default':0.002,'min':0.0001,'max':0.1,'step':0.001}),
            'min_gap': ('INT', {'default':4,'min':1,'max':64}),
            'min_line_extent': ('INT', {'default':8,'min':1,'max':128}),
            'padding': ('INT', {'default':2,'min':0,'max':32}),
            'max_lines': ('INT', {'default':32,'min':1,'max':64})}}
    RETURN_TYPES = ('IMAGE','KOHARU_OCR_LAYOUT')
    RETURN_NAMES = ('line_images','layout')
    OUTPUT_IS_LIST = (True,False)
    FUNCTION = 'split'
    CATEGORY = 'Koharu/OCR'
    def split(self, images, enabled=True, direction='auto', polarity='auto', background_window=31,
              contrast=16., ink_fraction=0.002, min_gap=4, min_line_extent=8, padding=2,max_lines=32):
        if len(images.shape)!=4 or images.shape[0]!=1 or images.shape[-1]<3:
            raise ValueError('Expected one RGB OCR crop')
        if type(enabled) is not bool or direction not in ('auto','horizontal','vertical_rtl','vertical_ltr') or polarity not in ('auto','bright','dark','both'):
            raise ValueError('Invalid OCR line options')
        for value,low,high in [(background_window,3,255),(min_gap,1,64),(min_line_extent,1,128),(padding,0,32),(max_lines,1,64)]:
            if type(value) is not int or not low<=value<=high:
                raise ValueError('Invalid OCR line integer option')
        if not math.isfinite(contrast) or not 1<=contrast<=128 or not math.isfinite(ink_fraction) or not 0.0001<=ink_fraction<=0.1:
            raise ValueError('Invalid OCR line contrast option')
        height,width = images.shape[1:3]
        layout={'width':width,'height':height,'direction':'unchanged','boxes':[[0,0,width,height]]}
        def unchanged(reason='insufficient_bands'):
            LOG.info('OCR line segmentation: source=%sx%s, crops=1, direction=unchanged, reason=%s',width,height,reason)
            return ([images],layout)
        if not enabled:
            return unchanged('disabled')
        if direction=='auto':
            if width >= height*1.25:direction='horizontal'
            elif height >= width*1.25:direction='vertical_rtl'
        from scipy import ndimage
        pixels = images[0].detach().cpu().numpy()[...,:3]
        if not np.isfinite(pixels).all():
            raise ValueError('OCR source contains nonfinite pixels')
        gray = pixels @ np.array([0.299,0.587,0.114],dtype=np.float32)*255.
        window=background_window if background_window%2 else background_window+1
        detail=gray-ndimage.uniform_filter(gray,size=window,mode='reflect')
        if polarity=='auto':
            positive=float(np.percentile(detail,99));negative=-float(np.percentile(detail,1))
            if max(positive,negative)<contrast:return unchanged('low_contrast')
            polarity='bright' if positive>negative*1.2 else 'dark' if negative>positive*1.2 else 'both'
        foreground=(detail>contrast if polarity=='bright' else detail < -contrast if polarity=='dark' else np.abs(detail)>contrast)
        def candidate(horizontal):
            cross=width if horizontal else height
            projection=foreground.sum(axis=1 if horizontal else 0)>=max(2,math.ceil(cross*ink_fraction))
            return bands(projection,min_gap,min_line_extent)
        if direction=='auto':
            # Near-square crops still contain rows or columns. Infer from
            # separated foreground bands instead of bypassing them on aspect
            # ratio alone. Ambiguous two-dimensional grids stay intact.
            choices=[]
            for name,horizontal in [('horizontal',True),('vertical_rtl',False)]:
                proposed=candidate(horizontal)
                if 2<=len(proposed)<=max_lines:
                    gaps=sum(right[0]-left[1] for left,right in zip(proposed,proposed[1:]))
                    span=proposed[-1][1]-proposed[0][0]
                    choices.append((gaps/span,name))
            choices.sort(reverse=True)
            if not choices:return unchanged('no_separated_rows_or_columns')
            if len(choices)==2 and (choices[0][0]<choices[1][0]*1.5 or choices[0][0]-choices[1][0]<0.05):
                return unchanged('ambiguous_rows_and_columns')
            direction=choices[0][1]
        horizontal=direction=='horizontal'
        found=candidate(horizontal)
        if not 2<=len(found)<=max_lines:return unchanged()
        # Cuts lie in detected gaps; every source pixel belongs to a crop.
        # Padding adds only context. Never discard short marks/blank margins.
        extent=height if horizontal else width
        cuts=[0]+[(left[1]+right[0])//2 for left,right in zip(found,found[1:])]+[extent]
        crops,boxes=[],[]
        for start,end in zip(cuts,cuts[1:]):
            start,end=max(0,start-padding),min(extent,end+padding)
            box=[0,start,width,end] if horizontal else [start,0,end,height]
            left,top,right,bottom=box
            crops.append(images[:,top:bottom,left:right,:])
            boxes.append(box)
        if direction=='vertical_rtl':
            crops.reverse();boxes.reverse()
        layout.update(direction=direction,boxes=boxes)
        LOG.info('OCR line segmentation: source=%sx%s, crops=%s, direction=%s',width,height,len(crops),direction)
        return (crops,layout)


class KoharuOCRJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'text': ('STRING', {'forceInput':True}), 'metadata': ('KOHARU_RESULT',),
                             'layout': ('KOHARU_OCR_LAYOUT',)}}
    RETURN_TYPES = ('STRING','KOHARU_RESULT')
    RETURN_NAMES = ('text','metadata')
    INPUT_IS_LIST = True
    FUNCTION = 'join'
    CATEGORY = 'Koharu/OCR'
    def join(self,text,metadata,layout):
        if len(layout)!=1 or len(text)!=len(layout[0]['boxes']) or len(metadata)!=len(text):
            raise ValueError('OCR line results differ from source layout')
        if any(not isinstance(value,str) or '\0' in value for value in text):
            raise ValueError('Invalid OCR line text')
        if any(not isinstance(item,dict) or item.get('protocol')!=1 or item.get('text')!=value for item,value in zip(metadata,text)):
            raise ValueError('Invalid OCR line metadata')
        identity={(item.get('model','workflow/ocr'),item.get('revision','workflow')) for item in metadata}
        if len(identity)!=1:
            raise ValueError('OCR lines used different model identities')
        merged='\n'.join(text)
        if len(merged.encode('utf-8'))>65536:
            raise ValueError('Joined OCR text exceeds result limit')
        model,revision=next(iter(identity))
        result={'protocol':1,'text':merged,'model':model,'revision':revision,
                'line_layout':layout[0], 'line_recoveries':[item.get('recovery') for item in metadata]}
        return (merged,result)


NODE_CLASS_MAPPINGS={'KoharuOCRLineCrops':KoharuOCRLineCrops,'KoharuOCRJoin':KoharuOCRJoin}
