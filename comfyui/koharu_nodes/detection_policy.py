"""User-selected detection classes at the installed client's protocol boundary."""
import logging
from .detection_masks import page_size

LOG = logging.getLogger(__name__)


def apply_policy(result, include_text=True, effects='keep', include_bubbles=True, include_panels=True):
    page_size(result)
    if any(type(v) is not bool for v in (include_text,include_bubbles,include_panels)) or effects not in ('keep','as_text','exclude'):
        raise ValueError('Invalid detection label policy')
    items = []
    promoted = excluded = 0
    keep = {'text':include_text,'bubble':include_bubbles,'panel':include_panels}
    for item in result['detections']:
        label = item['label']
        if label == 'onomatopoeia':
            if effects == 'exclude':
                excluded += 1
                continue
            if effects == 'as_text':
                # The existing client uses text for both OCR and auto cleanup.
                # Retain original identity for inspection; do not fabricate an
                # inpaint-only mode the installed protocol cannot represent.
                item = {**item,'label_id':0,'label':'text',
                        'original_label':'onomatopoeia','original_label_id':1}
                promoted += 1
        elif label in keep and not keep[label]:
            excluded += 1
            continue
        items.append(item)
    output = {**result,'detections':items,'detection_policy':{
        'include_text':include_text,'effects':effects,'include_bubbles':include_bubbles,
        'include_panels':include_panels,'promoted_effects':promoted,'excluded':excluded}}
    LOG.info('Koharu detection policy: effects=%s promoted=%s excluded=%s output=%s',
             effects,promoted,excluded,len(items))
    return output


class KoharuDetectionPolicy:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required':{'result':('KOHARU_RESULT',),
            'include_text':('BOOLEAN',{'default':True}),
            'effects':(['keep','as_text','exclude'],{'default':'keep'}),
            'include_bubbles':('BOOLEAN',{'default':True}),
            'include_panels':('BOOLEAN',{'default':True})}}
    RETURN_TYPES = ('KOHARU_RESULT',)
    FUNCTION = 'apply'
    CATEGORY = 'Koharu/Detection'
    def apply(self,result,**options):
        return (apply_policy(result,**options),)


NODE_CLASS_MAPPINGS = {'KoharuDetectionPolicy':KoharuDetectionPolicy}
