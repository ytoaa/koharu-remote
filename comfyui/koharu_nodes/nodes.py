"""All model work runs on ComfyUI's executor; output is tied to one request."""
from io import BytesIO
import logging, hashlib, json
import numpy as np
from PIL import Image
from .runtime import get_job, ready, loaded_models

def pil_image(images):
    if images.shape[0] != 1:
        raise ValueError('Expected one image')
    pixels = np.clip(images[0].detach().cpu().numpy() * 255, 0, 255).round().astype(np.uint8)
    return Image.fromarray(pixels).convert('RGB')

class KoharuInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'request_id': ('STRING', {'default': ''})},
                'optional': {'image': ('IMAGE',), 'mask': ('MASK',)}}
    RETURN_TYPES = ('IMAGE','MASK','STRING')
    RETURN_NAMES = ('image','mask','request_id')
    FUNCTION = 'load'
    CATEGORY = 'Koharu/Input'
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')
    def load(self, request_id, image=None, mask=None):
        import torch
        if not request_id:
            if image is None:
                raise ValueError('Connect a standard LoadImage for manual GUI testing')
            if image.shape[0] != 1:
                raise ValueError('Expected one source image')
            # Core LoadImage uses a 64x64 zero mask for RGB files. Represent
            # that absent mask at image dimensions; never resize a real mask.
            shape = (1, image.shape[1], image.shape[2])
            if mask is None or (tuple(mask.shape) != shape and not mask.any().item()):
                mask = torch.zeros(shape, dtype=image.dtype, device=image.device)
            if tuple(mask.shape) != shape:
                raise ValueError('Source mask must match image dimensions')
            return (image, mask, '')
        job = get_job(request_id)
        mask = job.mask if job.mask is not None else Image.new('L',job.image.size)
        return (torch.from_numpy(np.asarray(job.image.convert('RGB'),dtype=np.float32)/255.0)[None],
                torch.from_numpy(np.asarray(mask,dtype=np.float32)/255.0)[None],request_id)

class KoharuDetector:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images':('IMAGE',),'request_id':('STRING',{'default':''}),
            'model':('KOHARU_DETECTOR_MODEL',),
            **{k:('FLOAT',{'default':v,'min':0.,'max':1.,'step':0.01}) for k,v in
               {'text':0.25,'onomatopoeia':0.20,'bubble':0.50,'panel':0.50}.items()}}}
    RETURN_TYPES = ('KOHARU_RESULT',)
    FUNCTION = 'detect'
    CATEGORY = 'Koharu/Detection'
    def detect(self, images, model, request_id='', **thresholds):
        image = pil_image(images)
        digest=hashlib.sha256(image.tobytes()).hexdigest()[:16]
        logging.getLogger(__name__).info('Koharu detector input: request=%s, source=%sx%s, rgb_sha256=%s',
            request_id or 'manual',image.width,image.height,digest)
        return (model.detect(image,thresholds),)

class KoharuOCR:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'images':('IMAGE',),'request_id':('STRING',{'default':''}),
            'model':('KOHARU_OCR_MODEL',),
            'max_new_tokens':('INT',{'default':256,'min':1,'max':2048}),
            'repetition_penalty':('FLOAT',{'default':1.,'min':1.,'max':2.,'step':0.01}),
            'num_beams':('INT',{'default':1,'min':1,'max':4}),
            'length_penalty':('FLOAT',{'default':1.,'min':0.1,'max':2.,'step':0.1}),
            'early_stopping':('BOOLEAN',{'default':True})}, 'optional': {
            'max_patches':([256,512,1024], {'default':256}),
            'retry_on_repetition':('BOOLEAN',{'default':True}),
            'retry_repetition_penalty':('FLOAT',{'default':1.1,'min':1.,'max':2.,'step':0.01})}}
    RETURN_TYPES = ('STRING','KOHARU_RESULT')
    RETURN_NAMES = ('text','metadata')
    FUNCTION = 'ocr'
    CATEGORY = 'Koharu/OCR'
    def ocr(self, images, model, request_id='', max_patches=256, retry_on_repetition=True, retry_repetition_penalty=1.1, **values):
        from dgx.vision.options import OcrOptions
        if type(max_patches) is not int or max_patches not in (256,512,1024):
            raise ValueError('Hayai max_patches must be 256, 512 or 1024')
        from dgx.vision.hayai import HayaiOutputError
        image = pil_image(images)
        options = OcrOptions(**values)
        # Validate retry controls before the first inference, including API graphs.
        retry_options = OcrOptions(**{**values, 'repetition_penalty': retry_repetition_penalty})
        if not isinstance(retry_on_repetition, bool):
            raise ValueError('retry_on_repetition must be boolean')
        try:
            result = model.ocr(image, options, max_patches=max_patches)
        except HayaiOutputError as exc:
            if exc.code != 'ocr_repetition_loop' or not retry_on_repetition or retry_options.repetition_penalty <= options.repetition_penalty:
                raise
            logging.getLogger(__name__).warning(
                'Hayai repetition detected: crop=%sx%s; retry once with repetition_penalty=%s',
                image.width, image.height, retry_options.repetition_penalty)
            result = model.ocr(image, retry_options, max_patches=max_patches)
            result = {**result, 'recovery': {'attempts': 2, 'repetition_penalty': retry_options.repetition_penalty}}
            logging.getLogger(__name__).info('Hayai repetition retry completed')
        return (result['text'],result)

class KoharuDetectionOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'request_id':('STRING',{'default':''}), 'result':('KOHARU_RESULT',)}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'finish'
    CATEGORY = 'Koharu/Output'
    def finish(self, request_id, result):
        if not isinstance(result,dict) or result.get('protocol') != 1 or not isinstance(result.get('detections'),list):
            raise ValueError('Invalid detection result')
        try:
            from .client_mask_audit import audit
            report=audit(result)
            log=logging.getLogger(__name__)
            method=log.warning if report['lost_text_pixels'] else log.info
            method('Koharu client mask audit: request=%s %s',request_id or 'manual',
                   json.dumps(report,separators=(',',':')))
        except (ValueError,KeyError,TypeError) as exc:
            # An optional diagnostic must never change the HTTP payload/result.
            logging.getLogger(__name__).warning('Koharu client mask audit unavailable: %s',exc)
        if request_id:
            job = get_job(request_id)
            if (result.get('image_width'),result.get('image_height')) != job.image.size:
                raise ValueError('Detection output must use original image coordinates')
            job.result = result
            ready.add('detection');job.finished.set()
        preview = {k:v for k,v in result.items() if k != 'detections'}
        preview['detections'] = [{k:v for k,v in item.items() if k != 'mask'} for item in result['detections']]
        return {'ui':{'koharu':[preview]}}

class KoharuOCROutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'request_id':('STRING',{'default':''}),'text':('STRING',{'forceInput':True}),
             'model_id':('STRING',{'default':'workflow/ocr'}),'revision':('STRING',{'default':'workflow'})},
             'optional':{'metadata':('KOHARU_RESULT',)}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = 'finish'
    CATEGORY = 'Koharu/Output'
    def finish(self, request_id, text, model_id='workflow/ocr', revision='workflow', metadata=None):
        if not isinstance(text,str) or len(text.encode('utf-8'))>65536 or '\0' in text:
            raise ValueError('Invalid OCR text')
        result = {**(metadata or {}),'protocol':1,'text':text}
        result.setdefault('model',model_id);result.setdefault('revision',revision)
        if request_id:
            job=get_job(request_id);job.result=result
            ready.add('ocr');job.finished.set()
        return {'ui':{'text':[text]}}

class KoharuMaskFillHoles:
    """Fill bounded interior holes; morphology and compositing use core nodes."""
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'mask': ('MASK',), 'enabled': ('BOOLEAN', {'default': True}),
                'max_hole_area': ('INT', {'default': 4096, 'min': 0, 'max': 33554432})}}
    RETURN_TYPES = ('MASK',)
    FUNCTION = 'fill'
    CATEGORY = 'Koharu/Mask'
    def fill(self, mask, enabled=True, max_hole_area=4096):
        if not enabled or not max_hole_area:
            return (mask,)
        import torch
        from scipy import ndimage
        values = mask.detach().cpu().numpy()
        if values.ndim != 3 or not np.isfinite(values).all():
            raise ValueError('Expected finite batched masks')
        result = values.copy()
        for index, value in enumerate(values):
            labels, count = ndimage.label(value <= 0.5)
            sizes = np.bincount(labels.ravel(), minlength=count + 1)
            eligible = (sizes <= max_hole_area)
            eligible[0] = False
            border = np.unique(np.concatenate((labels[0], labels[-1], labels[:,0], labels[:,-1])))
            eligible[border] = False
            result[index][eligible[labels]] = 1.0
        return (torch.from_numpy(result).to(device=mask.device, dtype=mask.dtype),)


class KoharuInpaintOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'request_id': ('STRING', {'default': ''}), 'images': ('IMAGE',)}}
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('final_image',)
    OUTPUT_NODE = True
    FUNCTION = 'finish'
    CATEGORY = 'Koharu/Output'
    def finish(self, request_id, images):
        # Standard graph nodes own all pixel operations. Keep the RGB PNG
        # response contract consumed by the installed Koharu client.
        job = get_job(request_id) if request_id else None
        try:
            output = pil_image(images)
            if job is not None and output.size != job.image.size:
                raise ValueError('Final image must match request dimensions; resize/composite in the graph')
            if job is not None:
                buffer = BytesIO()
                output.save(buffer, format='PNG')
                job.result = buffer.getvalue()
                loaded_models.clear()
                loaded_models.update(job.selected_models)
                ready.add('inpainting')
            return {'ui': {'koharu': [{'width': output.width, 'height': output.height}]},
                    'result': (images,)}
        except Exception as exc:
            if job is not None:
                job.error = str(exc)
            raise
        finally:
            if job is not None:
                job.finished.set()

NODE_CLASS_MAPPINGS = {'KoharuInput': KoharuInput, 'KoharuInpaintOutput': KoharuInpaintOutput, 'KoharuMaskFillHoles': KoharuMaskFillHoles}

NODE_CLASS_MAPPINGS.update({'KoharuDetector':KoharuDetector,'KoharuOCR':KoharuOCR,'KoharuDetectionOutput':KoharuDetectionOutput,'KoharuOCROutput':KoharuOCROutput})

from .detection_masks import NODE_CLASS_MAPPINGS as DETECTION_MASK_NODES
NODE_CLASS_MAPPINGS.update(DETECTION_MASK_NODES)

from .ocr_lines import NODE_CLASS_MAPPINGS as OCR_LINE_NODES
NODE_CLASS_MAPPINGS.update(OCR_LINE_NODES)

from .loaders import NODE_CLASS_MAPPINGS as MODEL_LOADER_NODES
NODE_CLASS_MAPPINGS.update(MODEL_LOADER_NODES)

from .local_color import NODE_CLASS_MAPPINGS as LOCAL_COLOR_NODES
NODE_CLASS_MAPPINGS.update(LOCAL_COLOR_NODES)

from .text_masks import NODE_CLASS_MAPPINGS as TEXT_MASK_NODES
NODE_CLASS_MAPPINGS.update(TEXT_MASK_NODES)

from .detection_merge import NODE_CLASS_MAPPINGS as DETECTION_MERGE_NODES
NODE_CLASS_MAPPINGS.update(DETECTION_MERGE_NODES)

from .detection_policy import NODE_CLASS_MAPPINGS as DETECTION_POLICY_NODES
NODE_CLASS_MAPPINGS.update(DETECTION_POLICY_NODES)

from .ui_metadata import configure as configure_ui_metadata
NODE_DISPLAY_NAME_MAPPINGS = configure_ui_metadata(NODE_CLASS_MAPPINGS)
