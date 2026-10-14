"""ComfyUI model-folder selection and resident adapter loader nodes."""
from pathlib import Path
import os

KINDS={'detection': ('koharu_detection','detection','load_model.py','KoharuDetectorModelLoader','KoharuDetector'),
       'ocr': ('koharu_ocr','ocr','modeling_hayai.py','KoharuOCRModelLoader','KoharuOCR'),
       'text_mask': ('koharu_text_masks','text_masks','inference.py','KoharuTextMaskModelLoader','KoharuTextMask')}
CONFIG_FILES={'detection': 'inference_config.json', 'ocr': 'config.json','text_mask':'config.json'}


def register_model_paths():
    import folder_paths
    for category,subdirectory,*_ in KINDS.values():
        folder_paths.add_model_folder_path(category,str(Path(folder_paths.models_dir)/'koharu'/subdirectory))


def model_catalog(stage):
    import folder_paths
    register_model_paths()
    category,_,marker,*_ = KINDS[stage]
    found={}
    for root in folder_paths.get_folder_paths(category):
        root=Path(root)
        pending=[root]
        seen=set()
        while pending:
            directory=pending.pop()
            if not directory.is_dir():continue
            resolved=directory.resolve()
            if (directory/marker).is_file() and (directory/'model.safetensors').is_file() and (directory/CONFIG_FILES[stage]).is_file():
                relative=directory.relative_to(root).as_posix()
                # Folder roots contain named bundles; a bundle can also be the
                # root of an explicitly configured extra-model path.
                name=directory.name if relative=='.' else relative
                found.setdefault(name,resolved)
                continue
            if resolved in seen:continue
            seen.add(resolved)
            with os.scandir(directory) as entries:
                pending.extend(sorted((Path(entry.path) for entry in entries if entry.is_dir()),reverse=True))
    return dict(sorted(found.items()))


def resolve_model(stage, name):
    if not isinstance(name,str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Select a named model bundle from the ComfyUI model folder')
    catalog=model_catalog(stage)
    if name not in catalog:
        category,subdirectory,*_ = KINDS[stage]
        raise ValueError(f'{stage} model bundle not found: {name}; place it under models/koharu/{subdirectory} or register {category} in extra_model_paths.yaml')
    return catalog[name]


def workflow_model_selection(stage, settings, workflow_stage=None):
    from .workflows import load
    _,_,_,loader_kind,consumer_kind=KINDS[stage]
    owner=workflow_stage or stage
    path=settings.get('workflows',{}).get(owner) or Path(__file__).with_name('workflows')/(owner+'.api.json')
    graph=load(path)
    consumers=[node for node in graph.values() if node['class_type']==consumer_kind]
    if not consumers:return None  # Another architecture owns its own model lifecycle.
    model_names=set()
    for node in consumers:
        link=node['inputs'].get('model')
        if not isinstance(link,list) or len(link)!=2 or link[1]!=0:
            raise ValueError(f'Connect {loader_kind} to {consumer_kind}; update the saved graph')
        loader=graph.get(str(link[0]))
        if not loader or loader['class_type']!=loader_kind:
            raise ValueError(f'{consumer_kind} requires its model loader')
        model_names.add((loader['inputs'].get('model_name'),loader['inputs'].get('precision','bf16') if stage=='text_mask' else None))
    if len(model_names)!=1:
        raise ValueError('One model selection per Koharu stage is required')
    name,precision=next(iter(model_names))
    if stage=='text_mask' and precision not in ('bf16','fp32'):raise ValueError('Invalid SAM-TS precision')
    return resolve_model(stage,name),precision


def workflow_model_path(stage,settings):
    selection=workflow_model_selection(stage,settings)
    return selection[0] if selection is not None else None


class KoharuDetectorModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        names=list(model_catalog('detection')) or ['(no models found)']
        return {'required': {'model_name': (names,)}}
    RETURN_TYPES = ('KOHARU_DETECTOR_MODEL',)
    RETURN_NAMES = ('model',)
    FUNCTION = 'load'
    CATEGORY = 'Koharu/Model Loaders'
    def load(self,model_name):
        from .runtime import backend
        return (backend('detection',resolve_model('detection',model_name)),)


class KoharuOCRModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        names=list(model_catalog('ocr')) or ['(no models found)']
        return {'required': {'model_name': (names,)}}
    RETURN_TYPES = ('KOHARU_OCR_MODEL',)
    RETURN_NAMES = ('model',)
    FUNCTION = 'load'
    CATEGORY = 'Koharu/Model Loaders'
    def load(self,model_name):
        from .runtime import backend
        return (backend('ocr',resolve_model('ocr',model_name)),)


class KoharuTextMaskModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        names=list(model_catalog('text_mask')) or ['(no models found)']
        return {'required':{'model_name':(names,),'precision':(['bf16','fp32'],)}}
    RETURN_TYPES=('KOHARU_TEXT_MASK_MODEL',)
    RETURN_NAMES=('model',)
    FUNCTION='load'
    CATEGORY='Koharu/Model Loaders'
    def load(self,model_name,precision='bf16'):
        from .runtime import backend
        return (backend('text_mask',resolve_model('text_mask',model_name),precision=precision),)


NODE_CLASS_MAPPINGS={'KoharuDetectorModelLoader':KoharuDetectorModelLoader,
                     'KoharuOCRModelLoader':KoharuOCRModelLoader,
                     'KoharuTextMaskModelLoader':KoharuTextMaskModelLoader}
