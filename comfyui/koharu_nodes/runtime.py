"""Request ownership and resident adapters shared by routes and queued nodes."""
from pathlib import Path
from dataclasses import dataclass, field
import json, os, threading, time
from dgx.qwen.comfy_backend import DEFAULT_MODELS

@dataclass
class Job:
    image: object
    mask: object = None
    options: object = None
    result: object = None
    workflow: object = None
    selected_models: dict = field(default_factory=dict)
    error: str | None = None
    finished: threading.Event = field(default_factory=threading.Event)
    created: float = field(default_factory=time.monotonic)

jobs = {}
backends = {}
ready = set()
loaded_models = {}
lock = threading.Lock()

def config():
    path = Path(os.environ.get('KOHARU_COMFY_CONFIG', Path(__file__).with_name('config.json')))
    values = json.loads(path.read_text()) if path.exists() else {}
    allowed = {'models', 'detector_path', 'ocr_path', 'timeout', 'api_url', 'workflows'}
    if set(values) - allowed:
        raise ValueError('Unknown Koharu configuration key')
    workflows = values.get('workflows', {})
    if not isinstance(workflows, dict) or set(workflows) - {'inpainting','detection','ocr'}:
        raise ValueError('Invalid workflow stage mapping')
    workflows = {k: str((path.parent / v).resolve()) for k,v in workflows.items()}
    values['workflows'] = workflows
    models = {**DEFAULT_MODELS, **values.get('models', {})}
    if set(models) != set(DEFAULT_MODELS) or any(not isinstance(v, str) or not v for v in models.values()):
        raise ValueError('Invalid Comfy model selection')
    from .workflows import profile_path
    for stage in ('inpainting','detection','ocr'):
        saved = profile_path(stage)
        if saved.exists():
            workflows[stage] = str(saved)
    if 'inpainting' in workflows:
        from .workflows import load, models as graph_models
        models = graph_models(load(workflows['inpainting']), models)
    timeout = values.get('timeout', 1800)
    if type(timeout) is not int or not 30 <= timeout <= 7200:
        raise ValueError('Timeout must be 30..7200')
    return {**values, 'models': models, 'timeout': timeout,
            'api_url': values.get('api_url', 'http://127.0.0.1:8188')}

def backend(stage, folder, precision=None):
    if stage not in ('detection','ocr','text_mask'):
        raise ValueError('Invalid model stage')
    path = folder
    key = (stage, str(Path(path).resolve()))
    if stage=='text_mask':
        precision=precision or 'bf16'
        if precision not in ('bf16','fp32'):raise ValueError('Invalid SAM-TS precision')
        key=(*key,precision)
    # Called only from the Comfy executor. No CUDA work in HTTP handlers.
    if key not in backends:
        if stage == 'detection':
            from dgx.vision.model import Backend
        elif stage=='ocr':
            from dgx.vision.hayai import Backend
        else:
            from dgx.vision.text_sam import Backend
        candidate = Backend(path,precision=precision) if stage=='text_mask' else Backend(path)
        for old in list(backends):
            if old[0] == stage:
                del backends[old]
        backends[key] = candidate
    return backends[key]

def get_job(request_id):
    with lock:
        return jobs[request_id]


def is_ready(stage):
    if stage not in ready:
        return False
    settings = config()
    if stage == 'inpainting':
        return loaded_models == settings['models']
    from .loaders import workflow_model_path,workflow_model_selection
    try:
        selected=workflow_model_path(stage,settings)
        if stage=='detection':
            auxiliary=workflow_model_selection('text_mask',settings,workflow_stage='detection')
            if auxiliary is not None and ('text_mask',str(auxiliary[0]),auxiliary[1]) not in backends:
                return False
    except (ValueError,OSError,KeyError):
        return False
    return selected is None or (stage,str(selected)) in backends
