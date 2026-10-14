from rorem.app import create_app as shared_app
import os
from .options import Options, MODEL
from .model_files import REVISION


def create_app(factory=None, token=None):
    if factory is None:
        if os.environ.get('QWEN_BACKEND','diffusers') == 'comfyui':
            from .comfy_backend import ComfyBackend
            factory = ComfyBackend
        else:
            from .backend import QwenBackend
            factory = QwenBackend
    # Async preload is performed by the shared application's lifespan. Its
    # injectable factories are also run in the worker thread for resident CUDA loading.
    return shared_app(factory, token, options_type=Options, service_name='qwen-inpaint',
                      model_name=MODEL, service_version='0.6.0', strategy_name='comfy-instruction-edit-mask-composite' if os.environ.get('QWEN_BACKEND')=='comfyui' else 'reference-mask-edit',
                      extra_capabilities={'qwen_image21_turbo':True,
                                          'model_revision':REVISION,
                                          'reference_mask_edit':True,
                                          'configurable_models':os.environ.get('QWEN_BACKEND') == 'comfyui'})


app = create_app()
