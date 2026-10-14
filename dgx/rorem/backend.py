"""Full local RORem pipeline + verified LCM LoRA, loaded once on CUDA.

Aspect ratio is preserved within the requested <=1024 canvas, then restored.
Region partitioning is owned by the Remote client, not this model backend.
"""
import logging
import os
import secrets
import time
from .model_files import validate_models, LORA_NAME
from .strategy import FullPageStrategy
from importlib.metadata import version
import platform

log = logging.getLogger(__name__)

class RORemBackend:
    name = 'RORem-4S'

    def __init__(self):
        self.model_loaded = False
        self.strategy = FullPageStrategy()
        import torch
        import diffusers
        import transformers
        import peft
        from diffusers import AutoPipelineForInpainting, LCMScheduler
        start = time.monotonic()
        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback')
        files = validate_models(os.environ['ROREM_PIPELINE'], os.environ['ROREM_LORA'])
        kwargs = dict(dtype=torch.float16, local_files_only=True, use_safetensors=True)
        if files['variant']:
            kwargs['variant'] = files['variant']
        log.info('Loading full RORem pipeline; variant=%s', files['variant'] or 'default')
        self.pipe = AutoPipelineForInpainting.from_pretrained(files['pipeline'], **kwargs)
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.load_lora_weights(files['lora'], weight_name=LORA_NAME,
                                    local_files_only=True, use_safetensors=True)
        self.pipe.fuse_lora()
        self.pipe.to('cuda')
        self.pipe.set_progress_bar_config(disable=True)
        self.device = 'cuda'
        self.gpu_name = torch.cuda.get_device_name()
        self.details = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                        'diffusers': diffusers.__version__, 'transformers': transformers.__version__,
                        'peft': peft.__version__, 'variant': files['variant'],
                        'lora_sha256': files['lora_sha256'],
                        'accelerate': version('accelerate'), 'huggingface-hub': version('huggingface-hub'),
                        'python': platform.python_version(), 'architecture': platform.machine(),
                        'strategy': self.strategy.name,
                        'load_seconds': round(time.monotonic() - start, 2)}
        self.model_loaded = True
        log.info('RORem ready on %s in %.1fs', self.gpu_name, self.details['load_seconds'])

    def infer(self, source, mask, opts):
        prepared = self.strategy.prepare(source, mask, opts.max_side, opts.mask_dilation)
        size = prepared.image.size
        seed = secrets.randbits(31) if opts.seed == -1 else opts.seed
        generator = self.torch.Generator(device='cuda').manual_seed(seed)
        log.info('Inpaint seed: requested=%d, effective=%d', opts.seed, seed)
        with self.torch.inference_mode():
            result = self.pipe(prompt=opts.prompt, negative_prompt=opts.negative_prompt,
                               image=prepared.image, mask_image=prepared.mask,
                               height=size[1], width=size[0], guidance_scale=opts.guidance_scale,
                               num_inference_steps=opts.steps, strength=opts.strength,
                               generator=generator).images[0]
        return self.strategy.restore(result, prepared)
