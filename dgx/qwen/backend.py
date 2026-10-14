"""Qwen reference-image + reference-mask editing, followed by exact mask compositing.

QwenImage21Pipeline does NOT accept mask_image/strength. A separate binary mask
is a second reference image, interpreted through the editing instruction. The
saved Turbo sigma grid is authoritative; no call-time step/sigma override.
"""
import logging
import math
import numpy as np
import os
import secrets
import time
from types import SimpleNamespace
from PIL import Image, ImageFilter
from .model_files import validate_models, REPO, REVISION, DIFFUSERS_COMMIT, SAMPLE_SIGMAS
from .options import MODEL, Options

log = logging.getLogger('uvicorn.error')
MASK_INSTRUCTION = ('Image 1 is the original artwork. Image 2 is a binary selection mask '
                    'at the same coordinates: white means edit, black means preserve. '
                    'Edit Image 1 only in the white regions of Image 2. Remove source '
                    'lettering and fill those regions with the surrounding background. '
                    'Return the edited artwork, not the mask. Do not add text or annotations. ')


def prepare(source, mask, max_side, dilation):
    # Letterbox to one square so the pipeline's area-based reference resizing
    # preserves exact source/mask alignment and obeys the selected long-side cap.
    scale = min(1.0, max_side / max(source.size))
    size = tuple(max(1, round(n * scale)) for n in source.size)
    image = source.resize(size, Image.Resampling.LANCZOS).convert('RGBA')
    scaled_mask = mask.resize(size, Image.Resampling.NEAREST)
    if mask.getbbox() and not scaled_mask.getbbox():
        coverage = np.asarray(mask.convert('F').resize(size, Image.Resampling.BOX))
        scaled_mask = Image.fromarray(np.where(coverage > 0, 255, 0).astype(np.uint8))
    if dilation:
        scaled_mask = scaled_mask.filter(ImageFilter.MaxFilter(2 * dilation + 1))
    side = max(256, math.ceil(max(size) / 32) * 32)
    # max_side is aligned down; never let letterboxing exceed the configured cap.
    if side > max_side:
        return prepare(source, mask, max_side // 32 * 32, dilation)
    canvas = Image.new('RGBA', (side, side), (255, 255, 255, 255))
    canvas.paste(image, (0, 0))
    mask_canvas = Image.new('L', (side, side), 0)
    mask_canvas.paste(scaled_mask, (0, 0))
    return canvas, mask_canvas.convert('RGBA'), size


class QwenBackend:
    name = MODEL

    def __init__(self):
        self.model_loaded = False
        import torch
        import diffusers
        from diffusers import QwenImage21Pipeline
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback')
        files = validate_models(os.environ['QWEN_MODEL_DIR'])
        start = time.monotonic()
        self.torch = torch
        self.pipe = QwenImage21Pipeline.from_pretrained(files['path'], dtype=torch.bfloat16,
                                                       local_files_only=True)
        if list(self.pipe.config.sample_sigmas or []) != SAMPLE_SIGMAS:
            raise RuntimeError('Qwen Turbo sampling schedule mismatch')
        self.pipe.to('cuda')
        self.pipe.set_progress_bar_config(disable=True)
        self.device = 'cuda'
        self.gpu_name = torch.cuda.get_device_name()
        self.strategy = SimpleNamespace(name='reference-mask-edit')
        # Run all eight saved steps before becoming ready; use a small marked fixture.
        fixture = Image.new('RGB', (256, 256), 'white')
        mask = Image.new('L', fixture.size, 0)
        from PIL import ImageDraw
        ImageDraw.Draw(fixture).text((100, 100), 'TEST', fill='black')
        ImageDraw.Draw(mask).rectangle((96, 96, 136, 116), fill=255)
        warmup = time.monotonic()
        self.infer(fixture, mask, Options(max_side=256))
        self.details = {'torch':torch.__version__, 'cuda':torch.version.cuda,
                        'diffusers':diffusers.__version__, 'diffusers_commit':DIFFUSERS_COMMIT,
                        'model_repo':REPO, 'model_revision':REVISION,
                        'dtype':'bfloat16', 'steps':8, 'cfg':1.0,
                        'mask_mode':'reference-image-instruction',
                        'warmup_complete':True, 'warmup_seconds':round(time.monotonic()-warmup,3),
                        'load_seconds':round(time.monotonic()-start,3)}
        self.model_loaded = True
        log.info('Qwen Turbo loaded and warmed up on %s', self.gpu_name)

    def validate_options(self, opts):
        if opts.model not in ('',MODEL) or opts.text_encoder or opts.vae:
            raise ValueError('Diffusers server uses its resident pinned model; file selection needs ComfyUI backend')

    def infer(self, source, mask, opts):
        self.validate_options(opts)
        canvas, reference_mask, size = prepare(source, mask, opts.max_side, opts.mask_dilation)
        seed = secrets.randbits(31) if opts.seed == -1 else opts.seed
        generator = self.torch.Generator(device='cuda').manual_seed(seed)
        log.info('Qwen inpaint seed: requested=%d, effective=%d, canvas=%d',opts.seed,seed,canvas.width)
        with self.torch.inference_mode():
            result = self.pipe(prompt=MASK_INSTRUCTION + opts.prompt,
                               image=[canvas, reference_mask],
                               width=canvas.width, height=canvas.height,
                               output_resolution=canvas.width, true_cfg_scale=1.0,
                               use_kv_cache=opts.use_kv_cache, generator=generator).images[0]
        if result.size != canvas.size:
            raise RuntimeError('Qwen returned unexpected canvas dimensions')
        return result.crop((0, 0, *size)).resize(source.size, Image.Resampling.LANCZOS).convert('RGB')
