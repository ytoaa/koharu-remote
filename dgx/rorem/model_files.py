"""Offline validation; this module has no torch or hub dependency."""
from pathlib import Path
import hashlib
import json

LORA_NAME = 'pytorch_lora_weights.safetensors'
LORA_SHA256 = '5af337153ac9c622a551420ac23d2c802c49708ed227a6d4f4171c72684878c4'

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def weight_paths(component, variant):
    # Both diffusers and transformers support sharded safetensors.
    stems = ('diffusion_pytorch_model', 'model')
    suffix = '.fp16' if variant == 'fp16' else ''
    for stem in stems:
        whole = component / f'{stem}{suffix}.safetensors'
        indices = [component / f'{stem}.safetensors.index{suffix}.json',
                   component / f'{stem}{suffix}.safetensors.index.json',
                   component / f'{stem}.safetensors{suffix}.index.json']
        if whole.is_file() and whole.stat().st_size > 8:
            return [whole]
        for index in indices:
            if not index.is_file():
                continue
            data = json.loads(index.read_text())
            names = set(data.get('weight_map', {}).values())
            if not names:
                continue
            paths = [component / name for name in sorted(names)]
            if all(p.parent == component and p.is_file() and p.stat().st_size > 8 for p in paths):
                return paths
    return []

def validate_models(pipeline, lora, expected_hash=LORA_SHA256):
    pipeline, lora = Path(pipeline).resolve(), Path(lora).resolve()
    for label,path in [('RORem-HF',pipeline),('RORem-LCM',lora)]:
        if not path.is_dir(): raise ValueError(f'Missing model directory: {label}: {path}')
    weight = lora / LORA_NAME
    if not weight.is_file(): raise ValueError(f'Missing LCM LoRA: {weight}')
    index = json.loads((pipeline / 'model_index.json').read_text())
    if index.get('_class_name') != 'StableDiffusionXLInpaintPipeline':
        raise ValueError('Expected full RORem StableDiffusionXLInpaintPipeline, not a UNet-only directory')
    required = ['scheduler/scheduler_config.json', 'tokenizer/tokenizer_config.json',
                'tokenizer_2/tokenizer_config.json', 'unet/config.json', 'vae/config.json',
                'text_encoder/config.json', 'text_encoder_2/config.json']
    for name in required:
        path = pipeline / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing pipeline component: {name}')
        json.loads(path.read_text())
    components = [pipeline / c for c in ('unet', 'vae', 'text_encoder', 'text_encoder_2')]
    variant = None
    for candidate in (None, 'fp16'):
        groups = [weight_paths(c, candidate) for c in components]
        if all(groups):
            variant = candidate
            break
    else:
        raise ValueError('Missing/incomplete safetensors: all four components need the same default or fp16 variant')
    actual_hash = sha256(weight)
    if actual_hash != expected_hash:
        raise ValueError(f'LCM LoRA SHA256 mismatch: expected {expected_hash}, got {actual_hash}')
    return {'pipeline': str(pipeline), 'lora': str(lora), 'variant': variant,
            'lora_sha256': actual_hash,
            'pipeline_bytes': sum(p.stat().st_size for g in groups for p in g),
            'lora_bytes': weight.stat().st_size}
