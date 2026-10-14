"""Pinned SAM-TS-L release metadata; no GPU or optional imports."""
REPO='mayocream/koharu-text-sam-ts-l'
REV='5dd97423e0fbf2404264979136d47e8101144046'
FILES={
    'model.safetensors':'bcd9525291677f467f0603509a0ca3df35711b4e3417cefce8da6bfc97164f45',
    'config.json':'fcc8213df069f728e0f688c9e26d63330afbeeecd7d5dccca129bce09df68f52',
    'inference.py':'66fe733408dc812ec45fa9adf8f5904f831ab6445f7db5f8b06bbcb40eba1565',
}
URLS={name:f'https://huggingface.co/{REPO}/resolve/{REV}/{name}' for name in FILES}


def validate(folder):
    from .model_bundle import validate_bundle
    return validate_bundle(folder,REPO,REV,FILES,('config.json','inference.py'))
