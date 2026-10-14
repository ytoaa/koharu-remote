"""Pinned checkpoint identity and complete offline integrity checks, without torch."""
import hashlib
import json
from pathlib import Path

MANIFEST = json.loads(Path(__file__).with_name('model_manifest.json').read_text())
REPO = MANIFEST['repo']
MODEL = REPO.split('/', 1)[1]
REVISION = MANIFEST['revision']
DIFFUSERS_COMMIT = MANIFEST['diffusers_commit']
SAMPLE_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


def validate_models(directory):
    root = Path(directory).resolve()
    for name, expected in MANIFEST['files'].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected['size']:
            raise ValueError('Missing or incomplete pinned Qwen file: '+name)
        digest = hashlib.sha256() if 'sha256' in expected else hashlib.sha1()
        if 'git_blob_sha1' in expected:
            digest.update(f'blob {expected["size"]}\0'.encode())
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):
                digest.update(block)
        if digest.hexdigest() != expected.get('sha256',expected.get('git_blob_sha1')):
            raise ValueError('Pinned Qwen file checksum mismatch: '+name)
    index = json.loads((root/'model_index.json').read_text())
    if index.get('_class_name') != 'QwenImage21Pipeline' or index.get('sample_sigmas') != SAMPLE_SIGMAS:
        raise ValueError('Expected the complete Qwen Image 2.1 Turbo pipeline and eight-step grid')
    return {'path':str(root),'repo':REPO,'revision':REVISION}
