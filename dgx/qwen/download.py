"""Explicit staging-only download. The serving container mounts this folder read-only."""
import os
from huggingface_hub import snapshot_download
from .model_files import MANIFEST, REPO, REVISION, validate_models


def main():
    root = os.environ.get('QWEN_DOWNLOAD_DIR','/download')
    snapshot_download(REPO, revision=REVISION, local_dir=root,
                      allow_patterns=list(MANIFEST['files']), max_workers=4)
    validate_models(root)
    print('QWEN_CHECKPOINT_VERIFIED: '+REPO+'@'+REVISION,flush=True)


if __name__ == '__main__':
    main()
