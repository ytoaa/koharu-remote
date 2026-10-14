"""No keys or complete environment dump. Records the new container only."""

import json, os, platform
from importlib.metadata import version, PackageNotFoundError
from .model import REPO, REV, FILES, validate
from . import hayai_files


def main():
    import torch

    folder = validate(os.environ["VISION_MODEL_DIR"])
    ocr_folder = hayai_files.validate(os.environ["VISION_OCR_MODEL_DIR"])
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "torchaudio",
        "rfdetr",
        "transformers",
        "supervision",
        "numpy",
        "safetensors",
        "fastapi",
        "httpx",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    print(
        json.dumps(
            {
                "base_image": os.environ.get("VISION_BASE_IMAGE"),
                "python": platform.python_version(),
                "architecture": platform.machine(),
                "packages": packages,
                "cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name()
                if torch.cuda.is_available()
                else None,
                "model_path": str(folder),
                "model_repository": REPO,
                "model_revision": REV,
                "model_files": FILES,
                "ocr_model_path": str(ocr_folder),
                "ocr_repository": hayai_files.REPO,
                "ocr_revision": hayai_files.REV,
                "ocr_model_files": hayai_files.FILES,
                "verification": "DIAGNOSTICS_ONLY_INFERENCE_NOT_PROVEN",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
