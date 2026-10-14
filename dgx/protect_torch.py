"""Refuse dependency resolution that replaces NVIDIA's installed torch build."""

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import sys


def installed():
    result = {}
    for name in ("torch", "torchvision", "torchaudio"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    if "torch" not in result:
        raise RuntimeError("PyTorch base image does not contain torch")
    return result


if sys.argv[1] == "before":
    packages = installed()
    Path("/tmp/torch-before.json").write_text(json.dumps(packages))
    Path("/tmp/torch-constraints.txt").write_text(
        "".join(f"{k}=={v}\n" for k, v in packages.items())
    )
else:
    expected = json.loads(Path("/tmp/torch-before.json").read_text())
    if installed() != expected:
        raise RuntimeError("NVIDIA PyTorch packages changed during installation")
    print("Preserved NVIDIA torch:", expected["torch"])
