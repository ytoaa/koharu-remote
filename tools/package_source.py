"""Exclude all build outputs, private state, .env and weights from the source deliverable."""

from pathlib import Path
import json, shutil, zipfile
from package_windows import sums, sha
from verify_sums import verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    pin = json.loads((ROOT / "upstream.json").read_text())
    name = "koharu-remote-v" + pin["patch_version"] + "-source"
    folder = ROOT / "dist" / name
    if folder.exists():
        raise RuntimeError("Source package exists; choose a clean staging directory")
    folder.mkdir(parents=True)
    for directory in ("patches", "tools", "dgx", "comfyui", "tests", ".github"):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if any(
                part in ("__pycache__", ".pytest_cache", "state", "profiles", ".venv")
                for part in rel.parts
            ):
                continue
            if rel.as_posix() == "comfyui/koharu_nodes/config.json":
                continue
            if path.name.startswith(".env") and path.name != ".env.example":
                continue
            if path.suffix in (".pyc", ".safetensors", ".dll", ".exe", ".zip"):
                continue
            output = folder / rel
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)
    for filename in (
        "upstream.json",
        "tauri.remote.json",
        "start-remote.ps1",
        "rollback-custom-settings.ps1",
        "README.md",
        "CHANGES.md",
        "VALIDATION.md",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "LICENSE-MIT",
        "LICENSE-APACHE",
        ".gitignore",
        ".gitattributes",
    ):
        shutil.copyfile(ROOT / filename, folder / filename)
    sums(folder)
    verify(folder)
    archive = folder.parent / (name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as file:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                file.write(path, path.relative_to(folder.parent))
    with zipfile.ZipFile(archive) as file:
        if file.testzip() is not None:
            raise ValueError("Invalid ZIP")
        forbidden = [
            name
            for name in file.namelist()
            if Path(name).name == ".env" or Path(name).suffix == ".safetensors"
        ]
        if forbidden:
            raise ValueError("Private/model files found")
    archive.with_suffix(".zip.sha256").write_text(
        sha(archive) + "  " + archive.name + "\n"
    )
    print(archive)
    print("ZIP_SHA256: " + sha(archive))


if __name__ == "__main__":
    main()
