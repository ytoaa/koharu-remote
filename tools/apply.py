"""Apply only to a disposable, exact-revision checkout. Never installed Koharu."""

from pathlib import Path
import json, shutil, subprocess, sys, tempfile, tomllib

ROOT = Path(__file__).resolve().parents[1]


def apply(target):
    target = Path(target).resolve()
    pin = json.loads((ROOT / "upstream.json").read_text())
    head = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != pin["commit"]:
        raise RuntimeError("Upstream commit mismatch")
    version = tomllib.loads((target / "Cargo.toml").read_text())["workspace"][
        "package"
    ]["version"]
    if version != pin["version"]:
        raise RuntimeError("Upstream source/release version mismatch")
    if subprocess.check_output(
        ["git", "-C", str(target), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError(
            "Build checkout must be pristine; no reset/delete is performed"
        )
    patch = ROOT / "patches/remote-models.patch"
    modules = {
        "remote.rs": "crates/koharu-pipeline/src/remote.rs",
        "custom.rs": "crates/koharu-app/src/commands/custom.rs",
        "CustomPreferences.tsx": "packages/koharu/components/preferences/CustomPreferences.tsx",
    }
    for source, destination in modules.items():
        if not (ROOT / "patches" / source).is_file():
            raise RuntimeError("Missing patch module")
        if (target / destination).exists():
            raise RuntimeError("Unexpected existing module: " + destination)
    # Git for Windows may check out the patch artifact with CRLF. Normalize the
    # artifact, not upstream source, before both validation and application.
    with tempfile.TemporaryDirectory(prefix="koharu-remote-patch-") as temporary:
        normalized = Path(temporary) / "remote-models.patch"
        normalized.write_text(patch.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        subprocess.run(
            ["git", "-C", str(target), "apply", "--check", str(normalized)], check=True
        )
        subprocess.run(["git", "-C", str(target), "apply", str(normalized)], check=True)
    for source, destination in modules.items():
        shutil.copyfile(ROOT / "patches" / source, target / destination)
    print("REMOTE_PATCH_APPLIED_BUILD_NOT_RUN")


if __name__ == "__main__":
    apply(sys.argv[1])
