"""Package only after successful Actions build; this does not prove a GPU or live UI run."""

from pathlib import Path
import hashlib, json, shutil, subprocess, sys, zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sums(folder):
    lines = [
        f"{sha(p)}  {p.relative_to(folder).as_posix()}\n"
        for p in sorted(folder.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS"
    ]
    (folder / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def package(target):
    target = Path(target).resolve()
    pin = json.loads((ROOT / "upstream.json").read_text())
    head = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != pin["commit"]:
        raise RuntimeError("Built upstream commit mismatch")
    release = target / "target/release"
    if not (release / "koharu.exe").is_file():
        raise RuntimeError("Successful Windows build executable missing")
    name = f"koharu-remote-{pin['version']}-remote-{pin['patch_version']}-windows-x64"
    folder = ROOT / "dist" / name
    if folder.exists():
        raise RuntimeError("Package directory exists; use a clean Actions checkout")
    binary = folder / "bin"
    binary.mkdir(parents=True)
    shutil.copyfile(release / "koharu.exe", binary / "koharu-remote.exe")
    # Retain adjacent CEF/ML runtime files if the upstream build produces them.
    for item in release.iterdir():
        if item.is_file() and (
            item.suffix.lower() in (".dll", ".pak")
            or item.name
            in (
                "icudtl.dat",
                "v8_context_snapshot.bin",
                "snapshot_blob.bin",
                "vk_swiftshader_icd.json",
            )
        ):
            shutil.copyfile(item, binary / item.name)
        elif item.is_dir() and item.name in ("locales", "resources"):
            shutil.copytree(item, binary / item.name)
    for license in ("LICENSE-MIT", "LICENSE-APACHE"):
        shutil.copyfile(target / license, folder / license)
    for name_in in ("start-remote.ps1", "rollback-custom-settings.ps1", "README.md", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(ROOT / name_in, folder / name_in)
    info = {
        **pin,
        "source_repository": subprocess.check_output(
            ["git", "-C", str(ROOT), "config", "--get", "remote.origin.url"], text=True
        ).strip(),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "patch_sha256": sha(ROOT / "patches/remote-models.patch"),
        "cargo_lock_sha256": sha(target / "Cargo.lock"),
        "rust": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "verification": "WINDOWS_BUILD_AND_FOCUSED_TESTS_PASS_LIVE_UI_AND_GPU_NOT_RUN",
        "signed": False,
    }
    (folder / "BUILD_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
    sums(folder)
    archive = folder.parent / (folder.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as file:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                file.write(path, path.relative_to(folder.parent))
    with zipfile.ZipFile(archive) as file:
        if file.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
    archive.with_suffix(".zip.sha256").write_text(
        sha(archive) + "  " + archive.name + "\n"
    )
    print(archive)


if __name__ == "__main__":
    package(sys.argv[1])
