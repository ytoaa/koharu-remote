from pathlib import Path
import json, os, subprocess
import pytest, yaml

ROOT = Path(__file__).resolve().parents[1]


def test_scope_and_no_generated_protocol_patch():
    patch = (ROOT / "patches/remote-models.patch").read_text()
    paths = [line[6:] for line in patch.splitlines() if line.startswith("+++ b/")]
    assert len(paths) == 9
    assert "packages/bridge/src/protocol.ts" not in paths
    assert "crates/koharu-app/src/commands/processing.rs" not in paths
    assert not any("storage" in path or "scene/" in path for path in paths)


def test_remote_build_is_separate_and_official_updater_disabled():
    cfg = json.loads((ROOT / "tauri.remote.json").read_text())
    assert cfg["identifier"] == "dev.koharu.remote"
    assert (
        cfg["plugins"]["updater"]["endpoints"] == []
        and cfg["bundle"]["windows"]["signCommand"] is None
    )


def test_compose_does_not_manage_existing_services():
    cfg = yaml.safe_load((ROOT / "dgx/compose.yaml").read_text())
    assert set(cfg["services"]) == {"vision"}
    assert cfg["name"] == "koharu-remote-models"
    service = cfg["services"]["vision"]
    assert service["network_mode"] == "host" and "ports" not in service
    assert service["volumes"][0]["read_only"] is True
    assert not any("rorem4s" in str(v) for v in service["volumes"])
    dockerfile = (ROOT / "dgx/Dockerfile").read_text()
    assert '"127.0.0.1"' in dockerfile and "protect_torch.py after" in dockerfile


def test_watcher_builds_candidates_without_promoting_main():
    watcher = yaml.safe_load((ROOT / ".github/workflows/watch.yml").read_text())
    assert watcher["jobs"]["candidate"]["uses"] == "./.github/workflows/windows.yml"
    assert watcher["jobs"]["candidate"]["permissions"]["contents"] == "read"


def test_apply_preserves_original_postprocessing(tmp_path):
    fixture = Path(
        os.environ.get("UPSTREAM_FIXTURE", str(ROOT.parent / "upstream-koharu"))
    ).resolve()
    if not (fixture / ".git").exists():
        pytest.skip("Pristine pinned upstream fixture not present")
    target = tmp_path / "source"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(fixture), str(target)],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(target),
            "checkout",
            "--quiet",
            "--detach",
            json.loads((ROOT / "upstream.json").read_text())["commit"],
        ],
        check=True,
    )
    original = {
        name: (target / name).read_text()
        for name in [
            "crates/koharu-pipeline/src/stages/detection.rs",
            "crates/koharu-pipeline/src/stages/inpainting.rs",
        ]
    }
    subprocess.run(["python", str(ROOT / "tools/apply.py"), str(target)], check=True)
    detection = next(name for name in original if "detection" in name)
    assert (target / detection).read_text().split("fn build_patch(", 1)[1] == original[
        detection
    ].split("fn build_patch(", 1)[1]
    inpaint = next(name for name in original if "inpainting" in name)
    assert (target / inpaint).read_text().split("fn prepare(", 1)[1].split("#[cfg(test)]", 1)[0] == original[
        inpaint
    ].split("fn prepare(", 1)[1].split("#[cfg(test)]", 1)[0]
    assert (target / "crates/koharu-app/src/commands/custom.rs").is_file()
    assert (
        target / "packages/koharu/components/preferences/CustomPreferences.tsx"
    ).is_file()
    ui = (
        target / "packages/koharu/components/preferences/SettingsPage.tsx"
    ).read_text()
    assert "['custom', Cpu]" in ui and "tab === 'custom' && <CustomPreferences onOpenTab={setTab} />" in ui
    # Reapplying to a modified tree must refuse without altering it.
    before = subprocess.check_output(["git", "-C", str(target), "diff"])
    result = subprocess.run(
        ["python", str(ROOT / "tools/apply.py"), str(target)], capture_output=True
    )
    assert (
        result.returncode != 0
        and subprocess.check_output(["git", "-C", str(target), "diff"]) == before
    )


def test_apply_normalizes_crlf_patch_artifact(tmp_path, monkeypatch):
    import shutil
    from tools import apply
    fixture = Path(os.environ.get("UPSTREAM_FIXTURE", str(ROOT.parent / "upstream-koharu"))).resolve()
    if not (fixture / ".git").exists():
        pytest.skip("Pristine pinned upstream fixture not present")
    package = tmp_path / "package"
    shutil.copytree(ROOT / "patches", package / "patches")
    shutil.copyfile(ROOT / "upstream.json", package / "upstream.json")
    patch = package / "patches/remote-models.patch"
    patch.write_bytes(patch.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    target = tmp_path / "upstream"
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", str(fixture), str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--quiet", "--detach",
                    json.loads((ROOT / "upstream.json").read_text())["commit"]], check=True)
    monkeypatch.setattr(apply, "ROOT", package)
    apply.apply(target)
    source = (target / "crates/koharu-pipeline/src/stages/inpainting.rs").read_text()
    assert "async fn inpaint_remote(" in source
    assert (target / "crates/koharu-pipeline/src/remote.rs").read_bytes() == (ROOT / "patches/remote.rs").read_bytes()
