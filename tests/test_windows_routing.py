import yaml
import pytest
from tools.needs_windows import needed


@pytest.mark.parametrize("path", ["patches/remote.rs", "patches/CustomPreferences.tsx",
    "tools/package_windows.py", "tools/checkout.py", "upstream.json", "tauri.remote.json",
    "start-remote.ps1", "tests/test_patch.py", "unknown/native.rs", ".github/workflows/watch.yml"])
def test_native_or_unknown_changes_keep_windows_build(path):
    assert needed([path]) is True


def test_dgx_source_and_checks_do_not_need_a_new_windows_binary():
    assert needed(["dgx/manage.py", "dgx/rorem/backend.py", "dgx/compose.rorem.yaml",
                   "tests/test_dgx_management.py", "tests/test_rorem_auth.py", "tests/test_vision.py",
                   "tests/test_binding.py", "README.md"]) is False


def test_comfy_server_workflows_and_regressions_do_not_need_a_windows_binary():
    assert needed(['comfyui/koharu_nodes/nodes.py','comfyui/workflows/detection-sam-ts-l.json',
                   'tests/test_comfy_nodes.py','tests/test_local_color.py','tests/test_text_sam.py',
                   'tests/requirements.txt','tools/cleanup_branches.py']) is False


def test_automatic_checks_cannot_start_windows_or_repackage_old_artifacts():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    build=yaml.safe_load((root/'.github/workflows/windows.yml').read_text())
    events=build.get('on',build.get(True))
    assert events['workflow_dispatch']['inputs']['build_windows']['default'] is False
    assert events['workflow_call']['inputs']['build_windows']['default'] is False
    assert build['jobs']['windows']['if']=='inputs.build_windows == true'
    assert build['jobs']['offline']['permissions']['contents']=='read'
    assert all(job.get('permissions',{}).get('contents','read')=='read' for job in build['jobs'].values())
    repack=yaml.safe_load((root/'.github/workflows/repack.yml').read_text())
    assert set(repack.get('on',repack.get(True)))=={'workflow_dispatch'}
    assert 'default' not in repack.get('on',repack.get(True))['workflow_dispatch']['inputs']['source_run']
    watch=yaml.safe_load((root/'.github/workflows/watch.yml').read_text())
    assert watch['jobs']['candidate']['with']['build_windows'] is False
    assert watch.get('on',watch.get(True))['workflow_run']['workflows']==['Koharu CI and source']
    assert watch.get('on',watch.get(True))['workflow_run']['branches']==['main']
    assert "workflow_run.conclusion == 'success'" in watch['jobs']['cleanup']['if']
    assert watch['jobs']['cleanup']['permissions']['contents']=='write'


def test_only_routing_condition_changes_skip_windows_but_step_changes_do_not():
    previous = {"permissions": {"contents": "read"}, "jobs": {"windows": {
        "needs": "offline", "runs-on": "windows-latest", "steps": [{"run": "cargo build"}]}}}
    current = yaml.safe_load(yaml.safe_dump(previous))
    current["jobs"]["windows"]["if"] = "needs.offline.outputs.windows_needed == 'true'"
    assert needed([".github/workflows/windows.yml"], yaml.safe_dump(previous), yaml.safe_dump(current)) is False
    current["jobs"]["windows"]["steps"][0]["run"] = "cargo build --release"
    assert needed([".github/workflows/windows.yml"], yaml.safe_dump(previous), yaml.safe_dump(current)) is True


def test_missing_or_invalid_workflow_defaults_to_native_build():
    assert needed([".github/workflows/windows.yml"], "invalid: [", "{}") is True
