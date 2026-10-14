"""DGX-only changes run offline checks and packaging without recompiling Windows."""

import os
from pathlib import Path
import re
import subprocess

import yaml

WORKFLOW = ".github/workflows/windows.yml"
SAFE = {"README.md", "CHANGES.md", "VALIDATION.md", "tools/needs_windows.py",
        "tests/test_windows_routing.py", "tests/test_dgx_management.py",
        "tests/test_rorem_auth.py", "tests/test_model_files.py", "tests/test_binding.py",
        "tests/test_vision.py", "tests/test_hayai.py", "tests/test_comfy_nodes.py",
        "tests/test_comfy_backend.py", "tests/test_local_color.py", "tests/test_text_sam.py",
        "tests/requirements.txt", "tools/cleanup_branches.py", "tests/test_cleanup_branches.py"}


def needed(paths, previous_workflow=None, current_workflow=None):
    for path in paths:
        if path == WORKFLOW:
            try:
                previous, current = yaml.safe_load(previous_workflow), yaml.safe_load(current_workflow)
                def native_job(value):
                    return {k: v for k, v in value["jobs"]["windows"].items() if k != "if"}
                if (native_job(previous) != native_job(current)
                        or any(previous.get(k) != current.get(k) for k in ("env", "defaults", "permissions"))):
                    return True
            except (KeyError, TypeError, yaml.YAMLError):
                return True
        elif path not in SAFE and not path.startswith(("dgx/", "comfyui/", "tests/fixtures/rorem-v0.2.3/")):
            return True
    return False


def main():
    event = os.environ.get("EVENT_NAME", "")
    base, head = os.environ.get("BASE_SHA", ""), os.environ.get("HEAD_SHA", "")
    result = True
    if (event in ("push", "pull_request") and not os.environ.get("SOURCE_REF")
            and re.fullmatch(r"[0-9a-f]{40}", base) and base != "0" * 40
            and re.fullmatch(r"[0-9a-f]{40}", head)):
        def git(*args):
            return subprocess.check_output(["git", *args], text=True)
        try:
            paths = git("diff", "--name-only", base, head).splitlines()
            previous = current = None
            if WORKFLOW in paths:
                previous, current = (git("show", sha + ":" + WORKFLOW) for sha in (base, head))
            result = needed(paths, previous, current)
        except (subprocess.CalledProcessError, OSError):
            result = True
    value = str(result).lower()
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write("windows_needed=" + value + "\n")
    print("Native Windows source/build inputs changed: " + value)
    print("Windows compilation runs only when build_windows is explicitly enabled")


if __name__ == "__main__":
    main()
