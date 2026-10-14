import json, subprocess
import pytest
from tools import watch_upstream, lock


def test_release_candidate_created_once_and_never_promotes_main(tmp_path, monkeypatch):
    pin = {
        "repository": "koharu-rs/koharu",
        "version": "0.83.5",
        "commit": "a" * 40,
        "patch_version": "0.1.0",
    }
    path = tmp_path / "upstream.json"
    path.write_text(json.dumps(pin))
    output = tmp_path / "output"
    monkeypatch.setattr(watch_upstream, "ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    exists = False
    calls = []
    git = []

    def gh(*args):
        calls.append(args)
        if args[0] == "api" and args[1].endswith("releases/latest"):
            return json.dumps(
                {"tag_name": "v0.84.0", "draft": False, "prerelease": False}
            )
        if args[0] == "api":
            return json.dumps({"sha": "b" * 40})
        if args[:2] == ("pr", "list"):
            return json.dumps([{"number": 1}] if exists else [])
        if args[:2] == ("pr", "create"):
            return "https://github.com/example/pr/1"
        raise AssertionError("Unexpected external operation")

    def run(args, **kwargs):
        git.append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=("b" * 40 + " refs/heads/upstream/koharu-v0.84.0\n")
            if exists and "ls-remote" in args
            else "",
        )

    monkeypatch.setattr(watch_upstream, "gh", gh)
    monkeypatch.setattr(watch_upstream.subprocess, "run", run)
    watch_upstream.main()
    assert json.loads(path.read_text())["commit"] == "b" * 40
    assert output.read_text() == "branch=upstream/koharu-v0.84.0\n"
    pushes = [c for c in git if "push" in c]
    assert (
        len(pushes) == 1 and pushes[0][-1] == "HEAD:refs/heads/upstream/koharu-v0.84.0"
    )
    assert any(c[:2] == ("pr", "create") for c in calls)
    assert not any("merge" in c for c in calls)
    # Next scheduler invocation starts from the still-stable main pin.
    exists = True
    path.write_text(json.dumps(pin))
    output.unlink()
    git.clear()
    watch_upstream.main()
    assert not output.exists() and not any("push" in c for c in git)


@pytest.mark.parametrize("tag", ["v0.84.0-beta", "../main", "not-a-version"])
def test_unknown_release_tag_fails_before_mutation(tmp_path, monkeypatch, tag):
    (tmp_path / "upstream.json").write_text(
        json.dumps({"repository": "koharu-rs/koharu", "commit": "a" * 40})
    )
    monkeypatch.setattr(watch_upstream, "ROOT", tmp_path)
    monkeypatch.setattr(
        watch_upstream,
        "gh",
        lambda *args: json.dumps(
            {"tag_name": tag, "draft": False, "prerelease": False}
        ),
    )
    with pytest.raises(RuntimeError):
        watch_upstream.main()


def test_lock_refresh_refuses_version_drift(tmp_path, monkeypatch):
    path = tmp_path / "Cargo.lock"
    old = '[[package]]\nname="reqwest"\nversion="0.13.5"\n'
    path.write_text(old)

    def upgrade(*args, **kwargs):
        path.write_text(old.replace("0.13.5", "0.13.6"))

    monkeypatch.setattr(lock.subprocess, "run", upgrade)
    with pytest.raises(RuntimeError, match="versions/sources"):
        lock.refresh(tmp_path)
