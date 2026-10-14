"""Docker adoption and replacement contracts; no daemon, model weights or GPU."""

import copy
import json
from pathlib import Path
import subprocess

import pytest
import manage_rorem as rorem
import manage as entry

TOKEN = "retained-token-$-01234567890123456789"


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    root = tmp_path / "checkout/dgx"
    root.mkdir(parents=True)
    models = tmp_path / "models/rorem4s"
    pipeline, lora = models / "RORem-HF", models / "RORem-LCM"
    pipeline.mkdir(parents=True)
    lora.mkdir()
    cache = tmp_path / "cache/rorem"
    cache.mkdir(parents=True)
    monkeypatch.setattr(rorem, "ROOT", root)
    monkeypatch.setattr(rorem, "validate_models", lambda p, l: {"pipeline": str(p), "lora": str(l)})
    info = {
        "Id": "old-container", "Image": "sha256:verified-image", "State": {"Status": "running"},
        "Config": {"Env": ["ROREM_API_TOKEN=" + TOKEN, "ROREM_MAX_PIXELS=64000000"],
                   "Labels": {"com.docker.compose.project": rorem.PROJECT,
                              "com.docker.compose.service": "rorem",
                              "com.docker.compose.project.working_dir": "/home/user/old-deployment"}},
        "HostConfig": {"PortBindings": {"8765/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8765"}]}},
        "Mounts": [{"Type": "bind", "Destination": target, "Source": str(path), "RW": rw}
                   for target, path, rw in [("/models/pipeline", pipeline, False),
                                            ("/models/lora", lora, False), ("/cache", cache, True)]],
    }
    image = {"Id": info["Image"], "Architecture": "arm64",
             "Config": {"Labels": {"org.koharu.rorem.version": "0.2.3"}}}
    calls = []
    def docker(*args, capture=False):
        calls.append(args)
        if args[0] == "inspect":
            return json.dumps([info])
        if args[:2] == ("image", "inspect"):
            return json.dumps([image])
        return None
    monkeypatch.setattr(rorem, "docker", docker)
    return root, info, image, calls


def test_adopt_preserves_existing_credentials_mounts_image_and_vision_env(deployment, capsys):
    root, info, image, calls = deployment
    (root / ".env").write_text("VISION_API_TOKEN=existing-vision-token\n")
    rorem.adopt()
    c = rorem.config()
    assert c["ROREM_API_TOKEN"] == TOKEN and c["ROREM_AUTH_REQUIRED"] == "false"
    assert c["ROREM_BASE_IMAGE_ID"] == image["Id"]
    assert c["ROREM_PIPELINE_DIR"] == info["Mounts"][0]["Source"]
    assert c["ROREM_LORA_DIR"] == info["Mounts"][1]["Source"]
    assert c["ROREM_CACHE_DIR"] == info["Mounts"][2]["Source"]
    assert (root / ".env").read_text() == "VISION_API_TOKEN=existing-vision-token\n"
    assert (root / ".env.rorem").stat().st_mode & 0o777 == 0o600
    assert (root / "state/rorem/adoption.json").stat().st_mode & 0o777 == 0o600
    assert all(call[0] not in ("stop", "rm", "build", "compose") for call in calls)
    assert TOKEN not in capsys.readouterr().out
    before = (root / ".env.rorem").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        rorem.adopt()
    assert (root / ".env.rorem").read_bytes() == before


@pytest.mark.parametrize("change", ["project", "writable-model", "missing-mount", "architecture", "version", "port"])
def test_unknown_deployments_fail_before_adoption_writes(deployment, change):
    root, info, image, calls = deployment
    if change == "project": info["Config"]["Labels"]["com.docker.compose.project"] = "other"
    if change == "writable-model": info["Mounts"][0]["RW"] = True
    if change == "missing-mount": info["Mounts"].pop()
    if change == "architecture": image["Architecture"] = "amd64"
    if change == "version": image["Config"]["Labels"]["org.koharu.rorem.version"] = "0.2.2"
    if change == "port": info["HostConfig"]["PortBindings"]["8765/tcp"][0]["HostPort"] = "9999"
    with pytest.raises(ValueError):
        rorem.adopt()
    assert not (root / ".env.rorem").exists()
    assert not any(call[:2] == ("image", "tag") for call in calls)


def test_explicit_auth_setting_is_retained(deployment):
    root, info, image, calls = deployment
    info["Config"]["Env"].append("ROREM_AUTH_REQUIRED=true")
    rorem.adopt()
    assert rorem.config()["ROREM_AUTH_REQUIRED"] == "true"


def test_adoption_can_be_used_after_container_removal(deployment, monkeypatch):
    root, info, image, calls = deployment
    rorem.adopt()
    monkeypatch.setattr(rorem, "compose", lambda *a, **k: "")
    monkeypatch.setattr(rorem, "source_digest", lambda: "current-source")
    rorem.build()
    command = next(call for call in calls if call[0] == "build")
    assert "--network=none" in command and "--pull=false" in command
    assert "ROREM_BASE_IMAGE=" + rorem.config()["ROREM_BASE_IMAGE"] in command
    assert command[-1] == str(root)
    assert not any(call[0] in ("run", "pull", "stop", "rm") for call in calls)


def test_retained_image_identity_change_refuses_build(deployment, monkeypatch):
    root, info, image, calls = deployment
    rorem.adopt()
    image["Id"] = "sha256:changed"
    monkeypatch.setattr(rorem, "compose", lambda *a, **k: "")
    with pytest.raises(ValueError, match="identity changed"):
        rorem.build()
    assert not any(call[0] == "build" for call in calls)


def test_remove_targets_only_imported_rorem_and_preserves_configuration(deployment, monkeypatch):
    root, info, image, calls = deployment
    rorem.adopt()
    monkeypatch.setattr(rorem, "compose", lambda *a, **k: "")
    monkeypatch.setattr(rorem, "current", lambda: info)
    monkeypatch.setattr(rorem, "request_health", lambda **kw: {"busy": False})
    before = (root / ".env.rorem").read_bytes()
    rorem.remove()
    assert ("stop", info["Id"]) in calls and ("rm", info["Id"]) in calls
    assert (root / ".env.rorem").read_bytes() == before
    assert all("-v" not in call and "prune" not in call for call in calls)


def test_busy_service_is_not_removed_or_replaced(deployment, monkeypatch):
    root, info, image, calls = deployment
    rorem.adopt()
    monkeypatch.setattr(rorem, "compose", lambda *a, **k: "")
    monkeypatch.setattr(rorem, "current", lambda: info)
    monkeypatch.setattr(rorem, "request_health", lambda **kw: {"busy": True})
    with pytest.raises(ValueError, match="busy"):
        rorem.remove()
    with pytest.raises(ValueError, match="busy"):
        rorem.update()
    assert not any(call[0] in ("stop", "rm", "build") for call in calls)


def test_replacement_uses_same_project_and_only_rorem(deployment, monkeypatch):
    root, info, image, calls = deployment
    rorem.adopt()
    monkeypatch.setattr(rorem, "source_digest", lambda: "source")
    image["Config"]["Labels"].update({"org.koharu.rorem.deployment": "unified-1", "org.koharu.rorem.source": "source"})
    monkeypatch.setattr(rorem, "current", lambda: info)
    monkeypatch.setattr(rorem, "request_health", lambda **kw: {"busy": False})
    composed = []
    monkeypatch.setattr(rorem, "compose", lambda *a, **k: composed.append(a))
    rorem.up(expected=info["Id"])
    assert composed[-1] == ("up", "--no-build", "--no-deps", "-d", "--force-recreate", "rorem")
    assert all("vision" not in call and "vllm" not in call for call in composed)


def test_missing_and_changed_container_races_refuse_replacement(deployment, monkeypatch):
    root, info, image, calls = deployment
    rorem.adopt()
    monkeypatch.setattr(rorem, "compose", lambda *a, **k: "")
    monkeypatch.setattr(rorem, "ready_image", lambda c: None)
    monkeypatch.setattr(rorem, "current", lambda: info)
    for expected in (None, "another-container"):
        with pytest.raises(ValueError, match="changed during build"):
            rorem.up(expected=expected)


def test_stale_source_image_is_not_started(deployment, monkeypatch):
    root, info, image, calls = deployment
    rorem.adopt()
    monkeypatch.setattr(rorem, "compose", lambda *a, **k: "")
    with pytest.raises(ValueError, match="current checkout source"):
        rorem.up()


def test_compose_uses_new_root_private_env_and_stable_identity(deployment, monkeypatch):
    root, info, image, calls = deployment
    rorem.adopt()
    recorded = []
    def run(argv, **kwargs):
        recorded.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="")
    monkeypatch.setattr(rorem.subprocess, "run", run)
    rorem.compose("logs", "--tail", "100", "rorem")
    argv, kwargs = recorded[-1]
    assert argv[:4] == ["docker", "compose", "--project-name", "koharu-model-services"]
    assert str(root / "compose.rorem.yaml") in argv and str(root / ".env.rorem") in argv
    assert kwargs["env"]["ROREM_API_TOKEN"] == TOKEN


def test_dispatch_calls_correct_service_cli(monkeypatch):
    import manage_vision
    calls = []
    monkeypatch.setattr(manage_vision, "main", lambda a: calls.append(("vision", a)))
    monkeypatch.setattr(rorem, "main", lambda a: calls.append(("rorem", a)))
    entry.main(["vision", "health"])
    entry.main(["rorem", "adopt"])
    assert calls == [("vision", ["health"]), ("rorem", ["adopt"])]


def test_all_up_checks_both_configurations_before_any_start(monkeypatch):
    import manage_vision
    calls = []
    monkeypatch.setattr(manage_vision, "check", lambda: calls.append("vision-check"))
    monkeypatch.setattr(rorem, "check", lambda: calls.append("rorem-check"))
    monkeypatch.setattr(manage_vision, "main", lambda args: calls.append(("vision",args)))
    monkeypatch.setattr(rorem, "main", lambda args: calls.append(("rorem",args)))
    entry.main(["all","up"])
    assert calls == ["vision-check","rorem-check",("vision",["up"]),("rorem",["up"])]
    calls.clear()
    def fail():
        raise ValueError("RORem not configured")
    monkeypatch.setattr(rorem, "check", fail)
    with pytest.raises(ValueError, match="not configured"):
        entry.main(["all","up"])
    assert calls == ["vision-check"]


@pytest.mark.parametrize("ip", ["0.0.0.0", "192.0.2.10"])
def test_direct_bind_preserves_auth_and_uses_running_address_for_idle(deployment, monkeypatch, ip):
    root, info, image, calls = deployment
    rorem.adopt()
    before = rorem.config()
    rorem.main(["bind", ip])
    after = rorem.config()
    assert after == {**before, "BIND_ADDRESS": ip}
    assert after["ROREM_AUTH_REQUIRED"] == "false"
    rorem.validate_config(after)
    recorded = []
    monkeypatch.setattr(rorem, "request_health", lambda **kw: recorded.append(kw) or {"busy": False})
    rorem.idle(info)
    assert recorded == [{"bind_address": "127.0.0.1"}]
    assert not any(call[0] in ("stop", "rm", "build") for call in calls)
