import hashlib, json
import pytest
from vision import model
import manage_vision as manage


def test_prepare_missing_detector_fails_without_env_or_download(tmp_path, monkeypatch):
    monkeypatch.setattr(manage, "ROOT", tmp_path)
    monkeypatch.setattr(manage, "MODEL", tmp_path / "missing")
    monkeypatch.setattr(manage, "CACHE", tmp_path / "cache")
    with pytest.raises(FileNotFoundError):
        manage.prepare(False)
    assert not (tmp_path / ".env").exists() and not (tmp_path / "cache").exists()


def test_changed_detector_file_and_loader_fail_closed(tmp_path, monkeypatch):
    values = {name: name.encode() for name in model.FILES}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in values.items()}
    for name, data in values.items():
        (tmp_path / name).write_bytes(data)
    # Use tiny test data; no GPU or real model required.
    config = {
        "checkpoint_sha256": hashes["model.safetensors"],
        "rfdetr_version": "1.7.0",
    }
    (tmp_path / "inference_config.json").write_text(json.dumps(config))
    hashes["inference_config.json"] = hashlib.sha256(
        (tmp_path / "inference_config.json").read_bytes()
    ).hexdigest()
    monkeypatch.setattr(model, "FILES", hashes)
    monkeypatch.setattr(model, "WEIGHT_SHA", hashes["model.safetensors"])
    (tmp_path / "model-files.json").write_text(
        json.dumps({"repository": model.REPO, "revision": model.REV, "files": hashes})
    )
    assert model.validate(tmp_path) == tmp_path
    (tmp_path / "load_model.py").write_text("tampered")
    with pytest.raises(ValueError, match="pinned file hash"):
        model.validate(tmp_path)


def test_new_prepare_defaults_to_optional_auth_and_preserves_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(manage, "ROOT", tmp_path)
    monkeypatch.setattr(manage, "MODEL", tmp_path / "detector")
    monkeypatch.setattr(manage, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(manage, "validate", lambda folder: None)
    monkeypatch.setattr(manage.ocr_files, "validate", lambda folder: None)
    monkeypatch.setattr(manage, "OCR_MODEL", tmp_path / "ocr")
    calls = []
    monkeypatch.setattr(manage, "compose", lambda *args: calls.append(args))
    manage.prepare(False)
    values = manage.env()
    assert values["VISION_AUTH_REQUIRED"] == "false" and values["VISION_API_TOKEN"] == ""
    before = (tmp_path / ".env").read_bytes()
    manage.prepare(False)
    assert (tmp_path / ".env").read_bytes() == before
    assert calls == [("config", "--quiet"), ("config", "--quiet")]


def test_required_vision_auth_rejects_empty_token_before_compose(tmp_path, monkeypatch):
    monkeypatch.setattr(manage, "ROOT", tmp_path)
    monkeypatch.setattr(manage, "validate", lambda folder: None)
    monkeypatch.setattr(manage.ocr_files, "validate", lambda folder: None)
    monkeypatch.setattr(manage, "OCR_MODEL", tmp_path / "ocr")
    (tmp_path / ".env").write_text("VISION_MODEL_DIR=detector\nVISION_AUTH_REQUIRED=true\nVISION_API_TOKEN=\n")
    calls = []
    monkeypatch.setattr(manage, "compose", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="API token required"):
        manage.check()
    assert calls == []


def test_existing_prepare_adds_only_ocr_path_preserving_private_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(manage, "ROOT", tmp_path)
    monkeypatch.setattr(manage, "OCR_MODEL", tmp_path / "hayai")
    monkeypatch.setattr(manage, "validate", lambda folder: None)
    monkeypatch.setattr(manage.ocr_files, "validate", lambda folder: None)
    monkeypatch.setattr(manage, "compose", lambda *args: None)
    original = f"# private\nVISION_MODEL_DIR=existing-detector\nVISION_CACHE_DIR={tmp_path / 'cache'}\nVISION_API_TOKEN=keep-token\nVISION_BIND_ADDRESS=192.0.2.10\n"
    (tmp_path / ".env").write_text(original)
    manage.prepare()
    expected = original + f"\nVISION_OCR_MODEL_DIR={tmp_path / 'hayai'}\n"
    assert (tmp_path / ".env").read_text() == expected
    manage.prepare()
    assert (tmp_path / ".env").read_text() == expected
