"""Actual patched RORem HTTP handlers with a tiny in-memory inference backend."""

from pathlib import Path
import importlib, io, shutil, sys
import pytest
from fastapi.testclient import TestClient
from PIL import Image
SOURCE = Path(__file__).resolve().parents[1] / "dgx"
TOKEN = "test-token-012345678901234567890123"


@pytest.fixture
def app(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE))
    for name in list(sys.modules):
        if name == "rorem" or name.startswith("rorem."):
            monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module("rorem.app")
    yield module
    for name in list(sys.modules):
        if name == "rorem" or name.startswith("rorem."):
            sys.modules.pop(name)


class Backend:
    name = "RORem-4S"
    model_loaded = True
    device = "cuda"

    def infer(self, image, mask, options):
        return image.copy()


def images():
    def png(mode, color):
        stream = io.BytesIO()
        Image.new(mode, (8, 8), color).save(stream, "PNG")
        return stream.getvalue()
    return {"image": ("image.png", png("RGB", "white"), "image/png"),
            "mask": ("mask.png", png("L", 255), "image/png")}


def test_rorem_without_token_preloads_and_inpaints(app, monkeypatch):
    monkeypatch.delenv("ROREM_AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("ROREM_API_TOKEN", raising=False)
    with TestClient(app.create_app(Backend)) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["auth_required"] is False
        response = client.post("/v1/inpaint", files=images())
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(response.content)).size == (8, 8)


def test_rorem_stored_token_is_optional_until_auth_enabled(app, monkeypatch):
    monkeypatch.setenv("ROREM_API_TOKEN", TOKEN)
    monkeypatch.setenv("ROREM_AUTH_REQUIRED", "false")
    with TestClient(app.create_app(Backend)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/inpaint", files=images()).status_code == 200


def test_rorem_auth_enabled_rejects_missing_and_wrong_credentials(app, monkeypatch):
    monkeypatch.setenv("ROREM_API_TOKEN", TOKEN)
    monkeypatch.setenv("ROREM_AUTH_REQUIRED", "true")
    with TestClient(app.create_app(Backend)) as client:
        assert client.get("/health").status_code == 401
        assert client.post("/v1/inpaint", files=images()).status_code == 401
        assert client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
        headers = {"Authorization": "Bearer " + TOKEN}
        assert client.get("/health", headers=headers).json()["auth_required"] is True
        assert client.post("/v1/inpaint", files=images(), headers=headers).status_code == 200


def test_rorem_required_auth_needs_a_token(app, monkeypatch):
    monkeypatch.setenv("ROREM_AUTH_REQUIRED", "true")
    monkeypatch.delenv("ROREM_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="authentication is enabled"):
        with TestClient(app.create_app(Backend)):
            pass


def test_imported_rorem_source_matches_recorded_provenance():
    import hashlib, json
    provenance = json.loads((SOURCE / "rorem/provenance.json").read_text())
    for name, hashes in provenance["files"].items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == hashes.get("current_sha256", hashes["imported_sha256"])
        if name not in ("rorem/app.py", "rorem/healthcheck.py"):
            assert hashes["original_sha256"] == hashes["imported_sha256"]
    for name, digest in provenance.get("added_files", {}).items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest


def test_custom_inpainting_options_reach_backend_and_preserve_geometry(app, monkeypatch):
    monkeypatch.delenv("ROREM_AUTH_REQUIRED", raising=False)
    seen = []
    class Configured(Backend):
        def infer(self, image, mask, options):
            seen.append(options)
            return image.copy()
    with TestClient(app.create_app(Configured)) as client:
        response = client.post("/v1/inpaint", files=images(), data={"options":
            '{"prompt":"keep line art","negative_prompt":"letters","steps":6,"strength":0.95,"guidance_scale":1.2,"seed":123,"max_side":512,"mask_dilation":4}'})
        assert response.status_code == 200
        assert Image.open(io.BytesIO(response.content)).size == (8, 8)
        assert seen[0].prompt == "keep line art" and seen[0].steps == 6
        assert seen[0].max_side == 512 and seen[0].mask_dilation == 4 and seen[0].seed == 123
        assert client.get("/health").json()["capabilities"]["inpainting_options"] is True


@pytest.mark.parametrize("options", ['{"max_side":255}', '{"max_side":1025}',
                                   '{"mask_dilation":33}', '{"mask_dilation":true}',
                                   '{"seed":-2}', '{"seed":2147483648}', '{"seed":true}', '{"seed":1.5}'])
def test_invalid_custom_geometry_options_reject_before_inference(app, monkeypatch, options):
    monkeypatch.delenv("ROREM_AUTH_REQUIRED", raising=False)
    class Never(Backend):
        def infer(self, *args):
            pytest.fail("Invalid options reached inference")
    with TestClient(app.create_app(Never)) as client:
        assert client.post("/v1/inpaint", files=images(), data={"options": options}).status_code == 422


def test_model_space_dilation_preserves_original_compositing_mask():
    from rorem.strategy import FullPageStrategy, preserve_outside_mask
    source = Image.new("RGB", (512, 1024), "white")
    mask = Image.new("L", source.size)
    mask.putpixel((256, 512), 255)
    prepared = FullPageStrategy().prepare(source, mask, 512, 2)
    assert prepared.image.size == (256, 512)
    assert prepared.mask.getbbox() is not None
    result = preserve_outside_mask(Image.new("RGB", source.size, "red"), source, mask)
    assert result.getpixel((256, 512)) == (255, 0, 0)
    assert result.getpixel((255, 512)) == (255, 255, 255)
    assert mask.getbbox() == (256, 512, 257, 513)


def test_rorem_accepts_png_upload_above_old_32_mib_limit(app, monkeypatch):
    monkeypatch.delenv("ROREM_MAX_UPLOAD_BYTES", raising=False)
    files = images()
    name, data, mime = files["image"]
    # Valid PNG with trailing padding: exercises real multipart handling without a large pixel fixture.
    files["image"] = (name, data + b"\0" * (32 * 1024 * 1024), mime)
    with TestClient(app.create_app(Backend)) as client:
        assert client.get("/health").json()["max_upload_bytes"] == 128 * 1024 * 1024
        response = client.post("/v1/inpaint", files=files)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(response.content)).size == (8, 8)


@pytest.mark.parametrize("field", ["image", "mask"])
def test_rorem_configured_upload_limit_returns_structured_413_before_inference(app, monkeypatch, field):
    monkeypatch.setenv("ROREM_MAX_UPLOAD_BYTES", "1024")
    files = images()
    files[field] = (field + ".png", b"x" * 1025, "image/png")
    calls = []
    class NoInference(Backend):
        def infer(self, *args):
            calls.append(args)
            raise AssertionError("Oversized upload must not run inference")
    with TestClient(app.create_app(NoInference)) as client:
        response = client.post("/v1/inpaint", files=files)
        assert response.status_code == 413
        assert response.json() == {"detail": {"code": "upload_too_large", "field": field, "max_upload_bytes": 1024}}
        assert not calls
        assert client.get("/health").json()["last_inference_error"] is None


@pytest.mark.parametrize("value", ["0", "-1", "1023", "536870913", "1.5", "invalid", "１２３４"])
def test_upload_limit_invalid_configuration_is_rejected(value):
    from rorem.upload_limits import read_max_upload_bytes
    with pytest.raises(ValueError, match="ROREM_MAX_UPLOAD_BYTES"):
        read_max_upload_bytes(value)


def test_random_seed_is_resolved_per_request_and_fixed_zero_is_preserved(app, monkeypatch):
    from types import SimpleNamespace
    from contextlib import nullcontext
    from rorem import backend as real
    seeds = []
    class Generator:
        def __init__(self, device):
            assert device == "cuda"
        def manual_seed(self, value):
            seeds.append(value)
            return self
    backend = real.RORemBackend.__new__(real.RORemBackend)
    backend.model_loaded = True
    backend.device = "cuda"
    backend.strategy = real.FullPageStrategy()
    backend.torch = SimpleNamespace(Generator=Generator,inference_mode=nullcontext)
    backend.pipe = lambda **kwargs: SimpleNamespace(images=[kwargs["image"].copy()])
    draws = iter([1234,5678])
    bits = []
    def random_seed(count):
        bits.append(count)
        return next(draws)
    monkeypatch.setattr(real.secrets,"randbits",random_seed)
    monkeypatch.delenv("ROREM_AUTH_REQUIRED",raising=False)
    with TestClient(app.create_app(lambda: backend)) as client:
        assert client.get("/health").json()["capabilities"]["random_seed"] is True
        for seed in [0,-1,-1,7,7]:
            response = client.post("/v1/inpaint",files=images(),data={"options": '{"seed":'+str(seed)+'}'})
            assert response.status_code == 200
    assert seeds == [0,1234,5678,7,7] and bits == [31,31]
