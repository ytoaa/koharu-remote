"""Direct IP binding, private setting preservation and health readiness."""

import io
import json
from pathlib import Path
import urllib.request

import pytest
import yaml
from binding import health_url, set_address
import manage_vision as vision

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("ip,host", [
    ("0.0.0.0", "127.0.0.1"), ("::", "[::1]"),
    ("192.0.2.10", "192.0.2.10"), ("2001:db8::1", "[2001:db8::1]"),
])
def test_health_uses_specific_interface_or_wildcard_loopback(ip, host):
    assert health_url(ip, 8765) == f"http://{host}:8765/health"


def test_vision_bind_preserves_other_settings_and_requires_no_build(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "ROOT", tmp_path)
    path = tmp_path / ".env"
    original = "# keep\nVISION_API_TOKEN=secret\nVLLM_BASE_URL=http://127.0.0.1:8000/v1\nVISION_AUTH_REQUIRED=false\n"
    path.write_text(original)
    vision.main(["bind", "0.0.0.0"])
    assert path.read_text() == original + "VISION_BIND_ADDRESS=0.0.0.0\n"
    assert path.stat().st_mode & 0o777 == 0o600
    vision.main(["bind", "192.0.2.10"])
    assert path.read_text() == original + "VISION_BIND_ADDRESS=192.0.2.10\n"
    before = path.read_bytes()
    with pytest.raises(ValueError):
        vision.main(["bind", "invalid-host"])
    assert path.read_bytes() == before


def test_rorem_quoted_setting_preserves_literal_credentials(tmp_path):
    path = tmp_path / ".env.rorem"
    original = '# private\nROREM_API_TOKEN="dollar-$$-escaped"\nBIND_ADDRESS="127.0.0.1"\n'
    path.write_text(original)
    set_address(path, "BIND_ADDRESS", "0.0.0.0", quoted=True)
    assert path.read_text() == original.replace('BIND_ADDRESS="127.0.0.1"', 'BIND_ADDRESS="0.0.0.0"')


@pytest.mark.parametrize("ip", ["0.0.0.0", "192.0.2.10", "::"])
@pytest.mark.parametrize("ready", [True, False])
def test_existing_image_compose_health_follows_bind_and_rejects_unready(monkeypatch, ip, ready):
    cfg = yaml.safe_load((ROOT / "dgx/compose.yaml").read_text())["services"]["vision"]
    assert cfg["command"][cfg["command"].index("--host") + 1] == "${VISION_BIND_ADDRESS:-127.0.0.1}"
    test = cfg["healthcheck"]["test"]
    assert test[:3] == ["CMD", "python", "-c"]
    monkeypatch.setenv("VISION_BIND_ADDRESS", ip)
    monkeypatch.setenv("VISION_API_TOKEN", "preserved-token")
    captured = []
    data = {"protocol": 1, "service": "koharu-vision", "ready": ready, "model_loaded": True, "ocr_model_loaded": True,
                      "ocr_model": "JustANormalTinkerer/hayai-ocr-v2",
                      "ocr_revision": "4a4ce477c9a8841f208b94e1d9ed5c0938965e05"}
    class Opener:
        def open(self, request, timeout):
            captured.append(request)
            return io.StringIO(json.dumps(data))
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a: Opener())
    if ready:
        exec(test[3], {})
    else:
        with pytest.raises(AssertionError, match="Vision not ready"):
            exec(test[3], {})
    assert captured[0].full_url == health_url(ip, 8766)
    assert captured[0].get_header("Authorization") == "Bearer preserved-token"
