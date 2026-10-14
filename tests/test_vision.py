import asyncio, io, threading
import httpx, numpy as np, pytest
from fastapi.testclient import TestClient
from PIL import Image
from vision.app import create_app as make_app
from vision.hayai_files import REPO as OCR_REPO, REV as OCR_REV
from vision.model import rle

TOKEN = "unit-test-token"
HEADERS = {"Authorization": "Bearer " + TOKEN}


def png(size=(8, 9)):
    file = io.BytesIO()
    Image.new("RGB", size).save(file, "PNG")
    return file.getvalue()


def files(data=None):
    return {"image": ("image.png", png() if data is None else data, "image/png")}


class Mock:
    device = "cuda"
    gpu_name = "mock-GPU"

    def detect(self, image, thresholds):
        return {
            "protocol": 1,
            "image_width": image.width,
            "image_height": image.height,
            "detections": [],
            "thresholds": thresholds,
        }


class MockOcr:
    warmup_complete = True
    warmup_seconds = 0.01

    def ocr(self, image, options):
        return {"protocol": 1, "text": "テスト", "model": OCR_REPO, "revision": OCR_REV}


def create_app(factory=None, ocr_factory=MockOcr):
    return make_app(factory, ocr_factory)


@pytest.fixture(autouse=True)
def token(monkeypatch):
    monkeypatch.setenv("VISION_API_TOKEN", TOKEN)
    monkeypatch.setenv("VISION_AUTH_REQUIRED", "true")


def test_health_auth_and_preload_once():
    loaded = []

    def factory():
        loaded.append(True)
        return Mock()

    with TestClient(create_app(factory)) as client:
        assert client.get("/health").status_code == 401
        assert (
            client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code
            == 401
        )
        for _ in range(2):
            value = client.get("/health", headers=HEADERS).json()
            assert value["protocol"] == 1 and value["service"] == "koharu-vision"
            assert value["ready"] is True and value["model_loaded"] is True
            assert value["ocr_model_loaded"] is True
            assert value["ocr_model"] == OCR_REPO and value["ocr_revision"] == OCR_REV
        assert len(loaded) == 1


def test_failed_preload_is_not_ready():
    def fail():
        raise RuntimeError("CUDA unavailable")

    with TestClient(create_app(fail)) as client:
        response = client.get("/health", headers=HEADERS)
        assert response.status_code == 503 and response.json()["ready"] is False
        assert response.json()["model_loaded"] is False
        assert (
            client.post("/v1/detect", headers=HEADERS, files=files()).status_code == 503
        )


@pytest.mark.parametrize("missing", ["cv2", "supervision"])
def test_missing_prediction_dependency_cannot_report_ready(monkeypatch, missing):
    import sys
    from vision import model

    monkeypatch.setitem(sys.modules, "torch", object())
    original_import = model.importlib.import_module

    def import_module(name, *args, **kwargs):
        if name == missing:
            raise ImportError("libxcb.so.1: cannot open shared object file")
        if name in {"cv2", "supervision"}:
            return object()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(model.importlib, "import_module", import_module)
    with TestClient(create_app(lambda: model.Backend("/unused"))) as client:
        response = client.get("/health", headers=HEADERS)
        assert response.status_code == 503
        assert response.json()["ready"] is False
        assert response.json()["model_loaded"] is False
        assert response.json()["error"] == "Preload failed: ImportError"
        assert client.post("/v1/detect", headers=HEADERS, files=files()).status_code == 503


def test_optional_auth_accepts_no_header_and_preloads_without_token(monkeypatch):
    monkeypatch.delenv("VISION_AUTH_REQUIRED")
    monkeypatch.delenv("VISION_API_TOKEN")
    with TestClient(create_app(Mock)) as client:
        assert client.get("/health").json()["auth_required"] is False
        assert client.post("/v1/detect", files=files()).status_code == 200
        assert client.post("/v1/ocr", files=files()).json()["text"] == "テスト"


def test_stored_token_does_not_enable_auth_by_itself(monkeypatch):
    monkeypatch.setenv("VISION_AUTH_REQUIRED", "false")
    with TestClient(create_app(Mock)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/detect", files=files()).status_code == 200


def test_required_auth_without_token_cannot_preload(monkeypatch):
    monkeypatch.delenv("VISION_API_TOKEN")
    loaded = []
    with TestClient(create_app(lambda: loaded.append(True))) as client:
        assert client.get("/health").status_code == 401
        assert loaded == []
        assert client.app.state.status["backend"] is None


@pytest.mark.parametrize("data", [b"", b"not-an-image", b"\x89PNG\r\n"])
def test_invalid_image(data):
    with TestClient(create_app(Mock)) as client:
        assert (
            client.post("/v1/detect", headers=HEADERS, files=files(data)).status_code
            == 422
        )


@pytest.mark.parametrize(
    "options",
    ["[]", "bad", '{"text":-1}', '{"text":true}', '{"unknown":0.5}', '{"panel":NaN}'],
)
def test_invalid_thresholds(options):
    with TestClient(create_app(Mock)) as client:
        assert (
            client.post(
                "/v1/detect", headers=HEADERS, files=files(), data={"options": options}
            ).status_code
            == 422
        )


def test_original_dimensions_and_class_thresholds():
    with TestClient(create_app(Mock)) as client:
        response = client.post(
            "/v1/detect",
            headers=HEADERS,
            files=files(),
            data={"options": '{"text":0.3}'},
        )
        assert response.status_code == 200
        value = response.json()
        assert value["detections"] == []
        assert (value["image_width"], value["image_height"]) == (8, 9)
        assert value["thresholds"] == {
            "text": 0.3,
            "onomatopoeia": 0.2,
            "bubble": 0.5,
            "panel": 0.5,
        }


def test_gate_serializes_and_releases_after_failure():
    async def run():
        entered = threading.Event()
        release = threading.Event()
        active = 0
        maximum = 0
        calls = 0

        class Slow(Mock):
            def detect(self, image, thresholds):
                nonlocal active, maximum, calls
                calls += 1
                first = calls == 1
                active += 1
                maximum = max(maximum, active)
                try:
                    if first:
                        entered.set()
                        assert release.wait(3)
                        raise RuntimeError("mock OOM")
                    return super().detect(image, thresholds)
                finally:
                    active -= 1

        class QueuedOcr(MockOcr):
            def ocr(self, image, options):
                assert release.is_set(), "OCR overlapped active detection CUDA work"
                return super().ocr(image, options)
        app = create_app(Slow, QueuedOcr)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app),
                base_url="http://test",
                headers=HEADERS,
            ) as client:
                first = asyncio.create_task(client.post("/v1/detect", files=files()))
                assert await asyncio.to_thread(entered.wait, 2)
                second = asyncio.create_task(client.post("/v1/ocr", files=files()))
                await asyncio.sleep(0.03)
                assert app.state.status["busy"] and calls == 1
                release.set()
                a, b = await asyncio.gather(first, second)
                assert a.status_code == 503 and b.status_code == 200 and maximum == 1
                assert (
                    app.state.status["backend"] is not None
                    and not app.state.status["busy"]
                )

    asyncio.run(run())


def test_cancelled_client_keeps_gate_until_cuda_thread_finishes():
    async def run():
        entered = threading.Event()
        release = threading.Event()
        calls = 0

        class Slow(Mock):
            def detect(self, image, thresholds):
                nonlocal calls
                calls += 1
                if calls == 1:
                    entered.set()
                    assert release.wait(3)
                return super().detect(image, thresholds)

        app = create_app(Slow)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app),
                base_url="http://test",
                headers=HEADERS,
            ) as client:
                first = asyncio.create_task(client.post("/v1/detect", files=files()))
                assert await asyncio.to_thread(entered.wait, 2)
                first.cancel()
                second = asyncio.create_task(client.post("/v1/detect", files=files()))
                await asyncio.sleep(0.03)
                assert calls == 1 and app.state.status["busy"] and not first.done()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await first
                assert (await second).status_code == 200
                assert not app.state.status["busy"]

    asyncio.run(run())


def test_ocr_preloads_once_and_propagates_transcription():
    loaded = []
    def load_ocr():
        loaded.append(True)
        return MockOcr()
    with TestClient(create_app(Mock, load_ocr)) as client:
        for _ in range(2):
            result = client.post("/v1/ocr", headers=HEADERS, files=files())
            assert result.status_code == 200 and result.json()["text"] == "テスト"
            assert result.json()["model"] == OCR_REPO
        assert loaded == [True]


def test_ocr_failure_retains_model_and_can_recover():
    class FailsOnce(MockOcr):
        failed = False
        def ocr(self, image, options):
            if not self.failed:
                self.failed = True
                raise ValueError("Hayai output contains an excessive repetition loop")
            return super().ocr(image, options)
    with TestClient(create_app(Mock, FailsOnce)) as client:
        assert client.post("/v1/ocr", headers=HEADERS, files=files()).status_code == 503
        health = client.get("/health", headers=HEADERS).json()
        assert health["ready"] is True and health["last_inference_error"] == "ValueError"
        assert client.post("/v1/ocr", headers=HEADERS, files=files()).status_code == 200
        assert client.get("/health", headers=HEADERS).json()["last_inference_error"] is None


def test_failed_ocr_preload_is_not_ready():
    def fail():
        raise RuntimeError("Hayai unavailable")
    with TestClient(create_app(Mock, fail)) as client:
        response = client.get("/health", headers=HEADERS)
        assert response.status_code == 503
        assert response.json()["detector_loaded"] is True
        assert response.json()["ocr_model_loaded"] is False
        assert client.post("/v1/ocr", headers=HEADERS, files=files()).status_code == 503


def test_rle_roundtrip():
    for mask in [np.zeros((2, 3)), np.ones((2, 3)), np.array([[1, 0, 1], [0, 1, 0]])]:
        runs = rle(mask)
        decoded = []
        value = 0
        for count in runs:
            decoded.extend([value] * count)
            value = 1 - value
        assert decoded == mask.astype(int).reshape(-1).tolist()


@pytest.mark.parametrize("fail", [False, True])
def test_real_backend_warms_before_ready_and_failure_stays_unready(monkeypatch, tmp_path, fail):
    import sys
    from types import SimpleNamespace
    from contextlib import nullcontext
    from vision import model

    calls = []
    def predict(image, **options):
        calls.append((image.size, options))
        if fail:
            raise RuntimeError("warmup failed")
        return object()

    network = SimpleNamespace(model=SimpleNamespace(device="cuda", model=SimpleNamespace(eval=lambda: None)),
                              predict=predict)
    def load(module):
        module.load_model = lambda _: network

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        inference_mode=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "test-GPU",
                             synchronize=lambda: calls.append("synchronized"))))
    monkeypatch.setattr(model, "validate", lambda _: tmp_path)
    original = model.importlib.import_module
    monkeypatch.setattr(model.importlib, "import_module",
                        lambda name, *a, **kw: object() if name in {"cv2", "supervision"} else original(name, *a, **kw))
    monkeypatch.setattr(model.importlib.util, "spec_from_file_location",
                        lambda *_: SimpleNamespace(loader=SimpleNamespace(exec_module=load)))
    monkeypatch.setattr(model.importlib.util, "module_from_spec", lambda _: SimpleNamespace())
    with TestClient(create_app(lambda: model.Backend(str(tmp_path)))) as client:
        for _ in range(2):
            response = client.get("/health", headers=HEADERS)
            assert response.status_code == (503 if fail else 200)
            assert response.json()["ready"] is (not fail)
            assert response.json()["model_loaded"] is (not fail)
            if not fail:
                assert response.json()["warmup_complete"] is True
                assert response.json()["warmup_seconds"] >= 0
        assert calls[0][0] == (1280, 1536)
        assert calls[0][1]["include_source_image"] is False
        assert len([v for v in calls if isinstance(v, tuple)]) == (1 if fail else 2)
        assert ("synchronized" in calls) is (not fail)


@pytest.mark.parametrize("options", ['[]', 'bad', '{"unknown":1}', '{"max_tokens":true}',
                                   '{"max_tokens":0}', '{"temperature":3}', '{"prompt":""}', '{"max_new_tokens":0}',
                                   '{"max_new_tokens":2049}', '{"max_new_tokens":true}', '{"num_beams":5}',
                                   '{"repetition_penalty":NaN}', '{"repetition_penalty":0.5}',
                                   '{"length_penalty":3}', '{"early_stopping":"true"}'])
def test_stale_ocr_options_are_rejected_before_inference(options):
    class MustNotRun(MockOcr):
        def ocr(self, image, options):
            pytest.fail("Stale Gemma options reached Hayai")
    with TestClient(create_app(Mock, MustNotRun)) as client:
        assert client.post("/v1/ocr", headers=HEADERS, files=files(), data={"options": options}).status_code == 422


def test_hayai_defaults_and_custom_options_reach_resident_model():
    from vision.options import OcrOptions
    seen = []
    class Configured(MockOcr):
        def ocr(self, image, options):
            seen.append(options.model_dump())
            return super().ocr(image, options)
    custom = {"max_new_tokens":512,"num_beams":3,"repetition_penalty":1.2,
              "length_penalty":0.7,"early_stopping":False}
    import json
    with TestClient(create_app(Mock, Configured)) as client:
        health = client.get("/health", headers=HEADERS).json()
        assert health["capabilities"]["hayai_ocr_options"] is True
        assert health["ocr_defaults"] == OcrOptions().model_dump()
        assert client.post("/v1/ocr",headers=HEADERS,files=files()).status_code == 200
        assert client.post("/v1/ocr",headers=HEADERS,files=files(),data={"options":json.dumps(custom)}).status_code == 200
        assert client.post("/v1/ocr",headers=HEADERS,files=files()).status_code == 200
    assert seen == [OcrOptions().model_dump(),{"model":"",**custom},OcrOptions().model_dump()]
