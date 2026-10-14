import asyncio, hmac, io, json, logging, os
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from .model import Backend, DEFAULTS, REPO
from .auth import auth_required, validate_token
from .hayai import Backend as HayaiBackend
from .hayai_files import REPO as OCR_REPO, REV as OCR_REV
from .options import OcrOptions
from pydantic import ValidationError

LOG = logging.getLogger("uvicorn.error")
Image.MAX_IMAGE_PIXELS = 64_000_000


def create_app(factory=None, ocr_factory=None):
    required = auth_required(os.environ.get("VISION_AUTH_REQUIRED", "false"))
    api_token = os.environ.get("VISION_API_TOKEN", "")
    state = {"backend": None, "ocr": None, "error": None, "last_error": None, "busy": False}
    gate = asyncio.Lock()
    security = HTTPBearer(auto_error=False)

    @asynccontextmanager
    async def lifespan(app):
        try:
            validate_token(api_token, required)
            state["backend"] = await asyncio.to_thread(
                factory or (lambda: Backend(os.environ["VISION_MODEL_DIR"]))
            )
            state["ocr"] = await asyncio.to_thread(
                ocr_factory or (lambda: HayaiBackend(os.environ["VISION_OCR_MODEL_DIR"]))
            )
        except Exception as error:
            state["error"] = f"Preload failed: {type(error).__name__}"
            LOG.exception(state["error"])
        yield  # Model remains resident until container termination.

    app = FastAPI(lifespan=lifespan)
    app.state.status = state

    def auth(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
        if not required:
            return
        if (
            not api_token
            or not credentials
            or credentials.scheme.lower() != "bearer"
            or not hmac.compare_digest(credentials.credentials, api_token)
        ):
            raise HTTPException(
                401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"}
            )

    def ready(stage="backend"):
        if state[stage] is None:
            raise HTTPException(503, "Vision model not loaded")
        return state[stage]

    async def decode(upload):
        data = await upload.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024:
            raise HTTPException(413, "Upload too large")
        try:
            with Image.open(io.BytesIO(data)) as opened:
                if opened.width * opened.height > 64_000_000:
                    raise ValueError("Too many pixels")
                opened.load()
                return opened.convert("RGB")
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ):
            raise HTTPException(422, "Invalid image or size")

    async def infer(operation, *arguments):
        async with gate:
            state["busy"] = True
            try:
                # A disconnected client must not release the gate while its CUDA thread still runs.
                worker = asyncio.create_task(
                    asyncio.to_thread(operation, *arguments)
                )
                try:
                    result = await asyncio.shield(worker)
                except asyncio.CancelledError:
                    while not worker.done():
                        try:
                            await asyncio.shield(worker)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    if not worker.cancelled():
                        worker.exception()
                    raise
                state["last_error"] = None
                return result
            except Exception as error:
                state["last_error"] = type(error).__name__
                LOG.exception("Vision inference failed: %s", type(error).__name__)
                raise HTTPException(
                    503,
                    "Vision inference failed; model retained. Check service diagnostics",
                )
            finally:
                state["busy"] = False

    @app.get("/health", dependencies=[Depends(auth)])
    def health():
        backend = state["backend"]
        ocr_backend = state["ocr"]
        loaded = backend is not None and ocr_backend is not None
        return JSONResponse(
            {
                "protocol": 1,
                "service": "koharu-vision",
                "ready": loaded,
                "model_loaded": loaded,
                "model": getattr(backend,"model_name",REPO),
                "device": getattr(backend, "device", None),
                "gpu_name": getattr(backend, "gpu_name", None),
                "busy": state["busy"],
                "auth_required": required,
                "detector_loaded": backend is not None,
                "ocr_backend": "hayai",
                "ocr_model": getattr(ocr_backend,"model_name",OCR_REPO),
                "ocr_revision": getattr(ocr_backend,"revision",OCR_REV),
                "ocr_model_loaded": ocr_backend is not None,
                "ocr_warmup_complete": getattr(ocr_backend, "warmup_complete", None),
                "ocr_warmup_seconds": getattr(ocr_backend, "warmup_seconds", None),
                "error": state["error"],
                "last_inference_error": state["last_error"],
                "warmup_complete": getattr(backend, "warmup_complete", None),
                "warmup_seconds": getattr(backend, "warmup_seconds", None),
                "warmup_timings": getattr(backend, "warmup_timings", None),
                "ocr_defaults": OcrOptions().model_dump(),
                "capabilities": {"hayai_ocr": True, "hayai_ocr_options": True, "detection_thresholds": True},
            },
            status_code=200 if loaded else 503,
        )

    @app.post("/v1/detect", dependencies=[Depends(auth)])
    async def detect(image: UploadFile = File(...), options: str = Form("{}")):
        backend = ready()
        source = await decode(image)
        try:
            values = json.loads(options)
            if not isinstance(values, dict) or set(values) - (set(DEFAULTS) | {"model"}):
                raise ValueError()
            selected_model=values.pop("model", "")
            if selected_model and selected_model != getattr(backend,"model_name",REPO):
                raise ValueError("Selected detection model is not loaded")
            thresholds = {**DEFAULTS, **values}
            if any(
                type(v) not in (int, float) or not 0 <= v <= 1
                for v in thresholds.values()
            ):
                raise ValueError()
        except (ValueError, TypeError):
            raise HTTPException(422, "Invalid detection thresholds")
        return await infer(backend.detect, source, thresholds)

    @app.post("/v1/ocr", dependencies=[Depends(auth)])
    async def ocr(image: UploadFile = File(...), options: str = Form("{}")):
        backend = ready("ocr")
        try:
            opts = OcrOptions.model_validate_json(options)
            if opts.model and opts.model != getattr(backend,"model_name",OCR_REPO):
                raise HTTPException(409,"Selected OCR model is not loaded; update the Vision model path and restart")
        except ValidationError:
            raise HTTPException(422, "Invalid Hayai OCR options")
        source = await decode(image)
        return await infer(backend.ocr, source, opts)

    return app


app = create_app()
