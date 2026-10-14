"""Pinned RF-DETR inference; masks are local RLE in ORIGINAL page coordinates."""

from pathlib import Path
import hashlib, importlib.util, json, logging, time
from PIL import Image
import numpy as np

from .model_files import REPO, REV, WEIGHT_SHA, FILES

LABELS = ("text", "onomatopoeia", "bubble", "panel")
DEFAULTS = {"text": 0.25, "onomatopoeia": 0.20, "bubble": 0.50, "panel": 0.50}
LOG = logging.getLogger("uvicorn.error")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(folder):
    folder = Path(folder)
    from .model_bundle import validate_bundle
    validate_bundle(folder,REPO,REV,FILES,('load_model.py',))
    cfg = json.loads((folder / "inference_config.json").read_text())
    if cfg["checkpoint_sha256"] != sha(folder / "model.safetensors") or cfg["rfdetr_version"] != "1.7.0":
        raise ValueError("Detector config mismatch")
    return folder


def rle(mask):
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    change = np.flatnonzero(np.diff(flat) != 0) + 1
    runs = np.diff(np.r_[0, change, len(flat)]).astype(int).tolist()
    if len(flat) and flat[0]:
        runs.insert(0, 0)
    return runs


class Backend:
    def __init__(self, folder):
        import torch

        # RF-DETR imports supervision lazily inside predict(). Check its native
        # dependencies before reporting a loaded model through health.
        importlib.import_module("cv2")
        importlib.import_module("supervision")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU fallback disabled")
        folder = validate(folder)
        from .model_bundle import identity
        self.model_name,self.revision=identity(folder,REPO,REV)
        spec = importlib.util.spec_from_file_location(
            "pinned_koharu_layout", folder / "load_model.py"
        )
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        self.model = loader.load_model(folder / "model.safetensors")
        if str(self.model.model.device).split(":")[0] != "cuda":
            raise RuntimeError("Detector did not select CUDA")
        self.model.model.model.eval()
        self.gpu_name = torch.cuda.get_device_name()
        self.device = "cuda"
        self.warmup_complete = False
        self.warmup_seconds = None
        self.warmup_timings = []
        self.warmup()

    def warmup(self):
        import torch

        started = time.monotonic()
        LOG.info("Detector warmup started: input=1280x1536, model=1152x1152; health waits")
        # A non-square, larger source also exercises CUDA resize; an already
        # 1152-square source would skip that first-request preparation path.
        with torch.inference_mode():
            for run in range(2):
                step = time.monotonic()
                self.model.predict(Image.new("RGB", (1280, 1536), "white"),
                                   threshold=min(DEFAULTS.values()), shape=(1152, 1152),
                                   include_source_image=False)
                torch.cuda.synchronize()
                elapsed = round(time.monotonic() - step, 3)
                self.warmup_timings.append(elapsed)
                LOG.info("Detector warmup run %d completed: %.3fs", run + 1, elapsed)
        self.warmup_seconds = round(time.monotonic() - started, 3)
        self.warmup_complete = True
        LOG.info("Detector warmup completed: %.3fs", self.warmup_seconds)

    def detect(self, image, thresholds):
        import torch

        width, height = image.size
        started = time.monotonic()
        # Preserve the official RF-DETR input boundary: one model-side resize,
        # with original-resolution boxes/masks. No extra page resize or tile/crop inference.
        with torch.inference_mode():
            found = self.model.predict(
                image,
                threshold=min(thresholds.values()),
                shape=(1152, 1152),
                include_source_image=False,
            )
        predicted = time.monotonic()
        results = []
        mask_pixels = 0
        run_count = 0
        if found.mask is None and len(found):
            raise RuntimeError("Detector must return segmentation masks")
        for box, mask, class_id, score in zip(
            found.xyxy,
            found.mask if found.mask is not None else [],
            found.class_id,
            found.confidence,
        ):
            class_id = int(class_id)
            score = float(score)
            if not 0 <= class_id < 4:
                raise RuntimeError("Unknown detector class")
            label = LABELS[class_id]
            if score < thresholds[label]:
                continue
            if mask.shape != (height, width):
                raise RuntimeError("Detector mask coordinates differ from source")
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            x0 = int(xs.min())
            y0 = int(ys.min())
            x1 = int(xs.max()) + 1
            y1 = int(ys.max()) + 1
            local = mask[y0:y1, x0:x1]
            mask_pixels += local.size
            if local.size > 64 * 1024 * 1024 or mask_pixels > 256 * 1024 * 1024:
                raise RuntimeError("Detection masks exceed client memory contract")
            bounds = [
                max(0.0, float(box[0])),
                max(0.0, float(box[1])),
                min(float(width), float(box[2])),
                min(float(height), float(box[3])),
            ]
            if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                continue
            runs = rle(local)
            run_count += len(runs)
            if run_count > 4_000_000:
                raise RuntimeError("Detection RLE exceeds response limit")
            results.append(
                {
                    "label_id": class_id,
                    "label": label,
                    "score": score,
                    "bbox": bounds,
                    "mask": {
                        "x": x0,
                        "y": y0,
                        "width": x1 - x0,
                        "height": y1 - y0,
                        "runs": runs,
                    },
                }
            )
        LOG.info("Detection completed: %dx%d; %d regions; predict=%.3fs; encode=%.3fs",
                 width, height, len(results), predicted - started,
                 time.monotonic() - predicted)
        return {
            "protocol": 1,
            "image_width": width,
            "image_height": height,
            "model": self.model_name,
            "revision": self.revision,
            "detections": results,
        }
