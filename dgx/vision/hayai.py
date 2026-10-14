"""Resident Hayai OCR. Koharu's pinned weights, RGB NaFlex input and greedy defaults."""

import json, logging, math, time, types
import numpy as np
from PIL import Image
from .hayai_files import REPO, REV, validate
from .options import OcrOptions

MAX_NEW_TOKENS = 256
MAX_PATCHES = 256
PATCH_SIZE = 16
LOG = logging.getLogger(__name__)


def preprocess(image, max_patches=MAX_PATCHES):
    if type(max_patches) is not int or max_patches not in (256,512,1024):
        raise ValueError("Hayai max_patches must be 256, 512 or 1024")
    # Same binary search as Koharu's Hayai ImageProcessor, including tiny crops.
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Empty OCR crop")
    def scaled(scale, size):
        return max(PATCH_SIZE, math.ceil(size * scale / PATCH_SIZE) * PATCH_SIZE)
    low, high = 1e-6, 100.0
    while high - low >= 1e-5:
        mid = (low + high) / 2
        if scaled(mid, height) * scaled(mid, width) // PATCH_SIZE**2 <= max_patches:
            low = mid
        else:
            high = mid
    target_h, target_w = scaled(low, height), scaled(low, width)
    rgb = image.convert("RGB")
    if rgb.size != (target_w, target_h):
        rgb = rgb.resize((target_w, target_h), Image.Resampling.BILINEAR)
    pixels = np.asarray(rgb, dtype=np.float32) / np.float32(255)
    pixels = (pixels - np.float32(0.5)) / np.float32(0.5)
    rows, columns = target_h // PATCH_SIZE, target_w // PATCH_SIZE
    patches = pixels.reshape(rows, PATCH_SIZE, columns, PATCH_SIZE, 3)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(rows * columns, -1)
    padded = np.zeros((1, max_patches, PATCH_SIZE**2 * 3), dtype=np.float32)
    padded[0, :len(patches)] = patches
    mask = np.zeros((1, max_patches), dtype=np.int32)
    mask[0, :len(patches)] = 1
    return padded, mask, np.array([[rows, columns]], dtype=np.int64)


class HayaiOutputError(ValueError):
    MESSAGES = {
        'ocr_token_limit': 'Hayai output reached token limit without EOS/PAD',
        'ocr_invalid_output': 'Invalid Hayai output',
        'ocr_repetition_loop': 'Hayai output contains an excessive repetition loop',
    }
    def __init__(self, code):
        self.code = code
        super().__init__(code + ': ' + self.MESSAGES[code])


def validate_result(text, token_count, max_new_tokens=MAX_NEW_TOKENS):
    # Validation happens after decoding; never tune generation or silently trim text.
    if not isinstance(text, str) or len(text) > 65536 or "\0" in text:
        raise HayaiOutputError("ocr_invalid_output")
    # Long loops are suspicious, while manga effects such as どぅぅぅ stay valid.
    for period in range(1, 9):
        for start in range(max(0, len(text) - 32 + 1)):
            block = text[start:start + period]
            if block and text.startswith(block * max(8, math.ceil(32 / period)), start):
                raise HayaiOutputError("ocr_repetition_loop")
    # A loop can also exhaust the token budget. Classify its known cause first
    # so the Comfy node's existing bounded repetition retry can run. Ordinary
    # truncated text is still rejected and is never silently returned.
    if token_count >= max_new_tokens:
        raise HayaiOutputError("ocr_token_limit")
    return text


class DecodeRecorder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.token_to_id("<bos>")
        self.eos_token_id = tokenizer.token_to_id("<eos>")
        self.pad_token_id = tokenizer.token_to_id("<pad>")
        if None in (self.bos_token_id, self.eos_token_id, self.pad_token_id):
            raise ValueError("Hayai tokenizer lacks required special tokens")
        self.token_count = None

    def decode(self, tokens, skip_special_tokens=True):
        self.token_count = len(tokens)
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)


def load_model(folder, device="cuda"):
    import torch
    from safetensors.torch import load_file
    # Only hash-verified, immutable release code is executed. No trust_remote_code
    # discovery and no runtime downloads. Wire the release's package Config import.
    config_module = types.ModuleType("pinned_hayai_configuration")
    exec(compile((folder / "configuration_hayai.py").read_text(), "configuration_hayai.py", "exec"), config_module.__dict__)
    config = config_module.HayaiConfig(**json.loads((folder / "config.json").read_text()))
    config.vision_config = json.loads((folder / "vision-config.json").read_text())["vision_config"]
    source = (folder / "modeling_hayai.py").read_text()
    changes = {
        "try:\n    from .configuration_hayai import HayaiConfig\nexcept ImportError:\n    from configuration_hayai import HayaiConfig": "# Config supplied by the pinned local loader",
        'Siglip2VisionConfig.from_pretrained("google/siglip2-base-patch16-naflex")':
            "Siglip2VisionConfig.from_dict(config.vision_config)",
        # Koharu selects bf16 on CUDA capability >= 8.0, fp32 otherwise.
        # Disable the Python release's unconditional CUDA fp16 autocast.
        'enabled=(device.type == "cuda")': "enabled=False",
    }
    for original, replacement in changes.items():
        if source.count(original) != 1:
            raise ValueError("Pinned Hayai runtime adapter no longer matches")
        source = source.replace(original, replacement)
    module = types.ModuleType("pinned_hayai_model")
    module.HayaiConfig = config_module.HayaiConfig
    exec(compile(source, "modeling_hayai.py", "exec"), module.__dict__)
    model = module.HayaiModel(config)
    model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
    # Hayai consumes only last_hidden_state, just like Koharu's native encoder.
    # Transformers also computes an unused pooled output by default; its BF16
    # MultiheadAttention failed during GB10 startup. Retain/verify all checkpoint
    # weights, then skip only this unused branch, without changing OCR precision.
    model.vision_encoder.use_head = False
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
    return model.to(device=device, dtype=dtype).eval()


class Backend:
    def __init__(self, folder):
        import torch
        from tokenizers import Tokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU fallback disabled")
        folder = validate(folder)
        from .model_bundle import identity
        self.model_name,self.revision=identity(folder,REPO,REV)
        self.model = load_model(folder)
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        if self.tokenizer.get_vocab_size() != self.model.config.vocab_size:
            raise ValueError("Hayai vocabulary mismatch")
        self.device = "cuda"
        self.warmup_complete = False
        started = time.monotonic()
        self._generate(Image.new("RGB", (96, 64), "white"))
        torch.cuda.synchronize()
        self.warmup_seconds = round(time.monotonic() - started, 3)
        self.warmup_complete = True

    def _generate(self, image, options=None, max_patches=MAX_PATCHES):
        import torch
        options = options if options is not None else OcrOptions()
        pixels, mask, shape = preprocess(image, max_patches)
        LOG.info('Hayai input: crop=%sx%s, model=%sx%s, patch_budget=%s',
                 image.width, image.height, int(shape[0,1])*PATCH_SIZE, int(shape[0,0])*PATCH_SIZE, max_patches)
        recorder = DecodeRecorder(self.tokenizer)
        with torch.inference_mode():
            output = self.model.generate(
                pixel_values=torch.from_numpy(pixels).to(device=self.device, dtype=next(self.model.parameters()).dtype),
                pixel_attention_mask=torch.from_numpy(mask).to(self.device),
                spatial_shapes=torch.from_numpy(shape).to(self.device),
                tokenizer=recorder, **options.model_dump(exclude={"model"}),
            )
        if len(output) != 1 or recorder.token_count is None:
            raise ValueError("Hayai returned an invalid batch")
        return output[0], recorder.token_count

    def ocr(self, image, options, max_patches=MAX_PATCHES):
        text, count = self._generate(image, options, max_patches)
        try:
            text = validate_result(text, count, options.max_new_tokens)
        except HayaiOutputError as exc:
            LOG.warning('Hayai output rejected: code=%s, crop=%sx%s, tokens=%s/%s, characters=%s, patch_budget=%s, repetition_penalty=%s, num_beams=%s',
                        exc.code,image.width,image.height,count,options.max_new_tokens,len(text) if isinstance(text,str) else None,
                        max_patches,options.repetition_penalty,options.num_beams)
            raise
        return {"protocol": 1, "text": text,
                "model": self.model_name, "revision": self.revision}
