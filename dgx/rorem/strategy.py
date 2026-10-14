"""All image geometry and final mask compositing live here for comparison runs.

Each request is resized with its aspect ratio preserved. The Remote client owns
region tiling; this server only applies bounded resolution/model-mask options.
Final compositing still uses the original request mask, including with dilation.
"""
from dataclasses import dataclass
from array import array
from typing import Protocol
from PIL import Image, ImageFilter

@dataclass
class PreparedImage:
    image: Image.Image
    mask: Image.Image
    original_size: tuple[int, int]

class InferenceStrategy(Protocol):
    name: str
    def prepare(self, source: Image.Image, mask: Image.Image) -> PreparedImage: ...
    def restore(self, result: Image.Image, prepared: PreparedImage) -> Image.Image: ...

class FullPageStrategy:
    name = 'full-page'

    def prepare(self, source, mask, max_side=1024, mask_dilation=0):
        w, h = source.size
        scale = min(1., max_side / max(w, h))
        size = (max(8, round(w * scale / 8) * 8), max(8, round(h * scale / 8) * 8))
        prepared_mask = mask.resize(size, Image.Resampling.NEAREST)
        if mask.getbbox() is not None and prepared_mask.getbbox() is None:
            # Nearest sampling can drop a very thin, non-empty selection. Use
            # float area coverage only in that case; retain its page-space mask
            # for final compositing so no unselected pixels can be overwritten.
            coverage = mask.convert('F').resize(size, Image.Resampling.BOX)
            values = array('f')
            values.frombytes(coverage.tobytes())
            prepared_mask.putdata([255 if v > 0 else 0 for v in values])
        if mask_dilation:
            prepared_mask = prepared_mask.filter(ImageFilter.MaxFilter(mask_dilation * 2 + 1))
        return PreparedImage(source.resize(size, Image.Resampling.LANCZOS), prepared_mask, (w, h))

    def restore(self, result, prepared):
        return result.resize(prepared.original_size, Image.Resampling.LANCZOS).convert('RGB')

def preserve_outside_mask(result, source, mask):
    if result.size != source.size:
        raise RuntimeError('backend changed dimensions')
    return Image.composite(result.convert('RGB'), source, mask)
