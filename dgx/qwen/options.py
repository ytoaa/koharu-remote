from pydantic import BaseModel, ConfigDict, Field
from .model_files import MODEL

DEFAULT_PROMPT = 'Remove all text and sound effect lettering from <image1>. Keep all speech bubbles, their outlines, and white interiors unchanged. Restore the background behind the lettering and preserve the original artwork.'


class Options(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    model: str = Field(default=MODEL, max_length=512)
    text_encoder: str = Field(default='', max_length=512)
    vae: str = Field(default='', max_length=512)
    prompt: str = Field(default=DEFAULT_PROMPT, max_length=2000)
    seed: int = Field(default=0, ge=-1, le=2147483647, strict=True)
    max_side: int = Field(default=2048, ge=256, le=4096, strict=True)
    mask_dilation: int = Field(default=0, ge=0, le=32, strict=True)
    use_kv_cache: bool = Field(default=True, strict=True)
