"""Hayai generation controls; defaults match the pinned Koharu model."""
from pydantic import BaseModel, ConfigDict, Field


class OcrOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str = Field(default="", max_length=512)
    max_new_tokens: int = Field(default=256, ge=1, le=2048, strict=True)
    repetition_penalty: float = Field(default=1.0, ge=1, le=2, strict=True)
    num_beams: int = Field(default=1, ge=1, le=4, strict=True)
    length_penalty: float = Field(default=1.0, ge=0.1, le=2, strict=True)
    early_stopping: bool = Field(default=True, strict=True)
