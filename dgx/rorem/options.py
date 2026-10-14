from pydantic import BaseModel, ConfigDict, Field, model_validator

class Options(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    model: str = 'RORem-4S'
    steps: int = Field(default=4, ge=2, le=30)
    strength: float = Field(default=.99, gt=0, lt=1)
    guidance_scale: float = Field(default=1.0, ge=0, le=20)
    prompt: str = Field(default='', max_length=2000)
    negative_prompt: str = Field(default='', max_length=2000)
    seed: int = Field(default=0, ge=-1, le=2147483647, strict=True)
    max_side: int = Field(default=1024, ge=256, le=1024, strict=True)
    mask_dilation: int = Field(default=0, ge=0, le=32, strict=True)
    @model_validator(mode='after')
    def supported(self):
        if self.model != 'RORem-4S': raise ValueError('Unsupported model')
        if int(self.steps * self.strength) < 1: raise ValueError('steps * strength must be >= 1')
        return self
