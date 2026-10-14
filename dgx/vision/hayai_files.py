"""Immutable Hayai v2 release used by Koharu 0.83.5; stdlib-only metadata."""

REPO = "JustANormalTinkerer/hayai-ocr-v2"
REV = "4a4ce477c9a8841f208b94e1d9ed5c0938965e05"
VISION_REPO = "google/siglip2-base-patch16-naflex"
VISION_REV = "b53b807d3a2d5e2b3911292f2d69e5341cdc064c"
FILES = {
    "model.safetensors": "4c645b221db8428cda04991be234c18133bb8861142a3d87cba04c5099b02328",
    "config.json": "581b762f1dfd55d0108f3f84e3f157bc762524af37fb0c19a7172a18b75582e2",
    "configuration_hayai.py": "47abd38cf1bae7aef27d01f5b8b4aa0960a7bc625a8afad79c4762ff5e5ed970",
    "modeling_hayai.py": "3d78976206549964abd55f776ab059e002adc72d2167daf168e46a12a5f4ae62",
    "tokenizer.json": "f8a0a909c628a684fe463094614e236a8b1d3609e7770f77e7beafaf1056bf13",
    "vision-config.json": "c0b8c2e7f0527b0bea1b1d9abe0381c0f294352df92439c54590b8420e539118",
}
URLS = {
    name: f"https://huggingface.co/{REPO}/resolve/{REV}/{name}" for name in FILES
}
URLS["vision-config.json"] = f"https://huggingface.co/{VISION_REPO}/resolve/{VISION_REV}/config.json"


def validate(folder):
    from .model_bundle import validate_bundle
    return validate_bundle(folder,REPO,REV,FILES,('configuration_hayai.py','modeling_hayai.py','vision-config.json'))
