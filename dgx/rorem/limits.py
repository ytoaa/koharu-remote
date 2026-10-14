"""Bounded input decoding; does not change full-page inference geometry."""
import os

DEFAULT_MAX_PIXELS = 64_000_000
HARD_MAX_PIXELS = 128_000_000

def read_max_pixels(value=None):
    value = os.environ.get('ROREM_MAX_PIXELS', str(DEFAULT_MAX_PIXELS)) if value is None else value
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError('ROREM_MAX_PIXELS must be an integer between 64 and 128000000')
    number = int(value)
    if not 64 <= number <= HARD_MAX_PIXELS:
        raise ValueError('ROREM_MAX_PIXELS must be an integer between 64 and 128000000')
    return number
