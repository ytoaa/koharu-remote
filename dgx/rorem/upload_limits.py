"""Encoded upload size limit, independent of decoded pixel/inference limits."""

import os

DEFAULT_MAX_UPLOAD_BYTES = 128 * 1024 * 1024
HARD_MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def read_max_upload_bytes(value=None):
    value = os.environ.get('ROREM_MAX_UPLOAD_BYTES', str(DEFAULT_MAX_UPLOAD_BYTES)) if value is None else value
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError('ROREM_MAX_UPLOAD_BYTES must be an integer between 1024 and 536870912')
    number = int(value)
    if not 1024 <= number <= HARD_MAX_UPLOAD_BYTES:
        raise ValueError('ROREM_MAX_UPLOAD_BYTES must be an integer between 1024 and 536870912')
    return number
