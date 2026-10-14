"""Authentication is an explicit server setting, independent of stored credentials."""


def auth_required(value):
    value = value.strip().lower()
    if value in ("", "0", "false", "no"):
        return False
    if value in ("1", "true", "yes"):
        return True
    raise ValueError("AUTH_REQUIRED must be true or false")


def validate_token(value, required):
    if required and not value:
        raise ValueError("API token required when authentication is enabled")
    if value and (not value.isascii() or any(ord(c) <= 32 or ord(c) >= 127 for c in value)):
        raise ValueError("API token contains invalid whitespace or characters")
