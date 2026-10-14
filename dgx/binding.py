"""Host-side bind configuration; no model or dependency changes."""

import ipaddress
import json
import os
import tempfile


def address(value):
    return str(ipaddress.ip_address(value))


def health_url(value, port):
    ip = ipaddress.ip_address(value)
    host = ("::1" if ip.version == 6 else "127.0.0.1") if ip.is_unspecified else str(ip)
    if ip.version == 6:
        host = "[" + host + "]"
    return f"http://{host}:{port}/health"


def set_address(path, key, value, quoted=False):
    value = address(value)
    if path.is_symlink():
        raise ValueError("Refusing to replace a symlink configuration")
    lines = path.read_text().splitlines(keepends=True)
    entry = key + "=" + (json.dumps(value) if quoted else value) + "\n"
    output, replaced = [], False
    for line in lines:
        if line.strip().partition("=")[0].strip() == key:
            if not replaced:
                output.append(entry)
                replaced = True
        else:
            output.append(line)
    if not replaced:
        if output and not output[-1].endswith("\n"):
            output[-1] += "\n"
        output.append(entry)
    fd, temporary = tempfile.mkstemp(prefix=".bind-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write("".join(output))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
