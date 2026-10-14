"""Adopt and update RORem without the old source folder or dependency installation."""

import argparse
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from rorem.healthcheck import require_ready
from rorem.limits import read_max_pixels
from rorem.upload_limits import read_max_upload_bytes
from rorem.model_files import validate_models
from rorem.manifest import ensure_outside
from vision.auth import auth_required, validate_token
from binding import address, health_url, set_address

ROOT = Path(__file__).resolve().parent
PROJECT = "koharu-model-services"
DEFAULT_CONTAINER = "koharu-model-services-rorem-1"
IMAGE = "local/koharu-rorem:0.2.3"
UNSET = object()


def env_path():
    return ROOT / ".env.rorem"


def config():
    result = {}
    for line in env_path().read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        value = json.loads(value)
        if not isinstance(value, str):
            raise ValueError("RORem configuration values must be quoted strings")
        result[key] = value.replace("$$", "$")
    return result


def private_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(data)


def docker(*args, capture=False):
    result = subprocess.run(["docker", *args], check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout if capture else None


def compose(*args, capture=False):
    values = os.environ.copy()
    values.update(config())
    result = subprocess.run([
        "docker", "compose", "--project-name", PROJECT,
        "--project-directory", str(ROOT), "--file", str(ROOT / "compose.rorem.yaml"),
        "--env-file", str(env_path()), *args,
    ], check=True, text=True, env=values,
        stdout=subprocess.PIPE if capture else None)
    return result.stdout if capture else None


def inspect(identifier):
    records = json.loads(docker("inspect", identifier, capture=True))
    if len(records) != 1:
        raise ValueError("Expected exactly one RORem container")
    value = records[0]
    labels = value.get("Config", {}).get("Labels", {})
    if (labels.get("com.docker.compose.project") != PROJECT
            or labels.get("com.docker.compose.service") != "rorem"):
        raise ValueError("Container is not the expected Compose RORem service")
    return value


def current():
    identifiers = compose("ps", "--all", "-q", "rorem", capture=True).split()
    if not identifiers:
        return None
    if len(identifiers) != 1:
        raise ValueError("Expected one RORem service container")
    return inspect(identifiers[0])


def image_info(reference):
    values = json.loads(docker("image", "inspect", reference, capture=True))
    if len(values) != 1:
        raise ValueError("Expected one existing RORem image")
    value = values[0]
    if (value.get("Architecture") != "arm64"
            or value.get("Config", {}).get("Labels", {}).get("org.koharu.rorem.version") != "0.2.3"):
        raise ValueError("Expected the existing ARM64 RORem 0.2.3 image")
    return value


def validate_config(c):
    address(c["BIND_ADDRESS"])
    required = auth_required(c.get("ROREM_AUTH_REQUIRED", "false"))
    token = c.get("ROREM_API_TOKEN", "")
    validate_token(token, required)
    if required and len(token) < 24:
        raise ValueError("RORem authentication token must be at least 24 characters")
    read_max_pixels(c["ROREM_MAX_PIXELS"])
    read_max_upload_bytes(c.get("ROREM_MAX_UPLOAD_BYTES", "134217728"))
    models = validate_models(c["ROREM_PIPELINE_DIR"], c["ROREM_LORA_DIR"])
    ensure_outside(ROOT, c["ROREM_PIPELINE_DIR"], c["ROREM_LORA_DIR"])
    cache = Path(c["ROREM_CACHE_DIR"]).resolve()
    for key in ("ROREM_PIPELINE_DIR", "ROREM_LORA_DIR"):
        if cache.is_relative_to(Path(c[key]).resolve().parent):
            raise ValueError("Cache must be outside preserved model directories")
    if not cache.is_dir():
        raise ValueError("Existing RORem cache directory missing")
    return models


def adopt(container=DEFAULT_CONTAINER):
    if env_path().exists():
        raise ValueError(".env.rorem already exists; it will not be overwritten")
    if (ROOT / "state/rorem/adoption.json").exists():
        raise ValueError("Previous adoption record exists; inspect it before importing again")
    before = inspect(container)
    environment = dict(entry.split("=", 1) for entry in before["Config"]["Env"] if "=" in entry)
    mounts = {}
    for target in ("/models/pipeline", "/models/lora", "/cache"):
        matches = [m for m in before.get("Mounts", []) if m["Destination"] == target]
        if len(matches) != 1 or matches[0].get("Type") != "bind":
            raise ValueError("Expected existing bind mount for " + target)
        if target != "/cache" and matches[0].get("RW") is not False:
            raise ValueError("Preserved RORem models must be mounted read-only")
        mounts[target] = matches[0]["Source"]
    bindings = before["HostConfig"].get("PortBindings", {}).get("8765/tcp", [])
    if len(bindings) != 1 or bindings[0]["HostPort"] != "8765":
        raise ValueError("Expected one existing 8765/tcp binding")
    address = bindings[0]["HostIp"]
    default_auth = "false" if ipaddress.ip_address(address).is_loopback else "true"
    base = image_info(before["Image"])
    retained = "local/koharu-rorem:retained-" + uuid.uuid4().hex
    c = {
        "BIND_ADDRESS": address,
        "ROREM_PIPELINE_DIR": mounts["/models/pipeline"],
        "ROREM_LORA_DIR": mounts["/models/lora"],
        "ROREM_CACHE_DIR": mounts["/cache"],
        "ROREM_API_TOKEN": environment.get("ROREM_API_TOKEN", ""),
        "ROREM_AUTH_REQUIRED": environment.get("ROREM_AUTH_REQUIRED", default_auth),
        "ROREM_MAX_PIXELS": environment.get("ROREM_MAX_PIXELS", "64000000"),
        "ROREM_MAX_UPLOAD_BYTES": environment.get("ROREM_MAX_UPLOAD_BYTES", "134217728"),
        "ROREM_BASE_IMAGE": retained,
        "ROREM_BASE_IMAGE_ID": base["Id"],
        "ROREM_IMAGE": IMAGE,
    }
    validate_config(c)
    if inspect(container)["Id"] != before["Id"]:
        raise ValueError("RORem container changed during adoption")
    docker("image", "tag", base["Id"], retained)
    lines = ["# Adopted from Docker; private credentials, not committed."]
    lines += [key + "=" + json.dumps(value.replace("$", "$$")) for key, value in c.items()]
    private_write(env_path(), "\n".join(lines) + "\n")
    record = {"container_id": before["Id"], "image_id": base["Id"],
              "retained_image": retained, "models": mounts,
              "previous_working_directory": before["Config"].get("Labels", {}).get(
                  "com.docker.compose.project.working_dir")}
    private_write(ROOT / "state/rorem/adoption.json", json.dumps(record, indent=2) + "\n")
    print("ROREM_ADOPTED: configuration and image retained; live container unchanged")


def check():
    c = config()
    validate_config(c)
    base = image_info(c["ROREM_BASE_IMAGE"])
    if base["Id"] != c["ROREM_BASE_IMAGE_ID"]:
        raise ValueError("Retained RORem base image identity changed")
    compose("config", "--quiet")
    print("ROREM_CONFIG_AND_RETAINED_IMAGE_OK_GPU_NOT_TESTED")
    return c


def source_digest():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "rorem").rglob("*")):
        if path.is_file() and path.suffix in (".py", ".json"):
            digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build():
    c = check()
    docker("build", "--pull=false", "--network=none", "-f", str(ROOT / "Dockerfile.rorem"),
           "--build-arg", "ROREM_BASE_IMAGE=" + c["ROREM_BASE_IMAGE"],
           "--build-arg", "ROREM_SOURCE_SHA=" + source_digest(), "-t", c["ROREM_IMAGE"], str(ROOT))


def request_health(bind_address=None):
    c = config()
    headers = {}
    if c.get("ROREM_API_TOKEN"):
        headers["Authorization"] = "Bearer " + c["ROREM_API_TOKEN"]
    request = urllib.request.Request(health_url(bind_address or c["BIND_ADDRESS"], 8765), headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=10) as response:
        return require_ready(json.load(response))


def idle(container):
    if container and container["State"]["Status"] == "running":
        # The running container may still use the old address after editing .env.rorem.
        bindings = container["HostConfig"].get("PortBindings", {}).get("8765/tcp", [])
        if len(bindings) != 1 or bindings[0]["HostPort"] != "8765":
            raise ValueError("Expected one existing 8765/tcp binding")
        if request_health(bind_address=bindings[0]["HostIp"]).get("busy"):
            raise ValueError("RORem is busy; no service replacement performed")


def ready_image(c):
    labels = image_info(c["ROREM_IMAGE"])["Config"].get("Labels", {})
    if (labels.get("org.koharu.rorem.deployment") != "unified-1"
            or labels.get("org.koharu.rorem.source") != source_digest()):
        raise ValueError("RORem image does not contain current checkout source; run rorem build")


def up(expected=UNSET):
    c = check()
    ready_image(c)
    before = current()
    if expected is not UNSET and (before or {}).get("Id") != expected:
        raise ValueError("RORem container changed during build; no replacement performed")
    idle(before)
    if before:
        retained = "local/koharu-rorem:rollback-" + uuid.uuid4().hex
        docker("image", "tag", before["Image"], retained)
        private_write(ROOT / "state/rorem" / ("update-" + uuid.uuid4().hex + ".json"),
                      json.dumps({"previous_container": before["Id"], "rollback_image": retained,
                                  "new_source": source_digest()}, indent=2) + "\n")
    compose("up", "--no-build", "--no-deps", "-d", "--force-recreate", "rorem")


def wait_ready(timeout=900):
    deadline = time.monotonic() + timeout
    while True:
        try:
            data = request_health()
            if data.get("busy"):
                raise ValueError("RORem is busy")
            print(json.dumps(data, indent=2))
            return
        except (OSError, ValueError, RuntimeError, urllib.error.URLError):
            if time.monotonic() >= deadline:
                raise ValueError("RORem not ready; run manage.py rorem logs") from None
            print("Waiting for RORem model preload...", flush=True)
            time.sleep(10)


def update(timeout=900):
    check()
    before = current()
    idle(before)
    build()
    up(expected=before["Id"] if before else None)
    wait_ready(timeout)


def remove():
    check()
    before = current()
    if before:
        idle(before)
        docker("stop", before["Id"])
        docker("rm", before["Id"])
    print("RORem container removed; retained image, models, cache and configuration preserved")


def doctor():
    check()
    before = current()
    if not before:
        raise ValueError("RORem container missing")
    script = ("import json,platform; import torch; "
              "print(json.dumps(dict(python=platform.python_version(),architecture=platform.machine(),"
              "torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available())))")
    runtime = json.loads(compose("exec", "-T", "rorem", "python", "-c", script, capture=True))
    data = {"container_id": before["Id"], "image_id": before["Image"], "runtime": runtime,
            "health": request_health(), "source_sha256": source_digest(),
            "status": "DIAGNOSTIC_SNAPSHOT_NOT_INFERENCE_ACCEPTANCE"}
    path = ROOT / "state/rorem" / ("doctor-" + uuid.uuid4().hex + ".json")
    private_write(path, json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("adopt")
    p.add_argument("--container", default=DEFAULT_CONTAINER)
    p = sub.add_parser("bind")
    p.add_argument("address", help="Host IP address (for example 0.0.0.0)")
    for name in ("check", "build", "up", "health", "logs", "doctor", "stop", "remove"):
        sub.add_parser(name)
    for name in ("update", "wait"):
        p = sub.add_parser(name)
        p.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if args.command == "bind":
        set_address(env_path(), "BIND_ADDRESS", args.address, quoted=True)
        print("RORem bind configured; run manage.py rorem up to apply")
    elif args.command == "adopt":
        adopt(args.container)
    elif args.command == "health":
        print(json.dumps(request_health(), indent=2))
    elif args.command == "logs":
        compose("logs", "--tail", "100", "rorem")
    elif args.command == "stop":
        container = current()
        idle(container)
        if container:
            docker("stop", container["Id"])
    elif args.command in ("wait", "update"):
        (wait_ready if args.command == "wait" else update)(args.timeout)
    else:
        globals()[args.command]()


if __name__ == "__main__":
    main()
