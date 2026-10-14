"""Manage ONLY the new koharu-vision Compose project. Standard-library host tool."""

from pathlib import Path
import argparse, hashlib, json, os, subprocess, urllib.request
from vision.model_files import FILES, REPO, REV
from vision.auth import auth_required, validate_token
from vision import hayai_files as ocr_files
from binding import address, health_url, set_address

ROOT = Path(__file__).resolve().parent
MODEL = Path("/srv/koharu/models/koharu-layout-rfdetr-seg-2xl-1152")
OCR_MODEL = Path("/srv/koharu/models/hayai-ocr-v2")
CACHE = Path("/srv/koharu/cache/vision")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def env():
    values = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError("Invalid .env entry")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def validate(folder):
    from vision.model_bundle import validate_bundle
    return validate_bundle(folder,REPO,REV,FILES,('load_model.py',))


def compose(*args, capture=False):
    values = os.environ.copy()
    values.update(env())
    return subprocess.run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(ROOT),
            "-f",
            str(ROOT / "compose.yaml"),
            *args,
        ],
        check=True,
        text=True,
        capture_output=capture,
        env=values,
    )


def check():
    values = env()
    address(values.get("VISION_BIND_ADDRESS", "127.0.0.1"))
    validate(values["VISION_MODEL_DIR"])
    validate_token(values.get("VISION_API_TOKEN", ""),
                   auth_required(values.get("VISION_AUTH_REQUIRED", "false")))
    ocr_files.validate(values["VISION_OCR_MODEL_DIR"])
    if not Path(values["VISION_CACHE_DIR"]).is_dir():
        raise ValueError("Vision cache missing")
    compose("config", "--quiet")
    print("VISION_FILES_AND_COMPOSE_OK_GPU_NOT_TESTED")
    return values


def download_files(folder, repository, revision, files, urls):
    folder.mkdir(parents=True, exist_ok=True)
    for name, digest in files.items():
        destination = folder / name
        if destination.exists():
            if sha(destination) != digest:
                raise ValueError("Existing model file differs; refusing replacement: " + name)
            continue
        temporary = folder / (name + ".download")
        if temporary.exists():
            raise ValueError("Incomplete model download exists; inspect it before retrying: " + name)
        with urllib.request.urlopen(urls[name], timeout=120) as response, temporary.open("xb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
        if sha(temporary) != digest:
            raise ValueError("Downloaded model hash mismatch: " + name)
        # Exclusive creation: preserve existing files and incomplete downloads.
        os.link(temporary, destination)
        temporary.unlink()
    manifest = folder / "model-files.json"
    expected = {"repository": repository, "revision": revision, "files": files}
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError("Existing manifest differs")
    else:
        with manifest.open("x") as output:
            json.dump(expected, output, indent=2)


def prepare(download=False, download_ocr=False):
    values = env() if (ROOT / ".env").exists() else {}
    folder = Path(values.get("VISION_MODEL_DIR", MODEL))
    ocr_folder = Path(values.get("VISION_OCR_MODEL_DIR", OCR_MODEL))
    if download:
        urls = {name: f"https://huggingface.co/{REPO}/resolve/{REV}/{name}" for name in FILES}
        download_files(folder, REPO, REV, FILES, urls)
    validate(folder)
    if download_ocr:
        download_files(ocr_folder, ocr_files.REPO, ocr_files.REV, ocr_files.FILES, ocr_files.URLS)
    ocr_files.validate(ocr_folder)
    cache = Path(values.get("VISION_CACHE_DIR", CACHE))
    cache.mkdir(parents=True, exist_ok=True)
    if not values:
        data = {
            "VISION_BASE_IMAGE": "local/koharu-rorem:0.2.3",
            "VISION_MODEL_DIR": str(folder),
            "VISION_OCR_MODEL_DIR": str(ocr_folder),
            "VISION_CACHE_DIR": str(cache),
            "VISION_BIND_ADDRESS": "127.0.0.1",
            "VISION_AUTH_REQUIRED": "false",
            "VISION_API_TOKEN": "",
        }
        fd = os.open(ROOT / ".env", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as file:
            file.write("".join(f"{k}={v}\n" for k, v in data.items()))
    elif "VISION_OCR_MODEL_DIR" not in values:
        # Add only the new model path, preserving existing credentials and paths.
        with (ROOT / ".env").open("a") as file:
            file.write("\nVISION_OCR_MODEL_DIR=" + str(ocr_folder) + "\n")
    print("Prepared detector and Hayai for one Vision service. Tokens are not printed.")
    check()


def health():
    values = env()
    request = urllib.request.Request(
        health_url(values.get("VISION_BIND_ADDRESS", "127.0.0.1"), 8766),
        headers={"Authorization": "Bearer " + values["VISION_API_TOKEN"]}
        if values.get("VISION_API_TOKEN") else {},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=5) as response:
        if response.status != 200:
            raise ValueError("Vision health HTTP status mismatch")
        data = json.load(response)
    print(json.dumps(data, indent=2))
    if not (
        type(data.get("protocol")) is int
        and data["protocol"] == 1
        and data.get("service") == "koharu-vision"
        and data.get("ready") is True
        and data.get("model_loaded") is True
        and data.get("ocr_model_loaded") is True
        and data.get("ocr_model") == ocr_files.REPO
        and data.get("ocr_revision") == ocr_files.REV
    ):
        raise ValueError("Vision health not ready")


def doctor():
    check()
    result = json.loads(
        compose(
            "exec", "-T", "vision", "python", "-m", "vision.doctor", capture=True
        ).stdout
    )
    identifier = compose("ps", "-q", "vision", capture=True).stdout.strip()
    inspect = json.loads(
        subprocess.check_output(["docker", "inspect", identifier], text=True)
    )[0]
    result["image_id"] = inspect["Image"]
    image = json.loads(
        subprocess.check_output(
            ["docker", "image", "inspect", inspect["Image"]], text=True
        )
    )[0]
    result["image_digests"] = image.get("RepoDigests", [])
    result["image_architecture"] = image["Architecture"]
    state = ROOT / "state"
    state.mkdir(exist_ok=True)
    (state / "doctor.json").write_text(json.dumps(result, indent=2) + "\n")
    freeze = compose(
        "exec", "-T", "vision", "cat", "/app/vision-build-packages.txt", capture=True
    ).stdout
    (state / "build-packages.txt").write_text(freeze)
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "check", "build", "up", "health", "doctor", "logs", "stop", "bind", "models"],
    )
    parser.add_argument("address", nargs="?", help="IP address for bind (for example 0.0.0.0)")
    parser.add_argument(
        "--download-detector",
        action="store_true",
        help="Download only the NEW pinned layout detector; never RORem",
    )
    parser.add_argument("--download-ocr", action="store_true",
                        help="Download Koharu's pinned Hayai OCR v2 release")
    parser.add_argument("--ocr-dir")
    parser.add_argument("--detector-dir")
    args = parser.parse_args(argv)
    if (args.command == "bind") != (args.address is not None):
        parser.error("bind requires an IP address; other commands do not accept one")
    if (args.download_detector or args.download_ocr) and args.command != "prepare":
        parser.error("Model download flags require prepare")
    if args.command == "models":
        if not args.ocr_dir and not args.detector_dir: parser.error("models needs --ocr-dir or --detector-dir")
        values=env()
        if args.ocr_dir:
            ocr_files.validate(args.ocr_dir); values['VISION_OCR_MODEL_DIR']=str(Path(args.ocr_dir).resolve())
        if args.detector_dir:
            validate(args.detector_dir); values['VISION_MODEL_DIR']=str(Path(args.detector_dir).resolve())
        # Only model directory entries change; preserve authentication and bind values.
        text=(ROOT/'.env').read_text()
        for key in ('VISION_OCR_MODEL_DIR','VISION_MODEL_DIR'):
            lines=text.splitlines(); text='\n'.join(key+'='+values[key] if line.partition('=')[0].strip()==key else line for line in lines)+'\n'
        (ROOT/'.env').write_text(text)
        print('Model paths updated; run vision up to reload without build')
    elif args.command == "bind":
        set_address(ROOT / ".env", "VISION_BIND_ADDRESS", args.address)
        print("Vision bind configured; run manage.py vision up to apply")
    elif args.command == "prepare":
        prepare(args.download_detector, args.download_ocr)
    elif args.command == "check":
        check()
    elif args.command == "health":
        health()
    elif args.command == "doctor":
        doctor()
    elif args.command == "build":
        check()
        compose("build", "vision")
    elif args.command == "up":
        check()
        compose("up", "-d", "--no-build", "vision")
    elif args.command == "logs":
        compose("logs", "--tail", "100", "vision")
    elif args.command == "stop":
        compose("stop", "vision")


if __name__ == "__main__":
    main()
