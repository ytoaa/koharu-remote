"""Fetch immutable official source into a NEW build directory; never reset a user's tree."""

from pathlib import Path
import json, re, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]


def checkout(destination):
    pin = json.loads((ROOT / "upstream.json").read_text())
    destination = Path(destination).resolve()
    if destination.exists():
        raise RuntimeError("Destination already exists; use another build directory")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", pin["repository"]):
        raise ValueError("Invalid upstream repository")
    if not re.fullmatch(r"[0-9a-f]{40}", pin["commit"]):
        raise ValueError("Invalid immutable upstream commit")
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--no-checkout",
            "https://github.com/" + pin["repository"] + ".git",
            str(destination),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "origin", pin["commit"]], check=True
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", pin["commit"]],
        check=True,
    )


if __name__ == "__main__":
    checkout(sys.argv[1])
