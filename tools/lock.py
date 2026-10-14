"""Let Cargo regenerate derived lock data, then refuse dependency version/source drift."""

from pathlib import Path
import subprocess, sys, tomllib


def key(p):
    return p["name"], p["version"], p.get("source", ""), p.get("checksum", "")


def refresh(target):
    target = Path(target)
    path = target / "Cargo.lock"
    before = tomllib.loads(path.read_text())["package"]
    subprocess.run(["cargo", "update", "--workspace"], cwd=target, check=True)
    after = tomllib.loads(path.read_text())["package"]
    if {key(p) for p in before} != {key(p) for p in after}:
        raise RuntimeError(
            "Cargo changed dependency versions/sources. Candidate requires review"
        )
    prior = {key(p): p for p in before}
    for package in after:
        old = prior[key(package)]
        if package == old:
            continue
        allowed = {
            **old,
            "dependencies": sorted(
                set(old.get("dependencies", [])) | {"reqwest", "serde_json"}
            ),
        }
        if package["name"] != "koharu-pipeline" or package != allowed:
            raise RuntimeError("Unexpected lockfile change: " + package["name"])
    print("LOCK_REFRESHED_NO_DEPENDENCY_UPGRADES")


if __name__ == "__main__":
    refresh(sys.argv[1])
