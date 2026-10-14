from pathlib import Path
import hashlib, sys


def verify(root):
    root = Path(root).resolve()
    count = 0
    for line in (root / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Unsafe checksum path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Checksum mismatch: " + name)
        count += 1
    print(f"SHA256SUMS_OK: {count} files")


if __name__ == "__main__":
    verify(sys.argv[1] if len(sys.argv) > 1 else ".")
