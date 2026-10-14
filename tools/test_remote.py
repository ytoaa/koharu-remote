"""Offline Rust harness: real Custom module/config; result structs copied from pinned ML source.
This is explicitly NOT a full application or Windows build.
"""

from pathlib import Path
import argparse, json, re, subprocess, tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build/remote-contract-tests"
    )
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("Harness output exists; select another output directory")
    deps = tomllib.loads((upstream / "Cargo.toml").read_text())["workspace"][
        "dependencies"
    ]
    types = []
    for name, file in [
        ("KoharuLayoutThresholds", "config.rs"),
        ("KoharuLayoutDetections", "processor.rs"),
        ("KoharuLayoutDetection", "processor.rs"),
        ("KoharuLayoutMask", "processor.rs"),
    ]:
        source = (
            upstream / "crates/koharu-ml/src/koharu_layout_rfdetr_seg_2xl" / file
        ).read_text()
        matches = re.findall(r"pub struct " + name + r"\s*\{[^}]*\}", source)
        if len(matches) != 1:
            raise ValueError("Result type boundary changed: " + name)
        derive = (
            "Debug, Clone, Copy, Deserialize"
            if name == "KoharuLayoutThresholds"
            else (
                "Debug, Clone"
                if name == "KoharuLayoutMask"
                else "Debug, Clone, Serialize"
            )
        )
        types.append("#[derive(" + derive + ")]\n" + matches[0])
    (output / "src").mkdir(parents=True)

    def version(name):
        return deps[name] if isinstance(deps[name], str) else deps[name]["version"]

    manifest = '[package]\nname="remote-contract-tests"\nversion="0.1.0"\nedition="2024"\n[workspace]\n[dependencies]\n'
    for name in ("anyhow", "serde_json"):
        manifest += f"{name}={json.dumps('=' + version(name))}\n"
    manifest += f'image={{version={json.dumps("=" + version("image"))},default-features=false,features=["png"]}}\n'
    for name, features in [
        ("serde", ["derive"]),
        ("reqwest", ["stream", "json", "multipart"]),
        ("tokio", ["full"]),
    ]:
        manifest += f"{name}={{version={json.dumps('=' + version(name))},features={json.dumps(features)}}}\n"
    manifest += (
        "koharu-config={path="
        + json.dumps((upstream / "crates/koharu-config").as_posix())
        + "}\n"
    )
    (output / "Cargo.toml").write_text(manifest)
    source = (
        "extern crate self as koharu_ml;\npub mod koharu_layout_rfdetr_seg_2xl {use serde::{Deserialize,Serialize};\n"
        + "\n".join(types)
        + "\n}\n"
    )
    source += (
        "#[path="
        + json.dumps((ROOT / "patches/remote.rs").as_posix())
        + "]mod remote;\n"
    )
    (output / "src/lib.rs").write_text(source)
    subprocess.run(
        ["cargo", "test", "--manifest-path", str(output / "Cargo.toml"), "--lib"],
        check=True,
    )
    print("REMOTE_CONTRACT_HARNESS_PASS_FULL_APP_AND_GPU_NOT_RUN")


if __name__ == "__main__":
    main()
