"""Create a candidate PR only; existing stable main stays untouched."""

from pathlib import Path
import json, os, re, subprocess, tempfile

ROOT = Path(__file__).resolve().parents[1]


def gh(*args):
    return subprocess.check_output(["gh", *args], cwd=ROOT, text=True).strip()


def main():
    current = json.loads((ROOT / "upstream.json").read_text())
    repo = current["repository"]
    release = json.loads(gh("api", f"repos/{repo}/releases/latest"))
    tag = release["tag_name"]
    if (
        release["draft"]
        or release["prerelease"]
        or not re.fullmatch(r"v?\d+\.\d+\.\d+", tag)
    ):
        raise RuntimeError("Expected a stable SemVer release")
    commit = json.loads(gh("api", f"repos/{repo}/commits/{tag}"))["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid commit")
    if commit == current["commit"]:
        print("Upstream pin is current")
        return
    branch = "upstream/koharu-" + tag
    found = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", "refs/heads/" + branch],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not found:
        subprocess.run(["git", "checkout", "-b", branch], cwd=ROOT, check=True)
        current.update(version=tag.removeprefix("v"), commit=commit)
        (ROOT / "upstream.json").write_text(json.dumps(current, indent=2) + "\n")
        subprocess.run(
            ["git", "config", "user.name", "github-actions[bot]"], cwd=ROOT, check=True
        )
        subprocess.run(
            [
                "git",
                "config",
                "user.email",
                "41898282+github-actions[bot]@users.noreply.github.com",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(["git", "add", "upstream.json"], cwd=ROOT, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Candidate: Koharu " + tag], cwd=ROOT, check=True
        )
        subprocess.run(
            ["git", "push", "origin", "HEAD:refs/heads/" + branch], cwd=ROOT, check=True
        )
    # GITHUB_TOKEN events do not spawn another workflow. watch.yml calls reusable checks.
    prs = json.loads(
        gh("pr", "list", "--head", branch, "--state", "all", "--json", "number")
    )
    if not prs:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as file:
            file.write(
                f"Upstream release {tag}, immutable source `{commit}`.\n\nThe existing minimal patch is applied without regeneration, then tested by the watcher workflow. Windows compilation requires an explicit manual run with build_windows enabled. Conflicts, API changes, or failed checks block promotion. DGX GPU/OCR quality validation remains manual; no auto-merge.\n"
            )
            body = file.name
        try:
            gh(
                "pr",
                "create",
                "--base",
                "main",
                "--head",
                branch,
                "--title",
                "Candidate: Koharu " + tag,
                "--body-file",
                body,
            )
        finally:
            Path(body).unlink()
    if found:
        print(
            "Candidate already exists; checks run on PR changes, Windows builds require manual opt-in"
        )
        return  # Do not repeat an unchanged expensive Windows build every day.
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as file:
        file.write("branch=" + branch + "\n")
    print("Candidate queued: " + branch)


if __name__ == "__main__":
    main()
