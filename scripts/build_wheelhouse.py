"""Build the offline wheelhouse: every wheel the lock needs on one platform, plus the two libraries.

For the chosen platform the script reads ``uv.lock``, takes the best wheel of every registry package
for that platform, downloads it from the URL recorded in the lock and verifies its SHA-256. The two
libraries are cloned at their pinned commits and built once as pure-Python wheels into
``wheelhouse/common``. A ``requirements.txt`` with exact versions is written next to the wheels.

Needs network, git and uv. See OFFLINE.md for the installation from the result.

Usage:  uv run python scripts/build_wheelhouse.py --platform "Linux x86_64 (glibc 2.28)"
        uv run python scripts/build_wheelhouse.py --platform all
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_lock_platforms as lock_check

REPO = lock_check.REPO
LIBRARIES = {
    "symseq": (
        "https://github.com/symseqbench/SymSeq.git",
        "6c3d7236a276deebf8b3f26cbbaf92687f97de5b",
    ),
    "seqbench": (
        "https://github.com/symseqbench/SeqBench.git",
        "c3ce976a72f0c6d73dff3334bb5ecac3838efe40",
    ),
}


def slug(platform: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", platform.lower()).strip("-")


def best_wheel(package: dict, platform_tags: list[str]) -> dict:
    """The wheel of a locked package that the platform prefers."""
    priority = [
        *cpython_tags(lock_check.PYTHON, abis=["cp311", "abi3", "none"], platforms=platform_tags),
        *compatible_tags(lock_check.PYTHON, interpreter="cp311", platforms=platform_tags),
    ]
    rank = {tag: index for index, tag in enumerate(priority)}
    best, best_rank = None, len(rank)
    for wheel in package.get("wheels", []):
        tags = parse_wheel_filename(wheel["url"].rsplit("/", 1)[-1].replace("%2B", "+"))[3]
        wheel_rank = min((rank[tag] for tag in tags if tag in rank), default=len(rank))
        if wheel_rank < best_rank:
            best, best_rank = wheel, wheel_rank
    if best is None:
        raise LookupError(f"{package['name']} {package.get('version')}: no wheel for this platform")
    return best


def download(wheel: dict, directory: Path) -> Path:
    name = wheel["url"].rsplit("/", 1)[-1].replace("%2B", "+")
    target = directory / name
    if not target.exists():
        with urllib.request.urlopen(wheel["url"]) as response:
            target.write_bytes(response.read())
    expected = wheel.get("hash", "")
    if expected.startswith("sha256:"):
        found = hashlib.sha256(target.read_bytes()).hexdigest()
        if found != expected.removeprefix("sha256:"):
            target.unlink()
            raise ValueError(f"{name}: SHA-256 {found} differs from the lock")
    else:
        print(f"    {name}: no hash in the lock, not verified")
    return target


def build_libraries(out: Path) -> None:
    """Clone the two libraries at their pinned commits and build one pure-Python wheel of each."""
    common = out / "common"
    common.mkdir(parents=True, exist_ok=True)
    for name, (url, commit) in LIBRARIES.items():
        if list(common.glob(f"{name}-*.whl")):
            continue
        source = out / "_src" / name
        if not source.exists():
            subprocess.run(["git", "clone", "--quiet", url, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--quiet", commit], check=True)
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(common), str(source)], check=True
        )


def build_platform(lock: dict, platform: str, out: Path) -> None:
    sys_platform, machine, tags = lock_check.PLATFORMS[platform]
    directory = out / slug(platform)
    directory.mkdir(parents=True, exist_ok=True)
    requirements, size = [], 0
    for package in lock_check.resolve(lock, sys_platform, machine):
        requirements.append(f"{package['name']}=={package['version']}")
        if "registry" in package.get("source", {}):
            size += download(best_wheel(package, tags), directory).stat().st_size
    # the build backend of the project itself, needed to install it without network
    setuptools = next(p for p in lock["package"] if p["name"] == "setuptools")
    download(best_wheel(setuptools, tags), directory)
    (directory / "requirements.txt").write_text("\n".join(sorted(requirements)) + "\n")
    print(f"{platform}: {len(requirements)} packages, {size / 1e6:.0f} MB in {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--platform", default="all", choices=[*lock_check.PLATFORMS, "all"])
    parser.add_argument("--out", type=Path, default=REPO / "wheelhouse")
    arguments = parser.parse_args()
    lock = lock_check.load_lock()
    build_libraries(arguments.out)
    for platform in lock_check.PLATFORMS if arguments.platform == "all" else [arguments.platform]:
        build_platform(lock, platform, arguments.out)


if __name__ == "__main__":
    main()
