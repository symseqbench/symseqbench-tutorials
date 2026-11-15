"""Check that uv.lock can be installed from wheels on the platforms of the tutorial.

For each platform the script follows the dependencies of the project through ``uv.lock``, evaluating
every environment marker with the values Python reports on that platform, and checks that each
package reached from a registry has a wheel the platform accepts. Packages installed from git (the
two libraries) and the project itself are built from source and are listed separately.

It also reports the oldest operating system version for which every wheel exists.

Usage:  uv run python scripts/check_lock_platforms.py
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

from packaging.markers import Marker
from packaging.tags import compatible_tags, cpython_tags, mac_platforms
from packaging.utils import parse_wheel_filename

REPO = Path(__file__).resolve().parents[1]
PYTHON = (3, 11)


def marker_environment(sys_platform: str, machine: str) -> dict[str, str]:
    system = {"linux": "Linux", "darwin": "Darwin", "win32": "Windows"}[sys_platform]
    return {
        "implementation_name": "cpython",
        "implementation_version": "3.11.11",
        "os_name": "nt" if sys_platform == "win32" else "posix",
        "platform_machine": machine,
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_system": system,
        "platform_version": "",
        "python_full_version": "3.11.11",
        "python_version": "3.11",
        "sys_platform": sys_platform,
    }


def linux_tags(glibc_minor: int, machine: str = "x86_64") -> list[str]:
    tags = [f"manylinux_2_{minor}_{machine}" for minor in range(glibc_minor, 4, -1)]
    if glibc_minor >= 17:
        tags.append(f"manylinux2014_{machine}")
    if glibc_minor >= 12:
        tags.append(f"manylinux2010_{machine}")
    if glibc_minor >= 5:
        tags.append(f"manylinux1_{machine}")
    return [*tags, f"linux_{machine}"]


# name -> (sys_platform, platform_machine as Python reports it, wheel platform tags)
PLATFORMS: dict[str, tuple[str, str, list[str]]] = {
    "Linux x86_64 (glibc 2.28)": ("linux", "x86_64", linux_tags(28)),
    "macOS arm64 (macOS 12)": ("darwin", "arm64", list(mac_platforms((12, 0), "arm64"))),
    "macOS x86_64 (macOS 10.15)": ("darwin", "x86_64", list(mac_platforms((10, 15), "x86_64"))),
    "Windows x86_64": ("win32", "AMD64", ["win_amd64"]),
}


def supported_tags(platform_tags: list[str]) -> set:
    return set(cpython_tags(PYTHON, abis=["cp311", "abi3", "none"], platforms=platform_tags)) | set(
        compatible_tags(PYTHON, interpreter="cp311", platforms=platform_tags)
    )


def load_lock(path: Path = REPO / "uv.lock") -> dict:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _find(packages: list[dict], dependency: dict) -> dict:
    matches = [p for p in packages if p["name"] == dependency["name"]]
    if "version" in dependency:
        matches = [p for p in matches if p.get("version") == dependency["version"]]
    if "source" in dependency:
        matches = [p for p in matches if p.get("source") == dependency["source"]]
    if len(matches) != 1:
        raise LookupError(
            f"{dependency['name']}: {len(matches)} entries in the lock match {dependency}"
        )
    return matches[0]


def resolve(
    lock: dict, sys_platform: str, machine: str, root: str = "champalimaud-tutorial"
) -> list[dict]:
    """Packages installed on a platform: the root's dependencies and dev group, followed through the markers."""
    environment = marker_environment(sys_platform, machine)
    packages = lock["package"]
    start = next(p for p in packages if p["name"] == root)
    pending = [*start.get("dependencies", [])]
    for group in start.get("dev-dependencies", {}).values():
        pending.extend(group)
    reached: dict[tuple, dict] = {}
    while pending:
        dependency = pending.pop()
        if "marker" in dependency and not Marker(dependency["marker"]).evaluate(environment):
            continue
        package = _find(packages, dependency)
        key = (package["name"], package.get("version"), str(package.get("source")))
        if key not in reached:
            reached[key] = package
            pending.extend(package.get("dependencies", []))
    return sorted(reached.values(), key=lambda p: p["name"])


def check_platform(lock: dict, sys_platform: str, machine: str, platform_tags: list[str]) -> dict:
    """Split the packages of a platform into those with a usable wheel, those without, and those built from source."""
    accepted = supported_tags(platform_tags)
    with_wheel, without_wheel, from_source = [], [], []
    for package in resolve(lock, sys_platform, machine):
        label = f"{package['name']} {package.get('version', '')}".strip()
        if "registry" not in package.get("source", {}):
            from_source.append(label)
            continue
        usable = any(
            parse_wheel_filename(wheel["url"].rsplit("/", 1)[-1].replace("%2B", "+"))[3] & accepted
            for wheel in package.get("wheels", [])
        )
        (with_wheel if usable else without_wheel).append(label)
    return {"with_wheel": with_wheel, "without_wheel": without_wheel, "from_source": from_source}


def oldest_system(lock: dict) -> dict[str, str]:
    """Oldest operating system version on which every registry package has a wheel."""
    out = {}
    for minor in range(17, 40):
        if not check_platform(lock, "linux", "x86_64", linux_tags(minor))["without_wheel"]:
            out["Linux x86_64"] = f"glibc 2.{minor}"
            break
    for version in [(11, 0), (12, 0), (13, 0), (14, 0), (15, 0)]:
        if not check_platform(lock, "darwin", "arm64", list(mac_platforms(version, "arm64")))[
            "without_wheel"
        ]:
            out["macOS arm64"] = f"macOS {version[0]}"
            break
    for version in [(10, 9), (10, 12), (10, 13), (10, 14), (10, 15), (11, 0), (12, 0), (13, 0)]:
        if not check_platform(lock, "darwin", "x86_64", list(mac_platforms(version, "x86_64")))[
            "without_wheel"
        ]:
            out["macOS x86_64"] = (
                f"macOS {version[0]}.{version[1]}" if version[0] == 10 else f"macOS {version[0]}"
            )
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--lock", type=Path, default=REPO / "uv.lock")
    arguments = parser.parse_args()
    lock = load_lock(arguments.lock)
    failed = False
    for name, (sys_platform, machine, tags) in PLATFORMS.items():
        result = check_platform(lock, sys_platform, machine, tags)
        torch = next(
            label
            for label in result["with_wheel"] + result["without_wheel"]
            if label.startswith("torch ")
        )
        status = "OK" if not result["without_wheel"] else "MISSING WHEELS"
        print(f"{name}: {status}; {len(result['with_wheel'])} packages with wheels, {torch}")
        print(f"    built from source: {', '.join(result['from_source'])}")
        if result["without_wheel"]:
            failed = True
            print(f"    no wheel: {', '.join(result['without_wheel'])}")
    print("oldest system with a wheel for every package:", oldest_system(lock))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
