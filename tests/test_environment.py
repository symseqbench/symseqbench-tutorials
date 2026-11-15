"""The pinned environment: commits, CPU-only torch, wheels on the four platforms. Each test has a case it must reject."""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

from tutorial import env

REPO = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "check_lock_platforms", REPO / "scripts" / "check_lock_platforms.py"
)
lock_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lock_check)


def commits_in(text: str) -> dict[str, set[str]]:
    """Commits (40 hexadecimal digits) that follow the name of each library's repository in a text."""
    found: dict[str, set[str]] = {"symseq": set(), "seqbench": set()}
    for library, repository in (("symseq", "SymSeq"), ("seqbench", "SeqBench")):
        for match in re.finditer(
            rf"symseqbench/{repository}(?:\.git)?[^\n]*?([0-9a-f]{{40}})", text
        ):
            found[library].add(match.group(1))
    return found


def test_the_pinned_commits_agree_in_every_place():
    expected = {library: {commit} for library, commit in env.PINNED_COMMITS.items()}
    for name in ("pyproject.toml", "uv.lock", "README.md"):
        assert commits_in((REPO / name).read_text()) == expected, name
    # must reject: a text that names another commit
    altered = (REPO / "pyproject.toml").read_text().replace(env.PINNED_COMMITS["symseq"], "0" * 40)
    assert commits_in(altered) != expected


def test_the_installed_libraries_are_at_the_pinned_commits():
    """From git (``uv sync``) the commit is recorded; from the offline wheelhouse only the sources can be compared."""
    for library, commit in env.PINNED_COMMITS.items():
        recorded = env.installed_commit(library)
        assert recorded in (commit, None)
        assert env.source_fingerprint(library) == env.PINNED_FINGERPRINTS[library]
    assert (
        env.installed_commit("numpy") is None
    )  # must reject: a package that was not installed from git


def test_the_installed_sources_have_the_pinned_fingerprints(monkeypatch):
    for library, fingerprint in env.PINNED_FINGERPRINTS.items():
        assert env.source_fingerprint(library) == fingerprint
    assert (
        env.source_fingerprint("tutorial") not in env.PINNED_FINGERPRINTS.values()
    )  # must reject: other sources
    assert env.source_fingerprint("no_such_package_here") is None
    # an installation without a recorded commit (the offline wheelhouse) passes on the fingerprint alone,
    monkeypatch.setattr(env, "installed_commit", lambda package: None)
    assert env.environment_ok()
    # and must be rejected if the sources are not the pinned ones
    monkeypatch.setitem(env.PINNED_FINGERPRINTS, "seqbench", "0" * 64)
    assert [row["component"] for row in env.environment_report() if not row["ok"]] == ["seqbench"]


def test_install_check_passes_here_and_fails_on_a_wrong_pin(monkeypatch):
    rows = env.environment_report()
    assert env.environment_ok(rows), [row for row in rows if not row["ok"]]
    assert [row["component"] for row in rows][:3] == ["python", "symseq", "seqbench"]
    # must reject: the same environment checked against another commit and other sources
    monkeypatch.setitem(env.PINNED_COMMITS, "symseq", "0" * 40)
    monkeypatch.setitem(env.PINNED_FINGERPRINTS, "symseq", "0" * 64)
    wrong = env.environment_report()
    assert not env.environment_ok(wrong)
    assert [row["component"] for row in wrong if not row["ok"]] == ["symseq"]


def gpu_packages(lock: dict) -> list[str]:
    """Packages of a lock that only a CUDA build of torch pulls in."""
    return [
        package["name"]
        for package in lock["package"]
        if package["name"].startswith(("nvidia", "triton"))
    ]


def test_torch_is_cpu_only_and_numpy_is_the_same_everywhere():
    lock = lock_check.load_lock()
    assert gpu_packages(lock) == []
    # must reject: the same lock with one CUDA package added
    with_cuda = {"package": [*lock["package"], {"name": "nvidia-cublas-cu12", "version": "12.1"}]}
    assert gpu_packages(with_cuda) == ["nvidia-cublas-cu12"]
    torch = {
        package["version"]: package["source"]
        for package in lock["package"]
        if package["name"] == "torch"
    }
    assert set(torch) == {"2.2.2", "2.11.0", "2.11.0+cpu"}
    assert torch["2.11.0+cpu"] == {"registry": "https://download.pytorch.org/whl/cpu"}
    assert [package["version"] for package in lock["package"] if package["name"] == "numpy"] == [
        "1.26.4"
    ]
    for name, (sys_platform, machine, _) in lock_check.PLATFORMS.items():
        versions = [
            p["version"]
            for p in lock_check.resolve(lock, sys_platform, machine)
            if p["name"] == "torch"
        ]
        expected = (
            "2.2.2" if "macOS x86_64" in name else "2.11.0" if "macOS" in name else "2.11.0+cpu"
        )
        assert versions == [expected], name


@pytest.mark.parametrize("name", lock_check.PLATFORMS)
def test_every_package_has_a_wheel_on_the_platform(name):
    sys_platform, machine, tags = lock_check.PLATFORMS[name]
    result = lock_check.check_platform(lock_check.load_lock(), sys_platform, machine, tags)
    assert result["without_wheel"] == []
    assert len(result["with_wheel"]) > 60
    assert sorted(result["from_source"]) == ["seqbench 0.1.0", "symseq 0.1.0"]
    # must reject: the same platform asked for a wheel tag that no package carries
    nowhere = lock_check.check_platform(
        lock_check.load_lock(), sys_platform, machine, ["no_such_platform"]
    )
    assert any(label.startswith("numpy ") for label in nowhere["without_wheel"])
    assert any(label.startswith("torch ") for label in nowhere["without_wheel"])


def test_the_wheel_check_rejects_platforms_without_wheels():
    """must reject: Windows on ARM, an Apple-silicon Mac on macOS 11, and Linux with glibc 2.17."""
    lock = lock_check.load_lock()
    arm_windows = lock_check.check_platform(lock, "win32", "ARM64", ["win_arm64"])
    assert any(label.startswith("numpy ") for label in arm_windows["without_wheel"])
    old_mac = lock_check.check_platform(
        lock, "darwin", "arm64", list(lock_check.mac_platforms((11, 0), "arm64"))
    )
    assert old_mac["without_wheel"]
    old_linux = lock_check.check_platform(lock, "linux", "x86_64", lock_check.linux_tags(17))
    assert any(label.startswith("torch ") for label in old_linux["without_wheel"])


def test_pyproject_keeps_the_canonical_ruff_block_and_the_python_version():
    text = (REPO / "pyproject.toml").read_text()
    project = tomllib.loads(text)
    assert project["project"]["requires-python"] == ">=3.11,<3.12"
    assert project["tool"]["ruff"]["line-length"] == 100
    assert project["tool"]["ruff"]["lint"]["select"] == [
        "E",
        "W",
        "F",
        "I",
        "B",
        "UP",
        "SIM",
        "RUF",
    ]
    assert project["tool"]["ruff"]["lint"]["ignore"] == [
        "E501",
        "B008",
        "RUF001",
        "RUF002",
        "RUF003",
    ]
    template = (
        Path.home() / ".claude" / "templates" / "python-project" / "pyproject-ruff-block.toml"
    )
    if template.exists():  # only on the author's workstation
        assert text.endswith(template.read_text())
        assert not (text + "\n").endswith(
            template.read_text()
        )  # must reject: any change after the block
