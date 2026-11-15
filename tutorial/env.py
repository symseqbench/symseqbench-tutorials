"""Install check: the packages of the tutorial, their versions and the pinned commits of the two libraries."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from importlib import metadata
from pathlib import Path

# The commits of the two libraries. The same values are in pyproject.toml, uv.lock and README.md
# (tests/test_environment.py checks that the four agree).
PINNED_COMMITS = {
    "symseq": "6c3d7236a276deebf8b3f26cbbaf92687f97de5b",
    "seqbench": "c3ce976a72f0c6d73dff3334bb5ecac3838efe40",
}

# SHA-256 of the Python sources of each library at the pinned commit (``source_fingerprint``). An
# installation from the offline wheelhouse records no commit, so the install check accepts either
# the commit or this fingerprint.
PINNED_FINGERPRINTS = {
    "symseq": "06814650d776961390fe52d86cd9572baec74c2cb7668f624f672a797351e16f",
    "seqbench": "086931d6388e0eaabc991a5807aec5e49615906f33062ff07326d9ec8b3b74a8",
}

PYTHON = (3, 11)
PACKAGES = ("symseq", "seqbench", "marimo", "torch", "numpy", "scipy", "matplotlib", "pandas")


def installed_commit(package: str) -> str | None:
    """Commit a package was installed from, read from its ``direct_url.json`` (None if not from git)."""
    try:
        text = metadata.distribution(package).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id")


def source_fingerprint(package: str) -> str | None:
    """SHA-256 over the names and contents of the package's ``.py`` files, with line endings normalised."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(next(iter(spec.submodule_search_locations)))
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def environment_report() -> list[dict]:
    """One row per component: what is installed, what is expected, and whether they agree."""
    rows = [
        {
            "component": "python",
            "found": ".".join(map(str, sys.version_info[:3])),
            "expected": f"{PYTHON[0]}.{PYTHON[1]}.x",
            "ok": sys.version_info[:2] == PYTHON,
        }
    ]
    for package in PACKAGES:
        try:
            importlib.import_module(package)
            version = metadata.version(package)
            imported = True
        except Exception as error:  # the report must show a broken import, not stop on it
            version = f"import failed: {type(error).__name__}"
            imported = False
        if package in PINNED_COMMITS:
            commit = installed_commit(package)
            same_sources = imported and source_fingerprint(package) == PINNED_FINGERPRINTS[package]
            origin = f"commit {commit[:12]}" if commit else "no commit recorded"
            rows.append(
                {
                    "component": package,
                    "found": f"{version}, {origin}, sources {'as pinned' if same_sources else 'differ'}",
                    "expected": f"commit {PINNED_COMMITS[package][:12]} or its sources",
                    "ok": imported
                    and (commit == PINNED_COMMITS[package] or (commit is None and same_sources)),
                }
            )
        else:
            rows.append(
                {"component": package, "found": version, "expected": "importable", "ok": imported}
            )
    rows.append(_generation_check())
    return rows


def _generation_check() -> dict:
    """Generate and measure a few strings, so that the check covers more than imports."""
    try:
        from tutorial.generate import as_stream, generate_strings, two_state_grammar
        from tutorial.measures import string_measures

        strings = generate_strings(two_state_grammar(seed=0), 200)
        rate = string_measures(as_stream(strings, delimiter=False))["entropy_rate"]
        return {
            "component": "generate and measure",
            "found": f"200 strings, entropy rate {rate:.2f} bits per symbol",
            "expected": "between 0.5 and 0.8",
            "ok": 0.5 < rate < 0.8,
        }
    except Exception as error:
        return {
            "component": "generate and measure",
            "found": f"failed: {type(error).__name__}: {error}",
            "expected": "between 0.5 and 0.8",
            "ok": False,
        }


def environment_ok(rows: list[dict] | None = None) -> bool:
    """True if every row of the report is as expected."""
    return all(row["ok"] for row in (rows if rows is not None else environment_report()))
