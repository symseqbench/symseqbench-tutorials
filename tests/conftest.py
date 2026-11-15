"""Shared fixtures. Temporary files stay inside the repository (ignored by git), never in the system temp directory."""

from __future__ import annotations

import importlib.util
import shutil
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def work_dir():
    """An empty directory under ``.pytest_work/`` in the repository, removed after the test."""
    path = REPO / ".pytest_work" / uuid.uuid4().hex
    path.mkdir(parents=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def fallback_script():
    """``scripts/make_fallback.py`` as a module, so that a test can write a fallback folder of its own."""
    spec = importlib.util.spec_from_file_location(
        "make_fallback", REPO / "scripts" / "make_fallback.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
