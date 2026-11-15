"""The run script and the saved results: one file per run, summary, manifest, resuming, refusing.

Every test has a case it must reject. The script is run on a small part of the smoke set, in a
folder under ``.pytest_work``; the folders of the repository are only read.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from tutorial import config, results

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_precomputed.py"
RESULT_KEYS = ("history", "final", "optimum", "exact", "weights_sha256", "n_parameters", "data")


def run_script(*arguments: str) -> subprocess.CompletedProcess:
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def folder():
    """A results folder with the smoke runs of one grammar (3 seeds, 2 models)."""
    path = REPO / ".pytest_work" / "run_precomputed"
    shutil.rmtree(path, ignore_errors=True)
    done = run_script("--smoke", "--grammars", "nested", "--out", str(path))
    assert "6 runs computed" in done.stdout, done.stderr[-2000:]
    assert done.returncode == 1  # 60 of the 66 smoke runs are missing: the set is incomplete
    yield path
    shutil.rmtree(path, ignore_errors=True)


def test_one_file_per_run_a_summary_and_a_manifest_with_hashes(folder):
    files = sorted(p.name for p in (folder / "runs").glob("*"))
    assert files == sorted(
        f"nested__{model}__seed{seed}.json"
        for model in ("rnn", "linear-transformer")
        for seed in config.EXPLORATION_SEEDS
    )
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["kind"] == "smoke" and manifest["placeholder"] is True
    assert manifest["settings"] == config.SMOKE.as_dict()
    assert manifest["seeds"] == list(config.EXPLORATION_SEEDS)
    assert (manifest["expected"], manifest["present"]) == (66, 6)
    assert set(manifest["files"]) == {f"runs/{name}" for name in files}
    for name, digest in manifest["files"].items():
        assert results.sha256(folder / name) == digest and len(digest) == 64
    assert manifest["summary"]["sha256"] == results.sha256(folder / "summary.csv")
    table = pd.read_csv(folder / "summary.csv")
    assert len(table) == 6 and set(table["grammar"]) == {"nested"}
    assert set(table["seed"]) == set(config.EXPLORATION_SEEDS)
    run = json.loads((folder / "runs" / files[0]).read_text())
    assert run["kind"] == "smoke" and run["environment"]["threads"] == 1
    # must reject: the set is reported as incomplete, with the absent runs by name and without values
    assert manifest["complete"] is False and len(manifest["missing"]) == 60
    assert "two-state__rnn__seed101" in manifest["missing"]
    assert "two-state" not in set(table["grammar"])


def test_a_second_call_computes_nothing_and_a_deleted_run_is_recomputed_identically(folder):
    paths = sorted((folder / "runs").glob("*.json"))
    before = {path.name: (path.stat().st_mtime_ns, path.read_text()) for path in paths}
    again = run_script("--smoke", "--grammars", "nested", "--out", str(folder))
    assert "0 runs computed" in again.stdout and "(6 already saved)" in again.stdout
    assert all(path.stat().st_mtime_ns == before[path.name][0] for path in paths)

    victim = paths[2]
    victim.unlink()
    third = run_script("--smoke", "--grammars", "nested", "--out", str(folder))
    assert "1 runs computed" in third.stdout and "(5 already saved)" in third.stdout
    old, new = json.loads(before[victim.name][1]), json.loads(victim.read_text())
    for key in RESULT_KEYS:  # a second process, the same numbers and the same weights
        assert new[key] == old[key], key
    for path in paths:
        if path != victim:
            assert path.stat().st_mtime_ns == before[path.name][0]
    # must reject: a run saved with other settings is not taken for the requested one
    arguments = ("smoke", "nested", "rnn", config.EXPLORATION_SEEDS[0])
    assert results.saved_run(folder, *arguments, config.SMOKE.as_dict()) is not None
    assert results.saved_run(folder, *arguments, replace(config.SMOKE, epochs=4).as_dict()) is None
    assert results.saved_run(folder, "production", *arguments[1:], config.SMOKE.as_dict()) is None


def test_parts_split_the_jobs_and_do_not_write_the_manifest(work_dir):
    out = work_dir / "parts"
    first = run_script("--smoke", "--grammars", "two-state", "--shard", "1/3", "--out", str(out))
    assert "2 runs computed" in first.stdout and "one part" in first.stdout
    assert first.returncode == 0 and not (out / "manifest.json").exists()
    assert sorted(p.name for p in (out / "runs").glob("*")) == [
        "two-state__linear-transformer__seed101.json",
        "two-state__rnn__seed101.json",
    ]
    collected = run_script("--smoke", "--collect", "--out", str(out))
    assert "2 of 66 runs present" in collected.stdout and (out / "manifest.json").exists()
    # the folder of the figure without any run: nothing is trained, and the set is incomplete
    empty = work_dir / "production"
    nothing = run_script("--production", "--collect", "--out", str(empty))
    assert nothing.returncode == 1 and "0 of 66 runs present, 66 missing" in nothing.stdout
    manifest = json.loads((empty / "manifest.json").read_text())
    assert (manifest["kind"], manifest["placeholder"], manifest["complete"]) == (
        "production",
        False,
        False,
    )
    assert manifest["seeds"] == list(config.PLOTTED_SEEDS) and not (empty / "runs").exists()
    # must reject: a part that does not exist, an unknown grammar, and a call without a kind
    assert run_script("--smoke", "--shard", "4/3", "--out", str(out)).returncode == 2
    assert run_script("--smoke", "--grammars", "klingon", "--out", str(out)).returncode != 0
    assert run_script("--out", str(out)).returncode == 2
    assert run_script("--smoke", "--production", "--out", str(out)).returncode == 2
    assert run_script("--smok", "--out", str(out)).returncode == 2  # no abbreviations


def test_the_loader_refuses_files_that_do_not_match_the_manifest(folder, work_dir):
    copy = work_dir / "copy"
    shutil.copytree(folder, copy)
    results.collect(copy, "smoke")
    runs, manifest = results.load_runs(copy)
    assert len(runs) == 6 and manifest["kind"] == "smoke"

    # must reject: a run file changed after the manifest was written
    target = copy / "runs" / "nested__rnn__seed101.json"
    changed = json.loads(target.read_text())
    changed["final"]["accuracy_with_eos"] = 0.99
    target.write_text(json.dumps(changed, indent=1))
    with pytest.raises(ValueError, match="changed after the manifest"):
        results.load_runs(copy)
    # must reject: a listed file that is gone, and a run file the manifest does not know
    target.unlink()
    with pytest.raises(ValueError, match="absent"):
        results.load_runs(copy)
    results.collect(copy, "smoke")
    assert len(results.load_runs(copy)[0]) == 5
    shutil.copy(copy / "runs" / "nested__rnn__seed102.json", copy / "runs" / "stray.json")
    with pytest.raises(ValueError, match="not in the manifest"):
        results.load_runs(copy)
    with pytest.raises(FileNotFoundError):
        results.load_runs(work_dir / "nowhere")


def test_the_saved_smoke_results_are_complete_and_match_the_settings():
    """The placeholder results the notebook reads until the production runs exist."""
    runs, manifest = results.load_runs(results.FOLDERS["smoke"])
    assert manifest["kind"] == "smoke" and manifest["placeholder"] is True
    assert manifest["complete"] is True and manifest["missing"] == []
    assert manifest["checks_failed"] == []
    assert len(runs) == 66 == len(results.expected_runs("smoke"))
    assert {(run["rung"], run["model"], run["seed"]) for run in runs} == set(
        results.expected_runs("smoke")
    )
    # must reject: results of other settings, of the plotted seeds, or of another kind
    assert all(run["settings"] == config.SMOKE.as_dict() for run in runs)
    assert all(run["settings"] != config.PRODUCTION.as_dict() for run in runs)
    assert not {run["seed"] for run in runs} & set(config.PLOTTED_SEEDS)
    assert all(run["kind"] == "smoke" and run["checks"]["passed"] for run in runs)
    assert all(run["data"]["seqbench_seeds"]["heldout"] == 1_000_003 for run in runs)


def test_the_production_results_are_used_only_when_complete(work_dir, monkeypatch):
    folders = {"production": work_dir / "production", "smoke": work_dir / "smoke"}
    monkeypatch.setattr(results, "FOLDERS", folders)
    assert results.default_folder() == folders["smoke"]  # no production results at all
    folders["production"].mkdir()
    (folders["production"] / "manifest.json").write_text(json.dumps({"complete": False}))
    assert results.default_folder() == folders["smoke"]  # must reject: an incomplete set
    (folders["production"] / "manifest.json").write_text(json.dumps({"complete": True}))
    assert results.default_folder() == folders["production"]


def test_figure_data_has_every_seed_and_refuses_inconsistent_runs():
    runs, manifest = results.load_runs(results.FOLDERS["smoke"])
    data = results.figure_data(runs, config.LADDER, manifest["kind"])
    assert data["grammars"] == list(config.LADDER) and data["kind"] == "smoke"
    for model in config.MODELS:
        for measure in results.MEASURES:
            assert [len(seeds) for seeds in data["models"][model][measure]] == [3] * 6
        assert data["models"][model]["seeds"] == [list(config.EXPLORATION_SEEDS)] * 6
    first = next(r for r in runs if (r["rung"], r["model"], r["seed"]) == ("two-state", "rnn", 101))
    assert data["models"]["rnn"]["cross_entropy"][0][0] == first["final"]["cross_entropy"]
    assert data["optimum"]["accuracy_with_eos"][0] == first["optimum"]["accuracy_with_eos"]
    table = results.mean_table(data)
    assert len(table) == 12
    assert table.loc[("two-state", "tanh RNN"), "seeds"] == 3
    # another ladder can be drawn from the same runs
    other = results.figure_data(runs, ("two-state", "2-AX", "Dyck (depth 2)"), "smoke")
    assert other["families"] == ["regular", "regular", "context-free"]

    # must reject: runs of one grammar with different optima, a grammar without runs, a missing model
    tampered = json.loads(json.dumps(runs))
    tampered[0]["optimum"]["cross_entropy"] += 0.1
    with pytest.raises(ValueError, match="do not share one optimum"):
        results.figure_data(tampered, config.LADDER, "smoke")
    with pytest.raises(ValueError, match="No saved run for the grammar"):
        results.figure_data([r for r in runs if r["rung"] != "nested"], config.LADDER, "smoke")
    with pytest.raises(ValueError, match="No saved run of the rnn"):
        results.figure_data(
            [r for r in runs if (r["rung"], r["model"]) != ("nested", "rnn")],
            config.LADDER,
            "smoke",
        )


def test_settings_and_seed_sets_are_declared_once_and_are_disjoint():
    assert set(config.LADDER) | set(config.ALTERNATIVES) == set(config.CANDIDATES)
    assert len(config.LADDER) == 6 and len(config.ALTERNATIVES) == 5
    assert not set(config.LADDER) & set(config.ALTERNATIVES)
    assert config.LADDER[0] == "two-state" and config.LADDER[-1] == "Dyck (depth 3)"
    plotted, trial = set(config.PLOTTED_SEEDS), set(config.EXPLORATION_SEEDS)
    assert len(plotted) == 3 and not plotted & trial and config.LIVE_SEED not in plotted
    assert config.SEEDS == {"production": config.PLOTTED_SEEDS, "smoke": config.EXPLORATION_SEEDS}
    assert config.SMOKE.epochs < config.PRODUCTION.epochs
    assert replace(config.SMOKE, n_train=2000, n_heldout=500, epochs=30) == config.PRODUCTION
    # the candidates are those of the proposal table, with the same parameters
    spec = importlib.util.spec_from_file_location(
        "ladder_proposal", REPO / "scripts" / "ladder_proposal.py"
    )
    proposal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proposal)
    assert list(config.CANDIDATES.values()) == proposal.CANDIDATES
    listed = pd.read_csv(REPO / "scripts" / "output" / "ladder_proposal.csv")["rung"]
    assert list(listed) == list(config.CANDIDATES)
    # must reject: a seed used for both purposes makes the module refuse to load
    source = (REPO / "tutorial" / "config.py").read_text()
    assert "raise ValueError" in source.split("if set(PLOTTED_SEEDS) &")[1]
