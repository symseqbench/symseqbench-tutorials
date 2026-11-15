"""The runs of the complexity experiment: training, chosen weights, run files, figure data and the trend tests.

Each test has a case it must reject. All seeds are 101 or more; nothing is written into ``data/``.
"""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom
from test_complexity import UNIFILAR

from tutorial import complexity as cx
from tutorial import complexity_runs as runs
from tutorial import config

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_complexity.py"

# A grammar written by hand and a few gradient steps: the code of a run, not a result.
TINY = replace(
    config.COMPLEXITY_SMOKE,
    min_length=3,
    max_length=14,
    sizes=(50,),
    n_heldout=40,
    max_steps=60,
    eval_every=10,
)


@pytest.fixture(scope="module")
def job() -> runs.JobData:
    values = cx.describe(UNIFILAR, TINY.min_length, TINY.max_length)
    return runs.prepare(UNIFILAR, values, 101, TINY)


@pytest.fixture(scope="module")
def tiny_runs(job) -> dict:
    return {model: runs.train_run(job, model, 50, TINY) for model in config.MODELS}


def test_a_job_holds_what_the_measures_need(job):
    assert len(job.strings.heldout) == len(job.strings.ungrammatical) == 40
    assert job.width == TINY.max_length + 1 and job.classes == ("#", "A", "B", "C", "S")
    assert job.heldout_inputs.shape == (40, 15, 5) and job.p_heldout.shape == (40, 15, 5)
    assert job.exact["auc"] == 1.0 and job.exact["divergence_bits"] == 0.0
    assert job.exact["ungrammatical_possible"] == 0 and job.exact["largest_row_error"] < 1e-12
    assert job.walk_divergence > 0.0
    # D0, the divergence of the uniform guess, by a route that uses no function of the package:
    # log2 of the number of outcomes minus the entropy of the exact distribution at every target
    entropies = [
        -sum(x * math.log2(x) for x in job.p_heldout[row, t] if x > 0.0)
        for row, string in enumerate(job.strings.heldout)
        for t in range(len(string) + 1)
    ]
    by_hand = math.log2(len(job.classes)) - sum(entropies) / len(entropies)
    assert abs(job.uniform_divergence - by_hand) < 1e-12 and job.uniform_divergence > 0.0
    assert abs(job.exact["mean_entropy_bits"] - sum(entropies) / len(entropies)) < 1e-12
    assert job.exact["score"] == 1.0
    # must reject: the entropy of the generator in the place of the entropy on these targets
    assert abs(math.log2(len(job.classes)) - job.values["entropy"] - job.uniform_divergence) > 1e-3
    # the probability of a training string is the product of the exact predictor's steps
    string = next(iter(job.probability))
    assert abs(job.probability[string] - cx.string_probability(job.predictor, string)) < 1e-15
    # must reject: the probabilities of the distinct training strings do not exceed one in sum
    assert 0.0 < sum(job.probability.values()) < 1.0


def test_run_records_what_it_was_and_passes_its_checks(job, tiny_runs):
    for model, run in tiny_runs.items():
        assert (run["model"], run["seed"], run["n_train"], run["level"]) == (model, 101, 50, 0)
        assert (
            run["grammar"] == json.loads(json.dumps(UNIFILAR.as_dict()))
            or run["grammar"] == UNIFILAR.as_dict()
        )
        assert run["settings"] == TINY.as_dict() and run["network"] == config.PRODUCTION.as_dict()
        assert 19_800 <= run["n_parameters"] <= 20_200
        assert run["data"]["n_gradient"] + run["data"]["n_validation"] == 50
        assert run["data"]["n_distinct_training"] < 50  # draws: some strings were drawn twice
        assert run["history"]["step"] == list(range(0, 61, 10))
        assert run["environment"]["threads"] == 1
        assert run["checks"]["passed"] and len(run["checks"]) == 15
        assert run["measures"]["divergence_bits"] > 0.0
        # the prediction score: one minus the divergence over that of the uniform guess on the
        # held-out targets of this job, for the network, the first-order model and the references
        d0 = run["references"]["uniform_guess"]
        assert d0 == job.uniform_divergence
        for found in (run["measures"], run["first_order"]):
            assert abs(found["score"] - (1.0 - found["divergence_bits"] / d0)) < 1e-12
            assert found["score"] < 1.0
        assert set(run["reference_scores"]) == {"walk_without_bounds", "counted_transitions"}
        for name, score in run["reference_scores"].items():
            assert abs(score - (1.0 - run["references"][name] / d0)) < 1e-12
        assert run["exact"]["score"] == 1.0
        # must reject: the score made with the generator's entropy is another number
        generator = math.log2(run["n_classes"]) - run["grammar_values"]["entropy"]
        wrong = 1.0 - run["measures"]["divergence_bits"] / generator
        assert abs(wrong - run["measures"]["score"]) > 1e-4
        assert 0.0 < run["measures"]["top_choice_ratio"] <= 1.0
        assert 0.0 <= run["measures"]["auc"] <= 1.0
        assert set(run["references"]) == {
            "walk_without_bounds",
            "counted_transitions",
            "uniform_guess",
        }
        json.dumps(run)  # only numbers, strings, lists and dictionaries
    # the first-order model and the references do not depend on the network
    rnn, transformer = (tiny_runs[model] for model in config.MODELS)
    assert (
        rnn["first_order"] == transformer["first_order"]
        and rnn["references"] == transformer["references"]
    )
    # training moved the network towards the exact predictor
    assert rnn["history"]["divergence_bits"][0] > rnn["measures"]["divergence_bits"]
    # must reject: a size that is not one of the settings
    with pytest.raises(ValueError, match="must be one of"):
        runs.train_run(job, "rnn", 51, TINY)


def test_the_measured_weights_are_those_of_the_lowest_validation_loss(job, tiny_runs):
    for run in tiny_runs.values():
        history = run["history"]
        chosen = history["step"].index(run["best_step"])
        assert history["validation_loss"][chosen] == min(history["validation_loss"])
        assert chosen == history["validation_loss"].index(
            min(history["validation_loss"])
        )  # the first
        assert run["measures"]["validation_loss"] == history["validation_loss"][chosen]
        assert abs(history["divergence_bits"][chosen] - run["measures"]["divergence_bits"]) < 1e-9
        assert run["best_is_last"] == (run["best_step"] == history["step"][-1])
        # must reject: at every other step the divergence is another one
        others = [
            value for index, value in enumerate(history["divergence_bits"]) if index != chosen
        ]
        assert all(abs(value - run["measures"]["divergence_bits"]) > 1e-9 for value in others)
    # a run that goes on after its best step is measured at its best step, not at its last
    settings = replace(TINY, max_steps=400, eval_every=20)
    run = runs.train_run(job, "rnn", 50, settings)
    history = run["history"]
    chosen = history["step"].index(run["best_step"])
    assert not run["best_is_last"] and chosen == history["validation_loss"].index(
        min(history["validation_loss"])
    )
    assert abs(history["divergence_bits"][chosen] - run["measures"]["divergence_bits"]) < 1e-9
    # must reject: the weights after the last step give another divergence
    assert abs(history["divergence_bits"][-1] - run["measures"]["divergence_bits"]) > 1e-6


def test_identical_seeds_give_identical_runs(job, tiny_runs):
    again = runs.train_run(job, "rnn", 50, TINY)
    first = tiny_runs["rnn"]
    for key in (
        "history",
        "measures",
        "first_order",
        "references",
        "weights_sha256",
        "best_step",
        "data",
    ):
        assert again[key] == first[key], key
    # the strings of a job are drawn again identically
    values = cx.describe(UNIFILAR, TINY.min_length, TINY.max_length)
    rebuilt = runs.prepare(UNIFILAR, values, 101, TINY)
    assert (
        rebuilt.strings.sequence == job.strings.sequence
        and rebuilt.strings.heldout == job.strings.heldout
    )
    assert runs.train_run(rebuilt, "rnn", 50, TINY)["weights_sha256"] == first["weights_sha256"]
    # must reject: another training seed gives other strings, other weights and another history
    other = runs.train_run(runs.prepare(UNIFILAR, values, 102, TINY), "rnn", 50, TINY)
    assert other["weights_sha256"] != first["weights_sha256"]
    assert other["history"]["validation_loss"] != first["history"]["validation_loss"]
    assert runs.run_seed(101, UNIFILAR, 50) != runs.run_seed(101, UNIFILAR, 200)


def test_the_training_sample_is_the_one_of_the_settings(job, tiny_runs):
    """The settings train on draws; the other sample, N distinct strings, is another run."""
    assert TINY.training_sample == "draws" and job.strings.draws >= 50
    drawn_run = tiny_runs["rnn"]
    # 50 draws: fewer distinct strings, 50 rows, and the repeats of a string on one side of the split
    assert drawn_run["data"]["n_gradient"] + drawn_run["data"]["n_validation"] == 50
    assert drawn_run["data"]["n_distinct_training"] == len(set(job.strings.sequence[:50])) < 50
    assert drawn_run["data"]["n_distinct_validation"] == round(
        TINY.validation_fraction * drawn_run["data"]["n_distinct_training"]
    )
    values = cx.describe(UNIFILAR, TINY.min_length, TINY.max_length)
    settings = replace(TINY, training_sample="distinct")
    distinct = runs.prepare(UNIFILAR, values, 101, settings)
    run = runs.train_run(distinct, "rnn", 50, settings)
    assert run["data"]["n_gradient"] == 40 and run["data"]["n_validation"] == 10
    assert run["data"]["n_distinct_training"] == 50 and run["checks"]["passed"]
    # must reject: the two samples do not give the same run
    assert run["weights_sha256"] != drawn_run["weights_sha256"]
    assert run["first_order"] != drawn_run["first_order"]
    assert run["references"]["uniform_guess"] != drawn_run["references"]["uniform_guess"]


@pytest.fixture(scope="module")
def repeated_job() -> tuple[runs.JobData, config.ComplexitySettings]:
    """The job in which the code check found the defect: two held-out strings, one violation.

    The grammar of the second pilot level (level index 1, grammar seed 101) with training seed 201
    and the settings of the pilot, with a few gradient steps only. 198 of its 200 ungrammatical
    strings are distinct.
    """
    level = config.COMPLEXITY_PILOT.levels[1]
    design = config.COMPLEXITY_PILOT
    spec, reason = cx.candidate(
        1, 101, level.target_nats, design.n_states, design.alphabet_size, design.n_terminal_states
    )
    assert spec is not None, reason
    settings = replace(design, max_steps=20, eval_every=10)
    values = cx.describe(spec, settings.min_length, settings.max_length)
    return runs.prepare(spec, values, 201, settings), settings


def test_a_job_with_a_repeated_ungrammatical_string_runs_through(repeated_job):
    job, settings = repeated_job
    strings = job.strings
    assert len(strings.ungrammatical) == len(strings.heldout) == 200
    assert len(set(strings.ungrammatical)) == 198 and len(set(strings.heldout)) == 200
    # every one of them is ungrammatical, and none is a training or a held-out string
    assert job.exact["ungrammatical_possible"] == 0 and job.exact["auc"] == 1.0
    assert not set(strings.ungrammatical) & (set(strings.sequence) | set(strings.heldout))
    training = strings.training(50, settings.training_sample)
    assert training == {
        s: strings.sequence[:50].count(s) for s in dict.fromkeys(strings.sequence[:50])
    }

    def scores(table, some: tuple) -> np.ndarray:
        inputs, targets = cx.encode(some, job.classes, job.width)
        return cx.largest_surprisal(table[inputs.argmax(axis=-1)], targets)

    for model in config.MODELS:
        run = runs.train_run(job, model, 50, settings)
        assert run["checks"]["passed"], model
        assert (
            run["data"]["n_ungrammatical"] == 200 and run["data"]["distinct_ungrammatical"] == 198
        )
        assert run["data"]["n_heldout"] == 200 and 0.0 <= run["measures"]["auc"] <= 1.0
        # the measures are taken on all 200 ungrammatical strings as they are: the area of the
        # first-order model, computed here from all of them, is the stored one
        table = cx.first_order_log2(training, job.classes)
        grammatical = scores(table, strings.heldout)
        on_all = cx.auc(scores(table, strings.ungrammatical), grammatical)
        assert run["first_order"]["auc"] == on_all
        # must reject: the area over the 198 distinct ones is another number
        on_distinct = cx.auc(
            scores(table, tuple(dict.fromkeys(strings.ungrammatical))), grammatical
        )
        assert on_distinct != on_all

    # must reject: an ungrammatical string that is also a gradient, a validation or a held-out string
    distinct = list(training)
    gradient, validation = cx.split_training(
        len(distinct),
        settings.validation_fraction,
        np.random.default_rng(runs.run_seed(job.seed, job.spec, 50)),
    )
    for leaked in (distinct[gradient[0]], distinct[validation[0]], strings.heldout[7]):
        doctored = replace(strings, ungrammatical=(leaked, *strings.ungrammatical[1:]))
        with pytest.raises(ValueError, match="also a training or a held-out string"):
            runs.train_run(replace(job, strings=doctored), "rnn", 50, settings)
    # must reject: a held-out string that occurs twice, and one that is a training string
    twice = replace(strings, heldout=(strings.heldout[0], *strings.heldout[:-1]))
    with pytest.raises(ValueError, match="more than once"):
        runs.train_run(replace(job, strings=twice), "rnn", 50, settings)
    trained = replace(strings, heldout=(distinct[gradient[0]], *strings.heldout[1:]))
    with pytest.raises(ValueError, match="more than once"):
        runs.train_run(replace(job, strings=trained), "rnn", 50, settings)


def test_checks_of_a_run_can_fail(tiny_runs):
    run = tiny_runs["rnn"]
    assert runs.check_run(run)["passed"]
    d0_by_two_routes = "uniform guess: divergence is log2 of the outcomes minus the mean entropy of the exact predictor"
    score_is_eq_13 = "every score is 1 minus the divergence over that of the uniform guess"
    # must reject: each of these changed runs. A changed score or D0 also breaks the check that
    # ties every score to the stored divergences, so two checks fail for those.
    also = {
        "uniform guess: divergence is positive": {d0_by_two_routes, score_is_eq_13},
        "exact predictor: score is 1": {score_is_eq_13},
        "no score exceeds 1": {score_is_eq_13},
        "D0 changed": {d0_by_two_routes, score_is_eq_13},
        "exact predictor: divergence is 0": {score_is_eq_13},
        "divergence is never negative": {score_is_eq_13},
    }
    changes = {
        d0_by_two_routes: lambda r: r["exact"].update(
            mean_entropy_bits=r["exact"]["mean_entropy_bits"] + 1e-6
        ),
        "D0 changed": lambda r: r["references"].update(
            uniform_guess=r["references"]["uniform_guess"] + 1e-6
        ),
        "uniform guess: divergence is positive": lambda r: r["references"].update(
            uniform_guess=0.0
        ),
        "exact predictor: score is 1": lambda r: r["exact"].update(score=0.999),
        "no score exceeds 1": lambda r: r["first_order"].update(score=1.001),
        score_is_eq_13: lambda r: r["measures"].update(score=r["measures"]["score"] - 1e-6),
        "exact predictor: area under the curve is 1": lambda r: r["exact"].update(auc=0.999),
        "exact predictor: divergence is 0": lambda r: r["exact"].update(divergence_bits=1e-6),
        "exact predictor: no ungrammatical string is possible": lambda r: r["exact"].update(
            ungrammatical_possible=1
        ),
        "exact predictor: distributions sum to one": lambda r: r["exact"].update(
            largest_row_error=1e-3
        ),
        "divergence is never negative": lambda r: r["measures"].update(divergence_bits=-1e-3),
        "first choice never beats the best achievable": lambda r: r["measures"].update(
            top_choice_ratio=1.01
        ),
        "numbers of strings are those of the settings": lambda r: r["data"].update(n_heldout=39),
    }
    for name, change in changes.items():
        changed = copy.deepcopy(run)
        change(changed)
        found = runs.check_run(changed)
        failed = {check for check, value in found.items() if not value}
        expected = ({name} if name in found else set()) | also.get(name, set())
        assert failed == expected | {"passed"}, name


# --- the script and the run files -----------------------------------------------------------------


def script(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments], cwd=REPO, capture_output=True, text=True
    )


@pytest.fixture(scope="module")
def smoke_folder():
    """One smoke run of the script: two levels, one grammar each, one size, both networks."""
    import shutil
    import uuid

    folder = REPO / ".pytest_work" / f"complexity-{uuid.uuid4().hex}"
    done = script("--smoke", "--out", str(folder))
    assert done.returncode == 0, done.stderr[-2000:]
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


def numbers(run: dict) -> dict:
    """Everything of a run but its timings and the speed of the machine."""
    return {key: value for key, value in run.items() if key not in ("seconds", "environment")}


def test_script_writes_runs_summary_manifest_and_grammars(smoke_folder):
    saved, manifest, drawn = runs.load_runs(smoke_folder)
    assert manifest["kind"] == "smoke" and manifest["placeholder"] is True
    assert manifest["expected"] == manifest["present"] == len(saved) == 4
    assert manifest["complete"] and not manifest["missing"] and not manifest["checks_failed"]
    assert manifest["settings"] == json.loads(json.dumps(config.COMPLEXITY_SMOKE.as_dict()))
    for name, digest in manifest["files"].items():
        assert runs.sha256(smoke_folder / name) == digest
    assert runs.sha256(smoke_folder / "grammars.json") == manifest["grammars"]["sha256"]
    summary = (smoke_folder / "summary.csv").read_text().splitlines()
    assert len(summary) == 5 and "divergence (bits per target)" in summary[0]
    # the grammars: every candidate is listed with its values or the reason for which there is none
    for level in drawn["levels"]:
        assert len(level["admitted"]) == 1 and level["complete"]
        assert [entry["admitted"] for entry in level["candidates"]].count(True) == 1
        assert all(entry["reasons"] for entry in level["candidates"] if not entry["admitted"])
        admitted = level["admitted"][0]
        low, high = level["band_bits"]
        assert low <= admitted["values"]["topological_entropy_bits"] < high
        assert admitted["values"]["memory_gap"] >= config.COMPLEXITY.min_memory_gap_bits
        assert admitted["spec"]["seed"] >= 101
    assert all(run["seed"] == 101 and run["checks"]["passed"] for run in saved)
    # must reject: grammars drawn for another kind of run, or with other settings, are refused
    with pytest.raises(ValueError, match="were drawn for smoke runs"):
        runs.load_grammars(smoke_folder, "pilot", config.COMPLEXITY_SMOKE)
    with pytest.raises(ValueError, match="other settings"):
        runs.load_grammars(smoke_folder, "smoke", config.COMPLEXITY_PILOT)
    with pytest.raises(FileNotFoundError, match="Draw them first"):
        runs.load_grammars(smoke_folder / "runs")


def test_a_second_call_computes_nothing_and_a_deleted_run_is_recomputed_identically(smoke_folder):
    _, manifest, _ = runs.load_runs(smoke_folder)
    stamps = {name: (smoke_folder / name).stat().st_mtime_ns for name in manifest["files"]}
    done = script("--smoke", "--out", str(smoke_folder))
    assert done.returncode == 0, done.stderr[-2000:]
    assert "0 runs computed" in done.stdout and "(4 already saved)" in done.stdout
    assert {name: (smoke_folder / name).stat().st_mtime_ns for name in manifest["files"]} == stamps
    # delete one run: only that one is trained again, in another process, with the same numbers
    name = sorted(manifest["files"])[1]
    before = json.loads((smoke_folder / name).read_text())
    (smoke_folder / name).unlink()
    partial = script("--smoke", "--collect", "--out", str(smoke_folder))
    assert partial.returncode == 1 and "3 of 4 runs present, 1 missing" in partial.stdout
    assert json.loads((smoke_folder / "manifest.json").read_text())["complete"] is False
    done = script("--smoke", "--out", str(smoke_folder))
    assert done.returncode == 0 and "1 runs computed" in done.stdout
    after = json.loads((smoke_folder / name).read_text())
    assert numbers(after) == numbers(before)
    # must reject: the timings of the two processes are not what makes the runs equal
    assert after["seconds"] != before["seconds"]


def script_module():
    spec = importlib.util.spec_from_file_location("run_complexity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_failed_job_is_recorded_and_does_not_end_the_part(work_dir, monkeypatch, capsys):
    """One job of a smoke run is made to raise: the other job runs, and the folder says what is missing."""
    module = script_module()
    folder = work_dir / "smoke"
    assert module.main(["--smoke", "--draw", "--out", str(folder)]) == 0
    first, second = (spec for spec, _ in runs.admitted_grammars(runs.load_grammars(folder)))
    train = runs.train_run

    def failing(data, *arguments, **keywords):
        if data.spec.name == first.name:
            raise ValueError("The string F E F B B D occurs more than once.")
        return train(data, *arguments, **keywords)

    monkeypatch.setattr(runs, "train_run", failing)
    capsys.readouterr()
    # the part goes on after the failed job, and ends with an exit code that is not zero
    assert module.main(["--smoke", "--shard", "1/1", "--out", str(folder)]) == 1
    printed = capsys.readouterr().out
    assert "JOB FAILED: ValueError: The string F E F B B D occurs more than once." in printed
    assert "2 runs computed" in printed and f"FAILED JOB {runs.job_name(first, 101)}" in printed
    written = sorted(path.name for path in (folder / "runs").glob("*.json"))
    assert len(written) == 2 and all(f"grammar{second.seed}" in name for name in written)
    assert all(name.startswith(f"level{second.level + 1}-") for name in written)
    (record,) = runs.failed_jobs(folder)
    assert record["job"] == runs.job_name(first, 101) and record["grammar"] == first.name
    assert record["error"] == "ValueError: The string F E F B B D occurs more than once."
    assert "Traceback" in record["traceback"] and record["seed"] == 101
    # the collect step: the folder is incomplete, and says which job failed, why, and what is missing
    assert module.main(["--smoke", "--collect", "--out", str(folder)]) == 1
    printed = capsys.readouterr().out
    assert "2 of 4 runs present, 2 missing" in printed and "1 failed jobs" in printed
    assert f"FAILED JOB {record['job']}: {record['error']}" in printed
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["complete"] is False and len(manifest["missing"]) == 2
    assert all(name.startswith(f"level{first.level + 1}-") for name in manifest["missing"])
    assert all(name in printed for name in manifest["missing"])
    assert manifest["failed_jobs"] == [
        {key: record[key] for key in ("job", "grammar", "seed", "error")}
    ]
    # once the job runs through, its record is removed and the folder is complete
    monkeypatch.setattr(runs, "train_run", train)
    assert module.main(["--smoke", "--out", str(folder)]) == 0
    assert runs.failed_jobs(folder) == []
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["complete"] is True and manifest["failed_jobs"] == []
    assert manifest["present"] == 4 and "2 runs computed" in capsys.readouterr().out
    # must reject: a collect step that reports a complete folder when the record of a failed job
    # is there, even though every run file is present
    runs.record_failure(folder, "smoke", second, 101, RuntimeError("no strings"))
    again = runs.collect(folder, "smoke", config.COMPLEXITY_SMOKE)
    assert again["present"] == 4 and not again["missing"] and again["complete"] is False
    assert again["failed_jobs"][0]["error"] == "RuntimeError: no strings"
    assert module.main(["--smoke", "--collect", "--out", str(folder)]) == 1
    runs.clear_failure(folder, second, 101)
    assert runs.collect(folder, "smoke", config.COMPLEXITY_SMOKE)["complete"] is True


def test_loader_refuses_changed_files(smoke_folder, work_dir):
    import shutil

    copy_folder = work_dir / "copy"
    shutil.copytree(smoke_folder, copy_folder)
    runs.load_runs(copy_folder)
    manifest = json.loads((copy_folder / "manifest.json").read_text())
    name = sorted(manifest["files"])[0]
    # must reject: a run file changed after the manifest was written
    run = json.loads((copy_folder / name).read_text())
    run["measures"]["auc"] = 1.0
    (copy_folder / name).write_text(json.dumps(run, indent=1))
    with pytest.raises(ValueError, match="was changed after the manifest"):
        runs.load_runs(copy_folder)
    shutil.copy(smoke_folder / name, copy_folder / name)
    # must reject: a run file that the manifest does not list
    shutil.copy(copy_folder / name, copy_folder / "runs" / "extra.json")
    with pytest.raises(ValueError, match="not in the manifest"):
        runs.load_runs(copy_folder)
    (copy_folder / "runs" / "extra.json").unlink()
    # must reject: grammars changed after the manifest was written
    drawn = json.loads((copy_folder / "grammars.json").read_text())
    drawn["levels"][0]["admitted"][0]["values"]["topological_entropy_bits"] += 0.1
    (copy_folder / "grammars.json").write_text(json.dumps(drawn, indent=1))
    with pytest.raises(ValueError, match=r"grammars\.json was changed"):
        runs.load_runs(copy_folder)
    # must reject: a file that is listed and absent
    shutil.copy(smoke_folder / "grammars.json", copy_folder / "grammars.json")
    (copy_folder / name).unlink()
    with pytest.raises(ValueError, match="listed in the manifest and absent"):
        runs.load_runs(copy_folder)


def test_script_refuses_what_would_mix_results(work_dir):
    # must reject: a smoke run without a folder, another training sample without a folder, a bad shard
    assert "give --out FOLDER" in script("--smoke").stderr
    assert "give --out FOLDER" in script("--pilot", "--training-sample", "draws").stderr
    assert script("--pilot", "--shard", "3/2").returncode != 0
    assert script().returncode != 0  # the kind of run is required
    assert script("--pilot", "--prod").returncode != 0  # no abbreviations
    # a part cannot run before the grammars are drawn
    done = script("--smoke", "--shard", "1/2", "--out", str(work_dir / "empty"))
    assert done.returncode != 0 and "Draw them first" in done.stderr


def test_jobs_are_spread_over_the_parts():
    module = script_module()
    drawn = {
        "levels": [
            {
                "admitted": [
                    {
                        "spec": {**UNIFILAR.as_dict(), "level": level, "seed": 101 + rank},
                        "values": {},
                    }
                    for rank in range(3)
                ]
            }
            for level in range(4)
        ]
    }
    settings = replace(config.COMPLEXITY, seeds=(101, 102))
    jobs = module.jobs(drawn, settings)
    assert len(jobs) == 4 * 3 * 2 and len({(s.level, s.seed, seed) for s, _, seed in jobs}) == 24
    for part in (jobs[0::2], jobs[1::2]):
        levels = [spec.level for spec, _, _ in part]
        assert [levels.count(level) for level in range(4)] == [3, 3, 3, 3]
    # must reject: jobs listed level by level put two levels in each part
    naive = sorted(jobs, key=lambda job: (job[2], job[0].seed, job[0].level))
    assert {spec.level for spec, _, _ in naive[0::2]} == {0, 2}


# --- the numbers of the figure and the tests of the expectation ------------------------------------


def synthetic(n_levels: int = 4, per_level: int = 3, seeds=(101, 102)) -> tuple[list, dict, dict]:
    """Runs, manifest and grammars with known values.

    The divergence rises with the level and the divergence of the uniform guess falls (1.6 bit at
    the first level, 0.2 less at each further one), so the prediction score falls with the level;
    the area under the curve falls too. More training strings are better in every grammar.
    """
    drawn = {"levels": []}
    saved = []
    for level in range(n_levels):
        admitted = []
        guess = 1.6 - 0.2 * level
        for rank in range(per_level):
            spec = replace(
                UNIFILAR,
                name=f"level {level + 1}, grammar {101 + rank}",
                level=level,
                seed=101 + rank,
            )
            values = {
                "topological_entropy_bits": 1.0 + 0.25 * level + 0.01 * rank,
                "memory_gap": 0.1,
                "n_strings": 5000,
                "mean_length": 8.0,
            }
            admitted.append({"spec": spec.as_dict(), "values": values})
            for seed, n_train, model in itertools.product(seeds, (50, 200, 800), config.MODELS):
                base = 0.1 * (level + 1) + 0.01 * rank + (0.02 if model == "rnn" else 0.0)
                divergence = base * 50 / n_train + 0.001 * (seed - 101)
                saved.append(
                    {
                        "grammar": spec.as_dict(),
                        "grammar_values": values,
                        "level": level,
                        "model": model,
                        "seed": seed,
                        "n_train": n_train,
                        "measures": {
                            "divergence_bits": divergence,
                            "score": 1.0 - divergence / guess,
                            "top_choice_ratio": 0.9,
                            "auc": 1.0 - base / 2 + 0.0002 * n_train,
                        },
                        "first_order": {
                            "divergence_bits": 0.5,
                            "score": 1.0 - 0.5 / guess,
                            "top_choice_ratio": 0.8,
                            "auc": 0.7 - 0.01 * level,
                        },
                        "references": {"uniform_guess": guess},
                        "data": {
                            "n_distinct_training": 0.6 * n_train + level + (seed - 101),
                            "training_mass": 0.5 - 0.1 * level,
                            "mean_length_training": 9.0,
                            "mean_length_heldout": 12.0 + level,
                            "n_gradient": 0.8 * n_train - level,
                            "n_validation": 0.2 * n_train + level,
                            "n_distinct_validation": 0.1 * n_train + (seed - 101),
                        },
                    }
                )
        drawn["levels"].append(
            {
                "level": level,
                "band_bits": [1.0 + 0.25 * level, 1.2 + 0.25 * level],
                "admitted": admitted,
            }
        )
    manifest = {
        "kind": "production",
        "placeholder": False,
        "sizes": [50, 200, 800],
        "seeds": list(seeds),
        "models": list(config.MODELS),
    }
    return saved, manifest, drawn


def test_figure_data_holds_seeds_grammar_means_and_level_means():
    saved, manifest, drawn = synthetic()
    data = runs.figure_data(saved, manifest, drawn)
    assert [grammar["level"] for grammar in data["grammars"]] == [0] * 3 + [1] * 3 + [2] * 3 + [
        3
    ] * 3
    assert len(data["levels"]) == 4 and data["placeholder"] is False
    entry = data["models"]["rnn"]["200"]["divergence_bits"]
    # grammar 2 of level 3 (index 7): base 0.1 * 3 + 0.01 + 0.02, times 50 / 200, seeds add 0 and 0.001
    by_hand = [0.33 * 0.25, 0.33 * 0.25 + 0.001]
    assert (
        np.allclose(entry["seeds"][7], by_hand)
        and abs(entry["grammar"][7] - np.mean(by_hand)) < 1e-12
    )
    assert abs(entry["level"][2] - np.mean(entry["grammar"][6:9])) < 1e-12
    assert abs(data["levels"][2]["topological_entropy_bits"] - (1.5 + 0.01)) < 1e-12
    # the score of the same grammar: the uniform guess of level 3 is at 1.6 - 0.4 bit
    score = data["models"]["rnn"]["200"]["score"]
    assert np.allclose(score["seeds"][7], [1.0 - d / 1.2 for d in by_hand], rtol=0.0, atol=1e-12)
    assert abs(score["grammar"][7] - (1.0 - np.mean(by_hand) / 1.2)) < 1e-12
    assert abs(score["level"][2] - np.mean(score["grammar"][6:9])) < 1e-12
    # must reject: the score of a grammar is not one minus its mean divergence over another level's guess
    assert abs(score["grammar"][7] - (1.0 - np.mean(by_hand) / 1.4)) > 1e-3
    # the first-order model is counted once per grammar, size and seed, not once per network
    assert len(data["first_order"]["50"]["auc"]["seeds"][0]) == 2
    assert abs(data["first_order"]["50"]["score"]["grammar"][7] - (1.0 - 0.5 / 1.2)) < 1e-12
    table = runs.mean_table(data)
    assert table.shape[0] == 4 * 3
    for column in (
        "tanh RNN: score",
        "linear transformer: score",
        "first-order model: score",
        "first-order model: area",
        "tanh RNN: divergence (bits)",
    ):
        assert column in table.columns, column
    assert abs(table.loc[(3, 200), "tanh RNN: score"] - score["level"][2]) < 1e-12
    # must reject: a grammar without a run for one network and size
    with pytest.raises(ValueError, match="No saved run"):
        runs.figure_data(
            [
                r
                for r in saved
                if not (r["model"] == "rnn" and r["n_train"] == 800 and r["grammar"]["level"] == 1)
            ],
            manifest,
            drawn,
        )


def test_table_of_the_strings_of_the_runs_by_hand():
    saved, _, _ = synthetic()
    table = runs.job_table(saved)
    assert table.shape[0] == 4 * 3  # levels and training sizes
    row = table.loc[(3, 200)]  # level 3: index 2; three grammars and two seeds, one network counted
    assert row["distinct among them"] == pytest.approx(0.6 * 200 + 2 + 0.5)
    assert row["probability of the training strings"] == pytest.approx(0.3)
    assert row["mean length: generator"] == pytest.approx(8.0)
    assert row["mean length: held-out"] == pytest.approx(14.0)
    assert row["uniform guess: divergence (bits)"] == pytest.approx(1.2)
    # how a network divides the 200 strings: gradient steps, choice of the weights
    assert row["used for gradient steps"] == pytest.approx(0.8 * 200 - 2)
    assert row["used to choose the weights"] == pytest.approx(0.2 * 200 + 2)
    assert row["distinct among those"] == pytest.approx(0.1 * 200 + 0.5)
    assert row["used for gradient steps"] + row["used to choose the weights"] == pytest.approx(200)
    # must reject: the runs of one seed only give another number of distinct strings
    one_seed = runs.job_table([run for run in saved if run["seed"] == 101])
    assert one_seed.loc[(3, 200), "distinct among them"] == pytest.approx(0.6 * 200 + 2)


def test_tables_of_a_run_show_score_area_and_divergence(job, tiny_runs, monkeypatch):
    named = {config.MODEL_LABELS[model]: run for model, run in tiny_runs.items()}
    table = runs.measures_table(named)
    assert list(table.index) == [
        "tanh RNN",
        "linear transformer",
        "first-order model",
        "exact predictor",
        "uniform guess",
    ]
    for column in ("prediction score", "area under the curve", "divergence (bits per target)"):
        assert column in table.columns, column
    run = tiny_runs["rnn"]
    d0 = run["references"]["uniform_guess"]
    assert table.loc["tanh RNN", "prediction score"] == run["measures"]["score"]
    assert table.loc["exact predictor", "prediction score"] == 1.0
    assert table.loc["uniform guess", "prediction score"] == 0.0
    assert table.loc["uniform guess", "divergence (bits per target)"] == d0
    history = runs.history_table(named, every=1)
    scores = history["tanh RNN: score"].to_numpy()
    divergences = np.array(run["history"]["divergence_bits"])
    assert np.allclose(scores, 1.0 - divergences / d0, rtol=0.0, atol=1e-12)
    assert np.allclose(history["tanh RNN: divergence (bits)"], divergences, rtol=0.0, atol=0.0)
    # must reject: the score of the untrained network is not that of the measured weights
    assert abs(scores[0] - run["measures"]["score"]) > 1e-3

    # the row of the uniform guess: its area is one half, and it is what the measures of the other
    # rows give for a model that puts 1/K on every outcome at every position of the job's strings
    area, first_choice = "area under the curve", "first choice / best achievable"
    outcomes = len(job.classes)
    assert outcomes == run["n_classes"]
    guess_heldout = np.full(job.p_heldout.shape, -np.log2(outcomes))
    guess_ungrammatical = np.full((*job.ungrammatical_targets.shape, outcomes), -np.log2(outcomes))
    by_the_strings = cx.measures(
        guess_heldout,
        guess_ungrammatical,
        job.p_heldout,
        job.heldout_targets,
        job.ungrammatical_targets,
    )
    assert table.loc["uniform guess", area] == by_the_strings["auc"] == 0.5
    assert table.loc["uniform guess", "prediction score"] == 0.0
    assert by_the_strings["score"] == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert by_the_strings["divergence_bits"] == pytest.approx(d0, rel=0.0, abs=1e-12)
    # its first choice is not defined and the cell is empty: every outcome ties, so what Eq. (8)
    # gives depends on the order of the classes
    assert np.isnan(table.loc["uniform guess", first_choice])
    in_this_order = cx.top_choice_ratio(job.p_heldout, guess_heldout, job.heldout_targets)
    reversed_order = cx.top_choice_ratio(
        job.p_heldout[..., ::-1], guess_heldout, job.heldout_targets
    )
    assert abs(in_this_order - reversed_order) > 1e-3
    assert not np.isnan(
        table.loc["first-order model", first_choice]
    )  # must reject: an empty column
    # must reject: an area typed into the table. Eq. (10) is replaced, and the table follows it
    seen = []

    def another_area(ungrammatical, grammatical) -> float:
        seen.append((np.asarray(ungrammatical), np.asarray(grammatical)))
        return 0.25

    monkeypatch.setattr(cx, "auc", another_area)
    assert runs.measures_table(named).loc["uniform guess", area] == 0.25
    ((ungrammatical, grammatical),) = seen
    assert len(ungrammatical) == run["data"]["n_ungrammatical"] == len(job.strings.ungrammatical)
    assert len(grammatical) == run["data"]["n_heldout"] == len(job.strings.heldout)
    assert set(ungrammatical) | set(grammatical) == {np.log2(outcomes)}


def test_trend_table_reports_rho_and_a_permutation_p_value():
    saved, manifest, drawn = synthetic()
    table = runs.trend_table(runs.figure_data(saved, manifest, drawn))
    # two networks and three sizes: twelve cells of the expectation, then six rows of the divergence
    assert len(table) == 2 * 3 * 3 and set(table["grammars"]) == {12}
    tested = table[table["expected"] == "falls"]
    untested = table[table["expected"] == "none"]
    assert list(tested["measure"]) == ["prediction score"] * 6 + ["area under the curve"] * 6
    assert list(untested["measure"]) == ["divergence (bits per target)"] * 6
    assert list(table.index[:12]) == list(tested.index)  # the cells of the expectation come first
    assert np.allclose(tested["Spearman rho"], -1.0)
    assert (tested["verdict"] == "supported").all()
    assert (tested["p (one-sided, for a fall)"] < 1e-4).all()
    assert set(table["p from"]) == {f"{config.TREND_PERMUTATIONS} random orderings"}
    assert not table["at ceiling"].any()
    # the divergence rises here, and its row says so without a verdict
    assert np.allclose(untested["Spearman rho"], 1.0)
    assert (untested["verdict"] == "no expectation").all()
    assert (untested["p (one-sided, for a fall)"] > 0.99).all()
    # must reject: the same data with the complexity reversed support nothing
    reversed_data = runs.figure_data(saved, manifest, drawn)
    for grammar in reversed_data["grammars"]:
        grammar["topological_entropy_bits"] = 3.0 - grammar["topological_entropy_bits"]
    wrong = runs.trend_table(reversed_data)
    wrong_tested = wrong[wrong["expected"] == "falls"]
    assert (wrong_tested["verdict"] == "not supported").all()
    assert (wrong_tested["p (one-sided, for a fall)"] > 0.99).all()
    # must reject: a row of the divergence never carries a verdict, whatever its direction
    assert (wrong[wrong["expected"] == "none"]["verdict"] == "no expectation").all()
    assert (wrong[wrong["expected"] == "none"]["p (one-sided, for a fall)"] < 1e-4).all()

    # a measure at its best value in every grammar is marked as a ceiling and supports nothing
    def flat(score: float, area: float) -> dict:
        data = runs.figure_data(saved, manifest, drawn)
        for by_size in data["models"].values():
            for entry in by_size.values():
                entry["score"]["grammar"] = [score] * 12
                entry["auc"]["grammar"] = [area] * 12
        return runs.trend_table(data)

    ceiling = flat(0.995, 1.0)
    at_ceiling = ceiling[ceiling["expected"] == "falls"]
    assert at_ceiling["at ceiling"].all() and (at_ceiling["verdict"] == "not supported").all()
    assert at_ceiling["Spearman rho"].isna().all()
    assert not ceiling[ceiling["expected"] == "none"]["at ceiling"].any()
    assert config.CEILING_SCORE == 0.99 and config.CEILING_AUC == 0.99
    # must reject: grammars at a score of 0.98, and at an area of 0.98, are not at ceiling
    below = flat(0.98, 0.98)
    assert not below["at ceiling"].any()
    mixed = flat(0.995, 0.98)
    assert list(mixed[mixed["expected"] == "falls"]["at ceiling"]) == [True] * 6 + [False] * 6


def test_permutation_p_value_by_hand():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    # perfectly rising: only one of the 120 orderings is as rising
    assert runs.spearman(x, [2, 4, 5, 7, 9]) == pytest.approx(1.0)
    p, how = runs.permutation_p(x, [2, 4, 5, 7, 9], +1, 1000, 1)
    assert p == pytest.approx(1 / 120) and how == "all 120 orderings"
    # one swap: rho = 0.9; orderings with rho >= 0.9 are the identity and the four neighbour swaps
    assert runs.spearman(x, [2, 4, 5, 9, 7]) == pytest.approx(0.9)
    assert runs.permutation_p(x, [2, 4, 5, 9, 7], +1, 1000, 1)[0] == pytest.approx(5 / 120)
    # must reject: in the wrong direction the same data give p = 1
    assert runs.permutation_p(x, [2, 4, 5, 7, 9], -1, 1000, 1)[0] == pytest.approx(1.0)
    assert math.isnan(runs.permutation_p(x, [3, 3, 3, 3, 3], +1, 1000, 1)[0])
    # with more than 9 grammars orderings are drawn. The drawn share agrees with the exact one:
    # for 8 values both are computed, the drawn one by asking for it
    rng = np.random.default_rng(101)
    eight = np.arange(8.0)
    for noise in (2.0, 4.0, 8.0):
        y = eight + rng.normal(0.0, noise, 8)
        exact, how = runs.permutation_p(eight, y, +1, 100_000, 101)
        drawn, how_drawn = runs.permutation_p(eight, y, +1, 100_000, 101, exact_up_to=0)
        assert how == "all 40320 orderings" and how_drawn == "100000 random orderings"
        assert abs(drawn - exact) < 5.0 * math.sqrt(exact * (1.0 - exact) / 100_000) + 2e-5
        assert runs.permutation_p(eight, y, +1, 100_000, 101, exact_up_to=0)[0] == drawn  # seeded
    ten = np.arange(10.0)
    assert runs.permutation_p(ten, ten, +1, 1000, 101) == (
        pytest.approx(1 / 1001),
        "1000 random orderings",
    )
    # must reject: the drawn share of the reversed order is not the exact share of the order itself
    assert runs.permutation_p(eight, eight[::-1], +1, 100_000, 101, exact_up_to=0)[0] > 0.99


def test_size_table_is_a_sign_test_across_grammars():
    saved, manifest, drawn = synthetic()
    table = runs.size_table(runs.figure_data(saved, manifest, drawn))
    assert len(table) == 2 * 2 * 3  # two networks, two measures, three pairs of sizes
    assert set(table["measure"]) == {"prediction score", "area under the curve"}
    ends = table[table["sizes"] == "50 to 800"]
    assert (ends["better with more strings"] == 12).all()
    assert np.allclose(ends["p (sign test, one-sided)"], 0.5**12)
    assert binom.sf(11, 12, 0.5) == pytest.approx(0.5**12)
    # must reject: with the sizes exchanged in the data no grammar is better with more strings
    data = runs.figure_data(saved, manifest, drawn)
    for by_size in data["models"].values():
        by_size["50"], by_size["800"] = by_size["800"], by_size["50"]
    worse = runs.size_table(data)
    worse = worse[worse["sizes"] == "50 to 800"]
    assert (worse["better with more strings"] == 0).all() and np.allclose(
        worse["p (sign test, one-sided)"], 1.0
    )


def test_projection_of_the_production_time_by_hand():
    def run(name, seed, training, evaluation, data, probe):
        return {
            "grammar": {"name": name},
            "seed": seed,
            "seconds": {"training": training, "evaluation": evaluation, "data": data},
            "environment": {"speed_probe_us": [probe, probe]},
        }

    # one job of two runs: (18 + 2) s and (36 + 4) s at a probe of 10 us, data 5 s
    measured = [run("g", 101, 18.0, 2.0, 5.0, 10.0), run("g", 101, 36.0, 4.0, 5.0, 10.0)]
    found = runs.projected_seconds(measured, n_jobs=45, slow_probe_us=100.0, processes=2)
    # per job at 100 us: (2 + 4) * 10 * ... = two runs of 3 s per us on average, plus 0.5 s per us of data
    assert found["seconds_per_job"] == pytest.approx((3.0 * 2 + 0.5) * 100.0)
    assert found["hours"] == pytest.approx(650.0 * 45 / 2 / 3600.0)
    # must reject: a run measured on a machine ten times slower projects to the same time, not ten times more
    slow = [run("g", 101, 180.0, 20.0, 50.0, 100.0), run("g", 101, 360.0, 40.0, 50.0, 100.0)]
    assert runs.projected_seconds(slow, 45, 100.0, 2)["hours"] == pytest.approx(found["hours"])
    assert runs.projected_seconds(measured, 45, 10.0, 2)["hours"] == pytest.approx(
        found["hours"] / 10
    )
