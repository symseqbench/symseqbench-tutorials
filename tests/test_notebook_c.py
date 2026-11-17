"""Notebook C runs headless within a stated budget, on saved results and with its live branch.

Each test has a case it must reject.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import timeit
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_behaviour import broken_paragraphs
from test_conventions import cells as convention_cells

from tutorial import complexity, complexity_runs, config, results

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "c_networks.py"
FALLBACK = REPO / "fallback" / "c_networks"

# Stated budget for one headless run on saved results, in seconds of wall time. Measured on the
# author's workstation: a few seconds at full speed. When its processor is throttled to about a
# tenth of its speed the run takes about ten times longer; the budget leaves room for that.
BUDGET_S = 180.0

SWITCH_ON = 'load_saved = mo.ui.switch(value=True, label="load saved results")'
SWITCH_OFF = 'load_saved = mo.ui.switch(value=False, label="load saved results")'
LIVE_SETTINGS = "    live_settings = config.COMPLEXITY\n"
SHORT_SETTINGS = (
    '    live_settings = __import__("dataclasses").replace('
    "config.COMPLEXITY, max_steps=60, eval_every=20, n_heldout=40)\n"
)
LAST_LINES = "    views.full_width(figure)\n    return (figure,)"
REPORT = (
    '    print("REPORT " + __import__("json").dumps({"trained_here": trained_here, '
    '"measures": {model: run["measures"] for model, run in trained.items()}, '
    '"steps": {model: run["settings"]["max_steps"] for model, run in trained.items()}, '
    '"sizes": {model: run["n_train"] for model, run in trained.items()}, '
    '"levels": {model: run["level"] for model, run in trained.items()}, '
    '"seeds": {model: run["seed"] for model, run in trained.items()}}))\n'
)
SIX_LOADED = "    saved_runs, manifest = results.load_runs(results.default_folder())\n"
SIX_PLACEHOLDER = '    placeholder = manifest["placeholder"]\n'
COMPLEXITY_PLACEHOLDER = '    complexity_placeholder = complexity_manifest["placeholder"]\n'
COMPLEXITY_DATA = "    complexity_data = complexity_runs.figure_data(complexity_saved, complexity_manifest, grammars)\n"


def environment() -> dict:
    return dict(
        os.environ,
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )


def run(path: Path) -> tuple[int, float, str, str]:
    """Run a marimo notebook as a script. Returns exit code, seconds, stdout, stderr."""
    start = time.perf_counter()
    done = subprocess.run(
        [sys.executable, str(path)], cwd=REPO, env=environment(), capture_output=True, text=True
    )
    return done.returncode, time.perf_counter() - start, done.stdout, done.stderr


def variant(work_dir: Path, name: str, *replacements: tuple[str, str]) -> Path:
    """A copy of the notebook with some lines replaced; every line to replace must occur exactly once."""
    source = NOTEBOOK.read_text()
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    path = work_dir / name
    path.write_text(source)
    return path


def report(stdout: str) -> dict:
    return json.loads(next(line for line in stdout.splitlines() if line.startswith("REPORT "))[7:])


def within_budget(seconds: float, budget: float = BUDGET_S) -> bool:
    return seconds <= budget


def test_notebook_c_runs_headless_on_saved_results_within_the_budget(capsys):
    probe = timeit.timeit(lambda: sum(range(1000)), number=2000) / 2000 * 1e6
    code, seconds, _, stderr = run(NOTEBOOK)
    with capsys.disabled():
        print(
            f"\n[notebook C] {seconds:.1f} s of {BUDGET_S:.0f} s; speed probe {probe:.0f} us (about 10 us on an idle core)"
        )
    assert code == 0, stderr[-2000:]
    assert within_budget(seconds)
    # must reject: the same run against a budget it cannot meet
    assert not within_budget(seconds, budget=0.5)


def test_a_failing_cell_makes_the_headless_run_fail(work_dir):
    """must reject: the same notebook with one cell that raises exits with an error."""
    broken = variant(
        work_dir,
        "broken.py",
        (COMPLEXITY_DATA, COMPLEXITY_DATA + '    raise RuntimeError("injected failure")\n'),
    )
    code, _, _, stderr = run(broken)
    assert code != 0
    assert "injected failure" in stderr


def test_the_live_branch_trains_both_networks(work_dir):
    """With the switch off both networks are trained in the notebook, here for 60 gradient steps."""
    live = variant(
        work_dir,
        "live.py",
        (SWITCH_ON, SWITCH_OFF),
        (LIVE_SETTINGS, SHORT_SETTINGS),
        (LAST_LINES, REPORT + LAST_LINES),
    )
    code, _, stdout, stderr = run(live)
    assert code == 0, stderr[-2000:]
    trained = report(stdout)
    assert trained["trained_here"] is True
    assert set(trained["measures"]) == set(config.MODELS)
    assert trained["seeds"] == dict.fromkeys(config.MODELS, config.COMPLEXITY_LIVE_SEED)
    assert trained["steps"] == dict.fromkeys(config.MODELS, 60)
    assert trained["sizes"] == dict.fromkeys(config.MODELS, config.COMPLEXITY_LIVE_SIZE)
    assert trained["levels"] == dict.fromkeys(config.MODELS, config.COMPLEXITY_LIVE_LEVEL)
    for found in trained["measures"].values():
        assert found["divergence_bits"] > 0.0 and 0.0 <= found["auc"] <= 1.0
        assert 0.0 < found["top_choice_ratio"] <= 1.0

    # must reject: with the switch on nothing is trained and the saved run of the first seed is shown
    saved = variant(work_dir, "saved.py", (LAST_LINES, REPORT + LAST_LINES))
    code, _, stdout, stderr = run(saved)
    assert code == 0, stderr[-2000:]
    shown = report(stdout)
    assert shown["trained_here"] is False
    assert shown["seeds"] != trained["seeds"]
    assert shown["steps"] == dict.fromkeys(config.MODELS, config.COMPLEXITY.max_steps)
    assert shown["sizes"] == trained["sizes"] and shown["levels"] == trained["levels"]


def export_text(path: Path, out: Path) -> str:
    """The outputs of the notebook as marimo exports them to HTML, without the code of the cells."""
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            str(path),
            "-o",
            str(out),
            "--no-include-code",
            "--force",
        ],
        cwd=REPO,
        env=environment(),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return out.read_text()


SIX_WARNING = "The saved results of the six grammars are placeholders"
SIX_FIGURE_WARNING = "The saved results are short smoke runs"
COMPLEXITY_WARNING = "The saved results of the complexity experiment are placeholders"
COMPLEXITY_FIGURE_WARNING = "The saved results are pilot runs"


def numbers_in_prose(runs: list[dict]) -> dict[str, str]:
    """The sentences of section 3 that state numbers of the runs shown, computed here from the runs.

    Per level, as means over the runs of the largest training size: the divergence of the uniform
    guess, the number of distinct training strings, the probability the training strings hold and
    the mean lengths. The notebook gives the first and last level of the first and the range of the others.
    """
    largest = max(run["n_train"] for run in runs)
    smallest = min(run["n_train"] for run in runs)
    levels = sorted({run["level"] for run in runs})

    def by_level(value, size: int = largest) -> list[float]:
        return [
            float(
                np.mean(
                    [value(run) for run in runs if (run["level"], run["n_train"]) == (level, size)]
                )
            )
            for level in levels
        ]

    # how a network divides the strings of the smallest training size
    steps = by_level(lambda run: run["data"]["n_gradient"], smallest)
    choice = by_level(lambda run: run["data"]["n_validation"], smallest)
    choice_distinct = by_level(lambda run: run["data"]["n_distinct_validation"], smallest)

    guess = by_level(lambda run: run["references"]["uniform_guess"])
    distinct = by_level(lambda run: run["data"]["n_distinct_training"])
    mass = by_level(lambda run: 100 * run["data"]["training_mass"])
    heldout = by_level(lambda run: run["data"]["mean_length_heldout"])
    generator = by_level(lambda run: run["grammar_values"]["mean_length"])
    return {
        "uniform guess": f"is {guess[0]:.2f} bit per target at the lowest level and {guess[-1]:.2f} at the highest",
        "distinct": f"{min(distinct):.0f} to {max(distinct):.0f} of the {largest} training strings are distinct",
        "probability": f"hold {min(mass):.0f} to {max(mass):.0f} % of the probability of all strings",
        "length": f"has {min(heldout):.1f} to {max(heldout):.1f} symbols on average, "
        f"against {min(generator):.1f} to {max(generator):.1f} for the grammar itself",
        "choice of the weights": f"with {smallest} training strings, {min(steps):.0f} to {max(steps):.0f} "
        f"strings give gradient steps and {min(choice):.0f} to {max(choice):.0f} choose the weights, "
        f"which are {min(choice_distinct):.0f} to {max(choice_distinct):.0f} distinct strings",
        "under the figure": f"{min(choice):.0f} to {max(choice):.0f} strings of the {smallest} in the first column",
    }


def reading_in_prose(runs: list[dict], manifest: dict, drawn: dict) -> dict[str, str]:
    """The sentences beside the last figure that state counts of the runs shown, made here from the run files.

    The band that was not run is found from the settings: the band of the design that they leave
    out. A level mean is the mean over the grammars of a level of their means over the training
    seeds. The trend below the dashed part is the trend table of the runs of those levels alone.
    The counts of grammars in which one model is above another compare the means over the
    training seeds, at the largest number of training strings.
    """
    levels = sorted({run["level"] for run in runs})
    sizes = sorted({run["n_train"] for run in runs})
    grammars = sorted({(run["level"], run["grammar"]["seed"]) for run in runs})

    def value(level, seed, model, size, source, measure) -> float:
        return float(
            np.mean(
                [
                    run[source][measure]
                    for run in runs
                    if (run["level"], run["grammar"]["seed"], run["model"], run["n_train"])
                    == (level, seed, model, size)
                ]
            )
        )

    bands = [level["band_bits"] for level in manifest["settings"]["levels"]]
    left_out = [
        list(level.band_bits)
        for level in config.COMPLEXITY_DESIGN.levels
        if list(level.band_bits) not in bands
    ]
    (band,) = left_out
    lower = max(level for level in levels if bands[level][1] <= band[0])
    upper = min(level for level in levels if bands[level][0] >= band[1])
    entropy = {
        (spec.level, spec.seed): values["topological_entropy_bits"]
        for spec, values in complexity_runs.admitted_grammars(drawn)
    }
    highest = max(v for (level, _), v in entropy.items() if level == lower)
    lowest = min(v for (level, _), v in entropy.items() if level == upper)
    larger = combinations = 0
    for model in config.MODELS:
        for measure in ("score", "auc"):
            for size in sizes:
                mean = {
                    level: np.mean(
                        [
                            value(of_level, seed, model, size, "measures", measure)
                            for of_level, seed in grammars
                            if of_level == level
                        ]
                    )
                    for level in levels
                }
                larger += bool(mean[lower] - mean[upper] > mean[levels[0]] - mean[lower])
                combinations += 1
    table = complexity_runs.trend_table(complexity_runs.figure_data(runs, manifest, drawn))
    tested = table[table["expected"] == "falls"]
    p = tested["p (one-sided, for a fall)"]
    divided = int(((tested["verdict"] == "supported") & (p <= 0.05 / len(tested))).sum())
    fewer = complexity_runs.trend_table(
        complexity_runs.figure_data(
            [run for run in runs if run["level"] <= lower],
            manifest,
            {"levels": drawn["levels"][: lower + 1]},
        )
    )
    fewer = fewer[fewer["expected"] == "falls"]
    below = int((fewer["verdict"] == "supported").sum())
    n_below = sum(level <= lower for level, _ in grammars)
    size, total = sizes[-1], len(grammars)

    def above(first: str, second: str | None, measure: str) -> int:
        count = 0
        for level, seed in grammars:
            a = value(level, seed, first, size, "measures", measure)
            b = (
                value(level, seed, config.MODELS[0], size, "first_order", measure)
                if second is None
                else value(level, seed, second, size, "measures", measure)
            )
            count += a > b
        return count

    rnn, transformer = config.MODELS
    return {
        "dashed": f"The dashed part of a line joins two levels across a band of the design that was not run, {band[0]:.2f} to {band[1]:.2f} bits per symbol. "
        f"The grammars below it reach {highest:.2f} bits per symbol and those above it start at {lowest:.2f}; no grammar lies in between.",
        "not corrected": f"The {len(tested)} tests are not corrected for their number: {divided} of them keep a p-value of at most 0.05 divided by {len(tested)}. "
        f"Each test pools the {total} grammars of the {len(levels)} levels.",
        "below the dashed part": f"On the {n_below} grammars below the dashed part alone, {below} of the {combinations} combinations have a negative correlation with a p-value of at most 0.05.",
        "fall across": f"In {larger} of the {combinations} combinations the mean of a level falls more across the dashed part, from level {lower + 1} to level {upper + 1}, than from level {levels[0] + 1} to level {lower + 1}.",
        "largest size": f"With {size} training strings the tanh RNN is above the first-order model in {above(rnn, None, 'score')} of {total} grammars for the prediction score "
        f"and in {above(rnn, None, 'auc')} of {total} for the area under the curve; the linear transformer in {above(transformer, None, 'score')} of {total} and in {above(transformer, None, 'auc')} of {total}. "
        f"The tanh RNN is above the linear transformer in {above(rnn, transformer, 'score')} of {total} grammars for the prediction score and in {above(rnn, transformer, 'auc')} of {total} for the area under the curve.",
    }


def tables_of_the_export(page: str) -> list[list[dict]]:
    """The rows of every table that a static HTML export shows, in the order of the cells.

    marimo writes the session into the page as JSON; a table is an element that holds its rows as
    data. Each table is returned as a list of rows, a row as a dictionary from column to value.
    """
    start = page.index('"session":') + len('"session":')
    session, _ = json.JSONDecoder().raw_decode(page[start:].lstrip())
    tables = []
    for cell in session["cells"]:
        for output in cell.get("outputs", []):
            for content in output.get("data", {}).values():
                if not isinstance(content, str):
                    continue
                for attributes in re.findall(r"<marimo-table\b([^>]*)>", content):
                    held = re.search(r"data-data='([^']*)'", attributes)
                    rows = json.loads(html.unescape(held.group(1)))
                    tables.append(json.loads(rows) if isinstance(rows, str) else rows)
    return tables


def shows_with_decimals(tables: list[list[dict]], frame, decimals: int) -> bool:
    """Is one of the tables this frame, with every number rounded to that many decimals?

    A number too large for the page's own number type is written there as text, so every value of
    a column of numbers is read as a number before it is compared.
    """
    wanted = frame.round(decimals).to_dict("records")
    for rows in tables:
        if len(rows) != len(wanted) or set(rows[0]) != set(wanted[0]):
            continue
        if all(
            math.isclose(float(row[column]), value, rel_tol=1e-12, abs_tol=1e-12)
            if isinstance(value, float)
            else row[column] == value
            for row, expected in zip(rows, wanted, strict=True)
            for column, value in expected.items()
        ):
            return True
    return False


def test_placeholder_results_are_named_as_such(work_dir):
    _, manifest = results.load_runs(results.default_folder())
    shown, complexity_manifest, shown_drawn = complexity_runs.load_runs(
        complexity_runs.default_folder()
    )
    text = export_text(NOTEBOOK, work_dir / "c.html")
    assert "C. Networks" in text and "Grammars of controlled complexity" in text
    # section 3 states numbers of the runs it shows: each sentence is found with the values of the runs
    stated = numbers_in_prose(shown)
    for name, sentence in stated.items():
        assert sentence in text, name
    for words in ("prediction score", "uniform guess", "exact predictor"):
        assert words in text, words
    # the table of the grammars is shown with three decimals
    shown_tables = tables_of_the_export(text)
    assert len(shown_tables) == 10
    grammar_table = complexity_runs.grammar_table(shown_drawn)
    assert shows_with_decimals(shown_tables, grammar_table, 3)
    # must reject: the same table with one decimal, with two and with four
    for decimals in (1, 2, 4):
        assert not shows_with_decimals(shown_tables, grammar_table, decimals), decimals
    # beside the last figure: what the dashed part of a line is, what the tests rest on, and the
    # largest number of training strings. Each sentence is found with the counts of the runs shown.
    reading_of_the_default = None
    if not complexity_manifest["placeholder"]:
        reading_of_the_default = reading_in_prose(shown, complexity_manifest, shown_drawn)
        for name, sentence in reading_of_the_default.items():
            assert sentence in text, name
    # with placeholder results the note says that a level is its band and not its number
    band_not_number = (
        "a level is identified by its band of topological entropy and not by its number"
    )
    levels_named = (
        f"The pilot has {len(complexity_manifest['settings']['levels'])} levels and the "
        f"production runs have {len(config.COMPLEXITY.levels)}"
    )
    assert (band_not_number in text) == complexity_manifest["placeholder"]
    assert (levels_named in text) == complexity_manifest["placeholder"]
    # the runs of the six grammars exist, so their section carries no warning
    assert not manifest["placeholder"]
    assert SIX_WARNING not in text and SIX_FIGURE_WARNING not in text
    assert "Saved results: 66 runs" in text
    # the complexity experiment: pilot results are named as placeholders, production results are not
    assert (COMPLEXITY_WARNING in text) == complexity_manifest["placeholder"]
    assert (COMPLEXITY_FIGURE_WARNING in text) == complexity_manifest["placeholder"]
    # the notebook is also exported on the smoke results of the six grammars: both warnings are shown
    on_smoke = '    saved_runs, manifest = results.load_runs(results.FOLDERS["smoke"])\n'
    smoke = variant(work_dir, "smoke.py", (SIX_LOADED, on_smoke))
    text = export_text(smoke, work_dir / "smoke.html")
    assert SIX_WARNING in text and SIX_FIGURE_WARNING in text
    # ... and on the pilot results of the complexity experiment, whatever the default folder is
    # The pilot is read from a copy in which three stored numbers of every run are changed, so
    # that the sentences of the text must change with them.
    pilot_runs, _, _ = complexity_runs.load_runs(complexity_runs.FOLDERS["pilot"])
    changed_folder = work_dir / "changed_pilot"
    shutil.copytree(complexity_runs.FOLDERS["pilot"], changed_folder)
    for path in (changed_folder / "runs").glob("*.json"):
        run = json.loads(path.read_text())
        run["references"]["uniform_guess"] += 0.5
        run["data"]["n_distinct_training"] += 7
        run["data"]["mean_length_heldout"] += 3.0
        run["data"]["n_validation"] += 4
        run["data"]["n_gradient"] -= 4
        run["data"]["n_distinct_validation"] += 2
        path.write_text(json.dumps(run, indent=1))
    complexity_runs.collect(changed_folder, "pilot", config.COMPLEXITY_PILOT)
    changed_runs, _, _ = complexity_runs.load_runs(changed_folder)
    loaded = "    complexity_saved, complexity_manifest, grammars = complexity_runs.load_runs(complexity_runs.default_folder())\n"
    on_pilot = f"    complexity_saved, complexity_manifest, grammars = complexity_runs.load_runs({str(changed_folder)!r})\n"
    pilot = variant(work_dir, "pilot.py", (loaded, on_pilot))
    text = export_text(pilot, work_dir / "pilot.html")
    assert COMPLEXITY_WARNING in text and COMPLEXITY_FIGURE_WARNING in text
    # the note on the pilot says how its levels are numbered, whatever the default folder shows
    pilot_manifest = json.loads((changed_folder / "manifest.json").read_text())
    assert pilot_manifest["placeholder"] is True and band_not_number in text
    assert (
        f"The pilot has {len(pilot_manifest['settings']['levels'])} levels and the "
        f"production runs have {len(config.COMPLEXITY.levels)}"
    ) in text
    assert "Saved results of the complexity experiment:" not in text
    # the pilot has every level of the design, so no line has a dashed part; and nothing is read
    # from placeholders
    assert "The dashed part of a line" not in text and "lines join the means of the levels" in text
    assert "tests are not corrected for their number" not in text
    assert "training strings the tanh RNN is above" not in text
    # must reject: numbers typed into the text. On the changed runs the text gives the changed numbers
    before, after = numbers_in_prose(pilot_runs), numbers_in_prose(changed_runs)
    for name in (
        "uniform guess",
        "distinct",
        "length",
        "choice of the weights",
        "under the figure",
    ):
        assert after[name] in text and before[name] not in text and after[name] != before[name], (
            name
        )
    assert after["probability"] in text  # not changed in the copy, so it is the same sentence
    # must reject: when the results are not placeholders, no warning is shown. The production runs
    # are read from a copy in which the grammars beside the band that was not run are moved a
    # little, the first-order model is made perfect, and the linear transformer is put at the
    # uniform guess on every grammar of the level below the band: every count of the reading changes.
    production_runs, production_manifest, production_drawn = complexity_runs.load_runs(
        complexity_runs.FOLDERS["production"]
    )
    changed_folder = work_dir / "changed_production"
    shutil.copytree(complexity_runs.FOLDERS["production"], changed_folder)
    moved = json.loads((changed_folder / "grammars.json").read_text())
    for level, shift in ((2, 0.02), (3, -0.02)):
        for admitted in moved["levels"][level]["admitted"]:
            admitted["values"]["topological_entropy_bits"] += shift
    (changed_folder / "grammars.json").write_text(json.dumps(moved, indent=1))
    for path in (changed_folder / "runs").glob("*.json"):
        run = json.loads(path.read_text())
        run["first_order"].update(divergence_bits=0.0, score=1.0, auc=1.0)
        if run["level"] == 2 and run["model"] == "linear transformer":
            guess = run["references"]["uniform_guess"]
            run["measures"].update(divergence_bits=guess, score=0.0, auc=0.5)
        path.write_text(json.dumps(run, indent=1))
    assert complexity_runs.collect(changed_folder, "production", config.COMPLEXITY)["complete"]
    changed_runs, changed_manifest, changed_drawn = complexity_runs.load_runs(changed_folder)
    on_changed = f"    complexity_saved, complexity_manifest, grammars = complexity_runs.load_runs({str(changed_folder)!r})\n"
    genuine = variant(
        work_dir,
        "genuine.py",
        (SIX_PLACEHOLDER, "    placeholder = False\n"),
        (COMPLEXITY_PLACEHOLDER, "    complexity_placeholder = False\n"),
        (loaded, on_changed),
    )
    text = export_text(genuine, work_dir / "genuine.html")
    for warning in (SIX_WARNING, SIX_FIGURE_WARNING, COMPLEXITY_WARNING, COMPLEXITY_FIGURE_WARNING):
        assert warning not in text
    # must reject: the sentence on the numbering of the levels in a note on results that are no placeholders
    assert band_not_number not in text
    assert "Saved results: 66 runs" in text
    assert "Saved results of the complexity experiment:" in text
    # must reject: counts typed into the text beside the figure. On the changed runs every sentence
    # of the reading gives the changed counts, and none gives those of the production runs
    # (the production runs are the default folder whenever the default is no placeholder)
    before = reading_of_the_default or reading_in_prose(
        production_runs, production_manifest, production_drawn
    )
    after = reading_in_prose(changed_runs, changed_manifest, changed_drawn)
    assert list(after) == [
        "dashed",
        "not corrected",
        "below the dashed part",
        "fall across",
        "largest size",
    ]
    for name in after:
        assert after[name] in text and before[name] not in text and after[name] != before[name], (
            name
        )
    assert "Reading the figure." in text


def settings_in_prose(source: str) -> dict:
    """The hyperparameters and seeds as the text of the notebook states them."""
    sentence = re.search(
        r"Adam with learning rate ([\d.]+), batches of (\d+) samples, (\d+) epochs, "
        r"(\d+) training samples of (\d+) symbols, (\d+) held-out samples",
        source,
    )
    plotted = re.search(r"The figure uses seeds (\d+), (\d+) and (\d+)", source)
    trial = re.search(r"trial runs used seeds (\d+) to (\d+)", source)
    return {
        "learning_rate": float(sentence.group(1)),
        "batch_size": int(sentence.group(2)),
        "epochs": int(sentence.group(3)),
        "n_train": int(sentence.group(4)),
        "sample_length": int(sentence.group(5)),
        "n_heldout": int(sentence.group(6)),
        "plotted": tuple(int(x) for x in plotted.groups()),
        "trial": (int(trial.group(1)), int(trial.group(2))),
    }


def settings_in_config(settings: config.Hyperparameters) -> dict:
    return {
        "learning_rate": settings.learning_rate,
        "batch_size": settings.batch_size,
        "epochs": settings.epochs,
        "n_train": settings.n_train,
        "sample_length": settings.sample_length,
        "n_heldout": settings.n_heldout,
        "plotted": config.PLOTTED_SEEDS,
        "trial": (config.EXPLORATION_SEEDS[0], config.EXPLORATION_SEEDS[-1]),
    }


def complexity_in_prose(source: str) -> dict:
    """The size of the grammars and the admission threshold as the text of the notebook states them."""
    size = re.search(r"has (\d+) states and (\d+) symbols", source)
    gap = re.search(r"at least ([\d.]+) bit per target worse than the exact predictor", source)
    return {
        "n_states": int(size.group(1)),
        "alphabet_size": int(size.group(2)),
        "min_memory_gap_bits": float(gap.group(1)),
    }


def test_notebook_prose_states_the_settings_of_the_runs():
    source = NOTEBOOK.read_text()
    assert settings_in_prose(source) == settings_in_config(config.PRODUCTION)
    # must reject: the text against settings that differ in one number
    assert settings_in_prose(source) != settings_in_config(replace(config.PRODUCTION, epochs=31))
    assert "about 20 000 trainable parameters" in source.replace("close to", "about")
    assert config.PRODUCTION.parameter_budget == 20_000
    # the complexity experiment: the numbers written in the text are those of the settings, and every
    # other number of its text is read from the settings when the notebook runs
    stated = complexity_in_prose(source)
    assert stated == {key: getattr(config.COMPLEXITY, key) for key in stated}
    assert stated != {key: getattr(replace(config.COMPLEXITY, n_states=9), key) for key in stated}
    for name in ("sizes", "validation_fraction", "n_heldout", "max_steps", "training_sample"):
        assert f"live_settings.{name}" in source, name
    # the runs train on draws, and the text of the other sample is not the one that is shown
    assert config.COMPLEXITY.training_sample == "draws"
    assert "a string that is drawn several times is used several times" in source


def prose_texts(source: str) -> list[str]:
    """The text of every ``mo.md(...)`` call; a value computed in the notebook is written as X."""
    texts = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and ast.unparse(node.func) == "mo.md" and node.args):
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Constant):
            texts.append(argument.value)
        elif isinstance(argument, ast.JoinedStr):
            texts.append(
                "".join(
                    part.value if isinstance(part, ast.Constant) else "X"
                    for part in argument.values
                )
            )
    return texts


def conventions_broken(source: str) -> list[str]:
    """Every break of the conventions of the notebooks: hidden code, one line per paragraph.

    Which cell only shows text is decided by ``tests/test_conventions.py``: it returns nothing and
    what it displays is built by ``mo.md`` or ``mo.callout``, whether or not it computes numbers first.
    """
    found = []
    for only_text, hidden, _ in convention_cells(source):
        if only_text and not hidden:
            found.append("a cell that only shows text shows its code")
        if hidden and not only_text:
            found.append("a cell that does not only show text hides its code")
    found += [
        "a paragraph is broken over two lines"
        for text in prose_texts(source)
        if broken_paragraphs(text)
    ]
    return found


# what every table of the notebook shows, and the number of decimals it is rounded to
DECIMALS_OF_THE_TABLES = {
    "results.distance_table(distance)": 4,
    "complexity_runs.admission_table(grammars)": None,
    "complexity_runs.grammar_table(grammars)": 3,
    "complexity_runs.history_table(_named)": 3,
    "complexity_runs.measures_table(_named)": 3,
    "trends": 4,
    "sizes": 4,
    "complexity_runs.mean_table(complexity_data)": 3,
    "all_trends[all_trends['expected'] == 'none']": 4,
    "complexity_runs.job_table(complexity_saved)": 3,
}


def decimals_of_the_tables(source: str) -> dict:
    """For every ``views.table(...)`` of a notebook: what is shown, and the decimals given (None if none)."""
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "views.table":
            decimals = node.args[1].value if len(node.args) > 1 else None
            found[ast.unparse(node.args[0])] = decimals
    return found


def test_notebook_structure():
    """The switch is on by default, the install check comes first, and the notebook ends in its second figure."""
    source = NOTEBOOK.read_text()
    assert source.count(SWITCH_ON) == 1 and SWITCH_OFF not in source
    steps = [
        "# C. Networks",
        "How this notebook works",
        SWITCH_ON,
        "environment_ok",
        "## 1. The measure",
        "## 2. Six grammars",
        "results.load_runs",
        "plotting.figure_networks",
        "## 3. Grammars of controlled complexity",
        "complexity_runs.load_runs",
        "plotting.draw_grammar",
        "complexity_runs.train_run",
        "plotting.figure_complexity",
    ]
    positions = [source.index(step) for step in steps]
    assert positions == sorted(positions)
    # must reject: the same steps in the reverse order
    assert [source.index(step) for step in steps[::-1]] != sorted(positions)
    # two figures, in this order, and the notebook ends in the second
    figures = re.findall(r"plotting\.(figure_\w+)\(", source)
    assert figures == ["figure_networks", "figure_complexity"]
    assert figures[::-1] != ["figure_networks", "figure_complexity"]  # must reject: the other order
    last_cell = source.rsplit("@app.cell", 1)[1]
    assert "figure_complexity" in last_cell and "figure_networks" not in last_cell
    assert (
        "plt." not in source and source.count("plotting.draw_grammar(") == 1
    )  # one drawing of a grammar
    assert "mo.status.progress_bar" in source
    assert source.count("config.COMPLEXITY_LIVE_SEED") >= 2 and "PLOTTED_SEEDS[0]" not in source
    assert "COMPLEXITY.seeds" not in source  # the live run never uses a training seed of the figure
    # the measure is explained with the two numbers of the example, computed and not typed
    assert "non-adjacent" in source and "nested" in source and "ceiling_with_eos" in source
    assert "Kullback-Leibler" in source and "nothing about which grammar is harder" in source
    # section 1 does not say that the divergence in bits means the same on every grammar:
    # section 3 says that it cannot be compared across grammars as it is
    assert "and zero means the same on every grammar: the network knows what can be known" in source
    assert "it means the same on every grammar" not in source
    # section 3 defines the prediction score with one equation and names its two anchors
    assert source.count(r"$$S(q) = 1 - \frac{{D(q)}}{{D_0}}$$") == 1
    for words in (
        "uniform guess",
        "exact predictor",
        "prediction score",
        "less probable and longer",
    ):
        assert words in source, words
    assert "complexity_runs.job_table(complexity_saved)" in source  # its numbers come from the runs
    assert "divergence rises" not in source  # the expectation of the first design is gone
    # section 1 says which targets its two numbers count, and types no new number
    assert "Both numbers count the targets of the runs of section 2" in source
    assert "the first symbol of a string is not a target there" in source
    assert "0.60" not in source and "0.83" not in source and "0.67" not in source
    # what the first-order model uses of the training strings, where the data are described and
    # under the figure, with numbers that are read from the runs
    assert "Every model gets the same $N$ strings and uses them as it needs" in source
    assert "The grey line is not a network without memory under the same conditions" in source
    assert source.count("complexity_runs.job_table(complexity_saved)") >= 2
    # no raw accuracy and no raw cross-entropy is tabulated or drawn across grammars any more
    assert "results.mean_table" not in source and "results.figure_data" not in source
    lowered = source.lower()
    assert "maternal" not in lowered and "social" not in lowered
    assert "maternal" in (lowered + " maternal")  # must reject: a text that names a private dataset
    # conventions: a cell that only shows text hides its code, and a paragraph is one line
    cells = convention_cells(source)
    assert sum(only_text for only_text, _, _ in cells) >= 21 and conventions_broken(source) == []
    assert len(prose_texts(source)) >= 20  # the text of the computed cells is checked as well
    assert "—" not in source and "–" not in source  # no dash used as punctuation
    # must reject: a text cell without hide_code, a code cell with it, a broken paragraph
    assert conventions_broken(
        source.replace(
            '@app.cell(hide_code=True)\ndef _(mo):\n    mo.md(r"""\n    # C. Networks',
            '@app.cell\ndef _(mo):\n    mo.md(r"""\n    # C. Networks',
        )
    ) == ["a cell that only shows text shows its code"]
    # section 1 computes its two numbers first and is a text cell all the same: its code is hidden
    section_1 = (
        "@app.cell(hide_code=True)\ndef _(config, ladder, mo):\n    _ceilings = ladder.exact_table("
    )
    assert source.count(section_1) == 1
    assert conventions_broken(
        source.replace(section_1, section_1.replace("(hide_code=True)", ""))
    ) == ["a cell that only shows text shows its code"]
    hidden_code = source.replace(
        "@app.cell\ndef _(mo):\n    load_saved",
        "@app.cell(hide_code=True)\ndef _(mo):\n    load_saved",
    )
    assert conventions_broken(hidden_code) == ["a cell that does not only show text hides its code"]
    # the glue of the visible cells is in tutorial/notebook_views.py; the calls that teach stay
    for call in (
        "complexity_runs.load_runs(",
        "complexity.build_grammar(",
        "complexity.draw_strings(",
        "complexity.BoundedPredictor(",
        "complexity_runs.prepare(",
        "complexity_runs.train_run(",
        "complexity_runs.measures_table(",
        "complexity_runs.trend_table(",
        "plotting.figure_networks(",
        "plotting.figure_complexity(",
    ):
        assert call in source, call
    assert "from tutorial import notebook_views as views" in source and "inspect." not in source
    # the decimals of every table, written here a second time
    assert decimals_of_the_tables(source) == DECIMALS_OF_THE_TABLES
    # must reject: the table of the grammars with one decimal, and a table shown without rounding
    one_decimal = source.replace(
        "views.table(complexity_runs.grammar_table(grammars), 3)",
        "views.table(complexity_runs.grammar_table(grammars), 1)",
    )
    assert decimals_of_the_tables(one_decimal) != DECIMALS_OF_THE_TABLES and one_decimal != source
    unrounded = source.replace("views.table(sizes, 4)", "views.table(sizes)")
    assert decimals_of_the_tables(unrounded) != DECIMALS_OF_THE_TABLES and unrounded != source
    broken = source.replace(
        "The six grammars differ in kind. ", "The six grammars differ in kind.\n    "
    )
    assert conventions_broken(broken) == ["a paragraph is broken over two lines"]


def from_the_run_files(folder: Path) -> dict:
    """The numbers of the six-grammar figure read from the run files directly, without ``results``.

    Per grammar of the ladder and model: for every seed, in the order of the seeds, the
    cross-entropy after every epoch minus the cross-entropy of the exact predictor on the same
    held-out symbols.
    """
    runs = [json.loads(path.read_text()) for path in sorted((folder / "runs").glob("*.json"))]
    figure: dict = {"n_runs": len(runs), "models": {}}
    for model in config.MODELS:
        figure["models"][model] = []
        for grammar in config.LADDER:
            by_seed = {
                run["seed"]: run for run in runs if (run["rung"], run["model"]) == (grammar, model)
            }
            assert sorted(by_seed) == sorted(config.PLOTTED_SEEDS), (grammar, model)
            figure["models"][model].append(
                [
                    [
                        value - by_seed[seed]["optimum"]["cross_entropy_realised"]
                        for value in by_seed[seed]["history"]["cross_entropy"]
                    ]
                    for seed in sorted(by_seed)
                ]
            )
    return figure


def complexity_from_the_run_files(folder: Path) -> dict:
    """The numbers of the complexity figure read from the run files directly, with other code.

    Per model and training size: for every grammar, by level and grammar seed, the mean over the
    training seeds of the prediction score, of the area under the curve and of the divergence. The
    score of a run is made here, as one minus its divergence over the divergence of the uniform
    guess; the stored score is not read. ``first_order`` holds the score and the area of the
    first-order model in the same form.
    """
    runs = [json.loads(path.read_text()) for path in sorted((folder / "runs").glob("*.json"))]
    grammars = sorted({(run["level"], run["grammar"]["seed"]) for run in runs})
    figure: dict = {"n_runs": len(runs), "grammars": grammars, "models": {}, "first_order": {}}
    sizes = sorted({run["n_train"] for run in runs})

    def values_of(found: dict, run: dict) -> dict:
        return {
            "score": 1.0 - found["divergence_bits"] / run["references"]["uniform_guess"],
            "auc": found["auc"],
            "divergence_bits": found["divergence_bits"],
        }

    for model in config.MODELS:
        figure["models"][model] = {}
        for size in sizes:
            entry = {"score": [], "auc": [], "divergence_bits": []}
            first = {"score": [], "auc": []}
            for level, seed in grammars:
                of_grammar = [
                    run
                    for run in runs
                    if (run["level"], run["grammar"]["seed"], run["model"], run["n_train"])
                    == (level, seed, model, size)
                ]
                assert of_grammar, (level, seed, model, size)
                for measure in entry:
                    entry[measure].append(
                        float(
                            np.mean(
                                [values_of(run["measures"], run)[measure] for run in of_grammar]
                            )
                        )
                    )
                for measure in first:
                    first[measure].append(
                        float(
                            np.mean(
                                [values_of(run["first_order"], run)[measure] for run in of_grammar]
                            )
                        )
                    )
            figure["models"][model][str(size)] = entry
            figure["first_order"][str(size)] = first
    return figure


def uniform_guess_from_the_strings(run: dict) -> float:
    """D0 of a run from the strings of its job, drawn again here: log2 K minus the mean entropy.

    The held-out strings are drawn again with the settings and the seed stored in the run; the
    entropy of the exact distribution is summed target by target. Neither the function that
    computes D0 in the package nor the divergence is used.
    """
    settings = run["settings"]
    spec = complexity.GrammarSpec.from_dict(run["grammar"])
    strings = complexity.draw_strings(
        spec,
        run["seed"],
        max(settings["sizes"]),
        settings["n_heldout"],
        settings["min_length"],
        settings["max_length"],
        settings["max_draws"],
        settings["training_sample"],
    )
    predictor = complexity.BoundedPredictor(
        complexity.build_grammar(spec), settings["min_length"], settings["max_length"]
    )
    classes = complexity.classes_of(spec)
    p, _ = complexity.exact_distributions(
        predictor, strings.heldout, classes, settings["max_length"] + 1
    )
    entropies = [
        -sum(x * math.log2(x) for x in p[row, t] if x > 0.0)
        for row, string in enumerate(strings.heldout)
        for t in range(len(string) + 1)
    ]
    assert len(entropies) == run["data"]["n_targets_heldout"]
    return math.log2(len(classes)) - sum(entropies) / len(entropies)


def shown_by_the_export(html: str) -> str:
    """What a static HTML export shows: the outputs of its cells, without the code it also holds.

    marimo writes the session into the page as JSON, one entry per cell with its outputs by MIME
    type. The outputs are returned as one text, in the order of the cells.
    """
    start = html.index('"session":') + len('"session":')
    session, _ = json.JSONDecoder().raw_decode(html[start:].lstrip())
    return "\n".join(
        json.dumps(output.get("data", {}), ensure_ascii=False)
        for cell in session["cells"]
        for output in cell.get("outputs", [])
    )


def test_fallback_of_notebook_c_holds_the_figures_and_their_numbers(work_dir, fallback_script):
    production = results.FOLDERS["production"]
    complexity_folder = complexity_runs.default_folder()
    _, complexity_manifest, _ = complexity_runs.load_runs(complexity_folder)
    placeholder = complexity_manifest["placeholder"]
    fallback_script.fallback_c(work_dir / "fallback", no_html=True, allow_placeholder=placeholder)
    written = sorted(path.name for path in (work_dir / "fallback").iterdir())
    assert written == sorted(
        f"{figure}{ending}"
        for figure in ("figure_networks", "figure_complexity")
        for ending in ("-onwhite.png", ".data.json", ".pdf", ".png", ".svg")
    )
    assert all((work_dir / "fallback" / name).stat().st_size > 1000 for name in written)

    # first figure: every number, recomputed here from the run files
    data = json.loads((work_dir / "fallback" / "figure_networks.data.json").read_text())
    assert data["kind"] == "production" and data["grammars"] == list(config.LADDER)
    assert data["seeds"] == [1, 2, 3] and data["settings"]["epochs"] == config.PRODUCTION.epochs
    assert "exact predictor" in data["quantity"] and "accuracy" not in json.dumps(data)
    recomputed = from_the_run_files(production)
    assert recomputed["n_runs"] == 66
    for model in config.MODELS:
        assert data["models"][model]["distance"] == recomputed["models"][model]
    # the figure names its 36 run files, each with the SHA-256 the file has
    assert len(data["run_files"]) == len(config.LADDER) * len(config.MODELS) * 3
    for name, digest in data["run_files"].items():
        assert hashlib.sha256((production / name).read_bytes()).hexdigest() == digest, name

    # second figure: every number, recomputed here from the run files of the complexity experiment
    complexity = json.loads((work_dir / "fallback" / "figure_complexity.data.json").read_text())
    assert complexity["placeholder"] is placeholder
    assert complexity["kind"] == ("pilot" if placeholder else "production")
    again = complexity_from_the_run_files(complexity_folder)
    assert again["n_runs"] == len(complexity["run_files"]) == complexity_manifest["present"]
    assert [(g["level"], g["seed"]) for g in complexity["grammars"]] == again["grammars"]
    for model in config.MODELS:
        for size, entry in again["models"][model].items():
            for measure, values in entry.items():
                assert np.allclose(
                    complexity["models"][model][size][measure]["grammar"],
                    values,
                    rtol=0.0,
                    atol=1e-12,
                ), (model, size, measure)
            for measure, values in again["first_order"][size].items():
                assert np.allclose(
                    complexity["first_order"][size][measure]["grammar"],
                    values,
                    rtol=0.0,
                    atol=1e-12,
                ), (size, measure)
    for name, digest in complexity["run_files"].items():
        assert hashlib.sha256((complexity_folder / name).read_bytes()).hexdigest() == digest, name
    # twelve cells of the expectation (score and area) and six rows of the divergence without one
    assert len(complexity["trend"]) == 2 * len(complexity["sizes"]) * 3
    assert sorted({row["expected"] for row in complexity["trend"]}) == ["falls", "none"]
    assert set(complexity["quantities"]) == {"score", "auc", "divergence_bits"}
    # The data say which part of a line is dashed and why: the band of the design that the
    # settings of the runs leave out (found here from the settings), between the two levels beside it.
    held = [level["band_bits"] for level in complexity_manifest["settings"]["levels"]]
    left_out = [
        list(level.band_bits)
        for level in config.COMPLEXITY_DESIGN.levels
        if list(level.band_bits) not in held
    ]
    assert len(complexity["segments"]) == len(complexity["levels"]) - 1
    assert [segment["levels"] for segment in complexity["segments"]] == [
        [first["level"], second["level"]]
        for first, second in zip(complexity["levels"], complexity["levels"][1:], strict=False)
    ]
    dashed = [segment for segment in complexity["segments"] if segment["dashed"]]
    assert all(
        segment["bands_not_run"] == []
        for segment in complexity["segments"]
        if not segment["dashed"]
    )
    if placeholder:  # the pilot has every level of the design
        assert left_out == [] and dashed == []
    else:
        assert left_out == [[1.86, 1.98]] and len(dashed) == 1
        assert dashed[0]["bands_not_run"] == left_out and dashed[0]["levels"] == [2, 3]
        below, above = (complexity["levels"][level]["band_bits"] for level in dashed[0]["levels"])
        assert below[1] <= left_out[0][0] and left_out[0][1] <= above[0]
    # The divergence of the uniform guess, from the strings of the job, for the first job of every
    # level: it is the one stored with the run, on which the scores of the figure rest.
    saved_runs, _, _ = complexity_runs.load_runs(complexity_folder)
    for level in sorted({run["level"] for run in saved_runs}):
        run = min(
            (run for run in saved_runs if run["level"] == level),
            key=lambda run: (run["grammar"]["seed"], run["seed"], run["n_train"], run["model"]),
        )
        from_strings = uniform_guess_from_the_strings(run)
        assert abs(from_strings - run["references"]["uniform_guess"]) < 1e-9, level
        assert (
            abs(
                run["measures"]["score"] - (1.0 - run["measures"]["divergence_bits"] / from_strings)
            )
            < 1e-9
        )
        # must reject: the entropy of the generator in the place of the entropy on the held-out
        # targets. The held-out strings are not a sample of the generator, and the score differs.
        generator = math.log2(run["n_classes"]) - run["grammar_values"]["entropy"]
        assert abs(generator - from_strings) > 0.01, level
        wrong = 1.0 - run["measures"]["divergence_bits"] / generator
        assert abs(wrong - run["measures"]["score"]) > 1e-3, level

    # the folder of the repository holds the same numbers, and the static HTML export
    assert json.loads((FALLBACK / "figure_networks.data.json").read_text()) == data
    assert json.loads((FALLBACK / "figure_complexity.data.json").read_text()) == complexity
    assert sorted(path.name for path in FALLBACK.iterdir()) == sorted([*written, "c_networks.html"])
    html = (FALLBACK / "c_networks.html").read_text()
    shown = shown_by_the_export(html)
    assert len(html) > 100_000 and "Saved results: 66 runs" in shown  # exported on the saved runs
    assert "Grammars of controlled complexity" in shown
    # The export also holds the code of the cells, and with it the sentence of the warning
    # whatever the results are. What counts is what the outputs show.
    assert COMPLEXITY_WARNING in html
    assert (COMPLEXITY_WARNING in shown) == placeholder
    assert (COMPLEXITY_FIGURE_WARNING in shown) == placeholder
    assert ("Saved results of the complexity experiment:" in shown) != placeholder
    # must reject: an export whose outputs show the warning, and one that shows it only in its code
    in_an_output = json.dumps(
        {"session": {"cells": [{"outputs": [{"data": {"text/html": COMPLEXITY_WARNING}}]}]}}
    )
    assert COMPLEXITY_WARNING in shown_by_the_export(in_an_output)
    in_the_code = json.dumps(
        {"session": {"cells": [{"code": COMPLEXITY_WARNING, "outputs": [{"data": {}}]}]}}
    )
    assert COMPLEXITY_WARNING not in shown_by_the_export(in_the_code)

    # must reject: figure data with one number changed
    changed = copy.deepcopy(data)
    changed["models"]["rnn"]["distance"][2][1][30] += 1e-6
    assert changed["models"]["rnn"]["distance"] != recomputed["models"]["rnn"]
    for measure in ("score", "auc"):
        changed = copy.deepcopy(complexity)
        changed["models"]["rnn"]["200"][measure]["grammar"][0] += 1e-6
        assert not np.allclose(
            changed["models"]["rnn"]["200"][measure]["grammar"],
            again["models"]["rnn"]["200"][measure],
            rtol=0.0,
            atol=1e-12,
        )
    # must reject: the smoke results of the six grammars, which are placeholders; nothing is written
    with pytest.raises(SystemExit, match="placeholders"):
        fallback_script.fallback_c(
            work_dir / "smoke",
            no_html=True,
            folder=results.FOLDERS["smoke"],
            allow_placeholder=True,
        )
    assert not (work_dir / "smoke").exists()
    # must reject: pilot results of the complexity experiment, unless they are asked for by name
    with pytest.raises(SystemExit, match="pilot runs"):
        fallback_script.fallback_c(
            work_dir / "pilot", no_html=True, complexity_folder=complexity_runs.FOLDERS["pilot"]
        )
    assert not (work_dir / "pilot").exists()
    # must reject: a folder of production runs that is not complete
    partial = work_dir / "partial"
    shutil.copytree(production, partial)
    manifest = json.loads((partial / "manifest.json").read_text())
    absent = next(iter(manifest["files"]))
    (partial / absent).unlink()
    del manifest["files"][absent]
    manifest.update(complete=False, missing=[Path(absent).stem], present=65)
    (partial / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="incomplete"):
        fallback_script.fallback_c(
            work_dir / "from_partial", no_html=True, folder=partial, allow_placeholder=True
        )


def test_no_production_run_of_the_complexity_experiment_exists_before_it_is_released():
    """The stop point of the stage: the production folder is empty until the code has been verified.

    Once the production runs exist this test asserts the opposite: the notebook and the fallback
    then show them, and the shipped fallback is no longer a placeholder.
    """
    production = complexity_runs.FOLDERS["production"]
    if not production.exists():
        assert complexity_runs.default_folder() == complexity_runs.FOLDERS["pilot"]
        assert (
            json.loads((FALLBACK / "figure_complexity.data.json").read_text())["placeholder"]
            is True
        )
        return
    _, manifest, drawn = complexity_runs.load_runs(production)
    assert manifest["kind"] == "production" and manifest["complete"]
    assert all(spec.seed < 101 for spec, _ in complexity_runs.admitted_grammars(drawn))
    # must reject: a shipped fallback that still shows the pilot once the production runs exist
    assert (
        json.loads((FALLBACK / "figure_complexity.data.json").read_text())["placeholder"] is False
    )
