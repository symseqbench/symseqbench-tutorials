"""Notebook A runs headless from start to end within a stated time budget. Each test has a case it must reject."""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
import subprocess
import sys
import time
import timeit
from pathlib import Path

import pandas as pd
import pytest
from test_behaviour_measures import same_numbers

from tutorial import closed_forms as cf
from tutorial import worked_example
from tutorial.generate import two_state_grammar

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "a_generate_and_measure.py"
FALLBACK = REPO / "fallback" / "a_generate_and_measure"
FIGURE_SUFFIXES = ("-onwhite.png", ".data.json", ".pdf", ".png", ".svg")
PANELS = ("worked_example_words", "worked_example_block_entropy")

# Stated budget for one headless run, in seconds of wall time. Measured on the author's workstation:
# 6 s at full speed, 37 to 45 s when its processor is throttled to about a tenth of its speed (CPU
# frequency scaling), which happens often there. The budget leaves a factor of four above that.
BUDGET_S = 180.0


def run(path: Path) -> tuple[int, float, str]:
    """Run a marimo notebook as a script, with two threads at most. Returns exit code, seconds, stderr."""
    environment = dict(
        os.environ,
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    start = time.perf_counter()
    done = subprocess.run(
        [sys.executable, str(path)], cwd=REPO, env=environment, capture_output=True, text=True
    )
    return done.returncode, time.perf_counter() - start, done.stderr


def within_budget(seconds: float, budget: float = BUDGET_S) -> bool:
    return seconds <= budget


def test_notebook_a_runs_headless_within_the_budget(capsys):
    probe = timeit.timeit(lambda: sum(range(1000)), number=2000) / 2000 * 1e6
    code, seconds, stderr = run(NOTEBOOK)
    with capsys.disabled():
        print(
            f"\n[notebook A] {seconds:.1f} s of {BUDGET_S:.0f} s; speed probe {probe:.0f} us (about 10 us on an idle core)"
        )
    assert code == 0, stderr[-2000:]
    assert within_budget(seconds)
    # must reject: the same run against a budget it cannot meet
    assert not within_budget(seconds, budget=0.5)


def test_a_failing_cell_makes_the_headless_run_fail(work_dir):
    """must reject: the same notebook with one cell that raises exits with an error."""
    source = NOTEBOOK.read_text()
    marker = "    comparison = worked_example.comparison(p, sequence)\n"
    assert source.count(marker) == 1
    broken = work_dir / "broken_notebook.py"
    broken.write_text(
        source.replace(marker, marker + '    raise RuntimeError("injected failure")\n')
    )
    code, _, stderr = run(broken)
    assert code != 0
    assert "injected failure" in stderr


def test_a_failed_install_check_stops_the_notebook(work_dir):
    """must reject: with wrong pins (commit and sources) the install check raises and nothing after it runs."""
    source = NOTEBOOK.read_text()
    marker = "    install_rows = env.environment_report()\n"
    assert source.count(marker) == 1
    wrong = work_dir / "wrong_pin_notebook.py"
    wrong.write_text(
        source.replace(
            marker,
            '    env.PINNED_COMMITS["symseq"] = "0" * 40\n    env.PINNED_FINGERPRINTS["symseq"] = "0" * 64\n'
            + marker,
        )
    )
    code, _, stderr = run(wrong)
    assert code != 0
    assert "The install check failed for: symseq" in stderr


def test_notebook_prose_and_controls():
    """The notebook has the install check first, the four controls of the ladder, and ends in the figure."""
    source = NOTEBOOK.read_text()
    steps = ["environment_report", "worked_example", "ladder_table"]
    assert in_order(source, steps)
    for control in ("alphabet_slider", "states_slider", "contexts_slider", "depth_slider"):
        assert f"{control} = mo.ui.slider" in source
    last_cell = source.rsplit("@app.cell", 1)[1]
    assert "figure_generate_and_measure" in last_cell
    assert not names_private_data(source)
    # must reject: the steps in another order, and a text that names a private dataset
    assert not in_order(source, steps[::-1])
    assert names_private_data(source + "\n# maternal behaviour tables")


def test_worked_example_controls_bound_the_work():
    """At every setting of the sliders the worked example handles at most about 60 000 symbols.

    Measured at 60 000 symbols on the author's workstation at full speed: 1.0 s to generate the
    strings at p = 0.95, 0.16 s for the table and 0.12 s for the estimates against length.
    """
    source = NOTEBOOK.read_text()
    assert "length_slider = mo.ui.slider(3000, 60000, step=3000, value=18000" in source
    assert "p_slider = mo.ui.slider(0.05, 0.95, step=0.05, value=0.5" in source
    assert "worked_example.strings_for_length(p, length_slider.value)" in source
    assert worked_example.MAX_SYMBOLS == 60_000
    for step in range(1, 20):
        p = 0.05 * step
        for n_symbols in (3000, 18_000, 60_000):
            n_strings = worked_example.strings_for_length(p, n_symbols)
            mean_length = n_strings * (1 + 1 / p)  # a string has 1 + 1 / p symbols on average
            assert mean_length <= 1.01 * n_symbols + (1 + 1 / p)
    # the default control gives the sequence of the saved results
    assert worked_example.strings_for_length(0.5, 18_000) == 6000
    _, sequence = worked_example.joined_sequence(
        0.05, worked_example.strings_for_length(0.05, 60_000), seed=0
    )
    assert 50_000 < len(sequence) < 70_000
    # must reject: the earlier control, 20 000 strings at p = 0.05, is seven times the bound,
    earlier = 20_000 * (1 + 1 / 0.05)
    assert earlier == pytest.approx(7 * worked_example.MAX_SYMBOLS)
    assert "n_strings_slider" not in source
    with pytest.raises(ValueError):
        worked_example.strings_for_length(0.5, 60_001)  # and a longer sequence is refused


def in_order(source: str, names: list[str]) -> bool:
    """True if the first occurrences of the names follow each other in this order."""
    positions = [source.index(name) for name in names]
    return positions == sorted(positions)


def names_private_data(source: str) -> bool:
    return "maternal" in source.lower() or "social" in source.lower()


def test_the_two_panels_of_the_worked_example_come_before_the_ladder():
    """Two intermediate plots in the worked-example section; the notebook still ends in its one final figure."""
    source = NOTEBOOK.read_text()
    words = "plotting.figure_worked_example_words("
    block = "plotting.figure_worked_example_block_entropy("
    final = "plotting.figure_generate_and_measure("
    ladder_section = "## 2. A ladder of grammars"
    assert in_order(source, ["## 1. Worked example", words, block, ladder_section, final])
    assert all(source.count(call) == 1 for call in (words, block, final))
    assert source.count("plotting.figure_") == 3 and "plt." not in source
    last_cell = source.rsplit("@app.cell", 1)[1]
    assert final in last_cell and "figure_worked_example" not in last_cell
    assert "worked_example.block_entropy_panel(p)" in source  # the panel follows the control of p
    # must reject: the two panels after the ladder, and a second final figure
    assert not in_order(source, [ladder_section, words])
    assert not in_order(source, [ladder_section, block])
    assert (source + final).count(final) != 1


def test_fallback_of_notebook_a_holds_its_figures_and_their_numbers(work_dir, fallback_script):
    out = work_dir / "fallback"
    fallback_script.fallback_a(out, no_html=True)
    written = sorted(path.name for path in out.iterdir())
    figures = [
        f"{name}{suffix}"
        for name in (*PANELS, "figure_generate_and_measure")
        for suffix in FIGURE_SUFFIXES
    ]
    tables = ["ladder_exact.csv", "ladder_table.csv", "worked_example.csv"]
    assert written == sorted([*figures, *tables])
    assert all((out / name).stat().st_size > 1000 for name in figures if "data.json" not in name)

    # the numbers of the two panels, each against an independent route
    words = json.loads((out / "worked_example_words.data.json").read_text())
    listed = [
        sum("BB" not in "".join(word) for word in itertools.product("AB", repeat=length))
        for length in words["length"]
    ]
    assert words["words"] == listed and words["length"] == list(range(1, 11))
    assert words["topological_entropy_bits"] == pytest.approx(math.log2(cf.PHI), abs=1e-12)
    assert words["topological_entropy_nats"] == pytest.approx(math.log(cf.PHI), abs=1e-12)
    block = json.loads((out / "worked_example_block_entropy.data.json").read_text())
    _, chain = cf.stream_chain(two_state_grammar(0.5, 1.0), delimiter=False)
    enumerated = [cf.block_entropy_enumerated(chain, length) for length in block["length"]]
    assert block["block_entropy"] == pytest.approx(enumerated, abs=1e-9)
    assert (block["p"], block["length"]) == (0.5, list(range(1, 9)))
    assert block["entropy_rate"] == pytest.approx(2 / 3, abs=1e-9)
    assert block["excess_entropy"] == pytest.approx(math.log2(3) - 4 / 3, abs=1e-9)

    # the folder of the repository holds the same numbers, and the static HTML export
    for name in written:
        if name.endswith(".json"):
            assert same_numbers(
                json.loads((FALLBACK / name).read_text()), json.loads((out / name).read_text())
            ), name
        elif name.endswith(".csv"):
            pd.testing.assert_frame_equal(
                pd.read_csv(FALLBACK / name), pd.read_csv(out / name), check_exact=False, rtol=1e-9
            )
    assert sorted(path.name for path in FALLBACK.iterdir()) == sorted(
        [*written, "a_generate_and_measure.html"]
    )
    html = (FALLBACK / "a_generate_and_measure.html").read_text()
    assert len(html) > 100_000 and "figure_worked_example_block_entropy" in html
    # must reject: data in which one number differs from the listed words, or from the shipped file
    changed = copy.deepcopy(words)
    changed["words"][3] += 1
    assert changed["words"] != listed
    assert not same_numbers(
        changed, json.loads((FALLBACK / "worked_example_words.data.json").read_text())
    )
