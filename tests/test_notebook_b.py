"""Notebook B runs headless within stated time budgets, with saved results and with its live cell. Each test has a case it must reject."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import timeit
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest
from test_behaviour import broken_paragraphs
from test_behaviour_measures import same_numbers

from tutorial import behaviour
from tutorial import behaviour_measures as bm

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "b_behaviour.py"
FALLBACK = REPO / "fallback" / "b_behaviour"

# Stated budgets, in seconds of wall time. The headless run on saved results: 180 s, as for the
# other notebooks. The live cell of the largest dataset: 60 s in the fast state of the build
# workstation (speed probe at most 15 us before and after), in one thread.
BUDGET_S = 180.0
LIVE_BUDGET_S = 60.0
FAST_PROBE_US = 15.0

SWITCH_ON = "load_saved = mo.ui.switch(value=True"
DEFAULT_DATASET = 'value="zebrafish", label="dataset")'


def speed_probe() -> float:
    return timeit.timeit(lambda: sum(range(1000)), number=2000) / 2000 * 1e6


def run(path: Path) -> tuple[int, float, str]:
    """Run a marimo notebook as a script in one thread. Returns exit code, seconds, stderr."""
    environment = dict(
        os.environ,
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    start = time.perf_counter()
    done = subprocess.run(
        [sys.executable, str(path)], cwd=REPO, env=environment, capture_output=True, text=True
    )
    return done.returncode, time.perf_counter() - start, done.stderr


def variant(work_dir: Path, name: str, replacements: dict[str, str]) -> Path:
    """A copy of the notebook with some lines replaced; each line must occur exactly once."""
    source = NOTEBOOK.read_text()
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    path = work_dir / name
    path.write_text(source)
    return path


def within_budget(seconds: float, budget: float = BUDGET_S) -> bool:
    return seconds <= budget


def test_notebook_b_runs_headless_on_saved_results_within_the_budget(capsys):
    probe = speed_probe()
    code, seconds, stderr = run(NOTEBOOK)
    with capsys.disabled():
        print(
            f"\n[notebook B, saved results] {seconds:.1f} s of {BUDGET_S:.0f} s; speed probe {probe:.0f} us"
        )
    assert code == 0, stderr[-2000:]
    assert within_budget(seconds)
    # must reject: the same run against a budget it cannot meet
    assert not within_budget(seconds, budget=0.5)


def test_a_failing_cell_makes_the_headless_run_fail(work_dir):
    """must reject: the same notebook with one cell that raises exits with an error."""
    marker = "        result = saved_results[dataset_choice.value]\n"
    broken = variant(
        work_dir, "broken.py", {marker: marker + '        raise RuntimeError("injected failure")\n'}
    )
    code, _, stderr = run(broken)
    assert code != 0 and "injected failure" in stderr


def test_a_failed_install_check_or_a_changed_table_stops_the_notebook(work_dir):
    """must reject: wrong pins stop the notebook at the install check; so does a manifest that does not match."""
    marker = "    if not env.environment_ok():\n"
    wrong = variant(
        work_dir,
        "wrong_pin.py",
        {
            marker: '    env.PINNED_COMMITS["symseq"] = "0" * 40\n    env.PINNED_FINGERPRINTS["symseq"] = "0" * 64\n'
            + marker
        },
    )
    code, _, stderr = run(wrong)
    assert code != 0 and "The install check failed" in stderr
    marker = "    behaviour.check_folder()"
    other_folder = variant(
        work_dir, "other_folder.py", {marker: f'    behaviour.check_folder(r"{work_dir}")'}
    )
    (work_dir / "MANIFEST.sha256").write_text("")
    code, _, stderr = run(other_folder)
    assert code != 0 and "ManifestError" in stderr


def test_live_branch_runs_and_gives_the_saved_numbers(work_dir, capsys):
    """The notebook with the switch off, on the smallest dataset, with a pasted sequence of the participant."""
    pasted = "\\n".join(" ".join("ABCAB"[(i + j) % 5] for j in range(40)) for i in range(4))
    # In this copy the saved result of the seals is replaced by that of the turtles, so that a
    # figure drawn from the saved result in the place of the live one shows other numbers.
    checked = "    bm.check_against_tables(saved_results)  # each saved result was computed from the shipped table\n"
    live = variant(
        work_dir,
        "live.py",
        {
            SWITCH_ON: "load_saved = mo.ui.switch(value=False",
            DEFAULT_DATASET: 'value="seal", label="dataset")',
            checked: checked
            + '    saved_results = {**saved_results, "seal": saved_results["turtle"]}\n',
            'own_text = mo.ui.text_area(placeholder="A B C A B C A C ...",': f'own_text = mo.ui.text_area(value="{pasted}",',
            "    views.full_width(figure)\n    return figure, figure_data\n": "    views.full_width(figure)\n"
            f'    import json, pathlib\n    pathlib.Path(r"{work_dir / "live.json"}").write_text(json.dumps(figure_data))\n'
            "    return figure, figure_data\n",
        },
    )
    code, seconds, stderr = run(live)
    with capsys.disabled():
        print(
            f"\n[notebook B, live cell on the seals, own sequence] {seconds:.1f} s; speed probe {speed_probe():.0f} us"
        )
    assert code == 0, stderr[-2000:]
    shown = json.loads(live.with_suffix(".json").read_text())
    saved, _ = bm.load_results()
    assert shown["datasets"] == [*behaviour.DATASETS, "own"] and shown["highlight"] == [
        "seal",
        "own",
    ]
    assert same_numbers(
        shown["entropy_rate"]["seal"], bm.figure_data(saved)["entropy_rate"]["seal"]
    )
    assert shown["labels"]["own"] == "your sequence" and shown["n_tokens"]["own"] == 160
    # must reject: the numbers of another dataset (which a figure drawn from the saved result would
    # show in this copy), and a pasted sequence that is too short
    assert not same_numbers(
        shown["entropy_rate"]["seal"], bm.figure_data(saved)["entropy_rate"]["turtle"]
    )
    assert same_numbers(
        shown["entropy_rate"]["turtle"], bm.figure_data(saved)["entropy_rate"]["turtle"]
    )
    short = variant(
        work_dir,
        "short.py",
        {
            'own_text = mo.ui.text_area(placeholder="A B C A B C A C ...",': 'own_text = mo.ui.text_area(value="A B A B",'
        },
    )
    code, _, stderr = run(short)
    assert code == 0, stderr[-2000:]  # the notebook goes on and says why the sequence is refused


def test_live_cell_of_the_largest_dataset_meets_its_budget_in_the_fast_state(capsys):
    """Criterion: at most 60 s in the fast state, one thread. In the slow state the test reports, it does not pass."""
    saved, _ = bm.load_results()
    largest = max(saved, key=lambda name: saved[name]["n_tokens"])
    strings = behaviour.load_strings(largest).strings
    before, start = speed_probe(), time.perf_counter()
    live = bm.plain(bm.measure_dataset(strings))  # the call of the notebook's live cell
    seconds, after = time.perf_counter() - start, speed_probe()
    with capsys.disabled():
        print(
            f"\n[notebook B, live cell, {largest}, {saved[largest]['n_tokens']} symbols] {seconds:.1f} s of "
            f"{LIVE_BUDGET_S:.0f} s; speed probe {before:.0f} us before, {after:.0f} us after"
        )
    reference = {
        key: value
        for key, value in saved[largest].items()
        if key not in ("dataset", "table_sha256")
    }
    assert same_numbers(live, reference)  # the live cell gives the saved numbers
    assert not within_budget(seconds, budget=0.5)  # must reject: a budget it cannot meet
    if max(before, after) > FAST_PROBE_US:
        pytest.skip(
            f"unresolved: the workstation was not in its fast state (probe {before:.0f} and {after:.0f} us); "
            f"the live cell of {largest} took {seconds:.1f} s"
        )
    assert within_budget(seconds, budget=LIVE_BUDGET_S)


def markdown_blocks(source: str) -> list[str]:
    """The texts of the notebook's markdown cells, without the indentation of the code."""
    blocks = re.findall(r'mo\.md\(r"""\n(.*?)\n    """\)', source, flags=re.DOTALL)
    return [
        "\n".join(line[4:] if line.startswith("    ") else line for line in block.splitlines())
        for block in blocks
    ]


def quotes_preprint_numbers(text: str) -> bool:
    """True if a text holds one of the values of the preprint's behaviour figure (a few distinctive ones)."""
    return any(
        value in text
        for value in ("20.95", "12.78", "827,539", "827539", "0.240", "2.776", "9.254")
    )


def in_order(source: str, names: list[str]) -> bool:
    positions = [source.index(name) for name in names]
    return positions == sorted(positions)


def left_out_measures_named(text: str) -> list[str]:
    """Every mention of the two measures that notebook B leaves out, in prose or as a key."""
    pattern = r"permutation[ _]entropy|pairwise[ _]mi\b|pairwise mutual information|mutual_information_strings"
    return re.findall(pattern, text, flags=re.IGNORECASE)


def test_notebook_structure_and_prose():
    source = NOTEBOOK.read_text()
    steps = [
        "environment_ok",
        "dataset_choice = mo.ui.dropdown",
        "symbol_table",
        "surrogate_sets",
        "level_table",
        "own_string_set",
    ]
    assert in_order(source, steps)
    assert SWITCH_ON in source and DEFAULT_DATASET in source
    last_cell = source.rsplit("@app.cell", 1)[1]
    assert (
        "figure_behaviour" in last_cell and source.count("figure_behaviour(") == 1
    )  # one figure, at the end
    assert "cannot be identified from examples of its strings alone" in source
    blocks = markdown_blocks(source)
    assert len(blocks) >= 12 and all(broken_paragraphs(block) == [] for block in blocks)
    for name in ("notebooks/b_behaviour.py", "data/behaviour/SOURCES.md", "README.md"):
        assert not quotes_preprint_numbers((REPO / name).read_text()), name
    assert "behavior" not in " ".join(blocks)  # British spelling in the prose, as in notebook A
    # one sentence says why permutation entropy is absent; neither measure left out is named elsewhere
    assert left_out_measures_named(source) == ["Permutation entropy"]
    (reason,) = (
        line for block in blocks for line in block.splitlines() if "Permutation entropy" in line
    )  # one paragraph of one sentence
    assert reason.count(".") == 1 and reason.endswith(".")
    assert "order" in reason and "renamed" in reason
    # must reject: the steps in another order, a broken paragraph, a value of the preprint, the other
    # spelling, a text that names a measure left out as a measure of the notebook
    assert left_out_measures_named("the pairwise mutual information between strings") != []
    assert left_out_measures_named("permutation_entropy ± 0.01, pairwise_mi") != []
    assert not in_order(source, steps[::-1])
    assert broken_paragraphs("One sentence that\ncontinues on the next line.") != []
    assert quotes_preprint_numbers("the largest eigenvalue is 20.95")
    assert "behavior" in "animal behavior"


def make_fallback():
    spec = importlib.util.spec_from_file_location(
        "make_fallback", REPO / "scripts" / "make_fallback.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fallback_of_notebook_b_holds_the_figure_and_its_numbers(work_dir):
    make_fallback().fallback_b(work_dir / "fallback", no_html=True)
    written = sorted(path.name for path in (work_dir / "fallback").iterdir())
    assert written == [
        "figure_behaviour-onwhite.png",
        "figure_behaviour.data.json",
        "figure_behaviour.pdf",
        "figure_behaviour.png",
        "figure_behaviour.svg",
    ]
    assert all((work_dir / "fallback" / name).stat().st_size > 1000 for name in written)
    data = json.loads((work_dir / "fallback" / "figure_behaviour.data.json").read_text())
    saved, manifest = bm.load_results()
    expected = bm.figure_data(saved, highlight="zebrafish")
    assert all(same_numbers(data[key], bm.plain(expected)[key]) for key in expected)
    assert data["result_files"] == manifest["files"] and data["highlight"] == ["zebrafish"]
    # the folder of the repository holds the same numbers, and the static HTML export
    shipped = json.loads((FALLBACK / "figure_behaviour.data.json").read_text())
    assert same_numbers(shipped, data)
    assert sorted(path.name for path in FALLBACK.iterdir()) == sorted(
        [*written, "b_behaviour.html"]
    )
    html = (FALLBACK / "b_behaviour.html").read_text()
    assert len(html) > 100_000
    # the export shows no column of a measure left out (its prose names permutation entropy, with a space)
    assert not [name for name in left_out_measures_named(html) if "_" in name]
    assert f"{saved['zebrafish']['n_tokens']} symbols" in html  # exported on the shipped tables
    # must reject: figure data in which one number differs from the saved results
    changed = json.loads(json.dumps(data))
    changed["entropy_rate"]["seal"]["data"][2] += 0.01
    assert not same_numbers(changed["entropy_rate"], bm.plain(expected)["entropy_rate"])
