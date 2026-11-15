"""What the notebooks show around their results (``tutorial.notebook_views``), and how they show it.

Each test has a case it must reject. Nothing here trains a network or reads a saved run.
"""

from __future__ import annotations

import ast
import base64
import inspect
import re
import struct
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import marimo as mo
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from marimo._plugins.ui._impl.table import ColumnSummariesArgs
from test_complexity import UNIFILAR
from test_conventions import as_cell, cell_facts

from tutorial import complexity as cx
from tutorial import notebook_views as views
from tutorial import plotting

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("a_generate_and_measure", "b_behaviour", "c_networks")


def frame_of_twelve_rows() -> pd.DataFrame:
    """Twelve rows, so that marimo's own table would draw its column summaries: two numeric columns."""
    return pd.DataFrame(
        {
            "network": ["tanh RNN", "linear transformer"] * 6,
            "value": np.linspace(0.123456, 0.987654, 12),
            "count": list(range(12)),
        }
    )


# --- the module itself ----------------------------------------------------------------------------


def functions_of_the_module() -> dict:
    return {
        name: function
        for name, function in inspect.getmembers(views, inspect.isfunction)
        if function.__module__ == views.__name__
    }


def undocumented(functions: dict) -> list[str]:
    """Functions without a docstring that says what they return."""
    return [
        name
        for name, function in functions.items()
        if not function.__doc__ or "eturns" not in function.__doc__
    ]


def untested(names, test_source: str) -> list[str]:
    """Functions that no test of this file calls."""
    return [name for name in names if f"views.{name}(" not in test_source]


def needs_the_installation(source: str) -> list[str]:
    """What a module imports at its top that only a complete installation has."""
    imported = [
        alias.name if isinstance(node, ast.Import) else str(node.module)
        for node in ast.parse(source).body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    ]
    return sorted({name for name in imported if name.startswith(("tutorial", "symseq", "torch"))})


def test_every_function_says_what_it_returns_and_has_a_test():
    functions = functions_of_the_module()
    assert len(functions) >= 12
    assert undocumented(functions) == []
    assert untested(functions, Path(__file__).read_text()) == []
    # must reject: a function without a docstring, one whose docstring does not say what it
    # returns, and one that no test calls
    silent = {"silent": lambda: None, "vague": undocumented, **functions}
    assert undocumented(silent) == ["silent", "vague"]
    assert untested([*functions, "not_there"], Path(__file__).read_text()) == ["not_there"]
    # the module imports nothing of SymSeq when it is imported, so that notebook A can show its
    # install check through it
    assert needs_the_installation(inspect.getsource(views)) == []
    # must reject: an import of the grammars, or of the networks, at the top of the module
    assert needs_the_installation("import marimo\nfrom tutorial import complexity\n") == [
        "tutorial"
    ]
    assert needs_the_installation("import torch\nimport numpy as np\n") == ["torch"]


# --- tables ---------------------------------------------------------------------------------------


def summaries_of(shown) -> tuple:
    """The column summaries that marimo computes for a table when the editor asks, and what was warned."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summaries = shown._get_column_summaries(ColumnSummariesArgs())
    return summaries, [str(warning.message) for warning in caught]


def test_a_table_shows_every_row_and_computes_no_column_summaries():
    frame = frame_of_twelve_rows()
    shown = views.table(frame, 3)
    assert isinstance(shown, mo.ui.table)
    assert list(shown.data["value"]) == list(frame["value"].round(3))
    assert list(views.table(frame).data["value"]) == list(
        frame["value"]
    )  # not rounded unless asked
    summaries, warned = summaries_of(shown)
    assert summaries.bin_values == {} and summaries.stats == {} and warned == []
    for option in ("pagination", "show-column-summaries", "show-data-types", "show-download"):
        assert re.search(rf"data-{option}=\S{{0,12}}false", shown.text), option
    # must reject: marimo's own table of the same frame, which is what a data frame left as the
    # output of a cell becomes. It draws a histogram under every numeric column, and computing
    # the histograms is what makes pandas warn: this is the cause of the warning in the editor.
    default = mo.ui.table(frame, selection=None, pagination=None)
    summaries, warned = summaries_of(default)
    assert "value" in summaries.bin_values
    assert any("invalid value encountered in cast" in message for message in warned)
    # a table of fewer than eleven rows gets no summaries from marimo either
    summaries, warned = summaries_of(mo.ui.table(frame.head(10), selection=None, pagination=None))
    assert summaries.bin_values == {} and warned == []


def test_titled_tables_stack_titles_and_tables():
    frame = frame_of_twelve_rows()
    stack = views.titled_tables({"**Strings** (a remark)": frame, "**Grammars**": frame.head(3)})
    assert stack.text.count("<marimo-table") == 2
    assert stack.text.index("a remark") < stack.text.index("Grammars")
    assert "<strong>Strings</strong>" in stack.text.replace("&lt;", "<").replace("&gt;", ">")
    # must reject: nothing to show
    with pytest.raises(ValueError, match="At least one table"):
        views.titled_tables({})


# --- figures --------------------------------------------------------------------------------------


def png_size(element) -> tuple[int, int]:
    """Width and height in pixels of the PNG that an element of ``full_width`` holds."""
    found = re.fullmatch(
        r'<img src="data:image/png;base64,([A-Za-z0-9+/=]+)" alt="figure" '
        r'style="width: 100%; height: auto; max-height: 570px; object-fit: contain;" />',
        element.text,
    )
    assert found, element.text[:120]
    data = base64.b64decode(found.group(1))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_a_figure_fills_the_width_of_the_column():
    with matplotlib.rc_context({"savefig.bbox": None}):
        figure, ax = plt.subplots(figsize=(7.2, 3.8))
        ax.plot([0, 1], [0, 1])
        shown = views.full_width(figure)
        assert png_size(shown) == (1440, 760)  # 7.2 by 3.8 inches at 200 dots per inch
        # the figure itself is what it was: the files of the fallback are written from it
        assert tuple(figure.get_size_inches()) == (7.2, 3.8) and len(ax.lines) == 1
        # must reject: another resolution is another image, and the width is 100 % whatever it is
        assert png_size(views.full_width(figure, dpi=100)) == (720, 380)
        plt.close(figure)
        # a figure that is nearly square would be taller at full width than the editor shows of an
        # output (610 pixels, 32 of them margins): the same style holds its height at 570 pixels
        assert views.MAX_FIGURE_HEIGHT == 570 and views.MAX_FIGURE_HEIGHT + 32 <= 610
        square, _ = plt.subplots(figsize=(6.0, 5.4))
        assert png_size(views.full_width(square)) == (1200, 1080)
        plt.close(square)
    # must reject: something that is not a figure
    with pytest.raises(TypeError, match="matplotlib figure"):
        views.full_width("figure")


# --- text made from code and from samples ---------------------------------------------------------


def test_source_and_samples_are_shown_as_text():
    shown = views.source_of(views.advance, views.live_point).text
    assert shown.index("advance") < shown.index("live_point") and "progress" in shown
    assert "bar_settings" not in views.source_of(views.advance).text  # only what was asked for
    sample = views.samples("A X X B # .", "seed: 101").text
    assert "A X X B" in sample and "seed" in sample and "101" in sample
    assert "without target" in sample
    # must reject: nothing to show the source of
    with pytest.raises(ValueError, match="At least one"):
        views.source_of()


# --- the controls of notebook C -------------------------------------------------------------------


def admitted_grammars() -> list[tuple]:
    """Two grammars at the first level and one at the third, as ``admitted_grammars`` lists them."""
    return [
        (replace(UNIFILAR, level=0, seed=101), {"topological_entropy_bits": 1.304}),
        (replace(UNIFILAR, level=0, seed=102), {"topological_entropy_bits": 1.351}),
        (replace(UNIFILAR, level=2, seed=103), {"topological_entropy_bits": 1.698}),
    ]


def test_the_controls_name_the_levels_and_the_sizes():
    admitted = admitted_grammars()
    levels = views.level_dropdown(admitted, 2, label="level")
    assert levels.value == 2
    assert list(levels.options) == [
        "level 1 (1.30 bits per symbol)",
        "level 3 (1.70 bits per symbol)",
    ]
    # a default that is not one of the levels gives the lowest level
    assert views.level_dropdown(admitted, 1, label="level").value == 0
    sizes = views.size_radio((50, 200, 800), 200, label="training strings")
    assert sizes.value == 200 and list(sizes.options) == [
        "50 strings",
        "200 strings",
        "800 strings",
    ]
    spec, values = views.first_grammar(admitted, 0)
    assert spec.seed == 101 and values["topological_entropy_bits"] == 1.304
    assert views.first_grammar(admitted, 2)[0].seed == 103
    # must reject: no grammar, a default size that is not offered, a level without a grammar
    with pytest.raises(ValueError, match="no level to choose"):
        views.level_dropdown([], 0, label="level")
    with pytest.raises(ValueError, match="not one of the sizes"):
        views.size_radio((50, 200, 800), 100, label="training strings")
    with pytest.raises(ValueError, match="No admitted grammar at level 2"):
        views.first_grammar(admitted, 1)


# --- one grammar and a few of its strings ---------------------------------------------------------


@pytest.fixture(scope="module")
def six_strings():
    predictor = cx.BoundedPredictor(cx.build_grammar(UNIFILAR), 3, 14)
    strings = cx.draw_strings(UNIFILAR, 101, 6, 6, 3, 14, 10_000, "distinct")
    return predictor, strings, cx.classes_of(UNIFILAR)


def test_the_listing_gives_the_largest_surprisal_of_every_string(six_strings):
    predictor, strings, classes = six_strings
    found = views.largest_surprisals(predictor, strings.heldout, classes)
    # by another route: the probability of every symbol and of the end, one string at a time
    p, _ = cx.exact_distributions(predictor, strings.heldout, classes, 15)
    for row, string in enumerate(strings.heldout):
        targets = [*(classes.index(symbol) for symbol in string), cx.EOS_CLASS]
        by_hand = max(-np.log2(p[row, t, target]) for t, target in enumerate(targets))
        assert abs(found[row] - by_hand) < 1e-12 and np.isfinite(found[row])
    assert np.isinf(views.largest_surprisals(predictor, strings.ungrammatical, classes)).all()
    listing = views.surprisal_listing(strings, predictor, classes).splitlines()
    assert len(listing) == 1 + 6 + 2 + 6 and listing[7] == "" and "planted violation" in listing[8]
    assert all(re.search(r" \d+\.\d\d$", line) for line in listing[1:7])
    assert all(line.endswith("infinite") for line in listing[9:])
    assert listing[1].startswith(" ".join(strings.heldout[0]))
    # must reject: a grammatical string listed as a violation gets its number, not "infinite"
    swapped = SimpleNamespace(heldout=strings.heldout, ungrammatical=strings.heldout)
    assert not any(
        "infinite" in line
        for line in views.surprisal_listing(swapped, predictor, classes).splitlines()
    )


def test_the_card_of_a_grammar_holds_its_drawing_and_its_strings(six_strings):
    predictor, strings, _ = six_strings
    with matplotlib.rc_context():
        plotting.apply_style()
        graph = plotting.draw_grammar(cx.build_grammar(UNIFILAR))
        values = {"topological_entropy_bits": 1.0565}
        card = views.grammar_card(graph, UNIFILAR, values, strings, predictor).text
        plt.close(graph)
    assert UNIFILAR.name in card and "1.06 bits per symbol" in card
    assert f"start state {UNIFILAR.start}" in card and ", ".join(UNIFILAR.terminals) in card
    assert card.count("infinite") == 6 and " ".join(strings.heldout[0]) in card
    assert "image/png" in card or "<img" in card  # the drawing is in the row
    # must reject: the values of another kind of object, which have no topological entropy
    with pytest.raises(KeyError):
        views.grammar_card(None, UNIFILAR, {}, strings, predictor)


# --- one run: saved or trained in the notebook ----------------------------------------------------


def test_saved_runs_progress_bar_and_the_point_of_a_live_run():
    def run(model: str, seed: int, n_train: int = 200, grammar: str = "level 1, grammar 2") -> dict:
        return {
            "grammar": {"name": grammar},
            "model": model,
            "n_train": n_train,
            "seed": seed,
            "measures": {"score": seed / 10, "auc": 0.9},
        }

    saved = [run("rnn", 1, 50), run("rnn", 1), run("rnn", 2), run("linear transformer", 1)]
    found = views.runs_of(saved, "level 1, grammar 2", 200, ("rnn", "linear transformer"))
    assert list(found) == ["rnn", "linear transformer"]
    assert found["rnn"] is saved[1] and found["linear transformer"] is saved[3]  # the first seed
    # must reject: a model without a run of that grammar and size
    with pytest.raises(ValueError, match="No saved run of the linear transformer"):
        views.runs_of(saved, "level 1, grammar 2", 50, ("rnn", "linear transformer"))
    with pytest.raises(ValueError, match="No saved run of the rnn"):
        views.runs_of(saved, "level 2, grammar 3", 200, ("rnn",))

    settings = SimpleNamespace(max_steps=2000, eval_every=50)
    assert views.bar_settings("tanh RNN", settings) == {
        "total": 40,
        "title": "Training the tanh RNN",
        "remove_on_exit": True,
    }
    moved = []
    progress = views.advance(SimpleNamespace(update=lambda: moved.append(1)))
    for step in (0, 50, 100):
        progress(step, {"validation_loss": 1.0})
    assert (
        len(moved) == 2
    )  # must reject: the measurement of the untrained network moves the bar too

    point = views.live_point({"topological_entropy_bits": 1.64, "memory_gap": 0.1}, 200, found)
    assert point == {
        "topological_entropy_bits": 1.64,
        "n_train": 200,
        "rnn": saved[1]["measures"],
        "linear transformer": saved[3]["measures"],
    }
    assert point["rnn"] is not saved[2]["measures"]  # must reject: the measures of another run


# --- how the notebooks show tables and figures ----------------------------------------------------


FRAME_METHODS = ("round", "set_index", "droplevel", "head", "rename", "sort_values")


def made_by(node: ast.Call) -> str:
    """The leftmost name of what a call calls: ``mo`` for ``mo.ui.table(...)``."""
    function = node.func
    while isinstance(function, ast.Attribute):
        function = function.value
    return function.id if isinstance(function, ast.Name) else ""


def shown_as_it_is(node: ast.AST, made_here: set[str]) -> bool:
    """Is this displayed expression a data frame or a figure that is left to marimo as it is?

    marimo then draws its own table, with the column summaries that warn, or the figure at its
    own width. What can be told from the source: a name that the cell did not make with marimo or
    ``notebook_views``; a method that returns a frame (``.round(3)`` and the like); a part of a
    frame (``table["string"]``); ``pd.DataFrame(...)``; a function named ``..._table``,
    ``..._legend``, ``figure_...`` or ``draw_...``. What a layout of marimo holds is looked at as
    well; what is handed to ``notebook_views`` is not, because that is where it belongs.
    """
    if isinstance(node, ast.IfExp):
        return shown_as_it_is(node.body, made_here) or shown_as_it_is(node.orelse, made_here)
    if isinstance(node, ast.Name):
        return node.id not in made_here
    if isinstance(node, ast.Subscript):
        return True
    if not isinstance(node, ast.Call):
        return False
    if made_by(node) == "views":
        return False
    if made_by(node) == "mo":
        held = []
        for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
            if isinstance(argument, (ast.List, ast.Tuple)):
                held += argument.elts
            elif isinstance(argument, ast.Dict):
                held += argument.values
        return any(shown_as_it_is(element, made_here) for element in held)
    name = node.func.attr if isinstance(node.func, ast.Attribute) else ast.unparse(node.func)
    return (
        name in FRAME_METHODS
        or name == "DataFrame"
        or name.endswith(("_table", "_legend"))
        or name.startswith(("figure_", "draw_"))
    )


def cells_that_show_something_as_it_is(source: str) -> list[int]:
    """Numbers of the visible cells that leave a data frame or a figure to marimo as it is."""
    found = []
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.decorator_list
    ]
    for number, (node, facts) in enumerate(zip(functions, cell_facts(source), strict=True)):
        if facts["hidden"]:
            continue
        shown = [statement for statement in node.body if not isinstance(statement, ast.Return)]
        if not shown or not isinstance(shown[-1], ast.Expr):
            continue
        made_here = {
            target.id
            for statement in shown
            if isinstance(statement, ast.Assign)
            and isinstance(statement.value, ast.Call)
            and made_by(statement.value) in ("mo", "views")
            for target in statement.targets
            if isinstance(target, ast.Name)
        }
        if shown_as_it_is(shown[-1].value, made_here):
            found.append(number)
    return found


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_no_cell_shows_a_frame_or_a_figure_as_it_is(name):
    source = (REPO / "notebooks" / f"{name}.py").read_text()
    assert cells_that_show_something_as_it_is(source) == []
    assert "views.full_width(figure" in source and "views.table(" in source
    assert source.count("views.full_width(") == source.count("plotting.figure_")  # every figure
    accepted = (
        "views.table(frame.round(3))",
        "choice = mo.ui.slider(1, 5)\nchoice",
        'mo.vstack([views.table(value), mo.accordion({"more": views.table(value)})])',
        "views.titled_tables({'**One**': value}) if value is not None else None",
    )
    for body in accepted:
        assert cells_that_show_something_as_it_is(source + as_cell(body)) == [], body
    # a call that is made for what it does, not for what it shows, is not an output of this kind
    assert cells_that_show_something_as_it_is(source + as_cell("bm.check_folder(value)")) == []
    # must reject: a rounded frame, a frame from a function, a part of a frame, a frame inside a
    # layout, a bare figure, and a name that the cell did not make with marimo or notebook_views
    refused = (
        "value.round(3)",
        "pd.DataFrame(value)",
        "bm.checks_table(value)",
        'value["string"]',
        "plotting.figure_networks(value)",
        'mo.vstack([mo.md("**Tokens**"), value.round(3)])',
        'mo.accordion({"more": bm.settings_table()})',
        "figure = plotting.figure_networks(value)\nfigure",
        "value",
    )
    for body in refused:
        assert len(cells_that_show_something_as_it_is(source + as_cell(body))) == 1, body
