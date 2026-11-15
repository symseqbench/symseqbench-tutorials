"""What the notebooks show around their results: tables, figures, controls and short listings.

The notebooks keep the calls that teach something in their cells: loading the runs, drawing a
grammar, drawing strings, the exact predictor, the measures, training, the figures. The glue around
those calls is here, so that a code cell stays short enough to be read: how a table is shown, how a
figure fills the column, how the options of a control are named, how a few strings are listed.

Nothing in this module computes a result, and no script that produces results imports it. Every
function says in its docstring what it returns. The module itself imports nothing of SymSeq, so
that notebook A can show its install check through it when the installation is broken; the
functions that need the grammars import ``tutorial.complexity`` when they are called.
"""

from __future__ import annotations

import base64
import inspect
import io
from collections.abc import Callable, Iterable, Mapping, Sequence

import marimo as mo
import numpy as np
import pandas as pd

FIGURE_DPI = 200  # dots per inch of a figure drawn for the notebook's column
# The editor of marimo shows at most 610 pixels of an output and scrolls the rest inside the cell;
# 32 of them are the margins of the output. A figure is therefore never drawn taller than this.
MAX_FIGURE_HEIGHT = 570

# --- tables and figures ---------------------------------------------------------------------------


def table(frame: pd.DataFrame, digits: int | None = None):
    """A data frame as a plain table: every row, and numbers rounded to ``digits`` decimals if given.

    Returns a marimo table without selection, pages, search field, download button, data types and
    column summaries. The summaries are the small histograms that marimo draws under the header of
    every numeric column once a table has eleven rows. They are not part of what a table says
    here, and computing them is what makes pandas warn "invalid value encountered in cast" under
    the cell, once for every numeric column.
    """
    shown = frame if digits is None else frame.round(digits)
    return mo.ui.table(
        shown,
        selection=None,
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
        show_download=False,
        show_search=False,
    )


def titled_tables(tables: Mapping[str, pd.DataFrame]):
    """Several tables under one another, each under its title.

    Returns a vertical stack: for every entry of ``tables`` its title, a line of markdown such as
    ``"**Strings**"``, and below it the table, shown by ``table``. The frames are shown as they
    are given, so a cell rounds them itself.
    """
    if not tables:
        raise ValueError("At least one table is needed.")
    items = []
    for title, frame in tables.items():
        items += [mo.md(title), table(frame)]
    return mo.vstack(items)


def full_width(figure, dpi: int = FIGURE_DPI):
    """A matplotlib figure as an image that fills the width of the notebook's column.

    Returns an HTML element with one image of width 100 %. A figure that would then be taller than
    ``MAX_FIGURE_HEIGHT`` pixels, more than the editor shows of an output without scrolling inside
    it, is drawn at that height instead, in the middle of the column and with its proportions kept.
    The figure is not changed: it is drawn once more, into memory, as a PNG with the margins of the
    style in force, so the files that ``plotting.save_figure`` writes for the fallback and for
    slides stay what they are.
    """
    if not hasattr(figure, "savefig"):
        raise TypeError(f"A matplotlib figure is needed, got {type(figure).__name__}.")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    style = f"width: 100%; height: auto; max-height: {MAX_FIGURE_HEIGHT}px; object-fit: contain;"
    return mo.Html(f'<img src="data:image/png;base64,{encoded}" alt="figure" style="{style}" />')


# --- text made from code and from samples ---------------------------------------------------------


def source_of(*objects) -> mo.Html:
    """The source code of some functions or classes as one block of Python.

    Returns the text that ``inspect.getsource`` gives for each object, in the order given and
    separated by an empty line, as marimo markdown with Python highlighting.
    """
    if not objects:
        raise ValueError("At least one function or class is needed.")
    return mo.md("```python\n" + "\n\n".join(inspect.getsource(o) for o in objects) + "```")


def samples(text: str, settings: str) -> mo.Html:
    """Two samples of a dataset and the settings they were built with, as text.

    ``text`` and ``settings`` are what ``datasets.preview`` returns: the samples as lines of
    symbols, and the settings as YAML. Returns markdown with one block for each.
    """
    return mo.md(
        f"A dot marks a position without target.\n\n```\n{text}\n```\n\n```yaml\n{settings}\n```"
    )


# --- the controls of notebook C -------------------------------------------------------------------


def first_grammar(admitted: Iterable[tuple], level: int) -> tuple:
    """The first admitted grammar of a level.

    Returns its specification and the values of its admission. ``admitted`` is what
    ``complexity_runs.admitted_grammars`` returns, and ``level`` counts from 0. Raises
    ``ValueError`` if the level has no admitted grammar.
    """
    for spec, values in admitted:
        if spec.level == level:
            return spec, values
    raise ValueError(f"No admitted grammar at level {level + 1}.")


def level_dropdown(admitted: Iterable[tuple], default: int, label: str):
    """A dropdown of the levels that have an admitted grammar.

    Returns a marimo dropdown whose value is the level, counted from 0. Each option is named with
    the number of the level, counted from 1, and the topological entropy of its first grammar.
    ``default`` is selected if it is one of the levels, the lowest level otherwise.
    """
    levels: dict[str, int] = {}
    for spec, values in admitted:
        if spec.level not in levels.values():
            entropy = values["topological_entropy_bits"]
            levels[f"level {spec.level + 1} ({entropy:.2f} bits per symbol)"] = spec.level
    if not levels:
        raise ValueError("No admitted grammar, so there is no level to choose.")
    chosen = default if default in levels.values() else next(iter(levels.values()))
    selected = next(name for name, level in levels.items() if level == chosen)
    return mo.ui.dropdown(options=levels, value=selected, label=label)


def size_radio(sizes: Sequence[int], default: int, label: str):
    """A choice among the numbers of training strings.

    Returns a marimo radio group in one row whose value is the number chosen; ``default`` must be
    one of ``sizes``.
    """
    if default not in sizes:
        raise ValueError(f"The default {default} is not one of the sizes {tuple(sizes)}.")
    return mo.ui.radio(
        options={f"{size} strings": size for size in sizes},
        value=f"{default} strings",
        label=label,
        inline=True,
    )


# --- one grammar and a few of its strings ---------------------------------------------------------


def largest_surprisals(predictor, strings: Sequence[Sequence[str]], classes: Sequence[str]):
    """The largest surprisal of every string under the exact predictor, in bits.

    Returns one number per string: the largest of ``-log2 p(target | prefix)`` over its symbols
    and the end marker (Eq. 9 of ``tutorial.complexity``, with the exact predictor as the model).
    It is infinite for a string the grammar cannot produce.
    """
    from tutorial import complexity

    width = predictor.max_length + 1
    p, _ = complexity.exact_distributions(predictor, strings, classes, width)
    _, targets = complexity.encode(strings, classes, width)
    return complexity.largest_surprisal(complexity.log2_of(p), targets)


def surprisal_listing(strings, predictor, classes: Sequence[str]) -> str:
    """Grammatical strings and strings with one planted violation, each with its largest surprisal.

    ``strings`` has ``heldout`` and ``ungrammatical``, as ``complexity.draw_strings`` returns
    them. Returns plain text, one string per line, in two groups.
    """
    lines = ["grammatical                      largest surprisal under the exact predictor (bits)"]
    for string, value in zip(
        strings.heldout, largest_surprisals(predictor, strings.heldout, classes), strict=True
    ):
        lines.append(f"{' '.join(string):<32s}{value:.2f}")
    lines += ["", "with one planted violation"]
    for string, value in zip(
        strings.ungrammatical,
        largest_surprisals(predictor, strings.ungrammatical, classes),
        strict=True,
    ):
        shown = "infinite" if np.isinf(value) else format(value, ".2f")
        lines.append(f"{' '.join(string):<32s}{shown}")
    return "\n".join(lines)


def grammar_card(graph, spec, values: Mapping, strings, predictor):
    """The drawing of a grammar beside its name and a few of its strings, as one row.

    Returns a horizontal stack. Left: ``graph``, the figure that ``plotting.draw_grammar``
    returns. Right: the name of the grammar, its topological entropy, its start and terminal
    states, and the listing of ``surprisal_listing`` for ``strings`` under ``predictor``.
    """
    from tutorial import complexity

    listing = surprisal_listing(strings, predictor, complexity.classes_of(spec))
    return mo.hstack(
        [
            mo.as_html(graph),
            mo.md(
                f"**{spec.name}**: {values['topological_entropy_bits']:.2f} bits per symbol, "
                f"start state {spec.start}, terminal states {', '.join(spec.terminals)}. "
                f"Drawn by SymSeq; `#` is the end marker.\n\n```\n{listing}\n```"
            ),
        ],
        widths=[1, 1.4],
    )


# --- one run of notebook C: saved or trained in the notebook --------------------------------------


def runs_of(runs: Sequence[Mapping], grammar: str, n_train: int, models: Sequence[str]) -> dict:
    """The saved run of every model for one grammar and one number of training strings.

    Returns a dictionary from the model to its run: the first one in ``runs`` for that grammar,
    model and size, which is the run of the first training seed. Raises ``ValueError`` if a model
    has no such run.
    """
    found = {}
    for model in models:
        for run in runs:
            if (run["grammar"]["name"], run["model"], run["n_train"]) == (grammar, model, n_train):
                found[model] = run
                break
        else:
            raise ValueError(f"No saved run of the {model} for {grammar}, {n_train} strings.")
    return found


def bar_settings(network: str, settings) -> dict:
    """The arguments of the progress bar of one training run.

    Returns the keywords for ``mo.status.progress_bar``: one step for every measurement of the
    validation loss after the first, a title that names the network, and removal at the end.
    """
    return {
        "total": settings.max_steps // settings.eval_every,
        "title": f"Training the {network}",
        "remove_on_exit": True,
    }


def advance(bar) -> Callable[[int, dict], None]:
    """What a training run calls after every measurement, so that a progress bar moves.

    Returns a function of the step and the row of the measurement for the ``progress`` argument of
    ``complexity_runs.train_run``. It moves ``bar`` by one, except at step 0, the untrained network.
    """

    def progress(step: int, row: dict) -> None:
        if step:
            bar.update()

    return progress


def live_point(values: Mapping, n_train: int, trained: Mapping[str, Mapping]) -> dict:
    """A run trained in the notebook as the point that ``plotting.figure_complexity`` draws as a star.

    Returns the topological entropy of the grammar, the number of training strings and, for every
    model, the measures of its run.
    """
    return {
        "topological_entropy_bits": values["topological_entropy_bits"],
        "n_train": n_train,
        **{model: run["measures"] for model, run in trained.items()},
    }
