"""Plot style and figure helpers.

The sizes and colours are those of the talk's figure style (ticks 7 pt, labels 8 pt, panel letters
10 pt bold, lines 1.5, spines 0.8, no titles; full slide 7.2 x 3.8 in, half slide 3.5 x 3.0 in). The
canonical implementation of that style renders text through LaTeX. Participants' laptops may have no
LaTeX, so the default here is matplotlib's own math rendering with the Computer Modern fonts that
ship with matplotlib. ``apply_style(use_latex=True)`` switches to LaTeX.

Labels must be written so that both renderers accept them: plain text, and mathematics between
dollar signs using ``\\mathrm`` rather than ``\\text``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from symseq.viz.mc_graph import MarkovChain

FULL_SLIDE = (7.2, 3.8)
HALF_SLIDE = (3.5, 3.0)

WONG_PALETTE = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#ECE133",
    "#D55E00",
    "#CC78BC",
    "#56B4E9",
    "#7A4F00",
]

COLORS = {
    "estimate": "#0173B2",  # our measurement
    "second": "#DE8F05",  # a second measured quantity in the same panel
    "reference": "0.45",  # closed form or exact value, drawn first, line 1.2
    "limit": "0.45",  # a bound or chance level, dashed, line 1.0
    "ink": "#1A1A2E",
}
REFERENCE_LINEWIDTH = 1.2
LIMIT_LINEWIDTH = 1.0

# Sizes shared with the canonical style (compared with it in tests/test_plotting.py).
STYLE_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "ytick.major.size": 3.0,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "errorbar.capsize": 2,
    "legend.frameon": False,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#cccccc",
}

MATHTEXT_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
}

LATEX_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
}


def latex_available() -> bool:
    """True if a LaTeX installation that matplotlib can use is on the path."""
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


def apply_style(use_latex: bool = False) -> None:
    """Apply the figure style. Without LaTeX (the default) Computer Modern comes from matplotlib itself."""
    if use_latex and not latex_available():
        raise RuntimeError(
            "LaTeX rendering was requested but latex and dvipng are not installed. Use apply_style()."
        )
    rc = dict(STYLE_RC)
    rc["axes.prop_cycle"] = mpl.cycler(color=WONG_PALETTE)
    rc.update(LATEX_RC if use_latex else MATHTEXT_RC)
    mpl.rcParams.update(rc)


def panel_label(ax, letter: str, x: float = -0.2, y: float = 1.08) -> None:
    """Bold panel letter at the upper left of an axis, 10 pt."""
    ax.text(
        x, y, rf"$\mathbf{{{letter}}}$", transform=ax.transAxes, fontsize=10, va="top", ha="left"
    )


def save_figure(
    fig, out_dir: str | Path, basename: str, formats: tuple[str, ...] = ("pdf", "png", "svg")
) -> dict:
    """Save a figure as PDF, PNG (300 dpi) and SVG with a transparent background, plus a PNG on white."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for fmt in formats:
        written[fmt] = out_dir / f"{basename}.{fmt}"
        fig.savefig(written[fmt], format=fmt, transparent=True)
    if "png" in formats:
        written["png_onwhite"] = out_dir / f"{basename}-onwhite.png"
        fig.savefig(written["png_onwhite"], format="png", transparent=False, facecolor="white")
    return written


def save_figure_data(data: dict, out_dir: str | Path, basename: str) -> Path:
    """Save the numbers a figure shows next to it, as JSON, so that they are never retyped."""

    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    path = Path(out_dir) / f"{basename}.data.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=default))
    return path


def figure_generate_and_measure(convergence: dict, curve: dict, prescribed: dict):
    """Notebook A: the worked example and a grammar synthesised at a chosen complexity.

    ``convergence`` and ``curve`` are the two-state worked example. ``prescribed`` comes from
    ``tutorial.prescribed.figure_data``: the sampler's steps and topological entropy in bits, its
    target, and the achieved entropy on the open and closed graphs.
    """
    fig, axes = plt.subplots(
        1,
        4,
        figsize=FULL_SLIDE,
        layout="constrained",
        gridspec_kw={"width_ratios": [1.15, 1.15, 1.0, 0.8]},
    )
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.04)
    ax_a, ax_b, ax_c, ax_d = axes

    # A: estimates against sequence length, closed forms as reference lines (drawn first, in grey)
    ax_a.axhline(
        convergence["entropy_rate_exact"],
        color=COLORS["reference"],
        lw=REFERENCE_LINEWIDTH,
        label="closed form",
    )
    ax_a.axhline(
        convergence["excess_entropy_exact"], color=COLORS["reference"], lw=REFERENCE_LINEWIDTH
    )
    ax_a.plot(
        convergence["n"],
        convergence["entropy_rate"],
        "o-",
        color=COLORS["estimate"],
        label=r"entropy rate $h$",
    )
    ax_a.plot(
        convergence["n"],
        convergence["excess_entropy"],
        "s-",
        color=COLORS["second"],
        label=r"memory $E$",
    )
    ax_a.set_xscale("log")
    ax_a.set_xlabel(r"sequence length $N$ (symbols)")
    ax_a.set_ylabel("bits")
    ax_a.legend(loc="center right", handlelength=1.5)
    panel_label(ax_a, "A")

    # B: entropy rate as a function of p, with the topological entropy as its upper limit
    ax_b.plot(
        curve["p"],
        curve["entropy_rate"],
        color=COLORS["reference"],
        lw=REFERENCE_LINEWIDTH,
        label="closed form",
    )
    ax_b.axhline(
        curve["topological_entropy"],
        color=COLORS["limit"],
        lw=LIMIT_LINEWIDTH,
        ls="--",
        label="topological entropy",
    )
    ax_b.axvline(curve["p_max"], color=COLORS["limit"], lw=LIMIT_LINEWIDTH, ls=":")
    ax_b.plot(
        curve["p_estimated"],
        curve["entropy_rate_estimated"],
        "o",
        color=COLORS["estimate"],
        label="estimate",
    )
    ax_b.set_xlabel(r"$p(\mathrm{A} \rightarrow \mathrm{B})$")
    ax_b.set_ylabel(r"entropy rate $h$ (bits per symbol)")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 0.8)
    ax_b.legend(
        loc="center", bbox_to_anchor=(0.5, 0.22), handlelength=1.5
    )  # inside the arch of the curve
    panel_label(ax_b, "B")

    # C: the sampler approaching the requested topological entropy
    steps = np.asarray(prescribed["steps"], dtype=float)
    entropy_bits = np.asarray(prescribed["entropy_bits"], dtype=float)
    valid = np.isfinite(entropy_bits) & (entropy_bits > 0)
    ax_c.plot(
        steps[valid],
        entropy_bits[valid],
        color=COLORS["estimate"],
        lw=SERIES_LINEWIDTH,
        label="sampler",
    )
    ax_c.axhline(
        prescribed["target_bits"],
        color=COLORS["limit"],
        lw=LIMIT_LINEWIDTH,
        ls="--",
        label="target",
    )
    ax_c.set_xlabel("iteration")
    ax_c.set_ylabel(r"topological entropy $h_{\mathrm{top}}$ (bits per symbol)")
    ax_c.set_ylim(bottom=0)
    ax_c.legend(loc="lower right", handlelength=1.5)
    panel_label(ax_c, "C")

    # D: target and achieved values, on the open graph and after closing it with the end marker
    labels = ["target", "open graph", "closed graph"]
    values = [
        prescribed["target_bits"],
        prescribed["achieved_open_bits"],
        prescribed["closed_bits"],
    ]
    positions = np.arange(len(labels))[::-1]
    ax_d.barh(
        positions,
        values,
        height=0.55,
        color=[COLORS["reference"], COLORS["estimate"], COLORS["second"]],
    )
    ax_d.set_yticks(positions)
    ax_d.set_yticklabels(labels)
    ax_d.set_xlabel(r"$h_{\mathrm{top}}$ (bits per symbol)")
    ax_d.set_xlim(0, 1.15 * max(values))
    for position, value in zip(positions, values, strict=True):
        ax_d.text(value + 0.02 * max(values), position, f"{value:.2f}", va="center", fontsize=7)
    panel_label(ax_d, "D", x=-0.55)
    return fig


MODEL_COLORS = {"rnn": COLORS["estimate"], "linear transformer": COLORS["second"]}
MODEL_NAMES = {"rnn": "tanh RNN", "linear transformer": "linear transformer"}
PLACEHOLDER_COLOR = "#D55E00"
PLACEHOLDER_TEXT = "placeholder: short smoke runs"
PLACEHOLDER_TEXT_PILOT = "placeholder: pilot runs"
DISTANCE_FLOOR = 1e-4  # bits per target; smaller distances (and negative ones) are not drawn


def tick_label(grammar: str) -> str:
    """Name of a grammar on an axis; no underscore, so that LaTeX accepts it."""
    return grammar.replace("p_target", "target")


def mark_placeholder(ax, text: str) -> None:
    """Write above an axis that what it shows is a placeholder."""
    ax.text(
        0.5,
        1.02,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
        color=PLACEHOLDER_COLOR,
    )


def figure_networks(data: dict):
    """First figure of notebook C: each network's distance from the exact predictor over training.

    One small panel per grammar. The quantity is the excess of the network's cross-entropy on the
    held-out samples over the cross-entropy of the exact predictor on the same symbols, in bits per
    target (Eq. 1 of ``tutorial.results``), on one logarithmic scale for all grammars. Thin lines:
    single seeds; thick lines: their mean. The grey dashed line is the resolution of the estimate
    (Eq. 3 there): a distance below it cannot be told from zero. Distances at or below
    ``DISTANCE_FLOOR``, negative ones included, are not drawn. The panels compare the two networks
    within a grammar; the figure does not rank the grammars.

    Parameters
    ----------
    data : dict
        From ``tutorial.results.distance_data``: ``grammars``, ``epochs`` and ``resolution`` (one
        entry per grammar), ``models`` (per model: ``distance``, one curve per seed for each
        grammar, and ``mean``), ``kind``. Results of any kind other than "production" are marked as
        placeholders in every panel.
    """
    names = data["grammars"]
    columns = (len(names) + 1) // 2
    fig, axes = plt.subplots(
        2,
        columns,
        figsize=FULL_SLIDE,
        layout="constrained",
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.get_layout_engine().set(w_pad=0.04, wspace=0.03, h_pad=0.06, hspace=0.08)
    top = 1.0
    for index, (ax, name) in enumerate(zip(axes.ravel(), names, strict=False)):
        epochs = np.asarray(data["epochs"][index])
        # the reference first and in grey: what the held-out samples can resolve
        ax.axhline(
            data["resolution"][index], color=COLORS["limit"], lw=LIMIT_LINEWIDTH, ls="--", zorder=1
        )
        for model, values in data["models"].items():
            color = MODEL_COLORS[model]
            for curve in values["distance"][index]:
                curve = np.asarray(curve, dtype=float)
                ax.plot(
                    epochs,
                    np.where(curve > DISTANCE_FLOOR, curve, np.nan),
                    color=color,
                    lw=0.6,
                    alpha=0.5,
                )
            mean = np.asarray(values["mean"][index], dtype=float)
            ax.plot(epochs, np.where(mean > DISTANCE_FLOOR, mean, np.nan), color=color, lw=1.6)
            top = max(top, float(np.nanmax(mean)))
        ax.text(
            0.5,
            1.03 if data["kind"] == "production" else 1.14,
            tick_label(name),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["ink"],
        )
        if data["kind"] != "production":
            mark_placeholder(ax, PLACEHOLDER_TEXT)
    for ax in axes.ravel()[len(names) :]:
        ax.set_axis_off()
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(DISTANCE_FLOOR, 10.0 ** np.ceil(np.log10(top)))
    for ax in axes[-1]:
        ax.set_xlabel("epoch")
    fig.supylabel("distance from the exact predictor (bits per target)", fontsize=8)

    handles = [
        *(
            Line2D([], [], color=MODEL_COLORS[model], lw=1.6, label=f"{MODEL_NAMES[model]}, mean")
            for model in data["models"]
        ),
        Line2D([], [], color=COLORS["ink"], lw=0.6, alpha=0.5, label="single seeds"),
        Line2D(
            [],
            [],
            color=COLORS["limit"],
            lw=LIMIT_LINEWIDTH,
            ls="--",
            label="resolution of the held-out samples",
        ),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), handlelength=1.8)
    return fig


FIRST_ORDER = "first-order model"
FIRST_ORDER_COLOR = COLORS["reference"]  # grey: a reference model, not one of the two networks
# The anchors of a row (exact predictor, uniform guess, chance): thin dotted dark lines, in no
# colour and no line style that a model has.
ANCHOR_COLOR = COLORS["ink"]
ANCHOR_STYLE = ":"
ANCHOR_LINEWIDTH = 0.7
ANCHOR_FONTSIZE = 6.5
SERIES_LINEWIDTH = 1.4
# The line through the level means is dashed where it crosses a band of the design that was not
# run: no grammar was drawn in that range, so the line joins two levels and is no curve there.
NOT_RUN_STYLE = "--"
NOT_RUN_LABEL = "across a band that was not run"
# the two rows of the figure: measure, lower anchor and its name, label of the vertical axis
COMPLEXITY_ROWS = (
    ("score", 0.0, "uniform guess", "prediction score"),
    ("auc", 0.5, "chance", "grammaticality judgement\n(area under the ROC curve)"),
)


def level_stretches(segments: list[dict]) -> list[tuple[str, int, int]]:
    """How the line through the level means is drawn: in stretches, each solid or dashed.

    ``segments`` has one entry per pair of neighbouring levels, lowest first
    (``complexity_runs.level_segments``). Neighbouring segments of the same kind form one stretch.
    Returns, for each stretch, its line style and the positions of its first and last level.
    """
    stretches: list[tuple[str, int, int]] = []
    for position, segment in enumerate(segments):
        style = NOT_RUN_STYLE if segment["dashed"] else "-"
        if stretches and stretches[-1][0] == style:
            stretches[-1] = (style, stretches[-1][1], position + 1)
        else:
            stretches.append((style, position, position + 1))
    return stretches or [("-", 0, 0)]  # one level alone: a line of one point


def figure_complexity(data: dict, live: dict | None = None):
    """Final figure of notebook C: the two networks against the complexity of the grammar.

    One column per number of training strings. Upper row: the prediction score on held-out
    strings (Eq. 13 of ``tutorial.complexity``), which is 0 for the uniform guess and 1 for the
    exact predictor. Lower row: the area under the ROC curve of the grammaticality judgement
    (Eq. 10 there), which is 0.5 at chance and 1 for the exact predictor. The horizontal axis is
    the topological entropy of the grammar. Each panel holds three series: the two networks and,
    in grey, the first-order model with add-one smoothing. Points are grammars (mean over training
    seeds) and lines join the means of the levels. Between two levels that have a band of the
    design between them that was not run, the line is dashed, and the legend says so. The anchors
    are thin dotted dark lines, named in words in the first panel of their row, and always lie
    inside the axis range; the range goes below the lower anchor if a value does.

    Parameters
    ----------
    data : dict
        From ``tutorial.complexity_runs.figure_data``: ``grammars`` and ``levels`` (with their
        ``topological_entropy_bits``), ``segments`` (per pair of neighbouring levels, whether the
        line between them is ``dashed``), ``sizes``, ``models`` and ``first_order`` (per training
        size and measure: ``grammar``, one value per grammar, and ``level``, one per level),
        ``placeholder``. Placeholder results are marked as such above every panel.
    live : dict, optional
        A run trained in the notebook: ``topological_entropy_bits``, ``n_train`` and, per model,
        its ``score`` and ``auc``. It is drawn as a star in the column of its training size.
    """
    sizes = [str(size) for size in data["sizes"]]
    fig, axes = plt.subplots(
        2,
        len(sizes),
        figsize=FULL_SLIDE,
        layout="constrained",
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.02, h_pad=0.04, hspace=0.04)
    x_grammar = [grammar["topological_entropy_bits"] for grammar in data["grammars"]]
    x_level = [level["topological_entropy_bits"] for level in data["levels"]]
    series = [
        (FIRST_ORDER, FIRST_ORDER_COLOR, data["first_order"]),  # the reference model first
        *((MODEL_NAMES[m], MODEL_COLORS[m], by_size) for m, by_size in data["models"].items()),
    ]
    shown = live is not None and str(live["n_train"]) in sizes
    stretches = level_stretches(data["segments"])
    for row, (measure, anchor, anchor_name, label) in enumerate(COMPLEXITY_ROWS):
        lowest = anchor
        for column, size in enumerate(sizes):
            ax = axes[row, column]
            for value in (anchor, 1.0):
                ax.axhline(
                    value, color=ANCHOR_COLOR, ls=ANCHOR_STYLE, lw=ANCHOR_LINEWIDTH, zorder=1
                )
            for _, color, by_size in series:
                entry = by_size[size][measure]
                for style, first, last in stretches:
                    ax.plot(
                        x_level[first : last + 1],
                        entry["level"][first : last + 1],
                        style,
                        color=color,
                        lw=SERIES_LINEWIDTH,
                    )
                ax.plot(x_grammar, entry["grammar"], "o", color=color, ms=3, mec="none", alpha=0.85)
                lowest = min(lowest, *entry["grammar"], *entry["level"])
            if shown and str(live["n_train"]) == size:
                for model in data["models"]:
                    if model not in live:
                        continue
                    ax.plot(
                        live["topological_entropy_bits"],
                        live[model][measure],
                        "*",
                        ms=9,
                        color=MODEL_COLORS[model],
                        mec=COLORS["ink"],
                        mew=0.6,
                        zorder=3,
                    )
                    lowest = min(lowest, live[model][measure])
            if row == 0:
                ax.set_title(f"{size} training strings", pad=13 if data["placeholder"] else 5)
            if data["placeholder"]:
                mark_placeholder(ax, PLACEHOLDER_TEXT_PILOT)
        span = 1.0 - lowest  # both anchors, and every value below the lower one, are in the range
        axes[row, 0].set_ylim(lowest - 0.08 * span, 1.0 + 0.16 * span)
        axes[row, 0].set_ylabel(label)
        for value, name in ((anchor, anchor_name), (1.0, "exact predictor")):
            axes[row, 0].annotate(
                name,
                xy=(0.98, value),
                xycoords=axes[row, 0].get_yaxis_transform(),
                xytext=(0, 1.5),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=ANCHOR_FONTSIZE,
                color=ANCHOR_COLOR,
            )
    width = max(x_grammar) - min(x_grammar)
    axes[0, 0].set_xlim(min(x_grammar) - 0.08 * width - 0.02, max(x_grammar) + 0.08 * width + 0.02)
    fig.supxlabel(
        "topological entropy of the grammar (bits per symbol)",
        fontsize=plt.rcParams["axes.labelsize"],
    )
    panel_label(axes[0, 0], "A", x=-0.3, y=1.2)
    panel_label(axes[1, 0], "B", x=-0.3, y=1.08)

    handles = [
        Line2D([], [], color=color, lw=SERIES_LINEWIDTH, marker="o", ms=3, label=name)
        for name, color, _ in (*series[1:], series[0])
    ]
    if any(segment["dashed"] for segment in data["segments"]):
        handles.append(
            Line2D(
                [],
                [],
                color=COLORS["ink"],
                lw=SERIES_LINEWIDTH,
                ls=NOT_RUN_STYLE,
                label=NOT_RUN_LABEL,
            )
        )
    if shown:
        handles.append(
            Line2D(
                [],
                [],
                color=COLORS["ink"],
                marker="*",
                ls="none",
                ms=9,
                mfc="none",
                mew=0.6,
                label="trained here",
            )
        )
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), handlelength=2.2)
    return fig


SURROGATE_COLORS = {
    "data": COLORS["estimate"],
    "first-order": COLORS["second"],
    "shuffled": COLORS["limit"],
}
SURROGATE_NAMES = {
    "data": "data",
    "first-order": "first-order surrogate",
    "shuffled": "shuffled",
}
MI_FLOOR = 1e-4  # bits; smaller values (and negative ones) are not drawn on the logarithmic axis


def bold(text: str) -> str:
    """A label in bold through the math renderer, which both matplotlib and LaTeX accept."""
    return rf"$\mathbf{{{text.replace(' ', chr(92) + ' ')}}}$"


def figure_behaviour(data: dict):
    """The figure of notebook B: each dataset next to its two surrogates, in one column of two panels.

    Upper row: entropy of the next symbol given the k symbols before it, against k. The grey line is
    the closed form of the first-order surrogate, so the distance between that line and the
    surrogate's own points is the bias of the estimate. Lower row: mutual information between two
    symbols against their distance, with the band of shuffled sequences in grey.

    Parameters
    ----------
    data : dict
        From ``tutorial.behaviour_measures.figure_data``: ``datasets`` (the columns, in order),
        ``labels``, ``highlight`` (columns drawn with a heavy frame and a bold label: the chosen
        dataset and a participant's own sequence), and per dataset ``entropy_rate`` (one list per
        set, by order), ``closed_form``, ``markov_order`` and ``mi`` (``distances``, ``data``,
        ``first-order``, ``band_lower``, ``band_upper``).
    """
    names = data["datasets"]
    fig, axes = plt.subplots(2, len(names), figsize=FULL_SLIDE, layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.03, wspace=0.03, h_pad=0.04, hspace=0.06)
    for column, name in enumerate(names):
        upper, lower = axes[0, column], axes[1, column]
        rates = data["entropy_rate"][name]
        orders = np.arange(len(rates["data"]))
        # references first and in grey: the closed form of the surrogate, and the shuffled sequence
        upper.plot(
            orders[1:],
            np.full(len(orders) - 1, data["closed_form"][name]),
            color=COLORS["reference"],
            lw=REFERENCE_LINEWIDTH,
        )
        upper.plot(
            orders, rates["shuffled"], "--", color=SURROGATE_COLORS["shuffled"], lw=LIMIT_LINEWIDTH
        )
        upper.plot(
            orders, rates["first-order"], "s-", color=SURROGATE_COLORS["first-order"], ms=3.5
        )
        upper.plot(orders, rates["data"], "o-", color=SURROGATE_COLORS["data"], ms=3.5)
        upper.set_xticks(orders)
        upper.set_ylim(bottom=0)
        upper.set_xlabel(r"order $k$")
        upper.text(
            0.32,
            0.35,
            f"BIC: order {data['markov_order'][name]['data']}",
            transform=upper.transAxes,
            ha="left",
            va="center",
            fontsize=7,
            color=COLORS["ink"],
        )

        mi = data["mi"][name]
        distances = np.asarray(mi["distances"])
        edge = np.maximum(np.asarray(mi["band_upper"], dtype=float), MI_FLOOR)
        lower.fill_between(distances, MI_FLOOR, edge, color="0.85", lw=0)
        for kind, marker in (("first-order", "s-"), ("data", "o-")):
            values = np.asarray(mi[kind], dtype=float)
            lower.plot(
                distances,
                np.where(values > MI_FLOOR, values, np.nan),
                marker,
                color=SURROGATE_COLORS[kind],
                ms=3,
            )
        lower.set_yscale("log")
        lower.set_ylim(MI_FLOOR, 10)
        lower.set_xticks([1, *range(5, int(distances[-1]) + 1, 5)])
        lower.set_xlabel("distance (symbols)")

        chosen = name in data["highlight"]
        label = data["labels"][name]
        upper.text(
            0.5,
            1.05,
            bold(label) if chosen else label,
            transform=upper.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["ink"] if chosen else "0.35",
        )
        if chosen:
            for ax in (upper, lower):
                for side in ("left", "bottom"):
                    ax.spines[side].set_linewidth(1.6)
                    ax.spines[side].set_color(COLORS["ink"])
    axes[0, 0].set_ylabel(r"entropy rate $h_k$ (bits per symbol)")
    axes[1, 0].set_ylabel("mutual information (bits)")
    left = -0.02 - 0.1 * len(names)  # narrower panels need the letters further out
    panel_label(axes[0, 0], "A", x=left, y=1.2)
    panel_label(axes[1, 0], "B", x=left, y=1.2)

    handles = [
        Line2D([], [], color=SURROGATE_COLORS["data"], marker="o", ms=3.5, label="data"),
        Line2D(
            [],
            [],
            color=SURROGATE_COLORS["first-order"],
            marker="s",
            ms=3.5,
            label="first-order surrogate",
        ),
        Line2D(
            [],
            [],
            color=SURROGATE_COLORS["shuffled"],
            ls="--",
            lw=LIMIT_LINEWIDTH,
            label="shuffled",
        ),
        Line2D(
            [],
            [],
            color=COLORS["reference"],
            lw=REFERENCE_LINEWIDTH,
            label="closed form of the surrogate",
        ),
        Line2D([], [], color="0.85", lw=5, label="band of shuffles"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), handlelength=1.5)
    return fig


GRAPH_LABEL_PT = 8  # node labels of the grammar graph, the size of axis labels


def draw_grammar_graph(ax, symbols: list[str], allowed) -> None:
    """The graph of allowed transitions on an axis, drawn by SymSeq's own class for Markov chains.

    ``symseq.viz.mc_graph.MarkovChain`` is the class behind SymSeq's ``plot_grammar``. It is called
    directly, on the table of the symbols alone, for two reasons: ``plot_grammar`` draws the grammar
    object, in which the end marker is a third state, and at the pinned commit it fails when it is
    given an axis. Probabilities are not written on the arrows, because the number of allowed words
    depends only on which transitions exist. SymSeq puts the first state on the right, so the
    states are passed in reverse order to have the first symbol on the left.
    """
    order = list(range(len(symbols)))[::-1]
    table = np.asarray(allowed, dtype=float)[np.ix_(order, order)]
    figure = ax.get_figure()
    # SymSeq scales its fonts by figure width / 10 and rounds up; this base gives GRAPH_LABEL_PT
    base = (GRAPH_LABEL_PT - 0.5) * 10.0 / figure.get_size_inches()[0]
    with np.printoptions():  # the class changes numpy's print options; they are restored here
        chain = MarkovChain(
            table,
            [symbols[i] for i in order],
            annotate=False,
            transparency_func=lambda probability: 1.0,
            facecolor=COLORS["ink"],
            edgecolor=COLORS["ink"],
            ring_facecolor=COLORS["reference"],
            ring_edgecolor=COLORS["reference"],
            arrow_facecolor=COLORS["reference"],
            arrow_edgecolor=COLORS["reference"],
            node_fontsize=base,
            fontsize=base,
        )
        chain.draw(ax=ax, fig=figure)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.3, 2.2)
    ax.set_aspect("equal")


def draw_grammar(grammar, size: tuple[float, float] = (3.6, 3.6)):
    """A SymSeq grammar as a graph of its states, drawn by SymSeq's own class for Markov chains.

    Every state is a node labelled with its name (``A(0)`` and ``A(1)`` are two states that emit
    A), the end marker is the node with the double ring, and an arrow is an allowed transition.
    ``symseq.viz.mc_graph.MarkovChain`` is the class behind SymSeq's ``plot_grammar``; it is called
    directly because ``plot_grammar`` fails at the pinned commit when it is given an axis.
    Probabilities are not written: all transitions out of a state are equally probable.
    """
    names = [state.tostring() for state in grammar.states]
    table = np.asarray(grammar.transition_table).T  # SymSeq stores target x source
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    # SymSeq scales its fonts by figure width / 10 and rounds up; this base gives 7 pt labels
    base = 6.5 * 10.0 / size[0]
    with np.printoptions():  # the class changes numpy's print options; they are restored here
        chain = MarkovChain(
            table,
            names,
            annotate=False,
            transparency_func=lambda probability: 1.0,
            facecolor=COLORS["ink"],
            edgecolor=COLORS["ink"],
            ring_facecolor=COLORS["reference"],
            ring_edgecolor=COLORS["reference"],
            arrow_facecolor=COLORS["reference"],
            arrow_edgecolor=COLORS["reference"],
            node_fontsize=base,
            fontsize=base,
            arrow_width=0.04,
            arrow_head_width=0.28,
        )
        chain.draw(eos=grammar.eos, ax=ax, fig=fig)
    ax.set_aspect("equal")
    return fig


def figure_worked_example_words(data: dict, graph: bool = True):
    """Worked example, for the slides: the number of allowed words against their length.

    Exact values only. The counts lie on a logarithmic axis whose ticks are the counts themselves
    (2, 3, 5, 8, ...); the grey line is phi^(L+2) / sqrt(5), whose slope is the topological entropy.
    Beside the plot, the graph of the grammar.

    Parameters
    ----------
    data : dict
        From ``tutorial.worked_example.words_panel``: ``length``, ``words`` (exact counts), ``line``
        (the line at the same lengths), ``topological_entropy_bits`` and ``topological_entropy_nats``
        (its slope in both units), ``symbols`` and ``allowed`` (the graph).
    graph : bool
        Draw the graph of the grammar beside the plot (the default). Without it the plot has the
        half slide to itself.
    """
    if graph:
        fig, (ax, ax_graph) = plt.subplots(
            1, 2, figsize=HALF_SLIDE, layout="constrained", gridspec_kw={"width_ratios": [2.4, 1.0]}
        )
    else:
        fig, ax = plt.subplots(figsize=HALF_SLIDE, layout="constrained")
    lengths = np.asarray(data["length"])
    # the reference first and in grey: the line the counts follow
    ax.plot(lengths, data["line"], color=COLORS["reference"], lw=REFERENCE_LINEWIDTH)
    ax.plot(lengths, data["words"], "o", color=COLORS["ink"])
    ax.set_yscale("log", base=2)
    ax.set_yticks(data["words"])
    ax.set_yticklabels([str(count) for count in data["words"]])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xticks(lengths)
    ax.set_xlim(lengths[0] - 0.5, lengths[-1] + 0.5)
    ax.set_xlabel(r"word length $L$ (symbols)")
    ax.set_ylabel("number of allowed words")
    ax.text(
        0.04,
        0.97,
        "slope of the line:\n"
        rf"$\log_2 \varphi = {data['topological_entropy_bits']:.3f}$ bit per symbol"
        "\n"
        rf"$\ln \varphi = {data['topological_entropy_nats']:.3f}$ nat per symbol",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=COLORS["ink"],
    )
    if graph:
        draw_grammar_graph(ax_graph, data["symbols"], data["allowed"])
    return fig


def figure_worked_example_block_entropy(data: dict):
    """Worked example, for the slides: block entropy H(L) against the block length L.

    Exact values only. The grey line has the entropy rate h as its slope and meets L = 0 at the
    memory E, which is marked there.

    Parameters
    ----------
    data : dict
        From ``tutorial.worked_example.block_entropy_panel``: ``length``, ``block_entropy`` (H(L) in
        bits), ``entropy_rate`` (bits per symbol) and ``excess_entropy`` (bits).
    """
    fig, ax = plt.subplots(figsize=HALF_SLIDE, layout="constrained")
    lengths = np.asarray(data["length"])
    rate, memory = data["entropy_rate"], data["excess_entropy"]
    # the reference first and in grey: the straight line of slope h and intercept E
    line_at = np.array([0.0, lengths[-1]])
    ax.plot(line_at, memory + rate * line_at, color=COLORS["reference"], lw=REFERENCE_LINEWIDTH)
    ax.plot(lengths, data["block_entropy"], "o", color=COLORS["ink"])
    ax.plot(
        [0], [memory], "o", ms=6, mfc="white", mew=1.2, color=COLORS["reference"], clip_on=False
    )
    top = memory + rate * lengths[-1]
    ax.annotate(
        rf"memory $E = {memory:.4f}$ bit",
        xy=(0, memory),
        xytext=(0.9, memory + 0.02 * top),
        ha="left",
        va="center",
        fontsize=7,
        color=COLORS["ink"],
    )
    ax.text(
        0.05,
        0.95,
        rf"slope $h = {rate:.4f}$ bit per symbol",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=COLORS["ink"],
    )
    ax.set_xticks([0, *lengths])
    ax.set_xlim(0, lengths[-1] + 0.5)
    ax.set_ylim(0, 1.08 * top)
    ax.set_xlabel(r"block length $L$ (symbols)")
    ax.set_ylabel(r"block entropy $H(L)$ (bits)")
    return fig
