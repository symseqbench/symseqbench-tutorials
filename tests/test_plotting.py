"""The plot style works without LaTeX and keeps the sizes of the canonical style. Each test has a case it must reject."""

from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from symseq.metrics.grammar import topological_entropy

from tutorial import closed_forms as cf
from tutorial import complexity_runs, config, plotting, worked_example
from tutorial.generate import two_state_grammar

CANONICAL = Path.home() / "repos" / "hetsyn" / "defaults" / "figure_style.py"


def differences(ours: dict, canonical: dict) -> list[str]:
    """Keys of our style whose value differs from the canonical rcParams."""
    return [key for key, value in ours.items() if key in canonical and canonical[key] != value]


def follows_the_style_note(rc: dict) -> bool:
    """Ticks 7 pt, labels 8 pt, lines 1.5, spines 0.8, 300 dpi, no top and right spine."""
    return (
        (rc["xtick.labelsize"], rc["ytick.labelsize"], rc["axes.labelsize"]) == (7, 7, 8)
        and (rc["lines.linewidth"], rc["axes.linewidth"], rc["savefig.dpi"]) == (1.5, 0.8, 300)
        and not rc["axes.spines.top"]
        and not rc["axes.spines.right"]
    )


def test_sizes_are_those_of_the_style_note():
    assert follows_the_style_note(plotting.STYLE_RC)
    # must reject: one size changed, and matplotlib's own defaults
    assert not follows_the_style_note({**plotting.STYLE_RC, "xtick.labelsize": 8})
    assert not follows_the_style_note(matplotlib.rcParamsDefault)
    assert plotting.FULL_SLIDE == (7.2, 3.8) and plotting.HALF_SLIDE == (3.5, 3.0)
    assert plotting.COLORS["estimate"] == "#0173B2" and plotting.COLORS["reference"] == "0.45"
    assert (plotting.REFERENCE_LINEWIDTH, plotting.LIMIT_LINEWIDTH) == (1.2, 1.0)


@pytest.mark.skipif(
    not CANONICAL.exists(), reason="the canonical style file is only on the author's workstation"
)
def test_sizes_and_palette_equal_the_canonical_style():
    spec = importlib.util.spec_from_file_location("canonical_figure_style", CANONICAL)
    canonical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical)
    with matplotlib.rc_context():
        canonical.apply_style(use_latex=False)
        reference = dict(matplotlib.rcParams)
    shared = [key for key in plotting.STYLE_RC if key in reference]
    assert len(shared) == len(plotting.STYLE_RC)
    assert differences(plotting.STYLE_RC, reference) == []
    assert plotting.WONG_PALETTE == canonical.WONG_PALETTE
    # must reject: one changed size
    assert differences({**plotting.STYLE_RC, "axes.labelsize": 9}, reference) == ["axes.labelsize"]


def test_default_style_renders_without_latex(work_dir, monkeypatch):
    monkeypatch.setattr(plotting, "latex_available", lambda: False)
    with matplotlib.rc_context():
        plotting.apply_style()
        assert matplotlib.rcParams["text.usetex"] is False
        assert matplotlib.rcParams["mathtext.fontset"] == "cm"
        assert matplotlib.rcParams["font.serif"][0] == "cmr10"
        fig, ax = plt.subplots(figsize=plotting.HALF_SLIDE)
        ax.plot([1, 10, 100], [0.5, -0.6, 0.7], label=r"entropy rate $h$")
        ax.set_xscale("log")
        ax.set_xlabel(r"$p(\mathrm{A} \rightarrow \mathrm{B})$")
        ax.legend()
        plotting.panel_label(ax, "A")
        written = plotting.save_figure(fig, work_dir, "style")
        plt.close(fig)
    assert sorted(written) == ["pdf", "png", "png_onwhite", "svg"]
    assert all(path.stat().st_size > 1000 for path in written.values())
    # must reject: LaTeX requested where there is none
    with pytest.raises(RuntimeError, match="LaTeX"):
        plotting.apply_style(use_latex=True)


@pytest.mark.skipif(not plotting.latex_available(), reason="no LaTeX on this machine")
def test_latex_option_renders_where_latex_exists(work_dir):
    with matplotlib.rc_context():
        plotting.apply_style(use_latex=True)
        assert matplotlib.rcParams["text.usetex"] is True
        fig, ax = plt.subplots(figsize=plotting.HALF_SLIDE)
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel(r"$p(\mathrm{A} \rightarrow \mathrm{B})$")
        plotting.panel_label(ax, "A")
        written = plotting.save_figure(fig, work_dir, "latex", formats=("png",))
        plt.close(fig)
        assert written["png"].stat().st_size > 1000
        # must reject: a label that LaTeX cannot compile
        fig, ax = plt.subplots(figsize=plotting.HALF_SLIDE)
        ax.set_xlabel(r"$\nosuchcommand{x}$")
        with pytest.raises(RuntimeError):
            plotting.save_figure(fig, work_dir, "broken", formats=("png",))
        plt.close(fig)


def test_notebook_figure_has_four_panels_and_no_title(work_dir):
    convergence = {
        "n": [300, 1000, 3000],
        "entropy_rate": [0.60, 0.65, 0.66],
        "excess_entropy": [0.30, 0.26, 0.25],
        "entropy_rate_exact": 2 / 3,
        "excess_entropy_exact": 0.2516,
    }
    grid = np.linspace(0.01, 0.99, 50)
    curve = {
        "p": grid,
        "entropy_rate": grid * 0.5,
        "p_estimated": [0.2, 0.5],
        "entropy_rate_estimated": [0.6, 0.66],
        "topological_entropy": 0.694,
        "p_max": 0.382,
    }
    prescribed = {
        "target_bits": 1.4,
        "steps": [0, 10, 20],
        "entropy_bits": [0.8, 1.2, 1.39],
        "score": [0.5, 0.9, 0.99],
        "achieved_open_bits": 1.39,
        "closed_bits": 1.44,
        "memory_gap": 0.07,
        "mean_length": 9.1,
        "n_edges": 22,
    }
    with matplotlib.rc_context():
        plotting.apply_style()
        fig = plotting.figure_generate_and_measure(convergence, curve, prescribed)
        assert len(fig.axes) == 4
        assert tuple(fig.get_size_inches()) == plotting.FULL_SLIDE
        assert all(ax.get_title() == "" for ax in fig.axes) and fig._suptitle is None
        data = plotting.save_figure_data(
            {"convergence": convergence, "curve": curve, "prescribed": prescribed},
            work_dir,
            "figure",
        )
        plt.close(fig)
    assert data.read_text().count("entropy_rate") >= 3
    with pytest.raises(KeyError):
        plotting.figure_generate_and_measure({}, curve, prescribed)  # must reject: missing data
    plt.close("all")


def distance_data(kind: str = "production") -> dict:
    """Three grammars, two models, two seeds: distances that fall over six epochs."""
    epochs = list(range(6))
    grammars = ["two-state", "4-AX (p_target 0.9)", "nested"]

    def curves(scale: float) -> list[list[float]]:
        return [[scale * 0.5**epoch + 0.002 * (seed + 1) for epoch in epochs] for seed in range(2)]

    models = {}
    for model, scale in (("rnn", 1.0), ("linear transformer", 2.0)):
        distance = [curves(scale * (index + 1)) for index in range(len(grammars))]
        distance[0][1][5] = -0.0004  # an estimate below zero: possible by chance, not drawn
        models[model] = {
            "seeds": [[1, 2]] * len(grammars),
            "distance": distance,
            "mean": [np.mean(per_grammar, axis=0).tolist() for per_grammar in distance],
            "final_standard_error": [0.0005] * len(grammars),
        }
    return {
        "kind": kind,
        "grammars": grammars,
        "epochs": [epochs] * len(grammars),
        "n_targets": [16000] * len(grammars),
        "resolution": [8.0 / (16000 * math.log(2.0))] * len(grammars),
        "exact_cross_entropy": [0.5, 1.0, 0.2],
        "models": models,
    }


def test_networks_figure_shows_distance_from_the_exact_predictor_per_grammar():
    with matplotlib.rc_context():
        plotting.apply_style()
        data = distance_data()
        fig = plotting.figure_networks(data)
        panels = [ax for ax in fig.axes if ax.axison]
        assert (
            len(panels) == 3 and len(fig.axes) == 4
        )  # one panel per grammar, the fourth place empty
        assert tuple(fig.get_size_inches()) == plotting.FULL_SLIDE
        assert all(ax.get_title() == "" for ax in fig.axes) and fig._suptitle is None
        assert "distance from the exact predictor" in fig.get_supylabel()
        assert "bits per target" in fig.get_supylabel()
        # one logarithmic scale for all grammars
        assert all(ax.get_yscale() == "log" for ax in panels)
        assert (
            len({ax.get_ylim() for ax in panels}) == 1
            and panels[0].get_ylim()[0] == plotting.DISTANCE_FLOOR
        )
        names = [ax.texts[0].get_text() for ax in panels]
        assert names == ["two-state", "4-AX (target 0.9)", "nested"]
        # LaTeX does not accept an underscore in text. must reject: the names of the grammars as they are
        assert not any("_" in plotting.tick_label(name) for name in config.CANDIDATES)
        assert any("_" in name for name in config.CANDIDATES)
        for index, ax in enumerate(panels):
            lines = ax.get_lines()
            # the reference first: the resolution of the held-out samples, grey and dashed
            assert (
                lines[0].get_linestyle() == "--"
                and lines[0].get_color() == plotting.COLORS["limit"]
            )
            assert lines[0].get_ydata()[0] == pytest.approx(data["resolution"][index])
            thin = [line for line in lines[1:] if line.get_linewidth() < 1.0]
            thick = [line for line in lines[1:] if line.get_linewidth() > 1.0]
            assert len(thin) == 4 and len(thick) == 2  # two seeds and one mean per network
            for line, model in zip(thick, data["models"], strict=True):
                assert line.get_color() == plotting.MODEL_COLORS[model]
                assert np.allclose(line.get_ydata(), data["models"][model]["mean"][index])
        # a distance at or below the floor is not drawn, and nothing is drawn in its place
        drawn = panels[0].get_lines()[2].get_ydata()
        assert np.isnan(drawn[5]) and not np.isnan(drawn[:5]).any()
        legend = [text.get_text() for text in fig.legends[0].get_texts()]
        assert legend == [
            "tanh RNN, mean",
            "linear transformer, mean",
            "single seeds",
            "resolution of the held-out samples",
        ]
        texts = [text.get_text() for ax in fig.axes for text in ax.texts]
        assert plotting.PLACEHOLDER_TEXT not in texts
        # must reject: no axis shows raw accuracy or raw cross-entropy
        labels = " ".join([fig.get_supylabel(), *(ax.get_ylabel() for ax in fig.axes)])
        assert "accuracy" not in labels and "cross-entropy" not in labels
        plt.close(fig)

        # must reject: results that are not production runs carry the word in every panel
        smoke = plotting.figure_networks(distance_data("smoke"))
        marks = [text.get_text() for ax in smoke.axes for text in ax.texts]
        assert marks.count(plotting.PLACEHOLDER_TEXT) == 3
        assert "placeholder" in plotting.PLACEHOLDER_TEXT
        plt.close(smoke)
        with pytest.raises(KeyError):
            plotting.figure_networks({"kind": "smoke", "grammars": ["a"]})  # must reject: no data
    plt.close("all")


def test_distance_data_is_the_excess_over_the_exact_predictor_on_the_same_symbols():
    """Every number of the six-grammar figure, recomputed here from the run files."""
    import json

    from tutorial import results

    folder = results.FOLDERS["production"]
    saved, manifest = results.load_runs(folder)
    data = results.distance_data(saved, config.LADDER, manifest["kind"])
    assert data["grammars"] == list(config.LADDER) and data["kind"] == "production"
    files = [json.loads(path.read_text()) for path in sorted((folder / "runs").glob("*.json"))]
    old_reference_differs = 0
    for index, grammar in enumerate(config.LADDER):
        for model in config.MODELS:
            of_model = sorted(
                (run for run in files if (run["rung"], run["model"]) == (grammar, model)),
                key=lambda run: run["seed"],
            )
            assert (
                [run["seed"] for run in of_model]
                == [1, 2, 3]
                == data["models"][model]["seeds"][index]
            )
            for run, curve in zip(of_model, data["models"][model]["distance"][index], strict=True):
                realised = run["optimum"]["cross_entropy_realised"]
                assert curve == [value - realised for value in run["history"]["cross_entropy"]]
                assert len(curve) == 31 and curve[0] > 0.5 > curve[-1]
                # must reject: the distance from the EXPECTED cross-entropy of the exact predictor
                expected = run["optimum"]["cross_entropy"]
                old_reference_differs += (
                    abs(curve[-1] - (run["final"]["cross_entropy"] - expected)) > 1e-4
                )
            mean = np.mean(data["models"][model]["distance"][index], axis=0)
            assert np.allclose(data["models"][model]["mean"][index], mean)
        assert data["n_targets"][index] == of_model[0]["optimum"]["n_targets"]
        assert data["resolution"][index] == pytest.approx(
            8.0 / (data["n_targets"][index] * math.log(2.0))
        )
    # the two references differ only where an open choice is not between equally probable symbols:
    # in 4-AX (target probability 0.9) and in the Dyck language, six runs each
    assert old_reference_differs == 12
    table = results.distance_table(data)
    assert table.shape[0] == 12 and (table["largest"] >= table["smallest"]).all()
    # what the redrawn figure shows: every network ends within 0.03 bit of the exact predictor
    assert 0.0 < table["distance (bits per target)"].max() < 0.03
    # must reject: runs that do not share one exact predictor, and a grammar without runs
    tampered = [dict(run) for run in saved]
    tampered[0] = {
        **tampered[0],
        "optimum": {**tampered[0]["optimum"], "cross_entropy_realised": 0.0},
    }
    with pytest.raises(ValueError, match="do not share one exact predictor"):
        results.distance_data(tampered, config.LADDER, "production")
    with pytest.raises(ValueError, match="No saved run"):
        results.distance_data(
            [r for r in saved if r["rung"] != "nested"], config.LADDER, "production"
        )


def spread_of_the_distance(
    mixing: float, n_targets: int = 2000, repeats: int = 4000
) -> tuple[float, float]:
    """The mean divergence of a model q near p, and the spread of its estimate over repeated samples.

    q is p with a share ``mixing`` of another distribution, in 30 equally frequent contexts. The
    estimate of Eq. (1) is taken about the mean divergence of the contexts that were drawn, which
    is what the standard error of Eq. (2) describes.
    """
    rng = np.random.default_rng(101)
    n_contexts, n_classes = 30, 5
    p = rng.dirichlet(np.ones(n_classes), size=n_contexts)
    q = (1.0 - mixing) * p + mixing * rng.dirichlet(np.ones(n_classes), size=n_contexts)
    ratio = np.log2(p / q)
    per_context = (p * ratio).sum(axis=1)
    estimates = []
    for _ in range(repeats):
        contexts = rng.integers(n_contexts, size=n_targets)
        symbols = (rng.random(n_targets)[:, None] > np.cumsum(p[contexts], axis=1)).sum(axis=1)
        estimates.append(ratio[contexts, symbols].mean() - per_context[contexts].mean())
    return float(per_context.mean()), float(np.std(estimates))


def test_standard_error_of_a_distance_agrees_with_repeated_sampling():
    """Eq. (2) of ``tutorial.results``: sqrt(2 D / (n ln 2)), against the spread of 4000 estimates."""
    from tutorial import results

    n_targets = 2000
    # close to p the formula holds: within 10 % of the spread found
    divergence, spread = spread_of_the_distance(mixing=0.02)
    formula = results.distance_standard_error(divergence, n_targets)
    assert divergence < 0.002 and abs(formula / spread - 1.0) < 0.10
    # must reject: twice the variance, and half of it, are outside the 10 %
    assert abs(math.sqrt(2.0) * formula / spread - 1.0) > 0.10
    assert abs(formula / math.sqrt(2.0) / spread - 1.0) > 0.10
    # further from p it is an approximation: at the divergences of the saved runs it is within 25 %
    divergence, spread = spread_of_the_distance(mixing=0.1)
    formula = results.distance_standard_error(divergence, n_targets)
    assert 0.01 < divergence < 0.03 and abs(formula / spread - 1.0) < 0.25
    assert results.distance_resolution(n_targets) == pytest.approx(
        8.0 / (n_targets * math.log(2.0))
    )
    resolution = results.distance_resolution(n_targets)
    assert results.distance_standard_error(resolution, n_targets) == pytest.approx(resolution / 2.0)
    assert results.distance_standard_error(-0.001, n_targets) == 0.0


def complexity_data(
    placeholder: bool = False, lowest_score: float = 0.3, dashed: tuple[int, ...] = ()
) -> dict:
    """Three levels of two grammars: the score and the area fall as the entropy rises.

    ``lowest_score`` is the score of the first-order model on the last grammar at 50 strings.
    ``dashed`` holds the positions of the segments between neighbouring levels that cross a band
    which was not run (0: first to second level, 1: second to third).
    """
    grammars = [
        {
            "name": f"g{level}{rank}",
            "level": level,
            "seed": 101 + rank,
            "topological_entropy_bits": 1.3 + 0.3 * level + 0.04 * rank,
        }
        for level in range(3)
        for rank in range(2)
    ]

    def entry(values: list[float]) -> dict:
        return {
            "seeds": [[value] for value in values],
            "grammar": values,
            "level": [float(np.mean(values[2 * level : 2 * level + 2])) for level in range(3)],
        }

    def by_size(offset: float, last: float | None = None) -> dict:
        out = {}
        for size in (50, 200, 800):
            scores = [0.95 - 0.04 * i - offset - 10 / size for i in range(6)]
            if last is not None and size == 50:
                scores[-1] = last
            out[str(size)] = {
                "score": entry(scores),
                "auc": entry([0.99 - 0.03 * i - offset / 2 - 5 / size for i in range(6)]),
                "divergence_bits": entry(
                    [(0.02 + 0.03 * i + offset) * 50 / size for i in range(6)]
                ),
                "top_choice_ratio": entry([0.9] * 6),
            }
        return out

    return {
        "kind": "pilot" if placeholder else "production",
        "placeholder": placeholder,
        "sizes": [50, 200, 800],
        "seeds": [101],
        "grammars": grammars,
        "levels": [
            {
                "level": level,
                "band_bits": [1.25 + 0.3 * level, 1.4 + 0.3 * level],
                "topological_entropy_bits": 1.32 + 0.3 * level,
            }
            for level in range(3)
        ],
        "segments": [
            {
                "levels": [position, position + 1],
                # the bands of the levels end at 1.4 + 0.3 * level and begin at 1.25 + 0.3 * level
                "bands_not_run": [[1.45 + 0.3 * position, 1.5 + 0.3 * position]]
                if position in dashed
                else [],
                "dashed": position in dashed,
            }
            for position in range(2)
        ],
        "models": {"rnn": by_size(0.0), "linear transformer": by_size(0.02)},
        "first_order": by_size(0.2, last=lowest_score),
    }


def complexity_figure_problems(fig, data: dict) -> list[str]:
    """Everything in which a figure departs from the layout of the complexity figure; empty if nothing."""
    problems = []
    sizes = [str(size) for size in data["sizes"]]
    x_grammar = [grammar["topological_entropy_bits"] for grammar in data["grammars"]]
    x_level = [level["topological_entropy_bits"] for level in data["levels"]]
    series = {
        "tanh RNN": (plotting.MODEL_COLORS["rnn"], data["models"]["rnn"]),
        "linear transformer": (
            plotting.MODEL_COLORS["linear transformer"],
            data["models"]["linear transformer"],
        ),
        "first-order model": (plotting.COLORS["reference"], data["first_order"]),
    }
    model_colours = {matplotlib.colors.to_hex(colour) for colour, _ in series.values()}
    rows = (("score", 0.0, "uniform guess", "upper"), ("auc", 0.5, "chance", "lower"))
    if len(fig.axes) != 2 * len(sizes):
        return [f"{len(fig.axes)} panels"]
    for row, (measure, anchor, anchor_name, where) in enumerate(rows):
        panels = fig.axes[row * len(sizes) : (row + 1) * len(sizes)]
        if len({ax.get_ylim() for ax in panels}) != 1 or panels[0].get_yscale() != "linear":
            problems.append(f"the vertical axis of the {where} row is not shared and linear")
        for column, (size, ax) in enumerate(zip(sizes, panels, strict=True)):
            here = f"{where} row, {size} strings"
            title = f"{size} training strings" if row == 0 else ""
            if ax.get_title() != title:
                problems.append(f"{here}: title {ax.get_title()!r}")
            low, high = ax.get_ylim()
            anchors = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
            if sorted(float(line.get_ydata()[0]) for line in anchors) != [anchor, 1.0]:
                problems.append(f"{here}: the anchors are not {anchor} and 1")
            for line in anchors:
                if matplotlib.colors.to_hex(line.get_color()) in model_colours:
                    problems.append(f"{here}: an anchor has the colour of a model")
                if matplotlib.colors.to_hex(line.get_color()) != matplotlib.colors.to_hex(
                    plotting.COLORS["ink"]
                ):
                    problems.append(f"{here}: an anchor is not dark")
                if line.get_linewidth() >= 1.0:
                    problems.append(f"{here}: an anchor is not thin")
            if not (low < anchor and high > 1.0):
                problems.append(f"{here}: an anchor is outside the axis range")
            others = [line for line in ax.get_lines() if line.get_linestyle() != ":"]
            for name, (colour, by_size) in series.items():
                mine = [
                    line
                    for line in others
                    if matplotlib.colors.to_hex(line.get_color())
                    == matplotlib.colors.to_hex(colour)
                ]
                entry = by_size[size][measure]
                # the line through the level means: which neighbouring levels a line joins, and how
                joined: dict[int, list[str]] = {}
                strays = 0
                for line in mine:
                    if line.get_linestyle() not in ("-", "--"):
                        continue
                    xs, ys = line.get_xdata(), line.get_ydata()
                    pairs = 0
                    for k in range(len(xs) - 1):
                        for position in range(len(x_level) - 1):
                            if np.allclose(
                                [xs[k], xs[k + 1]], x_level[position : position + 2]
                            ) and np.allclose(
                                [ys[k], ys[k + 1]],
                                entry["level"][position : position + 2],
                                rtol=0.0,
                                atol=1e-12,
                            ):
                                joined.setdefault(position, []).append(line.get_linestyle())
                                pairs += 1
                    strays += pairs != len(xs) - 1  # a line that is not the level means
                every_pair_once = sorted(joined) == list(range(len(x_level) - 1)) and all(
                    len(styles) == 1 for styles in joined.values()
                )
                level = [1] if every_pair_once and not strays else []
                wanted_styles = {
                    position: ["--" if segment["dashed"] else "-"]
                    for position, segment in enumerate(data["segments"])
                }
                if level and joined != wanted_styles:
                    problems.append(
                        f"{here}: the line of {name} is not dashed exactly where a band was not run"
                    )
                points = [
                    line
                    for line in mine
                    if line.get_linestyle() == "None"
                    and len(line.get_xdata()) == len(x_grammar)
                    and np.allclose(line.get_xdata(), x_grammar)
                    and np.allclose(line.get_ydata(), entry["grammar"], rtol=0.0, atol=1e-12)
                ]
                if len(level) != 1 or len(points) != 1:
                    problems.append(f"{here}: {name} is missing or does not show its numbers")
                if min(entry["grammar"]) < low or max(entry["grammar"]) > high:
                    problems.append(f"{here}: a value of {name} is outside the axis range")
            drawn = {
                matplotlib.colors.to_hex(line.get_color())
                for line in others
                if line.get_marker() != "*"
            }
            if drawn != model_colours:
                problems.append(
                    f"{here}: the series are not the two networks and the first-order model"
                )
            names = {text.get_text() for text in ax.texts}
            wanted = {"exact predictor", anchor_name} if column == 0 else set()
            if names & {"exact predictor", "uniform guess", "chance"} != wanted:
                problems.append(f"{here}: the anchors are named {sorted(names)}")
    return problems


def test_complexity_figure_has_a_panel_per_training_size_and_named_anchors():
    with matplotlib.rc_context():
        plotting.apply_style()
        data = complexity_data()
        fig = plotting.figure_complexity(data)
        assert complexity_figure_problems(fig, data) == []
        assert tuple(fig.get_size_inches()) == plotting.FULL_SLIDE and fig._suptitle is None
        upper, lower = fig.axes[0], fig.axes[3]
        assert upper.get_ylabel() == "prediction score"
        assert "area under the ROC curve" in lower.get_ylabel()
        label = fig._supxlabel.get_text()
        assert "topological entropy" in label and "bits per symbol" in label
        assert fig._supxlabel.get_fontsize() == plt.rcParams["axes.labelsize"]
        legend = [text.get_text() for text in fig.legends[0].get_texts()]
        assert legend == ["tanh RNN", "linear transformer", "first-order model"]
        assert len(fig.legends) == 1 and all(ax.get_legend() is None for ax in fig.axes)
        # the anchors of a row are 0 and 1 for the score, 0.5 and 1 for the area, whatever the data
        assert upper.get_ylim()[0] < 0.0 < 1.0 < upper.get_ylim()[1]
        assert lower.get_ylim()[0] < 0.5 < 1.0 < lower.get_ylim()[1]
        texts = [text.get_text() for ax in fig.axes for text in ax.texts]
        assert plotting.PLACEHOLDER_TEXT_PILOT not in texts

        # must reject: the first-order model missing from the upper row, and from the lower row
        grey = matplotlib.colors.to_hex(plotting.COLORS["reference"])
        for panel, where in ((1, "upper row, 200 strings"), (5, "lower row, 800 strings")):
            broken = plotting.figure_complexity(data)
            for line in list(broken.axes[panel].get_lines()):
                if matplotlib.colors.to_hex(line.get_color()) == grey:
                    line.remove()
            found = complexity_figure_problems(broken, data)
            assert f"{where}: first-order model is missing or does not show its numbers" in found
            plt.close(broken)
        # must reject: a reference line drawn in a model's colour, and one drawn like a series
        for colour in (*plotting.MODEL_COLORS.values(), plotting.COLORS["reference"]):
            broken = plotting.figure_complexity(data)
            anchor = next(
                line for line in broken.axes[4].get_lines() if line.get_linestyle() == ":"
            )
            anchor.set_color(colour)
            found = complexity_figure_problems(broken, data)
            assert "lower row, 200 strings: an anchor has the colour of a model" in found
            plt.close(broken)
        broken = plotting.figure_complexity(data)
        next(
            line for line in broken.axes[0].get_lines() if line.get_linestyle() == ":"
        ).set_linestyle("-")
        assert complexity_figure_problems(broken, data) != []
        plt.close(broken)
        # must reject: a figure of other numbers, and the names of the anchors in every panel
        other = complexity_data()
        other["models"]["rnn"]["200"]["score"]["grammar"][2] += 1e-6
        assert complexity_figure_problems(fig, other) == [
            "upper row, 200 strings: tanh RNN is missing or does not show its numbers"
        ]
        broken = plotting.figure_complexity(data)
        broken.axes[2].text(1.5, 1.0, "exact predictor")
        assert complexity_figure_problems(broken, data) != []
        plt.close(broken)
        plt.close(fig)

        # a score below zero is shown: the range of the upper row goes down to it, and the anchors stay
        negative = complexity_data(lowest_score=-0.4)
        fig = plotting.figure_complexity(negative)
        assert complexity_figure_problems(fig, negative) == []
        assert fig.axes[0].get_ylim()[0] < -0.4 and fig.axes[0].get_ylim()[1] > 1.0
        assert fig.axes[3].get_ylim()[0] < 0.5  # the lower row keeps its own range
        # must reject: the same figure on an axis that starts at the uniform guess
        fig.axes[0].set_ylim(-0.05, 1.1)
        assert (
            "upper row, 50 strings: a value of first-order model is outside the axis range"
            in complexity_figure_problems(fig, negative)
        )
        plt.close(fig)

        # must reject: pilot results carry the words above every panel, and still show their numbers
        pilot_data = complexity_data(placeholder=True)
        pilot = plotting.figure_complexity(pilot_data)
        marks = [text.get_text() for ax in pilot.axes for text in ax.texts]
        assert marks.count(plotting.PLACEHOLDER_TEXT_PILOT) == 6
        assert plotting.PLACEHOLDER_TEXT_PILOT == "placeholder: pilot runs"
        assert complexity_figure_problems(pilot, pilot_data) == []
        plt.close(pilot)

        live = {
            "topological_entropy_bits": 1.6,
            "n_train": 200,
            "rnn": {"score": 0.5, "auc": 0.9},
            "linear transformer": {"score": -0.2, "auc": 0.85},
        }
        with_live = plotting.figure_complexity(data, live=live)
        assert [text.get_text() for text in with_live.legends[0].get_texts()][-1] == "trained here"
        for panel, measure in ((1, "score"), (4, "auc")):  # the column of 200 strings, both rows
            stars = [line for line in with_live.axes[panel].get_lines() if line.get_marker() == "*"]
            assert len(stars) == 2 and stars[0].get_xdata()[0] == 1.6
            assert sorted(star.get_ydata()[0] for star in stars) == sorted(
                live[model][measure] for model in ("rnn", "linear transformer")
            )
        assert not [
            line
            for panel in (0, 2, 3, 5)
            for line in with_live.axes[panel].get_lines()
            if line.get_marker() == "*"
        ]
        assert with_live.axes[0].get_ylim()[0] < -0.2  # the run trained here is inside the range
        assert complexity_figure_problems(with_live, data) == []
        plt.close(with_live)
        with pytest.raises(KeyError):
            plotting.figure_complexity(
                {"placeholder": False, "grammars": []}
            )  # must reject: no data
    plt.close("all")


def dashed_stretches(fig) -> list[list[tuple[float, ...]]]:
    """For every panel of a figure: the horizontal positions of each dashed line in it, sorted."""
    return [
        sorted(
            tuple(float(x) for x in line.get_xdata())
            for line in ax.get_lines()
            if line.get_linestyle() == "--"
        )
        for ax in fig.axes
    ]


def test_complexity_figure_dashes_the_line_across_a_band_that_was_not_run():
    # how the segments become stretches of one line style, by hand
    solid, dashed = {"dashed": False}, {"dashed": True}
    assert plotting.level_stretches([solid, solid, dashed]) == [("-", 0, 2), ("--", 2, 3)]
    assert plotting.level_stretches([dashed, solid]) == [("--", 0, 1), ("-", 1, 2)]
    assert plotting.level_stretches([solid, dashed, solid]) == [
        ("-", 0, 1),
        ("--", 1, 2),
        ("-", 2, 3),
    ]
    assert plotting.level_stretches([dashed, dashed]) == [("--", 0, 2)]
    assert plotting.level_stretches([solid] * 4) == [("-", 0, 4)]
    assert plotting.level_stretches([]) == [("-", 0, 0)]
    with matplotlib.rc_context():
        plotting.apply_style()
        # no band left out: every line solid, and the legend has nothing to say about dashes
        plain = complexity_data()
        fig = plotting.figure_complexity(plain)
        assert dashed_stretches(fig) == [[]] * 6
        assert plotting.NOT_RUN_LABEL not in [t.get_text() for t in fig.legends[0].get_texts()]

        # the band between the second and the third level was not run
        data = complexity_data(dashed=(1,))
        x_level = [level["topological_entropy_bits"] for level in data["levels"]]
        with_gap = plotting.figure_complexity(data)
        assert complexity_figure_problems(with_gap, data) == []
        # exactly one dashed stretch per series in every panel, between the right two levels
        assert dashed_stretches(with_gap) == [[(x_level[1], x_level[2])] * 3] * 6
        legend = [text.get_text() for text in with_gap.legends[0].get_texts()]
        assert legend == [
            "tanh RNN",
            "linear transformer",
            "first-order model",
            "across a band that was not run",
        ]
        handle = with_gap.legends[0].legend_handles[-1]
        assert handle.get_linestyle() == "--" and handle.get_marker() in ("None", "", None)
        # the dashes are not the dots of the anchors, and they have the colour of their series
        for ax in with_gap.axes:
            colours = sorted(
                matplotlib.colors.to_hex(line.get_color())
                for line in ax.get_lines()
                if line.get_linestyle() == "--"
            )
            assert colours == sorted(
                matplotlib.colors.to_hex(colour)
                for colour in (*plotting.MODEL_COLORS.values(), plotting.COLORS["reference"])
            )
        # must reject: a figure with every line solid, for data that leave a band out
        wrong = "the line of tanh RNN is not dashed exactly where a band was not run"
        assert f"upper row, 50 strings: {wrong}" in complexity_figure_problems(fig, data)
        # must reject: a dashed line for data that leave no band out
        assert f"lower row, 800 strings: {wrong}" in complexity_figure_problems(with_gap, plain)
        # must reject: the dashes between the wrong two levels
        other = complexity_data(dashed=(0,))
        assert f"upper row, 200 strings: {wrong}" in complexity_figure_problems(with_gap, other)
        assert complexity_figure_problems(plotting.figure_complexity(other), other) == []
        # must reject: a line that leaves a pair of levels unjoined
        broken = plotting.figure_complexity(data)
        next(line for line in broken.axes[0].get_lines() if line.get_linestyle() == "--").remove()
        assert complexity_figure_problems(broken, data) != []
        # with a run trained here the legend names the dashes before the star
        live = {"topological_entropy_bits": 1.6, "n_train": 200, "rnn": {"score": 0.5, "auc": 0.9}}
        starred = plotting.figure_complexity(data, live=live)
        assert [text.get_text() for text in starred.legends[0].get_texts()][-2:] == [
            "across a band that was not run",
            "trained here",
        ]
        # must reject: figure data that do not say which segments are dashed
        without = {key: value for key, value in data.items() if key != "segments"}
        with pytest.raises(KeyError, match="segments"):
            plotting.figure_complexity(without)

        # the saved runs: the production runs leave one band of the design out, between the third
        # and the fourth of their levels; the pilot has every level of the design
        for kind, position in (("production", 2), ("pilot", None)):
            saved, manifest, drawn = complexity_runs.load_runs(complexity_runs.FOLDERS[kind])
            real = complexity_runs.figure_data(saved, manifest, drawn)
            x_real = [level["topological_entropy_bits"] for level in real["levels"]]
            drawn_figure = plotting.figure_complexity(real)
            assert complexity_figure_problems(drawn_figure, real) == [], kind
            wanted = [] if position is None else [(x_real[position], x_real[position + 1])] * 3
            assert dashed_stretches(drawn_figure) == [wanted] * 6, kind
            if position is not None:
                # the two levels lie on the two sides of the band that was not run
                (band,) = real["segments"][position]["bands_not_run"]
                assert x_real[position] < band[0] < band[1] < x_real[position + 1]
                assert len(real["levels"]) == 4 and band == [1.86, 1.98]
            else:
                assert len(real["levels"]) == len(config.COMPLEXITY_DESIGN.levels) == 5
    plt.close("all")


def test_grammar_is_drawn_by_symseqs_own_class():
    from test_complexity import UNIFILAR

    from tutorial import complexity

    with matplotlib.rc_context():
        plotting.apply_style()
        before = np.get_printoptions()
        fig = plotting.draw_grammar(complexity.build_grammar(UNIFILAR))
        assert np.get_printoptions() == before  # SymSeq's class changes them; they are restored
        (ax,) = fig.axes
        labels = sorted({text.get_text() for text in ax.texts})
        assert labels == sorted(["#", "A(0)", "A(1)", "B", "C", "S"])
        assert not ax.axison and ax.get_aspect() == 1.0
        # must reject: no probability is written on an arrow
        assert not any(label.replace(".", "").isdigit() for label in labels)
        plt.close(fig)


def behaviour_data(own: bool = False) -> dict:
    names = ["zebrafish", "seal"] + (["own"] if own else [])
    rates = {
        "data": [2.9, 2.5, 2.4, 2.3, 2.0, 1.4],
        "first-order": [2.9, 2.5, 2.48, 2.36, 2.05, 1.46],
        "shuffled": [2.9, 2.9, 2.88, 2.7, 2.15, 1.1],
    }
    mi = {
        "distances": list(range(1, 21)),
        "data": [0.4 / d for d in range(1, 21)],
        "first-order": [0.4 * 0.2**d - 0.0002 for d in range(1, 21)],  # falls below zero
        "shuffled": [0.0] * 20,
        "band_lower": [-0.001] * 20,
        "band_upper": [0.001] * 20,
    }
    return {
        "datasets": names,
        "labels": {"zebrafish": "zebrafish", "seal": "seals and sea lions", "own": "your sequence"},
        "highlight": ["seal"] + (["own"] if own else []),
        "entropy_rate": {name: rates for name in names},
        "mi": {name: mi for name in names},
        "closed_form": {name: 2.5 for name in names},
        "markov_order": {name: {"data": 2, "first-order": 1, "shuffled": 0} for name in names},
    }


def test_behaviour_figure_has_a_column_per_dataset_and_marks_the_chosen_one():
    with matplotlib.rc_context():
        plotting.apply_style()
        fig = plotting.figure_behaviour(behaviour_data())
        assert len(fig.axes) == 4  # two datasets, two panels each
        assert tuple(fig.get_size_inches()) == plotting.FULL_SLIDE
        assert all(ax.get_title() == "" for ax in fig.axes) and fig._suptitle is None
        legend = [text.get_text() for text in fig.legends[0].get_texts()]
        assert legend[0] == "data" and legend[1:3] == [
            "first-order surrogate",
            "shuffled",
        ]  # the data first
        upper = [ax for ax in fig.axes if ax.get_xlabel() == r"order $k$"]
        lower = [ax for ax in fig.axes if ax.get_xlabel() == "distance (symbols)"]
        assert len(upper) == len(lower) == 2 and all(ax.get_yscale() == "log" for ax in lower)
        assert upper[0].get_ylabel() == r"entropy rate $h_k$ (bits per symbol)"
        assert lower[0].get_ylabel() == "mutual information (bits)"
        texts = [[text.get_text() for text in ax.texts] for ax in upper]
        assert "zebrafish" in texts[0] and plotting.bold("seals and sea lions") in texts[1]
        assert "BIC: order 2" in texts[0]
        # the chosen dataset has the heavy frame; must reject: the other dataset has not
        assert (
            upper[1].spines["left"].get_linewidth() == 1.6
            and lower[1].spines["bottom"].get_linewidth() == 1.6
        )
        assert upper[0].spines["left"].get_linewidth() == pytest.approx(0.8)
        # values at or below the floor are not drawn on the logarithmic axis
        drawn = lower[0].lines[0].get_ydata()
        assert np.isnan(drawn[-1]) and np.nanmin(drawn) > plotting.MI_FLOOR
        plt.close(fig)

        with_own = plotting.figure_behaviour(behaviour_data(own=True))
        assert len(with_own.axes) == 6  # a participant's sequence adds a column
        labels = [text.get_text() for ax in with_own.axes for text in ax.texts]
        assert plotting.bold("your sequence") in labels
        plt.close(with_own)
        assert "\\ " in plotting.bold("two words") and plotting.bold("word") == r"$\mathbf{word}$"
        with pytest.raises(KeyError):
            plotting.figure_behaviour({"datasets": ["zebrafish"]})  # must reject: missing data
    plt.close("all")


def words_without_bb(length: int) -> int:
    """Number of words of A and B of a given length without two B in a row, by listing all of them."""
    return sum("BB" not in "".join(word) for word in itertools.product("AB", repeat=length))


def test_words_panel_shows_the_counts_of_listed_words_and_the_topological_entropy():
    """Every plotted value of the panel against an independent route (enumeration of the words)."""
    data = worked_example.words_panel()
    _, chain = cf.stream_chain(two_state_grammar(0.5, 1.0), delimiter=False)  # SymSeq's own table
    with matplotlib.rc_context():
        plotting.apply_style()
        fig = plotting.figure_worked_example_words(data)
        ax, ax_graph = fig.axes
        assert tuple(fig.get_size_inches()) == plotting.HALF_SLIDE
        assert all(axis.get_title() == "" for axis in fig.axes) and fig._suptitle is None
        line, points = ax.lines  # the reference line first, then the counts
        lengths = [int(length) for length in points.get_xdata()]
        assert lengths == list(range(1, 11)) and list(line.get_xdata()) == lengths
        # the plotted counts: listed word by word here, and from SymSeq's table of the chain
        listed = [words_without_bb(length) for length in lengths]
        assert [int(count) for count in points.get_ydata()] == listed
        assert listed == [cf.count_words(chain, length) for length in lengths]
        assert listed[:6] == [2, 3, 5, 8, 13, 21]
        assert ax.get_yscale() == "log"
        assert [label.get_text() for label in ax.get_yticklabels()] == [str(n) for n in listed]
        # the plotted line: phi^(L+2) / sqrt(5), within one half of every count
        drawn = np.asarray(line.get_ydata(), dtype=float)
        assert drawn == pytest.approx(
            [cf.PHI ** (length + 2) / math.sqrt(5) for length in lengths], rel=1e-12
        )
        assert np.abs(drawn - listed).max() < 0.5
        # its slope is the topological entropy: bits per symbol in the figure, SymSeq's value in nats
        slopes = np.diff(np.log2(drawn))
        nats = float(
            topological_entropy(
                transitions=np.array([[1.0, 1.0], [1.0, 0.0]]), method="direct", verbose=False
            )
        )
        assert nats == pytest.approx(0.481212, abs=1e-6)
        assert slopes == pytest.approx(nats / math.log(2), abs=1e-9)
        assert data["topological_entropy_nats"] == pytest.approx(nats, abs=1e-9)
        assert data["topological_entropy_bits"] == pytest.approx(nats / math.log(2), abs=1e-9)
        growth = math.log2(words_without_bb(16) / words_without_bb(15))
        assert data["topological_entropy_bits"] == pytest.approx(growth, abs=1e-5)
        (note,) = (text.get_text() for text in ax.texts)
        assert "0.694$ bit per symbol" in note and "0.481$ nat per symbol" in note
        # the graph beside it: SymSeq's drawing of the two symbols, without probabilities
        assert sorted(text.get_text() for text in ax_graph.texts) == ["A", "B"]
        assert not ax_graph.axison and ax_graph.get_aspect() == 1.0
        assert np.get_printoptions()["precision"] == 8  # numpy's print options are left alone
        plt.close(fig)
        # without the graph the plot stands alone, with the same counts
        alone = plotting.figure_worked_example_words(data, graph=False)
        assert len(alone.axes) == 1 and tuple(alone.get_size_inches()) == plotting.HALF_SLIDE
        assert [int(count) for count in alone.axes[0].lines[1].get_ydata()] == listed
        plt.close(alone)
        # must reject: the counts of the sequence without the rule, and nats taken for bits
        assert [2**length for length in lengths] != listed
        assert data["topological_entropy_nats"] != pytest.approx(slopes[0], abs=0.1)
        with pytest.raises(KeyError):
            plotting.figure_worked_example_words({"length": [1, 2]})  # must reject: missing data
    with pytest.raises(ValueError):
        cf.allowed_word_counts(0)
    plt.close("all")


def test_block_entropy_panel_shows_the_block_entropies_of_the_stationary_chain():
    """Every plotted value of the panel against the entropy of all words under the stationary chain."""
    with matplotlib.rc_context():
        plotting.apply_style()
        for p in (0.5, 0.3):
            data = worked_example.block_entropy_panel(p)
            _, chain = cf.stream_chain(two_state_grammar(p, 1.0), delimiter=False)
            fig = plotting.figure_worked_example_block_entropy(data)
            (ax,) = fig.axes
            assert tuple(fig.get_size_inches()) == plotting.HALF_SLIDE
            assert ax.get_title() == "" and fig._suptitle is None
            line, points, mark = ax.lines  # the reference line first
            lengths = [int(length) for length in points.get_xdata()]
            assert lengths == list(range(1, 9))
            enumerated = [cf.block_entropy_enumerated(chain, length) for length in lengths]
            assert list(points.get_ydata()) == pytest.approx(enumerated, abs=1e-9)
            # the plotted line: slope h, value E at L = 0, from the stationary distribution of the chain
            (x0, x1), (y0, y1) = line.get_xdata(), line.get_ydata()
            slope = (y1 - y0) / (x1 - x0)
            assert x0 == 0 and slope == pytest.approx(cf.chain_entropy_rate(chain), abs=1e-9)
            assert y0 == pytest.approx(cf.chain_excess_entropy(chain), abs=1e-9)
            assert (mark.get_xdata()[0], mark.get_ydata()[0]) == (0, pytest.approx(y0))
            texts = " ".join(text.get_text() for text in ax.texts)
            assert f"h = {slope:.4f}$ bit per symbol" in texts and f"E = {y0:.4f}$ bit" in texts
            if p == 0.5:  # the values of the slides
                assert slope == pytest.approx(2 / 3, abs=1e-9)
                assert y0 == pytest.approx(math.log2(3) - 4 / 3, abs=1e-9)
                assert "0.6667" in texts and "0.2516" in texts
                half = list(points.get_ydata())
            plt.close(fig)
    # must reject: the block entropies at another p, and those of the sequence with the end marker
    _, other = cf.stream_chain(two_state_grammar(0.3, 1.0), delimiter=False)
    assert half != pytest.approx([cf.block_entropy_enumerated(other, n) for n in lengths], abs=0.01)
    _, marked = cf.stream_chain(two_state_grammar(0.5, 1.0), delimiter=True)
    with_marker = [cf.block_entropy_enumerated(marked, n) for n in lengths]
    assert half != pytest.approx(with_marker, abs=0.01)
    assert with_marker[-1] - with_marker[-2] == pytest.approx(
        0.5, abs=1e-9
    )  # h = 1/2 with the marker
    with pytest.raises(KeyError):
        plotting.figure_worked_example_block_entropy({"length": [1, 2]})  # must reject
    plt.close("all")
