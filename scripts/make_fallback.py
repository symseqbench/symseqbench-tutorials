"""Regenerate the saved results of a notebook in ``fallback/``.

Notebook A (the default), ``fallback/a_generate_and_measure/``: runs the computations of the
notebook with its default controls and saves the tables (CSV), the final figure (PDF, PNG, SVG, PNG
on white) with the numbers it shows (JSON), the two exact panels of the worked example
(``worked_example_words`` and ``worked_example_block_entropy``, each in the same five files), and a
static HTML export of the notebook.

        uv run python scripts/make_fallback.py

Notebook B, ``fallback/b_behaviour/``: the figure of the four datasets from the saved results of
``data/behaviour_measures/``, the numbers it shows, and the HTML export.

        uv run python scripts/make_fallback.py --notebook b

Notebook C, ``fallback/c_networks/``: its two figures with the numbers they show, and the HTML
export. ``figure_networks`` is made from the saved runs of ``data/precomputed/``, and
``figure_complexity`` from the saved runs of the complexity experiment. Placeholder results (smoke
runs, pilot runs) are refused, unless ``--allow-placeholder`` is given; the figure and its data then
say that they are placeholders.

        uv run python scripts/make_fallback.py --notebook c
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tutorial import (
    behaviour_measures,
    complexity_runs,
    config,
    ladder,
    plotting,
    prescribed,
    results,
    worked_example,
)

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "a_generate_and_measure.py"
NOTEBOOK_B = REPO / "notebooks" / "b_behaviour.py"
NOTEBOOK_C = REPO / "notebooks" / "c_networks.py"
DEFAULT_DATASET_B = "zebrafish"  # the dataset that notebook B shows first
FOLDERS = {"a": "a_generate_and_measure", "b": "b_behaviour", "c": "c_networks"}


def export_html(notebook: Path, target: Path) -> None:
    """Static HTML export of a notebook, run with its default controls."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "export",
            "html",
            str(notebook),
            "-o",
            str(target),
            "--force",
        ],
        check=True,
        cwd=REPO,
    )


def worked_example_panels(out: Path, p: float = 0.5) -> None:
    """The two exact panels of the worked example, each as a figure and the numbers it shows."""
    panels = (
        (
            "worked_example_words",
            worked_example.words_panel(),
            plotting.figure_worked_example_words,
        ),
        (
            "worked_example_block_entropy",
            worked_example.block_entropy_panel(p),
            plotting.figure_worked_example_block_entropy,
        ),
    )
    for name, data, draw in panels:
        figure = draw(data)
        plotting.save_figure(figure, out, name)
        plotting.save_figure_data(data, out, name)
        plt.close(figure)


def fallback_a(out: Path, no_html: bool) -> None:
    """The fallback of notebook A: its tables, its final figure, the two panels of the worked example."""
    out.mkdir(parents=True, exist_ok=True)

    # the default controls of the notebook
    p, n_symbols, seed, per_rung = 0.5, 18000, 0, 600
    n_strings = worked_example.strings_for_length(p, n_symbols)
    _, sequence = worked_example.joined_sequence(p, n_strings, seed)
    comparison = worked_example.comparison(p, sequence)
    convergence = worked_example.convergence(p, sequence)
    curve = worked_example.entropy_rate_curve(seed=seed)
    rungs = ladder.default_ladder()
    table = ladder.ladder_table(rungs, n_strings=per_rung, seed=seed)
    exact = ladder.exact_table(rungs)
    prescribed_result = prescribed.make_prescribed(1.4, seed=7)

    comparison.to_csv(out / "worked_example.csv")
    flat = table.copy()
    flat.columns = [f"{level}: {name}" if level else name for level, name in flat.columns]
    flat.to_csv(out / "ladder_table.csv")
    exact.to_csv(out / "ladder_exact.csv")

    plotting.apply_style()
    worked_example_panels(out, p)
    figure = plotting.figure_generate_and_measure(
        convergence, curve, prescribed.figure_data(prescribed_result)
    )
    plotting.save_figure(figure, out, "figure_generate_and_measure")
    plotting.save_figure_data(
        {
            "controls": {
                "p": p,
                "n_symbols": n_symbols,
                "n_strings": n_strings,
                "seed": seed,
                "strings_per_rung": per_rung,
            },
            "convergence": convergence,
            "curve": curve,
            "prescribed": prescribed.figure_data(prescribed_result),
        },
        out,
        "figure_generate_and_measure",
    )
    plt.close(figure)
    if not no_html:
        export_html(NOTEBOOK, out / "a_generate_and_measure.html")
    print(f"written {out}")


def fallback_b(out: Path, no_html: bool) -> None:
    """The fallback of notebook B: its figure from the saved results, the figure data, the HTML export."""
    out.mkdir(parents=True, exist_ok=True)
    saved, manifest = behaviour_measures.load_results()
    behaviour_measures.check_against_tables(saved)
    if not manifest["complete"] or manifest["checks_failed"]:
        raise SystemExit("The saved results of notebook B are incomplete or fail their checks.")
    data = behaviour_measures.figure_data(saved, highlight=DEFAULT_DATASET_B)
    plotting.apply_style()
    figure = plotting.figure_behaviour(data)
    plotting.save_figure(figure, out, "figure_behaviour")
    plotting.save_figure_data(
        {"settings": manifest["settings"], "result_files": manifest["files"], **data},
        out,
        "figure_behaviour",
    )
    plt.close(figure)
    if not no_html:
        export_html(NOTEBOOK_B, out / "b_behaviour.html")
    print(f"written {out}")


def records(table) -> list[dict]:
    """The rows of a table as plain values; a value that is not a number becomes null."""
    return json.loads(table.to_json(orient="records"))


def fallback_c(
    out: Path,
    no_html: bool,
    folder: Path | None = None,
    complexity_folder: Path | None = None,
    allow_placeholder: bool = False,
) -> None:
    """The fallback of notebook C: its two figures from the saved runs, their data, the HTML export.

    ``folder`` holds the six-grammar runs and ``complexity_folder`` the runs of the complexity
    experiment; by default the folders the notebook reads. Results that are incomplete or that fail
    their checks are refused. So are placeholders (smoke runs, pilot runs), unless
    ``allow_placeholder`` is set: the fallback shows the runs of the figures or says that it does not.
    """
    folder = results.default_folder() if folder is None else Path(folder)
    runs, manifest = results.load_runs(folder)
    if manifest["placeholder"] or not manifest["complete"] or manifest["checks_failed"]:
        raise SystemExit(
            f"The saved results in {folder} are placeholders, are incomplete or fail their checks; "
            "the fallback of notebook C needs the complete runs of the figure."
        )
    complexity_folder = (
        complexity_runs.default_folder() if complexity_folder is None else Path(complexity_folder)
    )
    saved, complexity_manifest, drawn = complexity_runs.load_runs(complexity_folder)
    if not complexity_manifest["complete"] or complexity_manifest["checks_failed"]:
        raise SystemExit(
            f"The saved results in {complexity_folder} are incomplete or fail their checks; "
            "the fallback of notebook C needs complete runs of the complexity experiment."
        )
    if complexity_manifest["placeholder"] and not allow_placeholder:
        raise SystemExit(
            f"The saved results in {complexity_folder} are placeholders (pilot runs). Run the "
            "production runs first, or give --allow-placeholder to export them marked as placeholders."
        )
    out.mkdir(parents=True, exist_ok=True)
    plotting.apply_style()

    data = results.distance_data(runs, config.LADDER, manifest["kind"])
    figure = plotting.figure_networks(data)
    plotting.save_figure(figure, out, "figure_networks")
    shown = [
        f"runs/{results.run_name(grammar, model, seed)}.json"
        for grammar in config.LADDER
        for model in config.MODELS
        for seed in manifest["seeds"]
    ]
    plotting.save_figure_data(
        {
            "quantity": "cross-entropy on the held-out samples minus that of the exact predictor "
            "on the same symbols, in bits per target, after every epoch",
            "settings": manifest["settings"],
            "seeds": manifest["seeds"],
            "run_files": {name: manifest["files"][name] for name in shown},
            **data,
        },
        out,
        "figure_networks",
    )
    plt.close(figure)

    complexity_data = complexity_runs.figure_data(saved, complexity_manifest, drawn)
    figure = plotting.figure_complexity(complexity_data)
    plotting.save_figure(figure, out, "figure_complexity")
    plotting.save_figure_data(
        {
            "quantities": {
                "score": "prediction score on the held-out strings: 1 minus the mean "
                "Kullback-Leibler divergence from the exact predictor to the model, divided by "
                "that of the uniform guess on the same targets (0: uniform guess, 1: exact predictor)",
                "auc": "area under the ROC curve of the largest surprisal of a string, "
                "ungrammatical against grammatical held-out strings (0.5: chance, 1: exact predictor)",
                "divergence_bits": "the divergence itself, in bits per target; it is not compared "
                "across grammars and is not drawn",
            },
            "settings": complexity_manifest["settings"],
            "network": complexity_manifest["network"],
            "run_files": complexity_manifest["files"],
            "grammars_file": complexity_manifest["grammars"],
            "trend": records(complexity_runs.trend_table(complexity_data)),
            "sizes_test": records(complexity_runs.size_table(complexity_data)),
            **complexity_data,
        },
        out,
        "figure_complexity",
    )
    plt.close(figure)
    if not no_html:
        export_html(NOTEBOOK_C, out / "c_networks.html")
    print(f"written {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--notebook", choices=tuple(FOLDERS), default="a")
    parser.add_argument(
        "--out", type=Path, default=None, help="default: the folder of the notebook in fallback/"
    )
    parser.add_argument(
        "--no-html", action="store_true", help="skip the static HTML export of the notebook"
    )
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="notebook C: export pilot results of the complexity experiment, marked as placeholders",
    )
    arguments = parser.parse_args()
    out = arguments.out or REPO / "fallback" / FOLDERS[arguments.notebook]
    if arguments.notebook == "c":
        fallback_c(out, arguments.no_html, allow_placeholder=arguments.allow_placeholder)
        return
    if arguments.allow_placeholder:
        raise SystemExit("--allow-placeholder applies to notebook C only.")
    {"a": fallback_a, "b": fallback_b}[arguments.notebook](out, arguments.no_html)


if __name__ == "__main__":
    main()
