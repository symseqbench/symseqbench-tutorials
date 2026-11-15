"""Saved training runs: one JSON file per run, a summary table and a manifest.

A results folder holds ``runs/<grammar>__<model>__seed<seed>.json`` (what ``training.train_run``
returned, plus the kind of run), ``summary.csv`` (one row per run, built from the run files) and
``manifest.json`` (settings, the runs that were expected, the runs that are missing, and the SHA-256
of every run file). ``load_runs`` refuses a folder whose files do not match the manifest.

Two folders: ``data/precomputed`` for the runs of the figure (kind "production") and
``data/precomputed_smoke`` for short runs that only exercise the code (kind "smoke"). Smoke results
are placeholders, and everything that shows them says so.

## Theory

Grammars differ in how much of a string is random by construction, so raw accuracy and raw
cross-entropy say more about the generator than about the network. What is compared across
grammars is a network's distance from the exact predictor of the grammar. With p the generator's
distribution of the next symbol, q the network's, and n held-out targets x_1 ... x_n:

    Eq. (1)  distance = (1 / n) sum_i [log2 p(x_i) - log2 q(x_i)]        bits per target,

the network's cross-entropy minus the cross-entropy of the exact predictor on the same symbols
(``optimum["cross_entropy_realised"]`` of a run). Its expectation is the mean Kullback-Leibler
divergence from p to q over the targets, which is zero only for a network that knows the grammar
and is never negative. Eq. (1) is an estimate of it from the symbols that happened to be drawn, so
it can be slightly negative by chance.

Sampling spread. Given its prefix, each term of Eq. (1) has the mean KL_i and, for q close to p, the
variance 2 KL_i / ln 2 (the variance of a log-likelihood ratio is twice its mean in nats). The
terms minus their means are uncorrelated, so with D the mean divergence in bits

    Eq. (2)  standard error of the distance = sqrt(2 D / (n ln 2))       bits per target,
    Eq. (3)  resolution = 8 / (n ln 2):  the D that equals two of its standard errors.

A distance below the resolution cannot be told from zero. Eq. (2) is an approximation for small
divergences; the saved runs hold no value per target, so the spread cannot be computed from them
exactly. ``tests/test_plotting.py`` checks Eq. (2) against repeated sampling: at a divergence of
0.016 bit it is 17 % above the spread found, so the resolution is a guide and not a test.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tutorial import config

REPO = Path(__file__).resolve().parents[1]
FOLDERS = {
    "production": REPO / "data" / "precomputed",
    "smoke": REPO / "data" / "precomputed_smoke",
}
MEASURES = ("accuracy_with_eos", "accuracy_within", "cross_entropy")


def run_name(rung: str, model: str, seed: int) -> str:
    """File name of a run, without the extension."""

    def slug(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

    return f"{slug(rung)}__{slug(model)}__seed{seed}"


def expected_runs(kind: str) -> list[tuple[str, str, int]]:
    """Every (grammar, model, seed) of a kind of run: all candidates, both models, the seeds of the kind."""
    return [
        (rung, model, seed)
        for rung in (*config.LADDER, *config.ALTERNATIVES)
        for seed in config.SEEDS[kind]
        for model in config.MODELS
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_run(result: dict, folder: Path, kind: str) -> Path:
    """Write one run. The file appears complete or not at all (written aside, then renamed)."""
    runs = Path(folder) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / f"{run_name(result['rung'], result['model'], result['seed'])}.json"
    aside = path.with_suffix(f".{os.getpid()}.part")
    aside.write_text(json.dumps({"kind": kind, **result}, indent=1))
    os.replace(aside, path)
    return path


def saved_run(
    folder: Path, kind: str, rung: str, model: str, seed: int, settings: dict
) -> dict | None:
    """The saved run with exactly this kind, grammar, model, seed and settings, or None."""
    path = Path(folder) / "runs" / f"{run_name(rung, model, seed)}.json"
    if not path.exists():
        return None
    try:
        run = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    same = (run.get("kind"), run.get("rung"), run.get("model"), run.get("seed")) == (
        kind,
        rung,
        model,
        seed,
    )
    return run if same and run.get("settings") == settings else None


def summary(runs: list[dict]) -> pd.DataFrame:
    """One row per run: the last epoch next to the optimum and to the values of the endless sequence."""
    rows = []
    for run in runs:
        final, optimum, endless = run["final"], run["optimum"], run["exact"]
        rows.append(
            {
                "grammar": run["rung"],
                "family": run["family"],
                "model": run["model"],
                "seed": run["seed"],
                "parameters": run["n_parameters"],
                "epochs": run["settings"]["epochs"],
                "accuracy (with #)": final["accuracy_with_eos"],
                "best achievable (with #)": optimum["accuracy_with_eos"],
                "ceiling, endless sequence (with #)": endless["ceiling_with_eos"],
                "previous symbol only (with #)": endless["previous_symbol_with_eos"],
                "accuracy (symbols only)": final["accuracy_within"],
                "best achievable (symbols only)": optimum["accuracy_within"],
                "ceiling, endless sequence (symbols only)": endless["ceiling_within"],
                "previous symbol only (symbols only)": endless["previous_symbol_within"],
                "cross-entropy": final["cross_entropy"],
                "best achievable cross-entropy": optimum["cross_entropy"],
                "entropy rate, endless sequence": endless["entropy_rate"],
                "checks passed": run["checks"]["passed"],
                "seconds, data": run["seconds"]["data"],
                "seconds, training": run["seconds"]["training"],
                "speed probe (us)": max(run["environment"]["speed_probe_us"]),
            }
        )
    return pd.DataFrame(rows)


def collect(folder: Path, kind: str) -> dict:
    """Build ``summary.csv`` and ``manifest.json`` of a folder from its run files, and return the manifest.

    A run that is expected and absent is listed as missing; it is never given a value.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    settings = config.SETTINGS[kind].as_dict()
    runs, files, missing = [], {}, []
    for rung, model, seed in expected_runs(kind):
        name = run_name(rung, model, seed)
        run = saved_run(folder, kind, rung, model, seed, settings)
        if run is None:
            missing.append(name)
            continue
        runs.append(run)
        files[f"runs/{name}.json"] = sha256(folder / "runs" / f"{name}.json")
    table = summary(runs)
    table.to_csv(folder / "summary.csv", index=False)
    manifest = {
        "kind": kind,
        "placeholder": kind != "production",
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": settings,
        "seeds": list(config.SEEDS[kind]),
        "grammars": [*config.LADDER, *config.ALTERNATIVES],
        "models": list(config.MODELS),
        "expected": len(expected_runs(kind)),
        "present": len(runs),
        "missing": missing,
        "complete": not missing,
        "checks_failed": [
            run_name(run["rung"], run["model"], run["seed"])
            for run in runs
            if not run["checks"]["passed"]
        ],
        "files": files,
        "summary": {"file": "summary.csv", "sha256": sha256(folder / "summary.csv")},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_runs(folder: Path) -> tuple[list[dict], dict]:
    """The runs of a folder and its manifest, after checking every file against the manifest.

    Raises ``FileNotFoundError`` without a manifest and ``ValueError`` if a listed file is absent or
    changed, or if the folder holds a run file that the manifest does not list.
    """
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    runs = []
    for name, digest in manifest["files"].items():
        path = folder / name
        if not path.exists():
            raise ValueError(f"{name} is listed in the manifest and absent from {folder}.")
        if sha256(path) != digest:
            raise ValueError(f"{name} was changed after the manifest was written.")
        runs.append(json.loads(path.read_text()))
    listed = {Path(name).name for name in manifest["files"]}
    unlisted = sorted(p.name for p in (folder / "runs").glob("*.json") if p.name not in listed)
    if unlisted:
        raise ValueError(f"Run files not in the manifest: {unlisted}. Collect the folder again.")
    return runs, manifest


def default_folder() -> Path:
    """The production results if they are complete, otherwise the smoke results."""
    manifest = FOLDERS["production"] / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["complete"]:
        return FOLDERS["production"]
    return FOLDERS["smoke"]


def figure_data(
    runs: list[dict], rungs: tuple[str, ...] = config.LADDER, kind: str = "production"
) -> dict:
    """Raw values after the last epoch: per grammar the optimum, and per model the value of every seed.

    These are accuracies and cross-entropies as they are, which differ between grammars for reasons
    that are not the networks' (see Theory). They are kept for the summary and the tests; no figure
    shows them across grammars any more, ``distance_data`` gives what the figure shows. The optimum
    of a grammar is the same in all its runs, because they share the held-out samples; the function
    raises if they disagree, or if a grammar has no run for one of the models.
    """
    data: dict = {
        "kind": kind,
        "grammars": list(rungs),
        "families": [],
        "optimum": {measure: [] for measure in MEASURES},
        "endless": {"ceiling_with_eos": [], "ceiling_within": [], "entropy_rate": []},
        "models": {
            model: {"seeds": [], **{measure: [] for measure in MEASURES}} for model in config.MODELS
        },
    }
    for rung in rungs:
        of_rung = [run for run in runs if run["rung"] == rung]
        if not of_rung:
            raise ValueError(f"No saved run for the grammar {rung!r}.")
        for measure in MEASURES:
            values = {run["optimum"][measure] for run in of_rung}
            if len(values) != 1:
                raise ValueError(f"The runs of {rung!r} do not share one optimum for {measure}.")
            data["optimum"][measure].append(values.pop())
        for key in data["endless"]:
            data["endless"][key].append(of_rung[0]["exact"][key])
        data["families"].append(of_rung[0]["family"])
        for model in config.MODELS:
            of_model = sorted(
                (run for run in of_rung if run["model"] == model), key=lambda run: run["seed"]
            )
            if not of_model:
                raise ValueError(f"No saved run of the {model} for the grammar {rung!r}.")
            data["models"][model]["seeds"].append([run["seed"] for run in of_model])
            for measure in MEASURES:
                data["models"][model][measure].append([run["final"][measure] for run in of_model])
    return data


def mean_table(data: dict) -> pd.DataFrame:
    """Per grammar and model: mean over seeds of the three measures, next to the optimum."""
    rows = []
    for index, rung in enumerate(data["grammars"]):
        for model in data["models"]:
            values = data["models"][model]
            rows.append(
                {
                    "grammar": rung,
                    "model": config.MODEL_LABELS[model],
                    "seeds": len(values["seeds"][index]),
                    "accuracy (with #)": np.mean(values["accuracy_with_eos"][index]),
                    "best achievable (with #)": data["optimum"]["accuracy_with_eos"][index],
                    "accuracy (symbols only)": np.mean(values["accuracy_within"][index]),
                    "best achievable (symbols only)": data["optimum"]["accuracy_within"][index],
                    "cross-entropy": np.mean(values["cross_entropy"][index]),
                    "best achievable cross-entropy": data["optimum"]["cross_entropy"][index],
                }
            )
    return pd.DataFrame(rows).set_index(["grammar", "model"])


def distance_standard_error(distance: float, n_targets: int) -> float:
    """Eq. (2): approximate standard error of a distance, in bits per target (0 for a distance below 0)."""
    return math.sqrt(2.0 * max(distance, 0.0) / (n_targets * math.log(2.0)))


def distance_resolution(n_targets: int) -> float:
    """Eq. (3): the distance that equals two of its standard errors, in bits per target."""
    return 8.0 / (n_targets * math.log(2.0))


def distance_data(
    runs: list[dict], rungs: tuple[str, ...] = config.LADDER, kind: str = "production"
) -> dict:
    """The numbers of the six-grammar figure: each network's distance from the exact predictor over training.

    Per grammar: the epochs, the number of held-out targets, the resolution of Eq. (3) and the
    cross-entropy of the exact predictor on the held-out symbols; per model: for every seed the
    distance of Eq. (1) after every epoch, their mean over seeds, and the approximate standard
    error of Eq. (2) of the mean after the last epoch (one seed's worth of targets: the seeds share
    the held-out samples). The function raises if the runs of a grammar do not share one exact
    predictor or one list of epochs, or if a grammar has no run for one of the models.
    """
    data: dict = {
        "kind": kind,
        "grammars": list(rungs),
        "epochs": [],
        "n_targets": [],
        "resolution": [],
        "exact_cross_entropy": [],
        "models": {
            model: {"seeds": [], "distance": [], "mean": [], "final_standard_error": []}
            for model in config.MODELS
        },
    }
    for rung in rungs:
        of_rung = [run for run in runs if run["rung"] == rung]
        if not of_rung:
            raise ValueError(f"No saved run for the grammar {rung!r}.")
        shared = {
            (run["optimum"]["cross_entropy_realised"], run["optimum"]["n_targets"])
            for run in of_rung
        }
        epochs = {tuple(run["history"]["epoch"]) for run in of_rung}
        if len(shared) != 1 or len(epochs) != 1:
            raise ValueError(
                f"The runs of {rung!r} do not share one exact predictor and one list of epochs."
            )
        exact, n_targets = shared.pop()
        data["epochs"].append(list(epochs.pop()))
        data["n_targets"].append(n_targets)
        data["resolution"].append(distance_resolution(n_targets))
        data["exact_cross_entropy"].append(exact)
        for model in config.MODELS:
            of_model = sorted(
                (run for run in of_rung if run["model"] == model), key=lambda run: run["seed"]
            )
            if not of_model:
                raise ValueError(f"No saved run of the {model} for the grammar {rung!r}.")
            curves = [
                [value - exact for value in run["history"]["cross_entropy"]] for run in of_model
            ]  # Eq. (1)
            mean = np.mean(curves, axis=0).tolist()
            values = data["models"][model]
            values["seeds"].append([run["seed"] for run in of_model])
            values["distance"].append(curves)
            values["mean"].append(mean)
            values["final_standard_error"].append(distance_standard_error(mean[-1], n_targets))
    return data


def distance_table(data: dict) -> pd.DataFrame:
    """Per grammar and model: the distance after the last epoch (mean and range over seeds) and what limits it."""
    rows = []
    for index, rung in enumerate(data["grammars"]):
        for model, values in data["models"].items():
            final = [curve[-1] for curve in values["distance"][index]]
            rows.append(
                {
                    "grammar": rung,
                    "model": config.MODEL_LABELS[model],
                    "seeds": len(final),
                    "distance (bits per target)": float(np.mean(final)),
                    "smallest": min(final),
                    "largest": max(final),
                    "standard error (approximate)": values["final_standard_error"][index],
                    "resolution": data["resolution"][index],
                    "held-out targets": data["n_targets"][index],
                }
            )
    return pd.DataFrame(rows).set_index(["grammar", "model"])
