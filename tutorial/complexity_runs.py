"""The runs of the complexity experiment: grammars, training, run files and the numbers of the figure.

One job is one grammar and one training seed: its strings are drawn once, and both networks are
trained on the first N of them for every training size N. Every setting is in
``tutorial.config`` (``ComplexitySettings``); the optimiser, the batch size and the size of the
networks are those of ``config.PRODUCTION``.

## Theory

Training. A string x_1 ... x_L is read as ``# x_1 ... x_L`` and the targets are ``x_1 ... x_L #``,
so every symbol and the marker are predicted. The loss is the cross-entropy of the targets,

    Eq. (1)  loss = -(1 / n) sum_targets log2 q(target | prefix)        bits per target.

Choice of the weights. A share of the N training strings is kept for validation and never used for
a gradient step. Every ``eval_every`` steps the loss of Eq. (1) is measured on it, and the weights
of the step with the lowest validation loss are the ones that are measured (the first such step if
several are equal). The number of gradient steps is the same for both networks and every N.

Measures. Eq. (7) to Eq. (13) of ``tutorial.complexity``, on the held-out strings and their
ungrammatical counterparts, against the exact predictor of the grammar. Across grammars the two
measures that are compared are the prediction score (Eq. 13 there: 0 for the uniform guess, 1 for
the exact predictor) and the area under the ROC curve (Eq. 10 there: 0.5 at chance, 1 for the exact
predictor). The divergence in bits (Eq. 7 there) is kept beside them and is not compared across
grammars: the divergence of the uniform guess itself differs from grammar to grammar.

Trend across grammars. The unit is the grammar: its value is the mean over training seeds. With r
the ranks of the topological entropy and s the ranks of a measure over the n grammars,

    Eq. (2)  rho = Pearson correlation of r and s        (Spearman's rank correlation),
    Eq. (3)  p = share of the orderings of s whose rho is at least as far in the expected direction.

All n! orderings are counted when n <= 9; otherwise ``TREND_PERMUTATIONS`` random orderings are
drawn and p = (1 + count) / (1 + number drawn). The expected direction is negative for the
prediction score and for the area under the curve: both are expected to fall as the topological
entropy rises. The divergence carries no expectation; its rho is listed with the p-value for a
fall, so that the row can be set beside that of the score, and without a verdict.

Ref: Prechelt (1998), "Early stopping, but when?", in Neural Networks: Tricks of the Trade,
LNCS 1524, 55-69 (choice of the weights). Spearman (1904), Am. J. Psychol. 15, 72-101 (Eq. 2). Good
(2005), Permutation, Parametric and Bootstrap Tests of Hypotheses, 3rd ed., Springer (Eq. 3).
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
import platform
import re
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import binom, rankdata

from tutorial import complexity as cx
from tutorial import config
from tutorial.config import PRODUCTION, ComplexitySettings, Hyperparameters
from tutorial.env import installed_commit
from tutorial.models import build_model, free_size, n_parameters
from tutorial.results import sha256
from tutorial.training import set_threads, speed_probe, weights_digest

REPO = Path(__file__).resolve().parents[1]
FOLDERS = {
    "production": REPO / "data" / "complexity",
    "pilot": REPO / "data" / "complexity_pilot",
}
MEASURES = ("score", "auc", "divergence_bits", "top_choice_ratio")
TOLERANCE = 1e-9  # numerical slack of the checks that hold by construction


# --- grammars of a kind of run --------------------------------------------------------------------


def draw_grammars(
    settings: ComplexitySettings, report: Callable[[str], None] | None = None
) -> dict:
    """Draw the grammars of every level and decide their admission by the written rule.

    For each level the grammar seeds are tried in order from ``first_grammar_seed`` until
    ``grammars_per_level`` grammars are admitted or ``max_candidates`` seeds were tried. Every
    candidate is listed with what was computed for it and with the reasons if it was not admitted.
    """
    levels = []
    for index, level in enumerate(settings.levels):
        candidates, admitted = [], []
        for seed in range(
            settings.first_grammar_seed, settings.first_grammar_seed + settings.max_candidates
        ):
            if len(admitted) == settings.grammars_per_level:
                break
            spec, reason = cx.candidate(
                index,
                seed,
                level.target_nats,
                settings.n_states,
                settings.alphabet_size,
                settings.n_terminal_states,
            )
            if spec is None:
                candidates.append({"seed": seed, "admitted": False, "reasons": [reason]})
                continue
            values = cx.describe(spec, settings.min_length, settings.max_length)
            reasons = cx.admission(
                values, level.band_bits, settings.min_memory_gap_bits, settings.min_strings
            )
            candidates.append({"seed": seed, "admitted": not reasons, "reasons": reasons, **values})
            if not reasons:
                admitted.append({"spec": spec.as_dict(), "values": values})
            if report is not None:
                report(
                    f"level {index + 1} seed {seed}: "
                    f"{'admitted' if not reasons else '; '.join(reasons)} "
                    f"(entropy {values['topological_entropy_bits']:.3f} bit, gap {values['memory_gap']:.3f} bit, "
                    f"{values['n_strings']} strings)"
                )
        levels.append(
            {
                "level": index,
                "target_nats": level.target_nats,
                "band_bits": list(level.band_bits),
                "candidates": candidates,
                "admitted": admitted,
                "complete": len(admitted) == settings.grammars_per_level,
            }
        )
    return {"settings": settings.as_dict(), "levels": levels}


def write_grammars(folder: Path, kind: str, settings: ComplexitySettings, report=None) -> dict:
    """Draw the grammars of a kind of run and save them with the list of every candidate."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    drawn = {"kind": kind, **draw_grammars(settings, report)}
    aside = folder / f"grammars.{os.getpid()}.part"
    aside.write_text(json.dumps(drawn, indent=1))
    os.replace(aside, folder / "grammars.json")
    return drawn


def load_grammars(
    folder: Path, kind: str | None = None, settings: ComplexitySettings | None = None
) -> dict:
    """The saved grammars of a folder. Raises if they were drawn for another kind or other settings."""
    path = Path(folder) / "grammars.json"
    if not path.exists():
        raise FileNotFoundError(f"No grammars in {folder}. Draw them first (--draw).")
    drawn = json.loads(path.read_text())
    if kind is not None and drawn["kind"] != kind:
        raise ValueError(
            f"The grammars in {folder} were drawn for {drawn['kind']} runs, not {kind}."
        )
    if settings is not None and drawn["settings"] != json.loads(json.dumps(settings.as_dict())):
        raise ValueError(f"The grammars in {folder} were drawn with other settings.")
    return drawn


def admitted_grammars(drawn: dict) -> list[tuple[cx.GrammarSpec, dict]]:
    """Every admitted grammar with the values of its admission, by level and grammar seed."""
    return [
        (cx.GrammarSpec.from_dict(entry["spec"]), entry["values"])
        for level in drawn["levels"]
        for entry in level["admitted"]
    ]


def grammar_table(drawn: dict) -> pd.DataFrame:
    """One row per admitted grammar: level, complexity, memory gap, number of strings, string length."""
    rows = [
        {
            "level": spec.level + 1,
            "grammar seed": spec.seed,
            "topological entropy (bits per symbol)": values["topological_entropy_bits"],
            "memory gap (bits per target)": values["memory_gap"],
            "previous symbol only: excess (bits per target)": values["first_order_gap"],
            "distinct strings": float(values["n_strings"]),
            "typical strings": 2.0 ** values["string_entropy"],
            "mean string length": values["mean_length"],
            "walks accepted": values["acceptance"],
            "exact predictor: cross-entropy (bits per target)": values["entropy"],
        }
        for spec, values in admitted_grammars(drawn)
    ]
    return pd.DataFrame(rows)


def admission_table(drawn: dict) -> pd.DataFrame:
    """Per level: the band, how many candidates were tried and how many failed each rule."""
    rows = []
    for level in drawn["levels"]:
        reasons = [reason for entry in level["candidates"] for reason in entry["reasons"]]
        rows.append(
            {
                "level": level["level"] + 1,
                "target (nats)": level["target_nats"],
                "band (bits per symbol)": f"{level['band_bits'][0]:.2f} to {level['band_bits'][1]:.2f}",
                "candidates": len(level["candidates"]),
                "admitted": len(level["admitted"]),
                **{reason: reasons.count(reason) for reason in sorted(set(reasons))},
            }
        )
    table = pd.DataFrame(rows).set_index("level")
    counted = [
        column
        for column in table.columns
        if column not in ("target (nats)", "band (bits per symbol)")
    ]
    table[counted] = table[counted].fillna(0).astype(int)
    return table


# --- data of a job --------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class JobData:
    """The strings of one grammar and training seed, with everything the measures need."""

    spec: cx.GrammarSpec
    values: dict  # what the admission computed for the grammar
    seed: int
    classes: tuple[str, ...]
    strings: cx.Strings
    width: int  # positions of a row: the longest string and the marker
    heldout_inputs: torch.Tensor
    heldout_targets: np.ndarray
    ungrammatical_inputs: torch.Tensor
    ungrammatical_targets: np.ndarray
    p_heldout: np.ndarray  # exact distribution at every held-out target
    exact: dict  # the measures of the exact predictor on these strings
    predictor: cx.BoundedPredictor
    probability: dict  # probability of every distinct training string under the grammar
    walk_divergence: float  # divergence of the walk without length bounds on the held-out strings
    uniform_divergence: float  # D0: divergence of the uniform guess on the held-out targets
    seconds: float


def prepare(spec: cx.GrammarSpec, values: dict, seed: int, settings: ComplexitySettings) -> JobData:
    """Draw the strings of a grammar and training seed and compute the exact predictor on them."""
    start = time.perf_counter()
    strings = cx.draw_strings(
        spec,
        seed,
        max(settings.sizes),
        settings.n_heldout,
        settings.min_length,
        settings.max_length,
        settings.max_draws,
        settings.training_sample,
    )
    classes = cx.classes_of(spec)
    width = settings.max_length + 1
    predictor = cx.BoundedPredictor(
        cx.build_grammar(spec), settings.min_length, settings.max_length
    )
    heldout_inputs, heldout_targets = cx.encode(strings.heldout, classes, width)
    ungrammatical_inputs, ungrammatical_targets = cx.encode(strings.ungrammatical, classes, width)
    p_heldout, possible = cx.exact_distributions(predictor, strings.heldout, classes, width)
    p_ungrammatical, possible_ungrammatical = cx.exact_distributions(
        predictor, strings.ungrammatical, classes, width
    )
    if not possible.all():
        raise ValueError(
            f"{spec.name}: the exact predictor gives a held-out string probability zero."
        )
    exact = cx.measures(
        cx.log2_of(p_heldout),
        cx.log2_of(p_ungrammatical),
        p_heldout,
        heldout_targets,
        ungrammatical_targets,
    )
    exact["ungrammatical_possible"] = int(possible_ungrammatical.sum())
    exact["largest_row_error"] = float(
        np.abs(p_heldout.sum(axis=-1) - 1.0)[heldout_targets != cx.PAD].max()
    )
    exact["mean_entropy_bits"] = cx.mean_entropy_bits(p_heldout, heldout_targets)
    walk = cx.state_model_log2(
        predictor, cx.counted_transitions(predictor), strings.heldout, classes, width
    )
    distinct = list(dict.fromkeys(strings.sequence))
    p_training, _ = cx.exact_distributions(predictor, distinct, classes, width)
    _, training_targets = cx.encode(distinct, classes, width)
    chosen = np.take_along_axis(p_training, np.maximum(training_targets, 0)[..., None], axis=-1)[
        ..., 0
    ]
    probability = np.where(training_targets != cx.PAD, chosen, 1.0).prod(axis=1)
    return JobData(
        spec=spec,
        values=values,
        seed=seed,
        classes=classes,
        strings=strings,
        width=width,
        heldout_inputs=torch.from_numpy(heldout_inputs),
        heldout_targets=heldout_targets,
        ungrammatical_inputs=torch.from_numpy(ungrammatical_inputs),
        ungrammatical_targets=ungrammatical_targets,
        p_heldout=p_heldout,
        exact=exact,
        predictor=predictor,
        probability=dict(zip(distinct, probability.tolist(), strict=True)),
        walk_divergence=cx.divergence_bits(p_heldout, walk, heldout_targets),
        uniform_divergence=cx.uniform_divergence_bits(p_heldout, heldout_targets),  # Eq. (12)
        seconds=time.perf_counter() - start,
    )


# --- one run --------------------------------------------------------------------------------------


@torch.no_grad()
def log2_q(model: torch.nn.Module, inputs: torch.Tensor) -> np.ndarray:
    """The model's log2-probability of every class at every position, as float64."""
    model.eval()
    return (F.log_softmax(model(inputs).double(), dim=-1) / math.log(2.0)).numpy()


@torch.no_grad()
def loss_bits(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Eq. (1) on a set of strings."""
    model.eval()
    scores = model(inputs)
    return float(
        F.cross_entropy(scores.flatten(0, 1), targets.flatten(), ignore_index=cx.PAD)
    ) / math.log(2.0)


def run_seed(seed: int, spec: cx.GrammarSpec, n_train: int) -> int:
    """The seed of the weights, the batch order and the validation split of one run."""
    return int(np.random.SeedSequence([seed, spec.seed, spec.level, n_train]).generate_state(1)[0])


def train_run(
    data: JobData,
    model_name: str,
    n_train: int,
    settings: ComplexitySettings,
    network: Hyperparameters = PRODUCTION,
    progress: Callable[[int, dict], None] | None = None,
) -> dict:
    """Train one network on the first ``n_train`` strings of a job and measure the chosen weights.

    The training seed of the job, the grammar and ``n_train`` fix the validation split, the initial
    weights and the order of the batches. ``progress(step, row)`` is called after every measurement
    of the validation loss. The result holds only numbers, strings, lists and dictionaries.
    """
    threads = set_threads(network.threads)
    probe = speed_probe()
    start = time.perf_counter()
    spec, strings = data.spec, data.strings
    if n_train not in settings.sizes:
        raise ValueError(f"n_train must be one of {settings.sizes}, got {n_train}.")
    seed = run_seed(data.seed, spec, n_train)
    training = strings.training(n_train, settings.training_sample)  # string -> times used
    distinct = list(training)
    gradient, validation = cx.split_training(
        len(distinct), settings.validation_fraction, np.random.default_rng(seed)
    )
    gradient_strings = [distinct[i] for i in gradient]
    validation_strings = [distinct[i] for i in validation]
    # No string twice among the gradient, validation and held-out strings. An ungrammatical string
    # is none of them; two held-out strings may give the same one, and it is then kept twice.
    cx.assert_disjoint(gradient_strings, validation_strings, strings.heldout)
    cx.assert_apart(strings.ungrammatical, gradient_strings, validation_strings, strings.heldout)

    def rows_of(chosen: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Input and target rows of some training strings, each as often as it is used."""
        repeated = [string for string in chosen for _ in range(training[string])]
        return tuple(torch.from_numpy(a) for a in cx.encode(repeated, data.classes, data.width))

    inputs, targets = rows_of(gradient_strings)
    validation_inputs, validation_targets = rows_of(validation_strings)
    used = (targets != cx.PAD).sum(dim=1)  # positions of every training string

    torch.manual_seed(seed)
    model = build_model(
        model_name,
        n_symbols=len(data.classes),
        budget=network.parameter_budget,
        width=network.transformer_width,
        heads=network.transformer_heads,
        layers=network.transformer_layers,
        max_length=data.width,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=network.learning_rate)
    order = torch.Generator().manual_seed(seed)

    history: dict[str, list] = {
        "step": [],
        "train_loss": [],
        "validation_loss": [],
        "divergence_bits": [],
    }
    seconds = {"training": 0.0, "evaluation": 0.0}
    best: dict = {"loss": math.inf, "step": None, "weights": None}

    def measure(step: int, train_loss: float | None) -> None:
        tick = time.perf_counter()
        row = {
            "step": step,
            "train_loss": train_loss,
            "validation_loss": loss_bits(model, validation_inputs, validation_targets),
            "divergence_bits": cx.divergence_bits(
                data.p_heldout, log2_q(model, data.heldout_inputs), data.heldout_targets
            ),
        }
        if row["validation_loss"] < best["loss"]:  # the first step with the lowest validation loss
            best.update(
                loss=row["validation_loss"], step=step, weights=copy.deepcopy(model.state_dict())
            )
        seconds["evaluation"] += time.perf_counter() - tick
        for key, value in row.items():
            history[key].append(value)
        if progress is not None:
            progress(step, row)

    measure(0, None)
    step, losses = 0, []
    tick = time.perf_counter()
    while step < settings.max_steps:
        for rows in torch.randperm(len(inputs), generator=order).split(network.batch_size):
            model.train()
            length = int(
                used[rows].max()
            )  # both networks are causal: later positions change nothing
            scores = model(inputs[rows, :length])
            loss = F.cross_entropy(
                scores.flatten(0, 1), targets[rows, :length].flatten(), ignore_index=cx.PAD
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if step % settings.eval_every == 0:
                seconds["training"] += time.perf_counter() - tick
                measure(step, float(np.mean(losses)) / math.log(2.0))
                losses, tick = [], time.perf_counter()
            if step == settings.max_steps:
                break

    model.load_state_dict(best["weights"])
    found = cx.measures(
        log2_q(model, data.heldout_inputs),
        log2_q(model, data.ungrammatical_inputs),
        data.p_heldout,
        data.heldout_targets,
        data.ungrammatical_targets,
    )
    table = cx.first_order_log2(training, data.classes)
    first_order = cx.measures(
        table[data.heldout_inputs.argmax(dim=-1).numpy()],
        table[data.ungrammatical_inputs.argmax(dim=-1).numpy()],
        data.p_heldout,
        data.heldout_targets,
        data.ungrammatical_targets,
    )
    counted = cx.state_model_log2(
        data.predictor,
        cx.counted_transitions(data.predictor, training),
        strings.heldout,
        data.classes,
        data.width,
    )
    counted_divergence = cx.divergence_bits(data.p_heldout, counted, data.heldout_targets)
    result = {
        "experiment": "complexity",
        "grammar": spec.as_dict(),
        "grammar_values": data.values,
        "level": spec.level,
        "model": model_name,
        "seed": data.seed,
        "n_train": n_train,
        "settings": settings.as_dict(),
        "network": network.as_dict(),
        "n_parameters": n_parameters(model),
        "model_size": free_size(model),
        "n_classes": len(data.classes),
        "data": {
            "n_gradient": len(inputs),
            "n_validation": len(validation_inputs),
            "n_distinct_training": len(distinct),
            "n_distinct_validation": len(validation_strings),
            "n_heldout": len(strings.heldout),
            "n_ungrammatical": len(strings.ungrammatical),
            "n_targets_heldout": int((data.heldout_targets != cx.PAD).sum()),
            "mean_length_training": float(
                np.average([len(s) for s in distinct], weights=list(training.values()))
            ),
            "mean_length_heldout": float(np.mean([len(s) for s in strings.heldout])),
            "distinct_ungrammatical": len(set(strings.ungrammatical)),
            "draws": strings.draws,
            "training_mass": float(sum(data.probability[string] for string in distinct)),
            "run_seed": seed,
        },
        "history": history,
        "best_step": best["step"],
        "best_is_last": best["step"] == history["step"][-1],
        "measures": {**found, "validation_loss": best["loss"]},
        "first_order": first_order,
        # divergences of three references on the held-out targets; the uniform guess is D0
        "references": {
            "walk_without_bounds": data.walk_divergence,
            "counted_transitions": counted_divergence,
            "uniform_guess": data.uniform_divergence,
        },
        "reference_scores": {  # Eq. (13) of tutorial.complexity
            "walk_without_bounds": cx.prediction_score(
                data.walk_divergence, data.uniform_divergence
            ),
            "counted_transitions": cx.prediction_score(counted_divergence, data.uniform_divergence),
        },
        "exact": data.exact,
        "weights_sha256": weights_digest(model),
    }
    result["checks"] = check_run(result)
    result["seconds"] = {"data": data.seconds, **seconds, "total": time.perf_counter() - start}
    result["environment"] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "symseq": installed_commit("symseq") or "sources of the pinned commit",
        "machine": platform.machine(),
        "system": platform.system(),
        "threads": threads,
        "speed_probe_us": [round(probe, 1), round(speed_probe(), 1)],
    }
    return result


def check_run(run: dict) -> dict:
    """What must hold in every run by construction. ``passed`` is False if any of it does not.

    A failure means that the exact predictor, the strings or a measure is not what it is said to be:
    the exact predictor does not separate the ungrammatical strings, its distributions do not sum
    to one, a divergence is negative, the first choice beats the best achievable value, the
    divergence of the uniform guess is not what Eq. (12) of ``tutorial.complexity`` says by its
    second route, or a prediction score is not Eq. (13) of the stored divergences.
    """
    exact, n = run["exact"], run["data"]
    steps = [*run["history"]["divergence_bits"], *run["references"].values()]
    models = [run["measures"], run["first_order"]]
    uniform = run["references"]["uniform_guess"]  # D0
    by_the_entropy = math.log2(run["n_classes"]) - exact["mean_entropy_bits"]
    scored = [  # every stored score with the divergence it was made from
        *((m["score"], m["divergence_bits"]) for m in (*models, exact)),
        *((score, run["references"][name]) for name, score in run["reference_scores"].items()),
    ]
    checks = {
        "uniform guess: divergence is log2 of the outcomes minus the mean entropy of the exact predictor": abs(
            uniform - by_the_entropy
        )
        <= TOLERANCE,
        "uniform guess: divergence is positive": uniform > 0.0,
        "exact predictor: score is 1": abs(exact["score"] - 1.0) <= TOLERANCE,
        "no score exceeds 1": max(score for score, _ in scored) <= 1.0 + TOLERANCE,
        "every score is 1 minus the divergence over that of the uniform guess": uniform > 0.0
        and all(abs(score - (1.0 - d / uniform)) <= TOLERANCE for score, d in scored),
        "exact predictor: area under the curve is 1": exact["auc"] == 1.0,
        "exact predictor: divergence is 0": abs(exact["divergence_bits"]) <= TOLERANCE,
        "exact predictor: first choice is the best achievable": abs(exact["top_choice_ratio"] - 1.0)
        <= TOLERANCE,
        "exact predictor: no ungrammatical string is possible": exact["ungrammatical_possible"]
        == 0,
        "exact predictor: distributions sum to one": exact["largest_row_error"] <= TOLERANCE,
        "divergence is never negative": min([*steps, *(m["divergence_bits"] for m in models)])
        >= -TOLERANCE,
        "first choice never beats the best achievable": max(m["top_choice_ratio"] for m in models)
        <= 1.0 + TOLERANCE,
        "area under the curve lies between 0 and 1": all(0.0 <= m["auc"] <= 1.0 for m in models),
        "numbers of strings are those of the settings": (
            n["n_gradient"] + n["n_validation"] == run["n_train"]
            and n["n_heldout"] == run["settings"]["n_heldout"]
            and n["n_ungrammatical"] == run["settings"]["n_heldout"]
        ),
    }
    return {**checks, "passed": all(checks.values())}


# --- run files ------------------------------------------------------------------------------------


def run_name(spec: cx.GrammarSpec, model: str, n_train: int, seed: int) -> str:
    """File name of a run, without the extension."""
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return f"level{spec.level + 1}-grammar{spec.seed}__{slug}__n{n_train}__seed{seed}"


def job_name(spec: cx.GrammarSpec, seed: int) -> str:
    """Name of a job: one grammar and one training seed, the runs of which share their strings."""
    return f"level{spec.level + 1}-grammar{spec.seed}__seed{seed}"


def record_failure(
    folder: Path, kind: str, spec: cx.GrammarSpec, seed: int, error: BaseException
) -> Path:
    """Write down that a job raised: which job, and the text of the error.

    One file per job in ``failed/`` of the results folder, so that parts that run side by side
    never write the same file. The collect step reads these files; a folder with one of them is
    not complete.
    """
    failed = Path(folder) / "failed"
    failed.mkdir(parents=True, exist_ok=True)
    path = failed / f"{job_name(spec, seed)}.json"
    aside = path.with_suffix(f".{os.getpid()}.part")
    record = {
        "kind": kind,
        "job": job_name(spec, seed),
        "grammar": spec.name,
        "seed": seed,
        "error": f"{type(error).__name__}: {error}",
        "traceback": "".join(traceback.format_exception(error)),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    aside.write_text(json.dumps(record, indent=1))
    os.replace(aside, path)
    return path


def clear_failure(folder: Path, spec: cx.GrammarSpec, seed: int) -> None:
    """Remove the record of a failed job, once the job has run through."""
    (Path(folder) / "failed" / f"{job_name(spec, seed)}.json").unlink(missing_ok=True)


def failed_jobs(folder: Path) -> list[dict]:
    """The records of the jobs of a folder that raised and have not run through since."""
    return [
        json.loads(path.read_text()) for path in sorted((Path(folder) / "failed").glob("*.json"))
    ]


def expected_runs(
    drawn: dict, settings: ComplexitySettings
) -> list[tuple[cx.GrammarSpec, str, int, int]]:
    """Every (grammar, model, training size, seed) of the admitted grammars."""
    return [
        (spec, model, n_train, seed)
        for spec, _ in admitted_grammars(drawn)
        for seed in settings.seeds
        for n_train in settings.sizes
        for model in config.MODELS
    ]


def write_run(result: dict, folder: Path, kind: str) -> Path:
    """Write one run. The file appears complete or not at all (written aside, then renamed)."""
    runs = Path(folder) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    spec = cx.GrammarSpec.from_dict(result["grammar"])
    path = runs / f"{run_name(spec, result['model'], result['n_train'], result['seed'])}.json"
    aside = path.with_suffix(f".{os.getpid()}.part")
    aside.write_text(json.dumps({"kind": kind, **result}, indent=1))
    os.replace(aside, path)
    return path


def saved_run(
    folder: Path,
    kind: str,
    spec: cx.GrammarSpec,
    model: str,
    n_train: int,
    seed: int,
    settings: ComplexitySettings,
    network: Hyperparameters = PRODUCTION,
) -> dict | None:
    """The saved run of exactly this kind, grammar, model, size, seed and settings, or None."""
    path = Path(folder) / "runs" / f"{run_name(spec, model, n_train, seed)}.json"
    if not path.exists():
        return None
    try:
        run = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    same = (
        run.get("kind") == kind
        and run.get("grammar") == json.loads(json.dumps(spec.as_dict()))
        and (run.get("model"), run.get("n_train"), run.get("seed")) == (model, n_train, seed)
        and run.get("settings") == json.loads(json.dumps(settings.as_dict()))
        and run.get("network") == network.as_dict()
    )
    return run if same else None


def summary(runs: list[dict]) -> pd.DataFrame:
    """One row per run: the grammar, the measures of the network and of the first-order model, the time."""
    rows = []
    for run in runs:
        values, found, first = run["grammar_values"], run["measures"], run["first_order"]
        rows.append(
            {
                "level": run["level"] + 1,
                "grammar seed": run["grammar"]["seed"],
                "topological entropy (bits per symbol)": values["topological_entropy_bits"],
                "memory gap (bits per target)": values["memory_gap"],
                "distinct strings": float(values["n_strings"]),
                "typical strings": 2.0 ** values["string_entropy"],
                "mean string length": values["mean_length"],
                "walks accepted": values["acceptance"],
                "model": run["model"],
                "training strings": run["n_train"],
                "seed": run["seed"],
                "parameters": run["n_parameters"],
                "prediction score": found["score"],
                "area under the curve": found["auc"],
                "divergence (bits per target)": found["divergence_bits"],
                "first choice / best achievable": found["top_choice_ratio"],
                "first-order model: prediction score": first["score"],
                "first-order model: area under the curve": first["auc"],
                "first-order model: divergence": first["divergence_bits"],
                "uniform guess: divergence": run["references"]["uniform_guess"],
                "graph known, transitions counted: prediction score": run["reference_scores"][
                    "counted_transitions"
                ],
                "graph known, transitions counted: divergence": run["references"][
                    "counted_transitions"
                ],
                "graph known, length bounds ignored: divergence": run["references"][
                    "walk_without_bounds"
                ],
                "distinct training strings": run["data"]["n_distinct_training"],
                "chosen step": run["best_step"],
                "chosen step is the last": run["best_is_last"],
                "probability of the training strings": run["data"]["training_mass"],
                "draws": run["data"]["draws"],
                "checks passed": run["checks"]["passed"],
                "seconds, data": run["seconds"]["data"],
                "seconds, training": run["seconds"]["training"],
                "seconds, evaluation": run["seconds"]["evaluation"],
                "speed probe (us)": max(run["environment"]["speed_probe_us"]),
            }
        )
    return pd.DataFrame(rows)


def collect(folder: Path, kind: str, settings: ComplexitySettings) -> dict:
    """Build ``summary.csv`` and ``manifest.json`` of a folder from its run files, and return the manifest.

    A run that is expected and absent is listed as missing; it is never given a value. A level with
    fewer admitted grammars than asked for makes the folder incomplete. So does the record of a
    job that raised (``record_failure``): the manifest lists the job and the text of its error.
    """
    folder = Path(folder)
    drawn = load_grammars(folder, kind, settings)
    failed = [
        {key: record[key] for key in ("job", "grammar", "seed", "error")}
        for record in failed_jobs(folder)
    ]
    runs, files, missing = [], {}, []
    for spec, model, n_train, seed in expected_runs(drawn, settings):
        name = run_name(spec, model, n_train, seed)
        run = saved_run(folder, kind, spec, model, n_train, seed, settings)
        if run is None:
            missing.append(name)
            continue
        runs.append(run)
        files[f"runs/{name}.json"] = sha256(folder / "runs" / f"{name}.json")
    summary(runs).to_csv(folder / "summary.csv", index=False)
    short = [level["level"] + 1 for level in drawn["levels"] if not level["complete"]]
    manifest = {
        "experiment": "complexity",
        "kind": kind,
        "placeholder": kind != "production",
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": settings.as_dict(),
        "network": PRODUCTION.as_dict(),
        "seeds": list(settings.seeds),
        "sizes": list(settings.sizes),
        "models": list(config.MODELS),
        "levels_short_of_grammars": short,
        "expected": len(expected_runs(drawn, settings)),
        "present": len(runs),
        "missing": missing,
        "failed_jobs": failed,
        "complete": not missing and not short and not failed,
        "checks_failed": [
            name.removeprefix("runs/").removesuffix(".json")
            for name, run in zip(files, runs, strict=True)
            if not run["checks"]["passed"]
        ],
        "files": files,
        "grammars": {"file": "grammars.json", "sha256": sha256(folder / "grammars.json")},
        "summary": {"file": "summary.csv", "sha256": sha256(folder / "summary.csv")},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_runs(folder: Path) -> tuple[list[dict], dict, dict]:
    """The runs, the manifest and the grammars of a folder, after checking every file against the manifest.

    Raises ``FileNotFoundError`` without a manifest and ``ValueError`` if a listed file is absent or
    changed, if the grammars were changed, or if the folder holds a run file the manifest does not list.
    """
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if sha256(folder / "grammars.json") != manifest["grammars"]["sha256"]:
        raise ValueError("grammars.json was changed after the manifest was written.")
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
    return runs, manifest, json.loads((folder / "grammars.json").read_text())


def default_folder() -> Path:
    """The production results if they are complete, otherwise the pilot results."""
    manifest = FOLDERS["production"] / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["complete"]:
        return FOLDERS["production"]
    return FOLDERS["pilot"]


# --- the numbers of the figure --------------------------------------------------------------------


def figure_data(runs: list[dict], manifest: dict, drawn: dict) -> dict:
    """The numbers of the complexity figure.

    ``grammars``: one entry per admitted grammar, by level and grammar seed. Per model and training
    size, ``seeds`` holds the value of every training seed for each grammar, ``grammar`` the mean
    over seeds (the unit of the figure and of the trend), and ``level`` the mean over the grammars
    of each level. ``first_order`` holds the same for the first-order model. The function raises if
    a grammar has no run for a model and size.
    """
    grammars = [
        {
            "name": spec.name,
            "level": spec.level,
            "seed": spec.seed,
            "topological_entropy_bits": values["topological_entropy_bits"],
            "memory_gap": values["memory_gap"],
            "n_strings": values["n_strings"],
            "mean_length": values["mean_length"],
        }
        for spec, values in admitted_grammars(drawn)
    ]
    levels = sorted({grammar["level"] for grammar in grammars})
    of_level = {
        level: [i for i, g in enumerate(grammars) if g["level"] == level] for level in levels
    }

    def gather(source: str, model: str | None) -> dict:
        out = {}
        for n_train in manifest["sizes"]:
            entry = {measure: {"seeds": [], "grammar": [], "level": []} for measure in MEASURES}
            for grammar in grammars:
                of_grammar = sorted(
                    (
                        run
                        for run in runs
                        if run["grammar"]["name"] == grammar["name"]
                        and run["n_train"] == n_train
                        and (model is None or run["model"] == model)
                    ),
                    key=lambda run: (run["seed"], run["model"]),
                )
                if model is None:  # the first-order model is the same in the runs of both networks
                    of_grammar = [run for run in of_grammar if run["model"] == config.MODELS[0]]
                if not of_grammar:
                    raise ValueError(
                        f"No saved run for {grammar['name']}, {model}, {n_train} strings."
                    )
                for measure in MEASURES:
                    values = [run[source][measure] for run in of_grammar]
                    entry[measure]["seeds"].append(values)
                    entry[measure]["grammar"].append(float(np.mean(values)))
            for measure in MEASURES:
                entry[measure]["level"] = [
                    float(np.mean([entry[measure]["grammar"][i] for i in of_level[level]]))
                    for level in levels
                ]
            out[str(n_train)] = entry
        return out

    return {
        "kind": manifest["kind"],
        "placeholder": manifest["placeholder"],
        "sizes": list(manifest["sizes"]),
        "seeds": list(manifest["seeds"]),
        "grammars": grammars,
        "levels": [
            {
                "level": level,
                "band_bits": drawn["levels"][level]["band_bits"],
                "topological_entropy_bits": float(
                    np.mean([grammars[i]["topological_entropy_bits"] for i in of_level[level]])
                ),
            }
            for level in levels
        ],
        "models": {model: gather("measures", model) for model in manifest["models"]},
        "first_order": gather("first_order", None),
    }


# --- the tests of the expectation -----------------------------------------------------------------


def spearman(x, y) -> float:
    """Eq. (2): Spearman's rank correlation, with average ranks for ties (nan if one side is constant)."""
    r, s = rankdata(x), rankdata(y)
    if np.ptp(r) == 0 or np.ptp(s) == 0:
        return float("nan")
    return float(np.corrcoef(r, s)[0, 1])


def permutation_p(
    x, y, direction: int, n_random: int, seed: int, exact_up_to: int = 9
) -> tuple[float, str]:
    """Eq. (3): one-sided p-value of Spearman's rho under reordering, and how it was obtained.

    ``direction`` is +1 if the expectation is a positive correlation and -1 if negative. All
    orderings are counted for at most ``exact_up_to`` values, and ``n_random`` are drawn otherwise.
    """
    r, s = rankdata(x), rankdata(y)
    if np.ptp(r) == 0 or np.ptp(s) == 0:
        return float("nan"), "one side is constant"
    r = (r - r.mean()) / r.std()
    s = (s - s.mean()) / s.std()
    observed = direction * float(r @ s) / len(r)
    slack = 1e-12
    if len(r) <= exact_up_to:
        orders = np.array(list(itertools.permutations(range(len(r)))))
        rho = direction * (s[orders] @ r) / len(r)
        return float((rho >= observed - slack).mean()), f"all {len(orders)} orderings"
    rng = np.random.default_rng(seed)
    orders = rng.permuted(np.tile(np.arange(len(r)), (n_random, 1)), axis=1)
    rho = direction * (s[orders] @ r) / len(r)
    return float(
        (1 + (rho >= observed - slack).sum()) / (1 + n_random)
    ), f"{n_random} random orderings"


# The two measures that carry the expectation. Both are expected to fall as the topological entropy
# rises (direction -1), both are better when larger, and each has its mark of a ceiling.
EXPECTED_DIRECTION = {"score": -1, "auc": -1}
CEILING = {"score": config.CEILING_SCORE, "auc": config.CEILING_AUC}
WITHOUT_EXPECTATION = ("divergence_bits",)  # listed beside them, never given a verdict
MEASURE_NAMES = {
    "score": "prediction score",
    "auc": "area under the curve",
    "divergence_bits": "divergence (bits per target)",
    "top_choice_ratio": "first choice / best achievable",
}
SUPPORTED, NOT_SUPPORTED, NO_EXPECTATION = "supported", "not supported", "no expectation"


def trend_table(data: dict) -> pd.DataFrame:
    """Per network, training size and measure: Spearman's rho across grammars and its permutation p-value.

    The first rows are the cells of the expectation: the prediction score and the area under the
    curve, twelve cells for two networks and three training sizes. A cell is ``supported`` if rho
    is negative and p is at most ``config.TREND_ALPHA``. ``at ceiling`` is True if every grammar is
    at or above ``config.CEILING_SCORE`` or ``config.CEILING_AUC``; such a cell cannot show a
    trend, and that is its result. The last rows are the divergence in bits: it carries no
    expectation, its p-value is the one-sided value for a fall, and its verdict is always
    ``no expectation``.
    """
    x = [grammar["topological_entropy_bits"] for grammar in data["grammars"]]
    rows = []
    for measure in (*EXPECTED_DIRECTION, *WITHOUT_EXPECTATION):
        tested = measure in EXPECTED_DIRECTION
        direction = EXPECTED_DIRECTION.get(measure, -1)
        for model, by_size in data["models"].items():
            for n_train, entry in by_size.items():
                y = entry[measure]["grammar"]
                rho = spearman(x, y)
                p, how = permutation_p(
                    x, y, direction, config.TREND_PERMUTATIONS, config.TREND_SEED
                )
                supported = not math.isnan(rho) and direction * rho > 0 and p <= config.TREND_ALPHA
                rows.append(
                    {
                        "network": config.MODEL_LABELS[model],
                        "training strings": int(n_train),
                        "measure": MEASURE_NAMES[measure],
                        "expected": "falls" if tested else "none",
                        "grammars": len(y),
                        "Spearman rho": rho,
                        "p (one-sided, for a fall)": p,
                        "p from": how,
                        "at ceiling": bool(tested and min(y) >= CEILING[measure]),
                        "verdict": (SUPPORTED if supported else NOT_SUPPORTED)
                        if tested
                        else NO_EXPECTATION,
                    }
                )
    return pd.DataFrame(rows)


def size_table(data: dict) -> pd.DataFrame:
    """Per network and measure: in how many grammars the measure is better with more training strings.

    For each pair of sizes the grammars are counted in which the larger size is better (a larger
    prediction score, a larger area under the curve); p is the one-sided sign test on the grammars
    that differ. The expectation "rises with N" is read from the pair of the smallest and largest size.
    """
    rows = []
    sizes = [str(size) for size in data["sizes"]]
    for model, by_size in data["models"].items():
        for measure in EXPECTED_DIRECTION:
            for small, large in itertools.combinations(sizes, 2):
                gain = np.array(by_size[large][measure]["grammar"]) - np.array(
                    by_size[small][measure]["grammar"]
                )
                better, differ = int((gain > 0).sum()), int((gain != 0).sum())
                rows.append(
                    {
                        "network": config.MODEL_LABELS[model],
                        "measure": MEASURE_NAMES[measure],
                        "sizes": f"{small} to {large}",
                        "grammars": len(gain),
                        "better with more strings": better,
                        "p (sign test, one-sided)": float(binom.sf(better - 1, differ, 0.5))
                        if differ
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def mean_table(data: dict) -> pd.DataFrame:
    """Per level, network and training size: the mean over the grammars of the level.

    The prediction score and the area under the curve of both networks and of the first-order
    model, and the divergence of the networks in bits per target, which is not compared across levels.
    """
    rows = []
    for position, level in enumerate(data["levels"]):
        for n_train in data["sizes"]:
            row = {
                "level": level["level"] + 1,
                "topological entropy (bits per symbol)": level["topological_entropy_bits"],
                "training strings": n_train,
            }
            series = {
                config.MODEL_LABELS[model]: by_size for model, by_size in data["models"].items()
            }
            series["first-order model"] = data["first_order"]
            for measure, name in (("score", "score"), ("auc", "area")):
                for label, by_size in series.items():
                    row[f"{label}: {name}"] = by_size[str(n_train)][measure]["level"][position]
            for model, by_size in data["models"].items():
                row[f"{config.MODEL_LABELS[model]}: divergence (bits)"] = by_size[str(n_train)][
                    "divergence_bits"
                ]["level"][position]
            rows.append(row)
    return pd.DataFrame(rows).set_index(["level", "training strings"])


def history_table(runs: dict[str, dict], every: int = 5) -> pd.DataFrame:
    """Validation loss, prediction score and divergence of several runs, every few measurements.

    The score of a step is Eq. (13) of ``tutorial.complexity`` from the divergence of that step and
    the divergence of the uniform guess stored with the run.
    """
    first = next(iter(runs.values()))
    steps = first["history"]["step"]
    rows = []
    for index, step in enumerate(steps):
        if index % every and index != len(steps) - 1:
            continue
        row = {"step": step}
        for name, run in runs.items():
            divergence = run["history"]["divergence_bits"][index]
            row[f"{name}: validation loss"] = run["history"]["validation_loss"][index]
            row[f"{name}: score"] = cx.prediction_score(
                divergence, run["references"]["uniform_guess"]
            )
            row[f"{name}: divergence (bits)"] = divergence
        rows.append(row)
    return pd.DataFrame(rows).set_index("step")


def measures_table(runs: dict[str, dict]) -> pd.DataFrame:
    """The measures of several runs, of the first-order model and of the two anchors of the score.

    The uniform guess gives every target the surprisal log2 K, with K the number of outcomes. Its
    divergence is the D0 stored with the run, so its score is 0 (Eq. 13). Every string then has the
    same largest surprisal (Eq. 9), so its area under the curve is decided by the rule for ties of
    Eq. (10), which gives one half: the area is computed by that function from the numbers of
    strings of the run. Its first choice is not defined, because every outcome ties and Eq. (8)
    has no rule for ties, so that cell stays empty.
    """
    first = next(iter(runs.values()))

    def row(found: dict) -> dict:
        return {
            MEASURE_NAMES["score"]: found["score"],
            MEASURE_NAMES["auc"]: found["auc"],
            MEASURE_NAMES["divergence_bits"]: found["divergence_bits"],
            MEASURE_NAMES["top_choice_ratio"]: found["top_choice_ratio"],
        }

    rows = {
        name: {**row(run["measures"]), "chosen step": run["best_step"]}
        for name, run in runs.items()
    }
    rows["first-order model"] = row(first["first_order"])
    rows["exact predictor"] = row(first["exact"])
    guess = first["references"]["uniform_guess"]  # D0
    surprisal = np.log2(first["n_classes"])  # of every target, so the largest of every string
    rows["uniform guess"] = {
        MEASURE_NAMES["score"]: cx.prediction_score(guess, guess),  # Eq. (13)
        MEASURE_NAMES["auc"]: cx.auc(  # Eq. (10)
            np.full(first["data"]["n_ungrammatical"], surprisal),
            np.full(first["data"]["n_heldout"], surprisal),
        ),
        MEASURE_NAMES["divergence_bits"]: guess,
    }
    return pd.DataFrame(rows).T


def job_table(runs: list[dict]) -> pd.DataFrame:
    """Per level and training size: what the strings of the runs are, as means over grammars and seeds.

    The number of distinct strings among the N training strings, the probability that the generator
    gives to one of them, the mean length of the training and of the held-out strings beside the
    generator's own, the divergence of the uniform guess on the held-out targets (D0), and how a
    network divides the N strings: those of its gradient steps and those that only choose its
    weights (a fifth of the distinct strings, each with all its repeats).
    """
    rows = []
    reference = [run for run in runs if run["model"] == config.MODELS[0]]  # one network: same jobs
    for level in sorted({run["level"] for run in reference}):
        for n_train in sorted({run["n_train"] for run in reference}):
            of_cell = [r for r in reference if r["level"] == level and r["n_train"] == n_train]
            rows.append(
                {
                    "level": level + 1,
                    "training strings": n_train,
                    "distinct among them": float(
                        np.mean([r["data"]["n_distinct_training"] for r in of_cell])
                    ),
                    "probability of the training strings": float(
                        np.mean([r["data"]["training_mass"] for r in of_cell])
                    ),
                    "mean length: generator": float(
                        np.mean([r["grammar_values"]["mean_length"] for r in of_cell])
                    ),
                    "mean length: training": float(
                        np.mean([r["data"]["mean_length_training"] for r in of_cell])
                    ),
                    "mean length: held-out": float(
                        np.mean([r["data"]["mean_length_heldout"] for r in of_cell])
                    ),
                    "uniform guess: divergence (bits)": float(
                        np.mean([r["references"]["uniform_guess"] for r in of_cell])
                    ),
                    # how a network uses the N strings; the first-order model counts on all of them
                    "used for gradient steps": float(
                        np.mean([r["data"]["n_gradient"] for r in of_cell])
                    ),
                    "used to choose the weights": float(
                        np.mean([r["data"]["n_validation"] for r in of_cell])
                    ),
                    "distinct among those": float(
                        np.mean([r["data"]["n_distinct_validation"] for r in of_cell])
                    ),
                }
            )
    return pd.DataFrame(rows).set_index(["level", "training strings"])


def projected_seconds(runs: list[dict], n_jobs: int, slow_probe_us: float, processes: int) -> dict:
    """Wall time of ``n_jobs`` jobs from the measured runs, scaled to a given speed of the workstation.

    The time of a run is taken as proportional to the speed probe (microseconds for
    ``sum(range(1000))``) measured at its start and end. A job is one grammar and seed: its data
    once, and every run of the two networks and the training sizes.
    """
    per_probe = [
        (run["seconds"]["training"] + run["seconds"]["evaluation"])
        / float(np.mean(run["environment"]["speed_probe_us"]))
        for run in runs
    ]
    data = {
        (run["grammar"]["name"], run["seed"]): run["seconds"]["data"]
        / float(np.mean(run["environment"]["speed_probe_us"]))
        for run in runs
    }
    jobs = {(run["grammar"]["name"], run["seed"]) for run in runs}
    runs_per_job = len(runs) / len(jobs)
    job = (
        float(np.mean(per_probe)) * runs_per_job + float(np.mean(list(data.values())))
    ) * slow_probe_us
    return {
        "seconds_per_job": job,
        "jobs": n_jobs,
        "processes": processes,
        "speed_probe_us": slow_probe_us,
        "hours": job * n_jobs / processes / 3600.0,
    }
