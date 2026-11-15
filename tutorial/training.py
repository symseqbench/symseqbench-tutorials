"""One training and evaluation loop, used by the networks notebook and by ``scripts/run_precomputed.py``.

## Theory

The network returns scores ``z_t`` over the classes at every position. With the model's distribution
``q_t = softmax(z_t)`` and the target ``x_{t+1}``:

    Eq. (1)  cross-entropy = -(1 / N) sum_targets log2 q_t(x_{t+1})      bits per target
    Eq. (2)  accuracy      =  (1 / N) sum_targets [argmax q_t == x_{t+1}]

Both are measured on held-out samples after every epoch, next to their optimum: what a predictor
that knows the grammar scores on the same targets (``tutorial.datasets.optimum``, computed from
``tutorial.exact``). Two conventions for the accuracy, as in ``tutorial.exact``:

- ``with_eos``: every target that follows a symbol, the end marker included;
- ``within``: the targets that are symbols, with the prediction restricted to symbols.

No model can do better than the optimum except by chance. ``check_against_optimum`` states how much
chance allows:

    Eq. (3)  N * (optimum cross-entropy - model cross-entropy) <= BITS_TOLERANCE bits
    Eq. (4)  model hits - expected optimum hits <= SIGMAS * sqrt(variance of the optimum hits)

Eq. (3): for held-out samples drawn from the grammar with distribution p and any model q fitted
without them, E_p[q(x) / p(x)] <= 1, so by Markov's inequality the model's total log-likelihood
exceeds the generator's by b bits with probability at most 2^-b. Eq. (4): the number of hits of
any predictor has a mean and a variance no larger than those of the best predictor, and SIGMAS = 5
standard deviations is a one-sided probability of about 3e-7 for a normal variable. A run that
violates either was not trained and scored as described: a target leaked into the input, or the
optimum belongs to another grammar or another set of targets.
"""

from __future__ import annotations

import hashlib
import math
import platform
import time
import timeit
from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F

from tutorial import exact
from tutorial.config import PRODUCTION, Hyperparameters
from tutorial.datasets import EOS_INDEX, PAD, RungData, Split, build_rung_data
from tutorial.env import installed_commit
from tutorial.ladder import Rung, build_generator
from tutorial.models import build_model, free_size, n_parameters

BITS_TOLERANCE = 20.0  # Eq. (3): probability at most 2^-20 per evaluation
SIGMAS = 5.0  # Eq. (4)
MEASURES = ("accuracy_with_eos", "accuracy_within", "cross_entropy")


def set_threads(threads: int) -> int:
    """Limit torch to ``threads`` threads for its own operations and return the number in force."""
    if not isinstance(threads, int) or threads < 1:
        raise ValueError(f"threads must be a positive integer, got {threads!r}.")
    torch.set_num_threads(threads)
    return torch.get_num_threads()


def speed_probe() -> float:
    """Microseconds for ``sum(range(1000))``: about 10 on an idle core of a recent laptop."""
    return timeit.timeit(lambda: sum(range(1000)), number=1000) / 1000 * 1e6


@torch.no_grad()
def evaluate(model: torch.nn.Module, split: Split) -> dict[str, float]:
    """Accuracy under both conventions and cross-entropy in bits per target.  Eq. (1), Eq. (2)"""
    model.eval()
    scores = model(split.inputs)
    targets, tokens = split.targets, split.tokens
    scored = targets != PAD
    log_q = F.log_softmax(scores, dim=-1)
    bits = -log_q[scored].gather(1, targets[scored].unsqueeze(1)).sum() / math.log(2)

    with_eos = scored & (tokens != EOS_INDEX)
    hits_with_eos = (scores.argmax(dim=-1) == targets)[with_eos].sum()
    within = with_eos & (targets != EOS_INDEX)
    symbol_guess = scores[..., EOS_INDEX + 1 :].argmax(dim=-1) + EOS_INDEX + 1
    hits_within = (symbol_guess == targets)[within].sum()
    return {
        "accuracy_with_eos": float(hits_with_eos) / int(with_eos.sum()),
        "accuracy_within": float(hits_within) / int(within.sum()),
        "cross_entropy": float(bits) / int(scored.sum()),
    }


def exact_values(rung: Rung) -> dict[str, float]:
    """The generator's own values for an endless sequence of joined strings (``tutorial.exact``).

    ``conditional_entropy`` is the best achievable cross-entropy per SeqBench target: the entropy of
    a string without that of its first symbol, which is never a target, divided by the mean number
    of targets per string. ``entropy_rate`` counts every symbol, the first one and the marker too.
    """
    predictor = exact.predictor_for(build_generator(rung), **rung.sample)
    values = exact.trial_statistics(predictor)
    first = predictor.distribution(predictor.initial())
    first_entropy = -sum(p * math.log2(p) for p in first.values() if p > 0.0)
    keep = (
        "mean_length",
        "string_entropy",
        "entropy_rate",
        "ceiling_with_eos",
        "ceiling_within",
        "previous_symbol_with_eos",
        "previous_symbol_within",
    )
    return {
        **{key: values[key] for key in keep},
        "first_symbol_entropy": first_entropy,
        "conditional_entropy": (values["string_entropy"] - first_entropy) / values["mean_length"],
    }


def check_against_optimum(history: dict, optimum: dict) -> dict:
    """Eq. (3) and Eq. (4) at every epoch. ``passed`` is False if any epoch beats the optimum."""
    n = optimum["n_targets"]
    advantage = [
        n * (optimum["cross_entropy_realised"] - value) for value in history["cross_entropy"]
    ]
    out = {
        "cross_entropy": {
            "largest_advantage_bits": max(advantage),
            "allowed_bits": BITS_TOLERANCE,
            "passed": max(advantage) <= BITS_TOLERANCE,
        }
    }
    for convention in ("with_eos", "within"):
        count = optimum[f"n_{convention}"]
        allowed = SIGMAS * math.sqrt(optimum[f"variance_{convention}"]) / count
        excess = max(
            value - optimum[f"accuracy_{convention}"] for value in history[f"accuracy_{convention}"]
        )
        out[f"accuracy_{convention}"] = {
            "largest_excess": excess,
            "allowed": allowed,
            "passed": excess <= allowed + 1e-12,
        }
    out["passed"] = all(entry["passed"] for entry in out.values())
    return out


def weights_digest(model: torch.nn.Module) -> str:
    """SHA-256 of all parameters, to compare the weights of two runs without storing them."""
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def train_run(
    rung: Rung,
    model_name: str,
    seed: int,
    settings: Hyperparameters = PRODUCTION,
    data: RungData | None = None,
    progress: Callable[[int, dict], None] | None = None,
) -> dict:
    """Train one model on one rung and measure it on held-out samples after every epoch.

    ``seed`` draws the training samples (unless ``data`` is given), the initial weights and the order
    of the batches. Entry ``e`` of every list in ``history`` is the measurement after ``e`` epochs;
    entry 0 is the untrained model. ``progress(epoch, row)`` is called after every measurement.
    The result holds only numbers, strings, lists and dictionaries.
    """
    threads = set_threads(settings.threads)
    probe = speed_probe()
    start = time.perf_counter()
    if data is None:
        data = build_rung_data(rung, seed, settings)
    train, heldout = data.train, data.heldout

    torch.manual_seed(seed)  # after the data: SeqBench reseeds torch when it builds a dataset
    model = build_model(
        model_name,
        n_symbols=len(data.symbols),
        budget=settings.parameter_budget,
        width=settings.transformer_width,
        heads=settings.transformer_heads,
        layers=settings.transformer_layers,
        max_length=settings.sample_length,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    order = torch.Generator().manual_seed(seed)

    history: dict[str, list] = {"epoch": [], "train_loss": [], **{key: [] for key in MEASURES}}
    seconds = {"training": 0.0, "evaluation": 0.0}

    def measure(epoch: int, train_loss: float | None) -> None:
        tick = time.perf_counter()
        row = {"epoch": epoch, "train_loss": train_loss, **evaluate(model, heldout)}
        seconds["evaluation"] += time.perf_counter() - tick
        for key, value in row.items():
            history[key].append(value)
        if progress is not None:
            progress(epoch, row)

    measure(0, None)
    for epoch in range(1, settings.epochs + 1):
        tick = time.perf_counter()
        model.train()
        losses = []
        for rows in torch.randperm(len(train), generator=order).split(settings.batch_size):
            scores = model(train.inputs[rows])
            loss = F.cross_entropy(
                scores.flatten(0, 1), train.targets[rows].flatten(), ignore_index=PAD
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        seconds["training"] += time.perf_counter() - tick
        measure(epoch, float(np.mean(losses)) / math.log(2))

    return {
        "rung": rung.name,
        "family": rung.family,
        "model": model_name,
        "seed": seed,
        "settings": settings.as_dict(),
        "n_parameters": n_parameters(model),
        "model_size": free_size(model),
        "n_classes": len(data.symbols),
        "data": {
            "n_train": len(train),
            "n_train_removed": data.n_train_removed,
            "n_heldout": len(heldout),
            "seqbench_seeds": data.seeds,
        },
        "exact": exact_values(rung),
        "optimum": data.optimum,
        "history": history,
        "final": {key: history[key][-1] for key in MEASURES},
        "weights_sha256": weights_digest(model),
        "checks": check_against_optimum(history, data.optimum),
        "seconds": {
            "data": data.seconds,
            **seconds,
            "total": time.perf_counter() - start,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "symseq": installed_commit("symseq") or "sources of the pinned commit",
            "seqbench": installed_commit("seqbench") or "sources of the pinned commit",
            "machine": platform.machine(),
            "system": platform.system(),
            "threads": threads,
            "speed_probe_us": [round(probe, 1), round(speed_probe(), 1)],
        },
    }


def reaches_optimum(
    result: dict, epoch: int, accuracy_margin: float = 0.02, bits_margin: float = 0.05
) -> bool:
    """True if after ``epoch`` epochs the accuracy (with the marker) and the cross-entropy are within the margins of the optimum."""
    history, optimum = result["history"], result["optimum"]
    return (
        history["accuracy_with_eos"][epoch] >= optimum["accuracy_with_eos"] - accuracy_margin
        and history["cross_entropy"][epoch] <= optimum["cross_entropy"] + bits_margin
    )


def history_table(runs: dict[str, dict], every: int = 5) -> list[dict]:
    """Held-out accuracy (with the marker) and cross-entropy of several runs side by side, every few epochs."""
    first = next(iter(runs.values()))
    epochs = first["history"]["epoch"]
    rows = []
    for index, epoch in enumerate(epochs):
        if epoch % every and epoch != epochs[-1]:
            continue
        row = {"epoch": epoch}
        for name, run in runs.items():
            row[f"{name}: accuracy"] = run["history"]["accuracy_with_eos"][index]
            row[f"{name}: cross-entropy"] = run["history"]["cross_entropy"][index]
        row["best achievable accuracy"] = first["optimum"]["accuracy_with_eos"]
        row["best achievable cross-entropy"] = first["optimum"]["cross_entropy"]
        rows.append(row)
    return rows
