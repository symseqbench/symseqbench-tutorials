"""The measures of notebook B: SymSeq's measures at four levels on a set of strings and its two surrogates.

Every number is the output of one SymSeq function, called with the arguments listed in ``MEASURES``;
nothing is re-implemented. The measures and their parameters are those of the behaviour figure of
the SymSeqBench preprint wherever the pinned SymSeq provides them. Where the time or memory budget
of a laptop needs a cheaper setting, it is in ``SETTINGS`` and nowhere else, next to the value of
the preprint (``PREPRINT``).

Two measures are left out on purpose. Permutation entropy is defined for ordered values: on
behaviour labels, which have no order, its value changes when the labels are renamed
(``tests/test_behaviour_measures.py`` shows it). A mutual information between two whole strings
needs strings that are aligned position by position, which behaviour strings are not. The edit
distance stays as the comparison of strings at the string-set level.

Units: entropies in bits. SymSeq returns topological entropy in nats; it is converted here
(division by ln 2).

## Theory

Entropy rate at order k (bits per symbol), from the maximised log-likelihood l_k (nats) that
``markov_order_selection`` returns for the order-k Markov fit of n targets inside the strings:

    h_k = -l_k / (n ln 2)                                                          Eq. (1)

This is the plug-in conditional entropy of the next symbol given the k symbols before it. In the
limit of much data it is an upper bound of the entropy rate and decreases with k; with few data it
falls below the true value, because contexts seen once predict their own continuation.

Topological entropy of the observed first-order graph, A[i, j] = 1 if symbol j ever follows symbol
i inside a string:

    h_top = log2(spectral radius of A) <= log2(number of symbols)                  Eq. (2)

Known answers on the surrogates (checks stored with every result):

- first-order surrogate: BIC selects order 1 and |h_1 - h| <= max(0.02, 5 s / sqrt(n)), with h and
  s the entropy rate and the surprisal standard deviation of Eq. (4) of ``tutorial.behaviour``;
- shuffled surrogate: BIC selects order 0; the memory h_0 - h_1 is at most the largest value
  among 500 other shuffles of the same set, each computed with numpy from its table of pairs
  (Eq. 3: h_0 - h_1 = sum_ij p_ij log2(p_ij / (p_i. p_.j)), the mutual information of neighbouring
  symbols); the mutual information lies outside SymSeq's 95 % shuffle band at no more distances
  than the 99.9 % point of a binomial count with probability 0.08 per distance (a 95 % normal band
  on a skewed statistic, from a small number of shuffles).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from symseq.metrics.grammar import (
    markov_order_selection,
    mi_decay_analysis,
    topological_entropy,
    vlmc_fit,
)
from symseq.metrics.string import (
    compute_compression_metrics_ensemble,
    compute_entropy_metrics_ensemble,
    entropy,
)
from symseq.metrics.stringset import edit_distance, string_set_entropy
from symseq.metrics.token import most_common_tokens, token_frequency
from symseq.utils.strtools import concatenate_string_set

from tutorial import behaviour

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "data" / "behaviour_measures"
TIMINGS = "timings.json"  # wall times of the last run; not part of the results and not hashed
LN2 = math.log(2.0)


@dataclass(frozen=True)
class Settings:
    """Every parameter of the measures. ``PREPRINT`` holds the values of the preprint's figure."""

    min_string_length: int = 10  # strings shorter than this are left out of the per-string measures
    max_block_size: int = 10  # block entropies up to min(this, shortest string // 2)
    n_compared: int = 50  # strings compared pairwise
    compared_length: int = 100  # tokens of each compared string that are used
    compared_seed: int = 42
    markov_max_order: int = 5
    vlmc_max_depth: int = 5
    mi_max_distance: int = 50  # also limited to a tenth of the sequence
    mi_shuffles: int = 100
    mi_confidence: float = 0.95
    mi_seed: int = 42
    lift_max: int = 15  # also limited to max(2, tokens // 1000)
    lift_min_tokens: int = 500
    lift_max_blocks: int = 1_000_000_000  # no limit in the preprint
    first_order_seed: int = 1
    shuffled_seed: int = 2
    band_shuffles: int = (
        500  # shuffles of the memory band that the shuffled surrogate is checked against
    )
    band_seed: int = 3
    convergence_tokens: tuple[int, ...] = (250, 500, 1000, 2000, 5000, 10_000, 20_000)

    def as_dict(self) -> dict:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


PREPRINT = Settings()

# The settings of the tutorial. Changed against the preprint, each for the reason given in
# ``CHEAPER``; everything else is the preprint's value.
SETTINGS = replace(
    PREPRINT,
    n_compared=30,
    compared_length=50,
    mi_max_distance=20,
    mi_shuffles=20,
    lift_max_blocks=1500,
)

CHEAPER = {
    "n_compared": "the edit distance is computed pair by pair in Python; 435 pairs instead of 1225",
    "compared_length": "the cost of one edit distance grows with the product of the two lengths",
    "mi_max_distance": "each distance costs one pass over the sequence for every shuffle",
    "mi_shuffles": "each shuffle costs one pass over the sequence for every distance",
    "lift_max_blocks": "the lift method builds a full table of size (number of distinct blocks) squared; "
    "it stops before a lift with more distinct blocks than this, so that the table stays below 20 MB",
}

SEEDS_NOTE = (
    "Seeds: 42 for every draw that the preprint's analysis also makes (compared strings, shuffles of "
    "the mutual-information baseline), 1 for the first-order surrogate, 2 for the shuffled surrogate, "
    "3 for the shuffles of the memory band."
)

# (key, level, SymSeq function and arguments, what the number is)
MEASURES: list[tuple[str, str, str, str]] = [
    ("types", "token", "token_frequency(joined)", "number of different symbols"),
    (
        "top_frequency",
        "token",
        "most_common_tokens(joined, n=1)",
        "relative frequency of the most frequent symbol",
    ),
    (
        "symbol_entropy",
        "string",
        "entropy(s, bias_correction=True), per string",
        "H_1: entropy of single symbols with the Miller-Madow correction, bits; mean over strings",
    ),
    (
        "entropy_rate",
        "string",
        'compute_entropy_metrics_ensemble(strings, max_block_size=L, min_sequence_length=10)["entropy_rate"]',
        "h_R of each string at the order R chosen by BIC, bits per symbol; mean over the strings long "
        "enough for an estimate (n says how many)",
    ),
    (
        "excess_entropy",
        "string",
        'compute_entropy_metrics_ensemble(...)["emc"]',
        "memory E_R = H_R - R h_R of each string, bits; mean over the same strings",
    ),
    (
        "gzip",
        "string",
        'compute_compression_metrics_ensemble(strings, min_sequence_length=10)["compressibility"]',
        "gzip payload in bytes divided by the length of the string; mean over strings",
    ),
    (
        "lzw",
        "string",
        'compute_compression_metrics_ensemble(strings, min_sequence_length=10)["lzw_complexity"]',
        "number of LZW codes divided by the length of the string; mean over strings",
    ),
    (
        "saturation",
        "string",
        'compute_entropy_metrics_ensemble(...)["saturation"]["k_star"]',
        "block length k* at which the block entropy of the strings stops growing (a description of the "
        "curve; few data flatten it too)",
    ),
    ("n_strings", "string set", "(no SymSeq call)", "number of strings"),
    ("mean_length", "string set", "(no SymSeq call)", "mean number of symbols per string"),
    (
        "set_entropy",
        "string set",
        "string_set_entropy(strings, use_frequencies=True)",
        "entropy of the distribution of whole strings, bits",
    ),
    (
        "edit_distance",
        "string set",
        "edit_distance(s, t) for every pair of the compared strings",
        "mean Levenshtein distance between two strings (a random subset of the strings, each cut to its first tokens)",
    ),
    ("states", "grammar", "context_counts(strings, max_depth=1)", "symbols that occur"),
    (
        "transitions",
        "grammar",
        "context_counts(strings, max_depth=1)",
        "different pairs of successive symbols that occur inside strings",
    ),
    (
        "density",
        "grammar",
        "(transitions / states squared)",
        "share of all possible pairs that occur",
    ),
    (
        "markov_order",
        "grammar",
        'markov_order_selection(strings, max_order=5, criterion="BIC")["optimal_order"]',
        "Markov order with the lowest BIC; no context crosses a string boundary",
    ),
    (
        "entropy_rate_by_order",
        "grammar",
        'markov_order_selection(...)["log_likelihoods"] / (n ln 2), sign changed',
        "h_k: entropy of the next symbol given the k symbols before it, bits per symbol, for k = 0 to 5",
    ),
    (
        "vlmc_contexts",
        "grammar",
        'vlmc_fit(strings, max_depth=5)["num_terminal_contexts"]',
        "number of terminal contexts of the fitted variable-length Markov chain",
    ),
    (
        "vlmc_depth",
        "grammar",
        'vlmc_fit(strings, max_depth=5)["max_depth_used"]',
        "length of its longest context",
    ),
    (
        "topological_direct",
        "grammar",
        'topological_entropy(transitions=A, method="direct") / ln 2',
        "log2 of the largest eigenvalue of the table of observed transitions, bits per symbol; at most "
        "log2 of the number of symbols",
    ),
    (
        "topological_lift",
        "grammar",
        'topological_entropy(joined, method="lift", max_lift=m)[1] / ln 2',
        "the same quantity estimated from blocks of growing length; it stops at the first lift that "
        "changes the value by less than 0.05 nats",
    ),
    (
        "mi",
        "grammar",
        'mi_decay_analysis(joined, max_distance=d, n_shuffles=n, confidence_level=0.95, units="bits")',
        "mutual information between two symbols d positions apart, with the band of shuffled sequences",
    ),
    (
        "distances above the shuffle band",
        "grammar",
        'mi_decay_analysis(...)["mi_values"] > mi_decay_analysis(...)["mi_ci_upper"]',
        "number of distances at which the mutual information is above the shuffle band; about one "
        "in forty is expected for a sequence without order",
    ),
]

LEVELS = ("token", "string", "string set", "grammar")


class _quiet:
    """Silence SymSeq's log messages and warnings on short or sparse sequences inside a block."""

    def __enter__(self):
        self._logger = logging.getLogger("symseq")
        self._level = self._logger.level
        self._logger.setLevel(logging.ERROR)
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.simplefilter("ignore")
        return self

    def __exit__(self, *exc):
        self._warnings.__exit__(*exc)
        self._logger.setLevel(self._level)
        return False


def joined(strings: Sequence[Sequence[str]]) -> list[str]:
    """The strings one after the other, without a marker between them (SymSeq's ``concatenate_string_set``)."""
    return concatenate_string_set([list(string) for string in strings])


def summarise(values: Sequence[float]) -> dict:
    """Mean and standard error of the values that exist, and how many exist."""
    array = np.asarray(values, dtype=float)
    present = array[~np.isnan(array)]
    n = int(present.size)
    return {
        "mean": float(present.mean()) if n else float("nan"),
        "sem": float(present.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "n": n,
        "of": int(array.size),
    }


# --------------------------------------------------------------------------------------------
# The four levels
# --------------------------------------------------------------------------------------------


def token_level(strings: Sequence[Sequence[str]]) -> dict:
    """Token level: how many symbols there are and how unevenly they are used."""
    stream = joined(strings)
    frequencies = token_frequency(stream, as_freq=True)
    top = most_common_tokens(stream, n=1)
    return {
        "n_tokens": len(stream),
        "types": len(frequencies),
        "top_frequency": float(next(iter(top.values()))),
        "frequencies": {symbol: float(frequencies[symbol]) for symbol in sorted(frequencies)},
    }


def string_level(strings: Sequence[Sequence[str]], settings: Settings = SETTINGS) -> dict:
    """String level, string by string: entropy, entropy rate, memory, compression."""
    kept = [list(s) for s in strings if len(s) >= settings.min_string_length]
    if not kept:
        return {
            "n_strings": 0,
            "max_block_size": 0,
            "summary": {},
            "per_string": {},
            "block_entropy": [],
        }
    max_block = min(settings.max_block_size, min(len(s) for s in kept) // 2)
    with _quiet():
        ensemble = compute_entropy_metrics_ensemble(
            kept, max_block_size=max_block, min_sequence_length=settings.min_string_length
        )
        compression = compute_compression_metrics_ensemble(
            kept, min_sequence_length=settings.min_string_length, n_jobs=1
        )
        per_string = {
            "symbol_entropy": [float(entropy(s, bias_correction=True)) for s in kept],
            "entropy_rate": [float(v) for v in ensemble["entropy_rate"]],
            "excess_entropy": [float(v) for v in ensemble["emc"]],
            "emc_order": [float(v) for v in ensemble["emc_order"]],
            "gzip": [float(v) for v in compression["compressibility"]],
            "lzw": [float(v) for v in compression["lzw_complexity"]],
        }
    blocks = np.array([record["block_entropies"][:max_block] for record in ensemble["records"]])
    return {
        "n_strings": len(kept),
        "max_block_size": int(max_block),
        "summary": {key: summarise(values) for key, values in per_string.items()},
        "per_string": per_string,
        "block_entropy": [float(v) for v in blocks.mean(axis=0)],
        "saturation": int(ensemble["saturation"]["k_star"]),
    }


def compared_strings(
    strings: Sequence[Sequence[str]], settings: Settings = SETTINGS
) -> list[list[str]]:
    """The strings that are compared pairwise: a seeded random subset, each cut to its first tokens."""
    strings = [list(s) for s in strings]
    if len(strings) > settings.n_compared:
        rng = np.random.default_rng(settings.compared_seed)
        chosen = sorted(rng.choice(len(strings), size=settings.n_compared, replace=False))
        strings = [strings[i] for i in chosen]
    return [s[: settings.compared_length] for s in strings]


def stringset_level(strings: Sequence[Sequence[str]], settings: Settings = SETTINGS) -> dict:
    """String-set level: length and diversity of the strings, and how much two strings differ."""
    strings = [list(s) for s in strings]
    compared = compared_strings(strings, settings)
    pairs = list(itertools.combinations(compared, 2))
    with _quiet():
        set_entropy = float(string_set_entropy(strings, use_frequencies=True))
        distances = [float(edit_distance(a, b)) for a, b in pairs]
    lengths = [len(s) for s in strings]
    return {
        "n_strings": len(strings),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "set_entropy": set_entropy,
        "n_compared": len(compared),
        "edit_distance": summarise(distances) if pairs else summarise([math.nan]),
    }


def observed_graph(strings: Sequence[Sequence[str]]) -> dict:
    """States, transitions and density of the observed first-order graph, and its topological entropy (Eq. 2)."""
    symbols, _, counts = behaviour.transition_matrix(strings)
    adjacency = (counts > 0).astype(float)
    with _quiet():
        value = float(topological_entropy(transitions=adjacency, method="direct", verbose=False))
    return {
        "states": len(symbols),
        "transitions": int(adjacency.sum()),
        "density": float(adjacency.sum() / len(symbols) ** 2),
        "topological_direct": value / LN2,
        "topological_bound": math.log2(len(symbols)),
    }


def entropy_rate_by_order(strings: Sequence[Sequence[str]], max_order: int) -> dict:
    """Markov order chosen by BIC and the entropy rate at every order (Eq. 1)."""
    with _quiet():
        fit = markov_order_selection(
            [list(s) for s in strings], max_order=max_order, criterion="BIC", eos=None
        )
    n = fit["n_observations"]
    return {
        "markov_order": int(fit["optimal_order"]),
        "entropy_rate_by_order": [-value / (n * LN2) for value in fit["log_likelihoods"]],
        "bic": [float(score) for score in fit["scores"]],
        "n_parameters": [int(count) for count in fit["n_parameters"]],
        "n_targets": int(n),
    }


def lift_limit(stream: Sequence[str], settings: Settings = SETTINGS) -> int:
    """Largest lift that is tried: the preprint's rule, cut where the table of blocks would get too large."""
    limit = min(settings.lift_max, max(2, len(stream) // 1000))
    for lift in range(1, limit + 1):
        blocks = set(zip(*(stream[offset:] for offset in range(lift)), strict=False))
        if len(blocks) > settings.lift_max_blocks:
            return max(1, lift - 1)
    return limit


def lift_estimate(stream: Sequence[str], settings: Settings = SETTINGS) -> dict:
    """SymSeq's lift estimate of the topological entropy of a sequence, in bits, with the value at every lift."""
    stream = list(stream)
    nothing = {"topological_lift": math.nan, "lift_values": [], "lift_limit": 0, "lift_note": ""}
    if len(stream) < settings.lift_min_tokens or len(set(stream)) < 2:
        return {**nothing, "lift_note": f"fewer than {settings.lift_min_tokens} symbols"}
    limit = lift_limit(stream, settings)
    try:
        with _quiet():
            _, value, values = topological_entropy(
                sequence=stream, method="lift", max_lift=limit, verbose=False
            )
    except Exception as error:  # the failure is kept in the result, not hidden
        return {
            **nothing,
            "lift_limit": limit,
            "lift_note": f"failed: {type(error).__name__}: {error}",
        }
    return {
        "topological_lift": round(float(np.real(value)) / LN2, 9),
        "lift_values": [round(float(np.real(v)) / LN2, 9) for v in values],
        "lift_limit": limit,
        "lift_note": "",
    }


def mi_decay(stream: Sequence[str], settings: Settings = SETTINGS) -> dict:
    """SymSeq's mutual information against distance, in bits, with the band of shuffled sequences."""
    stream = list(stream)
    distance = min(settings.mi_max_distance, len(stream) // 10)
    if distance < 1:
        return {
            "distances": [],
            "mi": [],
            "baseline": [],
            "lower": [],
            "upper": [],
            "mi_last_distance": 0,
        }
    with _quiet():
        result = mi_decay_analysis(
            stream,
            max_distance=distance,
            n_shuffles=settings.mi_shuffles,
            confidence_level=settings.mi_confidence,
            random_state=settings.mi_seed,
            units="bits",
        )
    return {
        "distances": [int(d) for d in result["distances"]],
        "mi": [float(v) for v in result["mi_values"]],
        "baseline": [float(v) for v in result["mi_baseline"]],
        "lower": [float(v) for v in result["mi_ci_lower"]],
        "upper": [float(v) for v in result["mi_ci_upper"]],
        "mi_last_distance": int(result["max_significant_distance_ci"]),
        "decay_model": str(result["best_model"]),
    }


def grammar_level(strings: Sequence[Sequence[str]], settings: Settings = SETTINGS) -> dict:
    """Grammar level: the observed transition graph, memory length, and dependence against distance."""
    strings = [list(s) for s in strings]
    stream = joined(strings)
    with _quiet():
        vlmc = vlmc_fit(strings, max_depth=settings.vlmc_max_depth, eos=None)
    return {
        **observed_graph(strings),
        **entropy_rate_by_order(strings, settings.markov_max_order),
        "vlmc_contexts": int(vlmc["num_terminal_contexts"]),
        "vlmc_depth": int(vlmc["max_depth_used"]),
        **lift_estimate(stream, settings),
        "mi": mi_decay(stream, settings),
    }


def measure_set(strings: Sequence[Sequence[str]], settings: Settings = SETTINGS) -> dict:
    """All measures of ``MEASURES`` for one set of strings."""
    return {
        "token": token_level(strings),
        "string": string_level(strings, settings),
        "string set": stringset_level(strings, settings),
        "grammar": grammar_level(strings, settings),
    }


# --------------------------------------------------------------------------------------------
# A dataset with its surrogates, the known answers and the estimates against the amount of data
# --------------------------------------------------------------------------------------------


def surrogate_sets(strings: Sequence[Sequence[str]], settings: Settings = SETTINGS) -> dict:
    """The data and its two surrogates, with the seeds of the settings."""
    seeds = {
        "data": 0,
        "first-order": settings.first_order_seed,
        "shuffled": settings.shuffled_seed,
    }
    return {kind: behaviour.surrogate(strings, kind, seeds[kind]) for kind in behaviour.SURROGATES}


def entropy_rate_tolerance(known: dict) -> float:
    """Allowed distance between the estimate on the first-order surrogate and its closed form, in bits."""
    return max(0.02, 5.0 * known["surprisal_sd"] / math.sqrt(known["n_transitions"]))


def pair_information(table: np.ndarray) -> float:
    """Eq. (3): mutual information of the two symbols of a pair, in bits, from the table of pair counts."""
    joint = table / table.sum()
    independent = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    seen = joint > 0
    return float((joint[seen] * np.log2(joint[seen] / independent[seen])).sum())


def pair_tables(strings: Sequence[Sequence[str]]) -> tuple[np.ndarray, np.ndarray, int]:
    """Token codes of a set, the positions that have a successor inside their string, and the alphabet size."""
    symbols = sorted({symbol for string in strings for symbol in string})
    code = {symbol: i for i, symbol in enumerate(symbols)}
    tokens = np.array([code[symbol] for string in strings for symbol in string])
    last = np.zeros(len(tokens), dtype=bool)
    last[np.cumsum([len(string) for string in strings]) - 1] = True
    return tokens, np.flatnonzero(~last), len(symbols)


def memory_of_shuffles(strings: Sequence[Sequence[str]], n_shuffles: int, seed: int) -> np.ndarray:
    """First-order memory h_0 - h_1 of independent shuffles of a set, computed with numpy (no SymSeq call).

    Each shuffle permutes all tokens and keeps the string lengths, as ``tutorial.behaviour.shuffled``.
    """
    tokens, previous, n_symbols = pair_tables(strings)
    rng = np.random.default_rng(seed)
    values = np.empty(n_shuffles)
    for index in range(n_shuffles):
        permuted = rng.permutation(tokens)
        pairs = permuted[previous] * n_symbols + permuted[previous + 1]
        table = np.bincount(pairs, minlength=n_symbols**2).reshape(n_symbols, n_symbols)
        values[index] = pair_information(table)
    return values


def allowed_outside(n_distances: int) -> int:
    """Number of distances that may lie outside the 95 % shuffle band for a sequence without order."""
    return int(stats.binom.ppf(0.999, n_distances, 0.08))


def above_band(mi: dict) -> int:
    """Number of distances at which the mutual information is above the shuffle band."""
    return int((np.asarray(mi["mi"]) > np.asarray(mi["upper"])).sum())


def outside_band(mi: dict) -> int:
    """Number of distances at which the mutual information is outside the shuffle band."""
    values, lower, upper = (np.asarray(mi[key]) for key in ("mi", "lower", "upper"))
    return int(((values < lower) | (values > upper)).sum())


def surrogate_checks(sets: dict, known: dict, band: Sequence[float]) -> dict:
    """The known answers of the two surrogates, checked on the measured sets (see the module docstring).

    ``sets`` maps "data", "first-order" and "shuffled" to the result of ``measure_set``; ``band`` holds
    the memory of other shuffles of the same set (``memory_of_shuffles``). Every check is reported
    with its numbers; ``passed`` is true if all hold.
    """
    first, shuffle, data = (sets[kind]["grammar"] for kind in ("first-order", "shuffled", "data"))
    tolerance = entropy_rate_tolerance(known)
    estimate = first["entropy_rate_by_order"][1]
    by_order = shuffle["entropy_rate_by_order"]
    memory = by_order[0] - by_order[1]
    threshold = float(np.max(band))
    n_distances = len(shuffle["mi"]["distances"])
    checks = {
        "first_order_selects_order_1": {
            "selected": first["markov_order"],
            "passed": first["markov_order"] == 1,
        },
        "first_order_entropy_rate": {
            "estimate": estimate,
            "closed_form": known["entropy_rate"],
            "tolerance": tolerance,
            "passed": abs(estimate - known["entropy_rate"]) <= tolerance,
        },
        "shuffled_selects_order_0": {
            "selected": shuffle["markov_order"],
            "passed": shuffle["markov_order"] == 0,
        },
        "shuffled_memory": {
            "memory": memory,
            "threshold": threshold,
            "band_mean": float(np.mean(band)),
            "band_shuffles": len(band),
            "memory_of_the_data": data["entropy_rate_by_order"][0]
            - data["entropy_rate_by_order"][1],
            "passed": memory <= threshold,
        },
        "shuffled_mutual_information": {
            "outside_band": outside_band(shuffle["mi"]),
            "allowed": allowed_outside(n_distances),
            "distances": n_distances,
            "data_outside_band": outside_band(data["mi"]),
            "passed": outside_band(shuffle["mi"]) <= allowed_outside(n_distances),
        },
    }
    checks["passed"] = all(check["passed"] for check in checks.values())
    return checks


def first_tokens(strings: Sequence[Sequence[str]], n_tokens: int) -> list[list[str]]:
    """The first ``n_tokens`` symbols of a set, as strings: whole strings in order, the last one cut."""
    kept, left = [], n_tokens
    for string in strings:
        if left <= 0:
            break
        kept.append(list(string[:left]))
        left -= len(kept[-1])
    return kept


def against_tokens(strings: Sequence[Sequence[str]], settings: Settings = SETTINGS) -> dict:
    """Estimates on the first n tokens of the data and of its first-order surrogate, for growing n.

    For each n: the observed transitions and the topological entropy of their graph (Eq. 2) for the
    data, and the entropy rate at orders 1 and 2 (Eq. 1) for the data and for the first-order
    surrogate. The surrogate has no structure beyond order 1, so its h_2 below its h_1 is the bias
    of the estimate at that amount of data.
    """
    sets = surrogate_sets(strings, settings)
    total = sum(len(s) for s in strings)
    sizes = [n for n in settings.convergence_tokens if n < total] + [total]
    rows: dict[str, list] = {
        key: []
        for key in (
            "n_tokens",
            "transitions",
            "topological_direct",
            "topological_bound",
            "h1_data",
            "h2_data",
            "h1_first_order",
            "h2_first_order",
        )
    }
    for size in sizes:
        data = first_tokens(sets["data"], size)
        surrogate = first_tokens(sets["first-order"], size)
        graph = observed_graph(data)
        data_rates = entropy_rate_by_order(data, 2)["entropy_rate_by_order"]
        surrogate_rates = entropy_rate_by_order(surrogate, 2)["entropy_rate_by_order"]
        rows["n_tokens"].append(size)
        rows["transitions"].append(graph["transitions"])
        rows["topological_direct"].append(graph["topological_direct"])
        rows["topological_bound"].append(graph["topological_bound"])
        rows["h1_data"].append(data_rates[1])
        rows["h2_data"].append(data_rates[2])
        rows["h1_first_order"].append(surrogate_rates[1])
        rows["h2_first_order"].append(surrogate_rates[2])
    return rows


def measure_dataset(
    strings: Sequence[Sequence[str]], settings: Settings = SETTINGS, progress=None
) -> dict:
    """Everything notebook B shows for one set of strings: the measures of the data and of its two
    surrogates, the known answers with their checks, and the estimates against the amount of data.

    ``progress``, if given, is called with the name of each part before it is computed.
    """
    strings = [list(s) for s in strings]
    sets = surrogate_sets(strings, settings)
    measured = {}
    for kind, surrogate in sets.items():
        if progress is not None:
            progress(behaviour.SURROGATE_LABELS[kind])
        measured[kind] = measure_set(surrogate, settings)
    if progress is not None:
        progress("known answers")
    known = behaviour.first_order_entropy_rate(strings)
    known.pop("symbols")
    return {
        "n_strings": len(strings),
        "n_tokens": sum(len(s) for s in strings),
        "settings": settings.as_dict(),
        "sets": measured,
        "known": known,
        "checks": surrogate_checks(
            measured, known, memory_of_shuffles(strings, settings.band_shuffles, settings.band_seed)
        ),
        "against_tokens": against_tokens(strings, settings),
    }


# --------------------------------------------------------------------------------------------
# Tables for the notebook
# --------------------------------------------------------------------------------------------


def _mean_sem(summary: dict) -> str:
    if summary["n"] == 0:
        return "no estimate"
    sem = "" if math.isnan(summary["sem"]) else f" ± {summary['sem']:.3f}"
    return f"{summary['mean']:.3f}{sem} (n = {summary['n']})"


def level_table(result: dict, level: str) -> pd.DataFrame:
    """One row for the data and one for each surrogate, with the measures of one level."""
    rows = {}
    for kind, measured in result["sets"].items():
        values = measured[level]
        if level == "token":
            row = {
                "symbols in all": values["n_tokens"],
                "types": values["types"],
                "top_frequency": values["top_frequency"],
            }
        elif level == "string":
            row = {
                key: _mean_sem(summary)
                for key, summary in values["summary"].items()
                if key != "emc_order"
            }
            row["saturation k*"] = values.get("saturation", math.nan)
        elif level == "string set":
            row = {
                "n_strings": values["n_strings"],
                "mean_length": values["mean_length"],
                "set_entropy": values["set_entropy"],
                "edit_distance": _mean_sem(values["edit_distance"]),
            }
        elif level == "grammar":
            lift = values["topological_lift"]
            row = {
                "states": values["states"],
                "transitions": values["transitions"],
                "density": values["density"],
                "markov_order": values["markov_order"],
                "vlmc_contexts": values["vlmc_contexts"],
                "vlmc_depth": values["vlmc_depth"],
                "topological_direct": values["topological_direct"],
                "topological_lift": lift if not math.isnan(lift) else values["lift_note"],
                "lifts tried": len(values["lift_values"]),
                "distances above the shuffle band": f"{above_band(values['mi'])} of {len(values['mi']['distances'])}",
            }
        else:
            raise KeyError(f"Unknown level {level!r}. Choose one of {LEVELS}.")
        rows[behaviour.SURROGATE_LABELS[kind]] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def order_table(result: dict) -> pd.DataFrame:
    """Entropy rate at every order (Eq. 1) for the data and the surrogates, next to the closed form."""
    table = pd.DataFrame(
        {
            behaviour.SURROGATE_LABELS[kind]: measured["grammar"]["entropy_rate_by_order"]
            for kind, measured in result["sets"].items()
        }
    )
    table.index.name = "order k"
    table["closed form of the first-order surrogate"] = [math.nan] + [
        result["known"]["entropy_rate"]
    ] * (len(table) - 1)
    return table


def tokens_table(result: dict) -> pd.DataFrame:
    """The estimates against the number of tokens used, as a table."""
    rows = result["against_tokens"]
    return pd.DataFrame(
        {
            "symbols used": rows["n_tokens"],
            "transitions seen": rows["transitions"],
            "topological entropy (bits)": rows["topological_direct"],
            "its upper limit, log2(symbols)": rows["topological_bound"],
            "h_1, data": rows["h1_data"],
            "h_2, data": rows["h2_data"],
            "h_1, first-order surrogate": rows["h1_first_order"],
            "h_2, first-order surrogate": rows["h2_first_order"],
        }
    ).set_index("symbols used")


def lift_table(result: dict) -> pd.DataFrame:
    """The lift estimate of the topological entropy at every lift, next to the value of the observed graph."""
    rows = {}
    for kind, measured in result["sets"].items():
        grammar = measured["grammar"]
        row = {f"lift {i + 1}": value for i, value in enumerate(grammar["lift_values"])}
        row["returned"] = grammar["topological_lift"]
        row["observed first-order graph"] = grammar["topological_direct"]
        rows[behaviour.SURROGATE_LABELS[kind]] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def checks_table(result: dict) -> pd.DataFrame:
    """The known answers of the surrogates, as stored with the result."""
    checks = result["checks"]
    rows = [
        {
            "check": "first-order surrogate: order chosen by BIC",
            "found": checks["first_order_selects_order_1"]["selected"],
            "expected": "1",
            "holds": checks["first_order_selects_order_1"]["passed"],
        },
        {
            "check": "first-order surrogate: entropy rate h_1 (bits)",
            "found": round(checks["first_order_entropy_rate"]["estimate"], 4),
            "expected": f"{checks['first_order_entropy_rate']['closed_form']:.4f} ± "
            f"{checks['first_order_entropy_rate']['tolerance']:.4f}",
            "holds": checks["first_order_entropy_rate"]["passed"],
        },
        {
            "check": "shuffled: order chosen by BIC",
            "found": checks["shuffled_selects_order_0"]["selected"],
            "expected": "0",
            "holds": checks["shuffled_selects_order_0"]["passed"],
        },
        {
            "check": "shuffled: memory h_0 - h_1 (bits)",
            "found": round(checks["shuffled_memory"]["memory"], 5),
            "expected": f"at most {checks['shuffled_memory']['threshold']:.5f} (largest of "
            f"{checks['shuffled_memory']['band_shuffles']} other shuffles)",
            "holds": checks["shuffled_memory"]["passed"],
        },
        {
            "check": "shuffled: distances with mutual information outside the shuffle band",
            "found": checks["shuffled_mutual_information"]["outside_band"],
            "expected": f"at most {checks['shuffled_mutual_information']['allowed']} of "
            f"{checks['shuffled_mutual_information']['distances']}",
            "holds": checks["shuffled_mutual_information"]["passed"],
        },
    ]
    return pd.DataFrame(rows).set_index("check")


def measures_legend() -> pd.DataFrame:
    """Which SymSeq call each measure comes from."""
    return pd.DataFrame(MEASURES, columns=["measure", "level", "SymSeq call", "meaning"]).set_index(
        "measure"
    )


def settings_table() -> pd.DataFrame:
    """The settings of the tutorial next to those of the preprint's figure, with the reason for each change."""
    ours, theirs = SETTINGS.as_dict(), PREPRINT.as_dict()
    rows = [
        {
            "setting": key,
            "here": str(ours[key]),
            "preprint": "no limit" if key == "lift_max_blocks" else str(theirs[key]),
            "why changed": CHEAPER.get(key, ""),
        }
        for key in ours
        if key
        not in (
            "first_order_seed",
            "shuffled_seed",
            "band_seed",
            "band_shuffles",
            "convergence_tokens",
        )
    ]
    return pd.DataFrame(rows).set_index("setting")


def figure_data(
    results: dict[str, dict], highlight: str | None = None, own: dict | None = None
) -> dict:
    """The numbers of the figure of notebook B.

    Parameters
    ----------
    results : dict
        Dataset name -> result of ``measure_dataset``, in the order of the panels.
    highlight : str, optional
        The dataset chosen in the notebook.
    own : dict, optional
        The result of ``measure_dataset`` for a participant's own sequence; it gets its own panels.
    """
    panels = {name: results[name] for name in results}
    labels = {name: behaviour.DATASETS[name]["short"] for name in results}
    if own is not None:
        panels["own"] = own
        labels["own"] = "your sequence"
    data: dict = {
        "datasets": list(panels),
        "labels": labels,
        "highlight": [
            name for name in (highlight, "own" if own is not None else None) if name in panels
        ],
        "entropy_rate": {},
        "mi": {},
        "closed_form": {},
        "markov_order": {},
        "n_tokens": {},
    }
    for name, result in panels.items():
        sets = result["sets"]
        data["entropy_rate"][name] = {
            kind: sets[kind]["grammar"]["entropy_rate_by_order"] for kind in behaviour.SURROGATES
        }
        data["markov_order"][name] = {
            kind: sets[kind]["grammar"]["markov_order"] for kind in behaviour.SURROGATES
        }
        mi = sets["data"]["grammar"]["mi"]
        data["mi"][name] = {
            "distances": mi["distances"],
            "data": mi["mi"],
            "first-order": sets["first-order"]["grammar"]["mi"]["mi"],
            "shuffled": sets["shuffled"]["grammar"]["mi"]["mi"],
            "band_lower": mi["lower"],
            "band_upper": mi["upper"],
        }
        data["closed_form"][name] = result["known"]["entropy_rate"]
        data["n_tokens"][name] = result["n_tokens"]
    return data


# --------------------------------------------------------------------------------------------
# Saved results: one JSON file per dataset and a manifest with the SHA-256 of each
# --------------------------------------------------------------------------------------------


def plain(value):
    """A copy made of dicts, lists, numbers and strings only, so that it can be written as JSON."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_result(dataset: str, result: dict, table_sha256: str, folder: Path = RESULTS) -> Path:
    """Write the result of one dataset. The file holds no time stamp: the same input gives the same bytes."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{dataset}.json"
    aside = path.with_suffix(f".{os.getpid()}.part")
    aside.write_text(
        json.dumps(plain({"dataset": dataset, "table_sha256": table_sha256, **result}), indent=1)
    )
    os.replace(aside, path)
    return path


def saved_result(
    dataset: str, table_sha256: str, settings: Settings, folder: Path = RESULTS
) -> dict | None:
    """The saved result of a dataset if it was computed from this table with these settings, else None."""
    path = Path(folder) / f"{dataset}.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    same = result.get("table_sha256") == table_sha256 and result.get("settings") == plain(
        settings.as_dict()
    )
    return result if same else None


def collect(folder: Path = RESULTS, settings: Settings = SETTINGS) -> dict:
    """Write ``manifest.json`` of a results folder from its files, and return it.

    A dataset without a result is listed as missing; it is never given a value.
    """
    folder = Path(folder)
    files, missing, failed = {}, [], []
    for dataset in behaviour.DATASETS:
        path = folder / f"{dataset}.json"
        if not path.exists():
            missing.append(dataset)
            continue
        files[path.name] = sha256(path)
        if not json.loads(path.read_text())["checks"]["passed"]:
            failed.append(dataset)
    manifest = {
        "settings": plain(settings.as_dict()),
        "datasets": list(behaviour.DATASETS),
        "missing": missing,
        "complete": not missing,
        "checks_failed": failed,
        "files": files,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_results(folder: Path = RESULTS) -> tuple[dict[str, dict], dict]:
    """The saved results of a folder and its manifest, after checking every file against the manifest.

    Raises ``FileNotFoundError`` without a manifest and ``ValueError`` if a listed file is absent or
    changed, if a result file is not listed, or if a result was computed from another table than the
    shipped one.
    """
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    results = {}
    for name, digest in manifest["files"].items():
        path = folder / name
        if not path.exists():
            raise ValueError(f"{name} is listed in the manifest and absent from {folder}.")
        if sha256(path) != digest:
            raise ValueError(f"{name} was changed after the manifest was written.")
        result = json.loads(path.read_text())
        results[result["dataset"]] = result
    unlisted = sorted(
        p.name
        for p in folder.glob("*.json")
        if p.name not in {*manifest["files"], "manifest.json", TIMINGS}
    )
    if unlisted:
        raise ValueError(f"Result files not in the manifest: {unlisted}. Run the script again.")
    return {name: results[name] for name in behaviour.DATASETS if name in results}, manifest


def check_against_tables(results: dict[str, dict], data_folder: Path = behaviour.DATA) -> None:
    """Raise ``ValueError`` if a saved result was not computed from the shipped table of its dataset."""
    shipped = behaviour.read_manifest(data_folder)
    for name, result in results.items():
        if result["table_sha256"] != shipped.get(f"{name}.csv.gz"):
            raise ValueError(
                f"The saved result of {name!r} was computed from another table than the shipped one."
            )
