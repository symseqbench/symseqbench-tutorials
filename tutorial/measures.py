"""Thin wrappers around SymSeq's measures at the four levels: tokens, strings, string sets, grammars.

Each value returned here is the output of one SymSeq function, called with the arguments listed in
``MEASURES``. Nothing is re-implemented. The functions take either a set of strings (a list of lists
of symbols) or the sequence obtained by joining them (``tutorial.generate.as_stream``).

Units: entropies in bits. SymSeq returns topological entropy in nats; it is converted to bits here
(division by ln 2) so that it can be read next to the entropy rate. SymSeq's own call on an open
grammar returns 0 with a warning; the closed grammar (``correct_terminal_sink=True``) is used.
"""

from __future__ import annotations

import itertools
import logging
import math
import warnings
from collections.abc import Sequence

import numpy as np
from symseq.generators import ArtificialGrammar
from symseq.metrics.grammar import markov_order_selection, topological_entropy, vlmc_fit
from symseq.metrics.string import compressibility, entropy, estimate_emc, lzw_complexity
from symseq.metrics.stringset import edit_distance, string_set_entropy
from symseq.metrics.token import most_common_tokens, token_frequency

from tutorial.generate import EOS, as_stream

# (key, level, SymSeq function and arguments, what the number is)
MEASURES: list[tuple[str, str, str, str]] = [
    ("types", "token", "token_frequency(stream)", "number of distinct symbols in the sequence"),
    (
        "top_frequency",
        "token",
        "most_common_tokens(stream, n=1)",
        "relative frequency of the most frequent symbol",
    ),
    (
        "symbol_entropy",
        "string",
        "entropy(stream, bias_correction=False)",
        "H_1: plug-in Shannon entropy of single symbols, bits",
    ),
    (
        "entropy_rate",
        "string",
        'estimate_emc(stream, order="bic", max_order=6)["entropy_rate"]',
        "h_R = H(X_{R+1} | X_1..X_R): plug-in conditional entropy at the Markov order R chosen by BIC, bits per symbol",
    ),
    (
        "excess_entropy",
        "string",
        'estimate_emc(stream, order="bic", max_order=6)["emc"]',
        "E_R = max(0, H_R - R h_R): plug-in block entropy of R symbols minus R times the rate, bits",
    ),
    (
        "emc_order",
        "string",
        'estimate_emc(stream, order="bic", max_order=6)["order"]',
        "the order R used by the two estimates above",
    ),
    (
        "lzw",
        "string",
        "lzw_complexity(stream)",
        "number of LZW codes divided by the sequence length",
    ),
    (
        "gzip",
        "string",
        "compressibility(stream)",
        "gzip payload in bytes divided by the sequence length (one byte per symbol before compression)",
    ),
    ("mean_length", "string set", "(no SymSeq call)", "mean number of symbols per string"),
    (
        "set_entropy",
        "string set",
        "string_set_entropy(strings, use_frequencies=True)",
        "plug-in entropy of the distribution of whole strings, bits",
    ),
    (
        "edit_distance",
        "string set",
        "edit_distance(s, t) for every pair among the first 60 strings",
        "mean Levenshtein distance between two strings",
    ),
    (
        "markov_order",
        "grammar",
        'markov_order_selection(strings, max_order=5, criterion="BIC", eos="#")["optimal_order"]',
        "Markov order with the lowest BIC, contexts never crossing a string boundary",
    ),
    (
        "vlmc_contexts",
        "grammar",
        'vlmc_fit(strings, max_depth=8, eos="#")["num_terminal_contexts"]',
        "number of terminal contexts of the fitted variable-length Markov chain",
    ),
    (
        "vlmc_depth",
        "grammar",
        'vlmc_fit(strings, max_depth=8, eos="#")["max_depth_used"]',
        "length of the longest context kept by the variable-length Markov chain",
    ),
    (
        "topological_entropy",
        "grammar",
        "generator.topological_entropy(correct_terminal_sink=True) / ln 2",
        "log of the largest eigenvalue of the Boolean transition table, with every exit looped back to the start, "
        "bits per symbol; only for generators that are an ArtificialGrammar",
    ),
]

LEVELS = ("token", "string", "string set", "grammar")


def token_measures(stream: Sequence[str]) -> dict[str, float]:
    """Token level: how many symbols there are and how unevenly they are used."""
    frequencies = token_frequency(list(stream))
    top = most_common_tokens(list(stream), n=1)
    return {"types": len(frequencies), "top_frequency": float(next(iter(top.values())))}


def string_measures(stream: Sequence[str], max_order: int = 6) -> dict[str, float]:
    """String level: entropy, entropy rate, excess entropy and compression of one long sequence."""
    stream = list(stream)
    estimate = estimate_emc(stream, order="bic", max_order=max_order)
    return {
        "symbol_entropy": float(entropy(stream, bias_correction=False)),
        "entropy_rate": _number(estimate["entropy_rate"]),
        "excess_entropy": _number(estimate["emc"]),
        "emc_order": _number(estimate["order"]),
        "lzw": float(lzw_complexity(stream)),
        "gzip": float(compressibility(stream)),
    }


def stringset_measures(strings: Sequence[Sequence[str]], n_compared: int = 60) -> dict[str, float]:
    """String-set level: length, diversity and mutual distance of the strings."""
    strings = [list(string) for string in strings]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # SymSeq warns below 20 strings
        set_entropy = float(string_set_entropy(strings, use_frequencies=True))
    compared = strings[:n_compared]
    distances = [edit_distance(a, b) for a, b in itertools.combinations(compared, 2)]
    return {
        "mean_length": float(np.mean([len(string) for string in strings])),
        "set_entropy": set_entropy,
        "edit_distance": float(np.mean(distances)) if distances else float("nan"),
    }


def grammar_measures(
    strings: Sequence[Sequence[str]],
    generator=None,
    eos: str = EOS,
    max_order: int = 5,
    max_depth: int = 8,
) -> dict[str, float]:
    """Grammar level: memory length inferred from the strings, and growth rate of the grammar itself."""
    strings = [list(string) for string in strings]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # max_order above the longest string
        order = markov_order_selection(strings, max_order=max_order, criterion="BIC", eos=eos)
    vlmc = vlmc_fit(strings, max_depth=max_depth, eos=eos)
    return {
        "markov_order": int(order["optimal_order"]),
        "vlmc_contexts": int(vlmc["num_terminal_contexts"]),
        "vlmc_depth": int(vlmc["max_depth_used"]),
        "topological_entropy": direct_topological_entropy(generator),
    }


def direct_topological_entropy(generator) -> float:
    """SymSeq's direct topological entropy of an ``ArtificialGrammar``, closed and converted from nats to bits.

    Returns NaN for generators without a transition table (non-adjacent dependencies, Dyck), for
    which SymSeq has no direct method.
    """
    if not isinstance(generator, ArtificialGrammar):
        return float("nan")
    return float(generator.topological_entropy(correct_terminal_sink=True)) / math.log(2.0)


def lift_topological_entropy(stream: Sequence[str]) -> float:
    """SymSeq's lift estimate of the topological entropy of a sequence, converted from nats to bits.

    Not part of the ladder table: the estimate stops at the first lift that changes the value by less
    than 0.05 nats, which on the sequences of this tutorial can be far from the limit (see README).
    """
    logger = logging.getLogger("symseq.metrics.grammar.topological")
    level = logger.level
    logger.setLevel(logging.ERROR)  # SymSeq logs a warning for sequences of 1000 symbols or fewer
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, value, _ = topological_entropy(sequence=list(stream), method="lift", verbose=False)
    finally:
        logger.setLevel(level)
    return float(np.real(value)) / math.log(2.0)


def measure(strings: Sequence[Sequence[str]], generator=None, eos: str = EOS) -> dict[str, float]:
    """All measures of ``MEASURES`` for one set of strings. The sequence is the strings joined with the end marker."""
    stream = as_stream(strings, delimiter=True, eos=eos)
    values: dict[str, float] = {}
    values.update(token_measures(stream))
    values.update(string_measures(stream))
    values.update(stringset_measures(strings))
    values.update(grammar_measures(strings, generator=generator, eos=eos))
    return {key: values[key] for key, *_ in MEASURES}


def _number(value) -> float:
    return float("nan") if value is None else float(value)
