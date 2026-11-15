"""The worked example on generated sequences: closed forms next to SymSeq's estimates.

The grammar is ``two_state_grammar(p)``: after A comes B with probability p, after B always A. Its
strings ``A^k B`` are joined without the end marker, which gives the two-symbol chain whose closed
forms are in ``tutorial.closed_forms``.

Which estimator each number comes from (all on the joined sequence, in bits):

- entropy rate: ``symseq.metrics.string.estimate_emc(sequence, order="bic", max_order=6)["entropy_rate"]``,
  the plug-in conditional entropy ``h_R = H(X_{R+1} | X_1..X_R)`` at the Markov order R that minimises
  SymSeq's BIC over observed contexts. No bias correction.
- single-symbol entropy: the ``"entropy"`` field of the same call, the plug-in entropy of symbol
  frequencies.
- memory (excess entropy): the ``"emc"`` field of the same call, ``max(0, H_R - R h_R)`` with H_R the
  plug-in entropy of blocks of R symbols. For R = 1 this is ``H_1 - h_1``.
- topological entropy: ``symseq.metrics.grammar.topological_entropy(sequence=..., method="lift")``,
  converted from nats to bits.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from symseq.metrics.string import estimate_emc

from tutorial.closed_forms import (
    LN2,
    PHI,
    allowed_word_counts,
    golden_mean_entropy_rate,
    two_state_block_entropies,
    two_state_closed_forms,
    word_count_line,
)
from tutorial.generate import as_stream, generate_strings, two_state_grammar
from tutorial.measures import lift_topological_entropy

MAX_ORDER = 6
MAX_SYMBOLS = 60_000  # longest sequence of the notebook; every cell then takes about a second


def strings_for_length(p: float, n_symbols: int) -> int:
    """Number of strings whose joined sequence has about ``n_symbols`` symbols.

    A string has ``1 + 1 / p`` symbols on average. Asking for a length in symbols keeps the work of
    every estimate the same for all p; asking for a number of strings would make the sequence
    twenty times longer at p = 0.05 than at p = 1.
    """
    if not 0 < n_symbols <= MAX_SYMBOLS:
        raise ValueError(f"n_symbols must be between 1 and {MAX_SYMBOLS}, got {n_symbols}.")
    return max(50, round(n_symbols * p / (1 + p)))


def joined_sequence(
    p: float = 0.5, n_strings: int = 6000, seed: int = 0
) -> tuple[list[list[str]], list[str]]:
    """Strings of the two-state grammar from SymSeq, and the sequence obtained by joining them without the marker."""
    grammar = two_state_grammar(p, 1.0)
    grammar.rng = np.random.default_rng(seed)
    strings = generate_strings(grammar, n_strings)
    return strings, as_stream(strings, delimiter=False)


def estimates(sequence: Sequence[str], topological: bool = True) -> dict[str, float]:
    """SymSeq's estimates on one sequence, with the keys of ``two_state_closed_forms``."""
    result = estimate_emc(list(sequence), order="bic", max_order=MAX_ORDER)
    out = {
        "entropy_rate": float(result["entropy_rate"]),
        "symbol_entropy": float(result["entropy"]),
        "excess_entropy": float(result["emc"]),
        "order": int(result["order"]),
    }
    if topological:
        out["topological_entropy"] = lift_topological_entropy(sequence)
    return out


def comparison(p: float, sequence: Sequence[str]) -> pd.DataFrame:
    """Closed forms and estimates side by side."""
    exact = two_state_closed_forms(p, 1.0, delimiter=False)
    measured = estimates(sequence)
    rows = [
        ("entropy rate h (bits per symbol)", "entropy_rate", 'estimate_emc(...)["entropy_rate"]'),
        ("single-symbol entropy H_1 (bits)", "symbol_entropy", 'estimate_emc(...)["entropy"]'),
        ("memory E = H_1 - h (bits)", "excess_entropy", 'estimate_emc(...)["emc"]'),
        (
            "topological entropy (bits per symbol)",
            "topological_entropy",
            'topological_entropy(..., method="lift")',
        ),
    ]
    return pd.DataFrame(
        [
            {
                "quantity": label,
                "closed form": exact[key],
                "estimate": measured[key],
                "difference": measured[key] - exact[key],
                "SymSeq call": call,
            }
            for label, key, call in rows
        ]
    ).set_index("quantity")


def convergence(
    p: float, sequence: Sequence[str], lengths: Sequence[int] = (300, 1000, 3000, 10_000)
) -> dict:
    """Estimates on the first N symbols of the sequence, for increasing N, with the closed forms."""
    exact = two_state_closed_forms(p, 1.0, delimiter=False)
    ns = [n for n in lengths if n < len(sequence)] + [len(sequence)]
    measured = [estimates(sequence[:n], topological=False) for n in ns]
    return {
        "n": ns,
        "entropy_rate": [m["entropy_rate"] for m in measured],
        "excess_entropy": [m["excess_entropy"] for m in measured],
        "entropy_rate_exact": exact["entropy_rate"],
        "excess_entropy_exact": exact["excess_entropy"],
    }


def entropy_rate_curve(
    ps: Sequence[float] = (0.1, 0.2, 1 / PHI**2, 0.5, 0.7, 0.9),
    n_symbols: int = 6000,
    seed: int = 0,
) -> dict:
    """h(p) = H(p) / (1 + p) on a grid, and SymSeq's estimate at a few values of p."""
    grid = np.linspace(0.001, 0.999, 999)
    measured = []
    for index, p in enumerate(ps):
        _, sequence = joined_sequence(p, strings_for_length(p, n_symbols), seed + index)
        measured.append(estimates(sequence, topological=False)["entropy_rate"])
    return {
        "p": grid,
        "entropy_rate": golden_mean_entropy_rate(grid),
        "p_estimated": list(ps),
        "entropy_rate_estimated": measured,
        "topological_entropy": float(np.log2(PHI)),
        "p_max": 1 / PHI**2,
    }


def words_panel(max_length: int = 10) -> dict:
    """The numbers of the panel "allowed words against their length": exact values only.

    ``words`` are the counts of Eq. (6) of ``tutorial.closed_forms``, ``line`` the values of Eq. (7)
    at the same lengths. The slope of the line is the topological entropy, given in bits per symbol
    and in nats per symbol (the unit SymSeq returns). ``allowed`` is the table of allowed
    transitions of the two symbols (rows: current symbol; columns: next symbol). Nothing here
    depends on p: the count of words depends only on which transitions exist.
    """
    lengths = list(range(1, max_length + 1))
    return {
        "length": lengths,
        "words": allowed_word_counts(max_length),
        "line": word_count_line(lengths).tolist(),
        "topological_entropy_bits": float(np.log2(PHI)),
        "topological_entropy_nats": float(np.log2(PHI) * LN2),
        "symbols": ["A", "B"],
        "allowed": [[1, 1], [1, 0]],
    }


def block_entropy_panel(p: float = 0.5, max_length: int = 8) -> dict:
    """The numbers of the panel "block entropy against block length": exact values only.

    ``block_entropy`` holds H(L) for L = 1 to ``max_length`` (Eq. 8 of ``tutorial.closed_forms``).
    The line through them has the entropy rate as its slope and the memory as its value at L = 0.
    """
    exact = two_state_closed_forms(p, 1.0, delimiter=False)
    return {
        "p": p,
        "length": list(range(1, max_length + 1)),
        "block_entropy": two_state_block_entropies(p, max_length),
        "entropy_rate": exact["entropy_rate"],
        "excess_entropy": exact["excess_entropy"],
    }
