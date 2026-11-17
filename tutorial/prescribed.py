"""The prescribed-complexity exercise of notebook A.

SymSeq synthesises a directed graph at a requested topological entropy by Glauber dynamics: one edge
is added or removed at every step and the change is accepted with a probability that depends on the
score. This module keeps the synthesis trace beside the grammar and its measured properties so that
the exercise shows both how the sampler works and what it produced.
"""

from __future__ import annotations

import warnings

import numpy as np
from symseq.generators.ag.synthesis import generate_grammar_with_target_te_complexity

from tutorial import complexity


def make_prescribed(
    target_bits: float,
    seed: int = 7,
    level: int = 0,
    n_states: int = 8,
    alphabet_size: int = 6,
    n_terminals: int = 2,
) -> dict:
    """One regular grammar and the trace that reached it.

    ``target_bits`` is the requested topological entropy of the open directed graph in bits per
    symbol. The returned ``achieved_open_bits`` is the same quantity measured again on the grammar;
    ``closed_bits`` is SymSeq's topological entropy after the end marker is linked back to the start.
    """
    if not 0 < target_bits < np.log2(n_states):
        raise ValueError(f"Need 0 < target_bits < log2({n_states}), got {target_bits}.")
    target_nats = target_bits * complexity.LN2
    spec, reason = complexity.candidate(
        level, seed, target_nats, n_states, alphabet_size, n_terminals
    )
    if spec is None:
        return {"reason": reason, "target_bits": target_bits}
    with warnings.catch_warnings():
        # The acceptance rule exponentiates a large score difference; 0 and 1 are the intended limits.
        warnings.simplefilter("ignore", RuntimeWarning)
        table, counts, entropies, densities, scores, steps, *_ = (
            generate_grammar_with_target_te_complexity(
                target_complexity=target_nats,
                n_nodes=n_states,
                rng=np.random.default_rng([seed, level]),
            )
        )
    edges = set(spec.edges)
    edge_table = np.array(
        [[float((source, target) in edges) for target in spec.states] for source in spec.states]
    )
    if not np.array_equal((np.asarray(table) > 0).astype(int), edge_table):
        raise RuntimeError("The synthesis trace does not belong to the admitted grammar.")
    values = complexity.describe(spec, 2, 40)
    return {
        "spec": spec,
        "reason": "",
        "target_bits": float(target_bits),
        "steps": np.asarray(steps, dtype=int),
        "entropy_bits": np.asarray(entropies) / complexity.LN2,
        "score": np.asarray(scores),
        "density": np.asarray(densities),
        "counts": np.asarray(counts),
        "achieved_open_bits": values["synthesised_nats"] / complexity.LN2,
        "closed_bits": values["topological_entropy_bits"],
        "memory_gap": values["memory_gap"],
        "mean_length": values["mean_length"],
        "n_edges": values["n_edges"],
    }


def example_strings(prescribed: dict, seed: int = 101, n: int = 6) -> list[str]:
    """Distinct strings of the admitted grammar, for inspection in the notebook."""
    strings = complexity.draw_strings(prescribed["spec"], seed, n, 0, 2, 40, 10_000, "distinct")
    return list(dict.fromkeys(" ".join(string) for string in strings.sequence))


def figure_data(prescribed: dict) -> dict:
    """Only the numbers drawn in notebook A's final figure."""
    return {
        key: prescribed[key]
        for key in (
            "target_bits",
            "steps",
            "entropy_bits",
            "score",
            "achieved_open_bits",
            "closed_bits",
            "memory_gap",
            "mean_length",
            "n_edges",
        )
    }
