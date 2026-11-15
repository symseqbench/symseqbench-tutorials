"""Candidate grammars for the ladder of the networks notebook: measured complexity and accuracy ceiling.

For every candidate the script reports SymSeq's measures on generated strings (the ladder table of
notebook A, with more strings) and the exact values from the generator: entropy rate of the joined
sequence and the best achievable next-symbol accuracy, with and without the end marker as a target.
It decides nothing: the table is the material for choosing five or six grammars.

Usage:  uv run python scripts/ladder_proposal.py --out scripts/output/ladder_proposal.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tutorial.ladder import Rung, exact_table, ladder_table

REPO = Path(__file__).resolve().parents[1]

CANDIDATES = [
    Rung("two-state", "two_state", {"p": 0.5, "q": 1.0}),
    Rung(
        "regular (6 states)",
        "regular",
        {"n_states": 6, "alphabet_size": 4, "transition_density": 0.3, "seed": 0},
    ),
    Rung("2-AX", "nax", {"n": 2, "p_target": 0.6}),
    Rung("2-AX (p_target 0.9)", "nax", {"n": 2, "p_target": 0.9}),
    Rung("4-AX", "nax", {"n": 4, "p_target": 0.6}),
    Rung("4-AX (p_target 0.9)", "nax", {"n": 4, "p_target": 0.9}),
    Rung(
        "non-adjacent",
        "nad",
        {"n_deps": 3, "n_fillers": 3},
        {"filler_len": 2, "randomize_fillers": True},
    ),
    Rung("nested", "nested", {"n_deps": 3}),
    Rung("crossed", "crossed", {"n_deps": 3}),
    Rung("Dyck (depth 2)", "dyck", {"k": 2, "depth": 2, "pairs": 4}),
    Rung("Dyck (depth 3)", "dyck", {"k": 2, "depth": 3, "pairs": 4}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--n-strings", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=REPO / "scripts" / "output" / "ladder_proposal.csv"
    )
    arguments = parser.parse_args()

    measured = ladder_table(CANDIDATES, n_strings=arguments.n_strings, seed=arguments.seed)
    exact = exact_table(CANDIDATES)
    table = pd.DataFrame(
        {
            "family": measured[("", "family")],
            "symbols": measured[("token", "types")],
            "mean length": exact["mean_length"],
            "h exact": exact["entropy_rate"],
            "h estimate": measured[("string", "entropy_rate")],
            "E estimate": measured[("string", "excess_entropy")],
            "order (emc)": measured[("string", "emc_order")],
            "Markov order (BIC)": measured[("grammar", "markov_order")],
            "VLMC depth": measured[("grammar", "vlmc_depth")],
            "VLMC contexts": measured[("grammar", "vlmc_contexts")],
            "TE direct": measured[("grammar", "topological_entropy")],
            "LZW": measured[("string", "lzw")],
            "ceiling (with #)": exact["ceiling_with_eos"],
            "previous symbol only (with #)": exact["previous_symbol_with_eos"],
            "ceiling (symbols only)": exact["ceiling_within"],
            "previous symbol only (symbols only)": exact["previous_symbol_within"],
        }
    )
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(arguments.out)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(table.round(3).to_string())
    print(
        f"\n{arguments.n_strings} strings per grammar, seed {arguments.seed}; entropies in bits; written {arguments.out}"
    )


if __name__ == "__main__":
    main()
