"""The complexity ladder: grammars, their parameters, and the table of measures.

A rung is a generator family (``kind``), the parameters of the generator (``params``) and the
arguments of each call that draws strings (``sample``). ``default_ladder`` lists the rungs of the
tutorial. A participant extends the ladder by appending a ``Rung`` to that list.

``family`` is declared, not measured. It names the class of the unbounded language family in the
Chomsky hierarchy. Every rung generated here is a finite-state instance of its family, because the
number of pairs and the nesting depth are bounded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tutorial import generate
from tutorial.measures import MEASURES, measure

GENERATORS: dict[str, Callable] = {
    "two_state": generate.two_state_grammar,
    "regular": generate.regular_grammar,
    "nax": generate.nax_grammar,
    "nad": generate.nad,
    "nested": generate.nested_nad,
    "crossed": generate.crossed_nad,
    "dyck": generate.dyck,
}

FAMILIES: dict[str, str] = {
    "two_state": "regular",
    "regular": "regular",
    "nax": "regular",
    "nad": "regular",
    "nested": "context-free",
    "crossed": "context-sensitive",
    "dyck": "context-free",
}


@dataclass(frozen=True)
class Rung:
    """One grammar of the ladder."""

    name: str
    kind: str
    params: dict = field(default_factory=dict)
    sample: dict = field(default_factory=dict)

    @property
    def family(self) -> str:
        return FAMILIES[self.kind]


def default_ladder(
    n_states: int = 6,
    alphabet_size: int = 4,
    n_contexts: int = 2,
    dependency_length: int = 2,
    depth: int = 2,
) -> list[Rung]:
    """The rungs of the tutorial, from the worked example to the Dyck language.

    The arguments are the interactive controls of notebook A: number of states and alphabet size of
    the sampled regular grammar, number of contexts of nAX, number of fillers between the two
    elements of a non-adjacent dependency, and nesting depth of the Dyck language.
    """
    return [
        Rung("two-state", "two_state", {"p": 0.5, "q": 1.0}),
        Rung(
            f"regular ({n_states} states)",
            "regular",
            {
                "n_states": n_states,
                "alphabet_size": alphabet_size,
                "transition_density": 0.3,
                "seed": 0,
            },
        ),
        Rung(f"{n_contexts}-AX", "nax", {"n": n_contexts, "p_target": 0.6}),
        Rung(
            "non-adjacent",
            "nad",
            {"n_deps": 3, "n_fillers": 3},
            {"filler_len": dependency_length, "randomize_fillers": True},
        ),
        Rung("nested", "nested", {"n_deps": 3}),
        Rung("crossed", "crossed", {"n_deps": 3}),
        Rung(f"Dyck (depth {depth})", "dyck", {"k": 2, "depth": depth, "pairs": 4}),
    ]


def build_generator(rung: Rung, seed: int = 0):
    """Build the SymSeq generator of a rung and seed its sampling of strings.

    The structure of a sampled regular grammar is fixed by the ``seed`` entry of the rung's
    parameters. ``seed`` given here only drives which strings are drawn, by replacing the
    generator's ``rng`` (as SeqBench does).
    """
    if rung.kind not in GENERATORS:
        raise KeyError(f"Unknown kind {rung.kind!r}. Known kinds: {sorted(GENERATORS)}.")
    generator = GENERATORS[rung.kind](**rung.params)
    generator.rng = np.random.default_rng(seed)
    return generator


def sample_strings(rung: Rung, n: int, seed: int = 0) -> list[list[str]]:
    """Draw ``n`` strings of a rung with SymSeq's ``generate_string_set``."""
    return generate.generate_strings(build_generator(rung, seed), n, **rung.sample)


def ladder_table(rungs: Sequence[Rung], n_strings: int = 400, seed: int = 0) -> pd.DataFrame:
    """Measures at the four levels for every rung: one row per rung, one column per measure.

    The columns carry two labels, the level and the measure, in the order of ``tutorial.measures.MEASURES``.
    """
    rows = []
    for rung in rungs:
        generator = build_generator(rung, seed)
        strings = generate.generate_strings(generator, n_strings, **rung.sample)
        rows.append(
            {"rung": rung.name, "family": rung.family, **measure(strings, generator=generator)}
        )
    table = pd.DataFrame(rows).set_index("rung")
    levels = {key: level for key, level, *_ in MEASURES}
    table.columns = pd.MultiIndex.from_tuples(
        [(levels.get(column, ""), column) for column in table.columns]
    )
    return table


def measures_legend() -> pd.DataFrame:
    """Which SymSeq function each column of the ladder table comes from."""
    return pd.DataFrame(MEASURES, columns=["measure", "level", "SymSeq call", "meaning"]).set_index(
        "measure"
    )


def exact_table(rungs: Sequence[Rung]) -> pd.DataFrame:
    """Exact values computed from each generator instead of from generated strings (``tutorial.exact``).

    ``entropy_rate`` is that of the strings joined with the end marker, in bits per symbol. The two
    ceilings are the best achievable next-symbol accuracies, with and without the end marker as a
    target; the two ``previous_symbol`` columns are the best accuracies of a predictor that sees only
    the previous symbol. Rungs whose generator is not covered get empty values.
    """
    from tutorial.exact import predictor_for, trial_statistics

    columns = (
        "mean_length",
        "entropy_rate",
        "ceiling_with_eos",
        "ceiling_within",
        "previous_symbol_with_eos",
        "previous_symbol_within",
    )
    rows = []
    for rung in rungs:
        try:
            statistics = trial_statistics(predictor_for(build_generator(rung), **rung.sample))
        except NotImplementedError:
            statistics = dict.fromkeys(columns, float("nan"))
        rows.append({"rung": rung.name, **{column: statistics[column] for column in columns}})
    return pd.DataFrame(rows).set_index("rung")


def figure_rungs(table: pd.DataFrame, exact: pd.DataFrame) -> dict:
    """The ladder panels of the notebook figure: estimated and exact entropy rate, estimated excess entropy."""
    return {
        "names": list(table.index),
        "entropy_rate": list(table[("string", "entropy_rate")]),
        "entropy_rate_exact": list(exact.loc[table.index, "entropy_rate"]),
        "excess_entropy": list(table[("string", "excess_entropy")]),
    }
