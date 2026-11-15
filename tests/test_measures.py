"""The wrapped measures reproduce SymSeq's own values on a fixed seed. Every test has a case it must reject."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from symseq.metrics.grammar import markov_order_selection, vlmc_fit
from symseq.metrics.string import compressibility, entropy, estimate_emc, lzw_complexity
from symseq.metrics.stringset import pairwise_distances, string_set_entropy
from symseq.metrics.token import most_common_tokens, token_frequency

from tutorial import generate
from tutorial.ladder import Rung, build_generator, default_ladder, ladder_table, measures_legend
from tutorial.measures import MEASURES, lift_topological_entropy, measure

SEED = 11
N = 300
RUNGS = [r for r in default_ladder() if r.kind in ("two_state", "regular", "nested", "dyck")]


def symseq_values(strings, generator) -> dict[str, float]:
    """The same quantities, each computed by calling SymSeq directly."""
    stream = [symbol for string in strings for symbol in [*string, "#"]]
    emc = estimate_emc(stream, order="bic", max_order=6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        order = markov_order_selection(strings, max_order=5, criterion="BIC", eos="#")
        distances = pairwise_distances(strings[:60], metric="edit")
        te = (
            generator.topological_entropy(correct_terminal_sink=True) / math.log(2)
            if hasattr(generator, "transition_table")
            else float("nan")
        )
    vlmc = vlmc_fit(strings, max_depth=8, eos="#")
    return {
        "types": len(token_frequency(stream)),
        "top_frequency": next(iter(most_common_tokens(stream, n=1).values())),
        "symbol_entropy": entropy(stream, bias_correction=False),
        "entropy_rate": emc["entropy_rate"],
        "excess_entropy": emc["emc"],
        "emc_order": emc["order"],
        "lzw": lzw_complexity(stream),
        "gzip": compressibility(stream),
        "mean_length": float(np.mean([len(s) for s in strings])),
        "set_entropy": string_set_entropy(strings, use_frequencies=True),
        "edit_distance": float(np.mean(distances[np.triu_indices(len(distances), 1)])),
        "markov_order": order["optimal_order"],
        "vlmc_contexts": vlmc["num_terminal_contexts"],
        "vlmc_depth": vlmc["max_depth_used"],
        "topological_entropy": te,
    }


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(
        (math.isnan(a[k]) and math.isnan(b[k]))
        or math.isclose(a[k], b[k], rel_tol=1e-12, abs_tol=1e-12)
        for k in a
    )


@pytest.mark.parametrize("rung", RUNGS, ids=lambda r: r.name)
def test_wrapped_measures_equal_symseq(rung):
    generator = build_generator(rung, SEED)
    strings = generate.generate_strings(generator, N, **rung.sample)
    wrapped = measure(strings, generator=generator)
    assert same(wrapped, symseq_values(strings, generator))

    # must reject: SymSeq's values on the strings of another seed
    other_generator = build_generator(rung, SEED + 1)
    other = generate.generate_strings(other_generator, N, **rung.sample)
    assert not same(wrapped, symseq_values(other, other_generator))


def test_legend_lists_exactly_the_measures_returned():
    rung = default_ladder()[0]
    generator = build_generator(rung, SEED)
    values = measure(generate.generate_strings(generator, 100), generator=generator)
    assert list(values) == [key for key, *_ in MEASURES]
    assert list(measures_legend().index) == list(values)
    assert {level for _, level, *_ in MEASURES} == {"token", "string", "string set", "grammar"}
    # must reject: a legend with one measure missing, and one with two measures exchanged
    assert list(values) != [key for key, *_ in MEASURES[:-1]]
    assert list(values) != [key for key, *_ in [MEASURES[1], MEASURES[0], *MEASURES[2:]]]


def test_topological_entropy_is_the_direct_method_and_is_absent_without_a_transition_table():
    table = ladder_table(default_ladder(), n_strings=100, seed=SEED)
    te = table[("grammar", "topological_entropy")]
    grammar = build_generator(default_ladder()[0])
    assert te["two-state"] == pytest.approx(
        grammar.topological_entropy(correct_terminal_sink=True) / math.log(2)
    )
    assert te["two-state"] == pytest.approx(0.5515, abs=1e-4)
    for name in ("non-adjacent", "nested", "crossed", "Dyck (depth 2)"):
        assert math.isnan(te[name])  # SymSeq has no direct method for these generators
    # must reject: the open grammar, for which SymSeq returns 0 with a warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert grammar.topological_entropy() == 0.0


def test_lift_estimate_is_kept_out_of_the_table_for_a_reason():
    """SymSeq's lift estimate stops early on the Dyck sequence: 1.85 bits where the exact value is 0.86."""
    rung = Rung("Dyck (depth 3)", "dyck", {"k": 2, "depth": 3, "pairs": 4})
    stream = generate.as_stream(generate.generate_strings(build_generator(rung, SEED), 2000))
    exact = (
        math.log2(13 * 2**4) / 9
    )  # 13 bracket structures, 16 type assignments, 8 symbols and the marker

    def close(estimate: float, value: float) -> bool:
        """The criterion a measure of the table has to meet: within 10 % of the exact value."""
        return abs(estimate - value) <= 0.1 * value

    # must reject: the lift estimate on the Dyck sequence, which is more than twice the exact value
    assert lift_topological_entropy(stream) > 2 * exact
    assert not close(lift_topological_entropy(stream), exact)
    # the same estimator is right on the worked example (must not reject a correct case)
    two = generate.as_stream(generate.generate_strings(generate.two_state_grammar(seed=SEED), 2000))
    assert lift_topological_entropy(two) == pytest.approx(0.5515, abs=1e-3)
    assert close(lift_topological_entropy(two), 0.5515)


def test_ladder_table_has_one_row_per_rung_and_two_level_columns():
    rungs = default_ladder()
    table = ladder_table(rungs, n_strings=100, seed=SEED)
    assert list(table.index) == [rung.name for rung in rungs]
    assert list(table[("", "family")]) == [rung.family for rung in rungs]
    assert [level for level, _ in table.columns[1:]] == [level for _, level, *_ in MEASURES]
    # the table can be extended with a rung defined by the participant
    extended = ladder_table(
        [*rungs, Rung("three brackets", "dyck", {"k": 3, "depth": 2, "pairs": 3})], 100, SEED
    )
    assert extended.index[-1] == "three brackets" and len(extended) == len(rungs) + 1
    with pytest.raises(KeyError):
        ladder_table([Rung("unknown", "no such kind")], 10, SEED)  # must reject
