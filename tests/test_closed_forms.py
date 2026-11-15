"""The closed forms of the worked example, against brute force and against SymSeq's estimators.

Every test has a case it must reject, marked "must reject".
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from symseq.metrics.grammar import topological_entropy
from symseq.metrics.string.entropy import estimate_emc

from tutorial import closed_forms as cf
from tutorial.generate import as_stream, generate_strings, nax_grammar, two_state_grammar

TOL = 1e-9
CASES = [(0.5, 1.0), (0.3, 1.0), (0.5, 0.6), (0.2, 0.9)]


@pytest.mark.parametrize("delimiter", [False, True])
@pytest.mark.parametrize("p,q", CASES)
def test_entropy_rate_matches_enumeration_of_strings(p, q, delimiter):
    """Closed-form entropy rate (hand formula and matrix route) against the enumeration of all strings."""
    grammar = two_state_grammar(p, q)
    brute = cf.renewal_statistics(cf.string_distribution(grammar), delimiter=delimiter)
    assert brute["enumerated_mass"] == pytest.approx(1.0, abs=1e-10)
    hand = cf.two_state_closed_forms(p, q, delimiter=delimiter)
    _, P = cf.stream_chain(grammar, delimiter=delimiter)
    assert hand["entropy_rate"] == pytest.approx(brute["entropy_rate"], abs=TOL)
    assert cf.chain_entropy_rate(P) == pytest.approx(brute["entropy_rate"], abs=TOL)
    assert hand["symbol_entropy"] == pytest.approx(brute["symbol_entropy"], abs=TOL)

    # must reject: the closed form of a slightly different grammar
    other = cf.two_state_closed_forms(
        min(p + 0.01, 1.0) if p < 0.99 else p - 0.01, q, delimiter=delimiter
    )
    assert abs(other["entropy_rate"] - brute["entropy_rate"]) > 1e-4
    # must reject: a formula that weights the rows equally instead of by the stationary distribution
    wrong = float(np.mean([cf.entropy_bits(row) for row in P]))
    assert abs(wrong - brute["entropy_rate"]) > 1e-3


def test_worked_example_values():
    """p = 1/2: h = 2/3, H_1 = log2(3) - 2/3, E = log2(3) - 4/3, TE = log2(phi), pi = (2/3, 1/3)."""
    values = cf.two_state_closed_forms(0.5, 1.0, delimiter=False)
    assert values["pi_A"] == pytest.approx(2 / 3, abs=TOL)
    assert values["pi_B"] == pytest.approx(1 / 3, abs=TOL)
    assert values["entropy_rate"] == pytest.approx(2 / 3, abs=TOL)
    assert values["symbol_entropy"] == pytest.approx(math.log2(3) - 2 / 3, abs=TOL)
    assert values["excess_entropy"] == pytest.approx(math.log2(3) - 4 / 3, abs=TOL)
    assert values["topological_entropy"] == pytest.approx(math.log2(cf.PHI), abs=TOL)
    # published values of the golden-mean process, three decimals (Crutchfield and Feldman 2003)
    assert round(values["entropy_rate"], 3) == 0.667
    assert round(values["excess_entropy"], 3) == 0.252

    # must reject: with the end marker kept the sequence is another process, with other values
    marked = cf.two_state_closed_forms(0.5, 1.0, delimiter=True)
    assert marked["entropy_rate"] == pytest.approx(0.5, abs=TOL)
    assert marked["excess_entropy"] == pytest.approx(1.0, abs=TOL)
    assert abs(marked["entropy_rate"] - values["entropy_rate"]) > 0.1


@pytest.mark.parametrize("delimiter", [False, True])
@pytest.mark.parametrize("p,q", CASES)
def test_excess_entropy_matches_enumeration_of_words(p, q, delimiter):
    """E = H_1 - h against H_L - L h_L computed from the probability of every word of L symbols."""
    _, P = cf.stream_chain(two_state_grammar(p, q), delimiter=delimiter)
    block = [cf.block_entropy_enumerated(P, length) for length in range(1, 9)]
    rate = block[-1] - block[-2]
    assert rate == pytest.approx(cf.chain_entropy_rate(P), abs=TOL)
    for length, value in enumerate(block, start=1):
        assert value - length * rate == pytest.approx(cf.chain_excess_entropy(P), abs=1e-8)
    assert cf.chain_excess_entropy(P) == pytest.approx(
        cf.two_state_closed_forms(p, q, delimiter=delimiter)["excess_entropy"], abs=TOL
    )
    # must reject: the single-symbol entropy alone is not the excess entropy
    assert abs(block[0] - cf.chain_excess_entropy(P)) > 1e-3


def test_closed_forms_refuse_grammars_with_hidden_states():
    """must reject: a grammar with several states per symbol is not a first-order chain over its symbols."""
    with pytest.raises(ValueError, match="one state per symbol"):
        cf.stream_chain(nax_grammar(2))


def test_topological_entropy_counts_words():
    """log2(phi) against the growth of the number of allowed words, and against SymSeq's direct method."""
    _, P = cf.stream_chain(two_state_grammar(0.5, 1.0), delimiter=False)
    counts = [cf.count_words(P, length) for length in range(1, 16)]
    assert counts[:6] == [2, 3, 5, 8, 13, 21]  # Fibonacci numbers
    assert math.log2(counts[-1] / counts[-2]) == pytest.approx(math.log2(cf.PHI), abs=1e-5)
    assert cf.chain_topological_entropy(P) == pytest.approx(math.log2(cf.PHI), abs=TOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        symseq_nats = topological_entropy(
            transitions=np.array([[1, 1], [1, 0]]), method="direct", verbose=False
        )
    assert symseq_nats / math.log(2) == pytest.approx(math.log2(cf.PHI), abs=TOL)

    # must reject: the unconstrained two-symbol sequence grows faster (one bit per symbol)
    full = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert cf.chain_topological_entropy(full) == pytest.approx(1.0, abs=TOL)
    assert abs(cf.chain_topological_entropy(full) - math.log2(cf.PHI)) > 0.3


def test_symseq_topological_entropy_of_the_grammar_object_includes_the_end_marker():
    """SymSeq's method on the grammar object closes the graph through '#': 0.5515 bits, not log2(phi)."""
    grammar = two_state_grammar(0.5, 1.0)
    closed_bits = grammar.topological_entropy(correct_terminal_sink=True) / math.log(2)
    assert closed_bits == pytest.approx(
        cf.two_state_closed_forms(0.5, 1.0, delimiter=True)["topological_entropy"], abs=TOL
    )
    assert closed_bits == pytest.approx(0.5515, abs=1e-4)
    # must reject: it is not the value of the two-symbol chain
    assert abs(closed_bits - math.log2(cf.PHI)) > 0.1


def test_entropy_rate_curve_has_its_maximum_at_the_topological_entropy():
    """h(p) = H(p) / (1 + p) is largest at p = 1 / phi^2, where it equals log2(phi)."""
    grid = np.linspace(0.001, 0.999, 99_801)
    rates = cf.golden_mean_entropy_rate(grid)
    assert grid[int(np.argmax(rates))] == pytest.approx(1 / cf.PHI**2, abs=2e-5)
    assert float(np.max(rates)) == pytest.approx(math.log2(cf.PHI), abs=1e-9)
    assert np.all(rates <= math.log2(cf.PHI) + 1e-12)
    for p in (0.1, 0.3, 0.5, 0.9):
        _, P = cf.stream_chain(two_state_grammar(p, 1.0), delimiter=False)
        assert float(cf.golden_mean_entropy_rate(p)) == pytest.approx(
            cf.chain_entropy_rate(P), abs=TOL
        )
    # must reject: at p = 1/2 the entropy rate is below the topological entropy
    assert math.log2(cf.PHI) - float(cf.golden_mean_entropy_rate(0.5)) > 0.02


def test_symseq_estimates_converge_to_the_closed_forms():
    """SymSeq's estimate_emc on a long generated sequence against the closed forms.

    Acceptance, fixed before the run: with at least 20 000 symbols the estimates of h, H_1 and E are
    within 0.01 bit of the closed forms, at order 1.
    """
    strings = generate_strings(two_state_grammar(0.5, 1.0, seed=2026), 8000)
    stream = as_stream(strings, delimiter=False)
    assert len(stream) >= 20_000
    estimate = estimate_emc(stream, order="bic", max_order=6)
    exact = cf.two_state_closed_forms(0.5, 1.0, delimiter=False)
    assert estimate["status"] == "computed"
    assert estimate["order"] == 1
    assert estimate["entropy_rate"] == pytest.approx(exact["entropy_rate"], abs=0.01)
    assert estimate["entropy"] == pytest.approx(exact["symbol_entropy"], abs=0.01)
    assert estimate["emc"] == pytest.approx(exact["excess_entropy"], abs=0.01)

    # must reject: the closed forms of the sequence with the end marker
    marked = cf.two_state_closed_forms(0.5, 1.0, delimiter=True)
    assert abs(estimate["entropy_rate"] - marked["entropy_rate"]) > 0.1
    assert abs(estimate["emc"] - marked["excess_entropy"]) > 0.5
    # and the estimates on the sequence with the marker agree with those closed forms instead
    with_marker = estimate_emc(as_stream(strings, delimiter=True), order="bic", max_order=6)
    assert with_marker["entropy_rate"] == pytest.approx(marked["entropy_rate"], abs=0.01)
    assert with_marker["emc"] == pytest.approx(marked["excess_entropy"], abs=0.02)
