"""The prescribed-complexity exercise produces its own grammar, trace and measured values."""

import numpy as np
import pytest

from tutorial import complexity, prescribed


def test_prescribed_grammar_trace_and_examples_agree():
    result = prescribed.make_prescribed(1.4, seed=7)
    assert result["reason"] == ""
    assert result["achieved_open_bits"] == pytest.approx(1.4, abs=0.05)
    assert result["entropy_bits"][-1] == pytest.approx(result["achieved_open_bits"], abs=1e-12)
    assert result["target_bits"] == 1.4
    assert result["closed_bits"] > 0
    assert len(result["steps"]) == len(result["entropy_bits"]) == len(result["score"])
    spec = result["spec"]
    edges = set(spec.edges)
    table = np.array(
        [[float((source, target) in edges) for target in spec.states] for source in spec.states]
    )
    assert np.isclose(np.log(max(np.abs(np.linalg.eigvals(table)))) / complexity.LN2, result["entropy_bits"][-1])
    examples = prescribed.example_strings(result)
    assert len(examples) == 6
    grammar = complexity.build_grammar(spec)
    assert all(grammar.is_grammatical(example.split()) for example in examples)
    # must reject: a target the grammar cannot have, above log2 of its state count
    with pytest.raises(ValueError):
        prescribed.make_prescribed(np.log2(9), seed=7)


def test_prescribed_target_changes_the_result():
    low = prescribed.make_prescribed(0.8, seed=7)
    high = prescribed.make_prescribed(1.4, seed=7)
    assert low["achieved_open_bits"] < high["achieved_open_bits"]
    assert low["spec"].edges != high["spec"].edges
    assert prescribed.figure_data(low)["target_bits"] == 0.8
    assert prescribed.figure_data(high)["achieved_open_bits"] == high["achieved_open_bits"]
