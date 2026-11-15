"""Exact entropy and next-symbol accuracy ceiling, against brute force and against SymSeq's samples.

Every test has a case it must reject.
"""

from __future__ import annotations

import itertools
import math

import pytest

from tutorial import closed_forms, exact, generate
from tutorial.ladder import Rung, build_generator, default_ladder

LADDER = [
    *default_ladder(),
    Rung("Dyck (depth 3)", "dyck", {"k": 2, "depth": 3, "pairs": 4}),
    Rung("4-AX", "nax", {"n": 4}),
]


def table_two_state():
    grammar = generate.two_state_grammar(0.5, 1.0)
    return (
        grammar,
        {},
        closed_forms.string_distribution(grammar),
    )  # paths of the grammar, no predictor involved


def table_nad():
    generator = generate.nad(3, 3)
    sample = {"filler_len": 2, "randomize_fillers": True}
    vocabulary = generator.generate_vocabulary(
        verbose=False, **sample
    )  # SymSeq's own enumeration, uniform
    return generator, sample, {tuple(string): 1 / len(vocabulary) for string in vocabulary}


def table_ordered(nested: bool):
    generator = generate.nested_nad(3) if nested else generate.crossed_nad(3)
    table = {}
    for order in itertools.permutations(range(3)):
        closing = reversed(order) if nested else order
        table[(*(f"A{i}" for i in order), *(f"B{i}" for i in closing))] = 1 / 6
    return generator, {}, table


def table_dyck(depth: int):
    """All 4^8 sequences over two bracket types, filtered by an independent validity check; uniform."""
    generator = generate.dyck(k=2, depth=depth, pairs=4)
    close_of = {"a": "A", "b": "B"}

    def valid(string) -> bool:
        stack = []
        for symbol in string:
            if symbol in close_of:
                stack.append(symbol)
                if len(stack) > depth:
                    return False
            elif not stack or close_of[stack.pop()] != symbol:
                return False
        return not stack

    strings = [s for s in itertools.product("aAbB", repeat=8) if valid(s)]
    return generator, {}, {string: 1 / len(strings) for string in strings}


TABLES = {
    "two-state": table_two_state,
    "non-adjacent": table_nad,
    "nested": lambda: table_ordered(True),
    "crossed": lambda: table_ordered(False),
    "Dyck (depth 2)": lambda: table_dyck(2),
    "Dyck (depth 3)": lambda: table_dyck(3),
}


# For the case that must be rejected: a language whose ceiling is different.
OTHER = {
    "two-state": "nested",
    "non-adjacent": "nested",
    "nested": "non-adjacent",
    "crossed": "non-adjacent",
    "Dyck (depth 2)": "Dyck (depth 3)",
    "Dyck (depth 3)": "Dyck (depth 2)",
}


@pytest.mark.parametrize("name", TABLES)
def test_ceiling_and_entropy_match_brute_force(name):
    """The forward computation against a table of all strings built without the predictor."""
    generator, sample, table = TABLES[name]()
    assert sum(table.values()) == pytest.approx(1.0, abs=1e-10)
    statistics = exact.trial_statistics(exact.predictor_for(generator, **sample))
    brute = exact.ceiling_from_strings(table)
    entropy = -sum(p * math.log2(p) for p in table.values())
    mean_length = sum(p * len(s) for s, p in table.items())
    assert statistics["ceiling_with_eos"] == pytest.approx(brute["ceiling_with_eos"], abs=1e-9)
    assert statistics["ceiling_within"] == pytest.approx(brute["ceiling_within"], abs=1e-9)
    for key in ("previous_symbol_with_eos", "previous_symbol_within"):
        assert statistics[key] == pytest.approx(brute[key], abs=1e-9)
        assert statistics[key] <= statistics[key.replace("previous_symbol", "ceiling")] + 1e-12
    assert statistics["string_entropy"] == pytest.approx(entropy, abs=1e-8)
    assert statistics["mean_length"] == pytest.approx(mean_length, abs=1e-9)
    assert statistics["entropy_rate"] == pytest.approx(entropy / (mean_length + 1), abs=1e-9)
    # the predictor's own enumeration gives the same table
    enumerated = exact.enumerate_strings(exact.predictor_for(generator, **sample))
    likely = {
        s for s, p in table.items() if p > 1e-9
    }  # the two enumerations cut their tails at different lengths
    assert likely <= enumerated.keys()
    assert all(enumerated[s] == pytest.approx(table[s], abs=1e-12) for s in likely)
    assert sum(p for s, p in enumerated.items() if s not in likely) < 1e-8

    # must reject: the brute-force ceiling of another language
    _, _, other = TABLES[OTHER[name]]()
    assert (
        abs(exact.ceiling_from_strings(other)["ceiling_with_eos"] - statistics["ceiling_with_eos"])
        > 1e-2
    )


def test_values_derived_by_hand():
    two = exact.trial_statistics(exact.predictor_for(generate.two_state_grammar(0.5, 1.0)))
    assert two["entropy_rate"] == pytest.approx(
        0.5, abs=1e-9
    )  # 2 bits per string of 3 symbols plus the marker
    assert two["ceiling_with_eos"] == pytest.approx(
        2 / 3, abs=1e-9
    )  # two coin flips after A, one certain end after B
    assert two["ceiling_within"] == pytest.approx(0.5, abs=1e-9)
    nested = exact.trial_statistics(exact.predictor_for(generate.nested_nad(3)))
    assert nested["entropy_rate"] == pytest.approx(math.log2(6) / 7, abs=1e-9)
    assert nested["ceiling_with_eos"] == pytest.approx(
        5.5 / 6, abs=1e-9
    )  # one of six outcomes is a coin flip
    assert nested["ceiling_within"] == pytest.approx(4.5 / 5, abs=1e-9)
    dyck = exact.trial_statistics(exact.predictor_for(generate.dyck(2, 2, 4)))
    assert dyck["entropy_rate"] == pytest.approx(
        7 / 9, abs=1e-9
    )  # 8 structures x 16 type assignments = 2^7 strings
    # seeing only the previous symbol: nothing is lost on the two-state grammar, much on the nested one
    assert two["previous_symbol_with_eos"] == pytest.approx(two["ceiling_with_eos"], abs=1e-12)
    # after any symbol of the nested language three outcomes are equally likely: 2 of 6 predictions are right
    assert nested["previous_symbol_with_eos"] == pytest.approx(2 / 6, abs=1e-9)
    assert nested["previous_symbol_within"] == pytest.approx(2 / 5, abs=1e-9)
    # must reject: nested and crossed have the same ceiling, so the ceiling cannot tell them apart
    crossed = exact.trial_statistics(exact.predictor_for(generate.crossed_nad(3)))
    assert crossed["ceiling_with_eos"] == pytest.approx(nested["ceiling_with_eos"], abs=1e-12)


@pytest.mark.parametrize("rung", LADDER, ids=lambda r: r.name)
def test_predictors_agree_with_strings_generated_by_symseq(rung):
    """The predictor scored on SymSeq's strings reaches its own ceiling and its own entropy.

    Acceptance, fixed before the run: on 4000 strings no string is impossible, the log loss per
    string is within 5 standard errors of the exact string entropy, and both accuracies are within
    5 binomial standard errors of the exact ceilings.
    """
    n = 4000
    generator = build_generator(rung, seed=21)
    strings = generate.generate_strings(generator, n, **rung.sample)
    predictor = exact.predictor_for(generator, **rung.sample)
    statistics = exact.trial_statistics(predictor)
    scored = exact.evaluate_on_strings(predictor, strings)
    assert scored["impossible"] == 0
    assert statistics["dropped_mass"] < 1e-9
    loss_error = 5 * scored["log_loss_sd"] / math.sqrt(n) + 1e-9
    assert abs(scored["log_loss"] - statistics["string_entropy"]) <= loss_error
    for convention in ("with_eos", "within"):
        ceiling = statistics[f"ceiling_{convention}"]
        error = 5 * math.sqrt(ceiling * (1 - ceiling) / scored[f"n_{convention}"]) + 1e-9
        assert abs(scored[f"accuracy_{convention}"] - ceiling) <= error
    # must reject: the same strings read backwards are not strings of the generator
    backwards = exact.evaluate_on_strings(predictor, [string[::-1] for string in strings])
    assert backwards["impossible"] > 0


WRONG = [
    # (strings come from, predictor built for, what must happen)
    (
        Rung("2-AX", "nax", {"n": 2, "p_target": 0.6}),
        Rung("2-AX", "nax", {"n": 2, "p_target": 0.9}),
        "loss",
    ),
    (
        Rung("Dyck 2", "dyck", {"k": 2, "depth": 2, "pairs": 4}),
        Rung("Dyck 3", "dyck", {"k": 2, "depth": 3, "pairs": 4}),
        "loss",
    ),
    (
        Rung("two", "two_state", {"p": 0.5, "q": 1.0}),
        Rung("two", "two_state", {"p": 0.7, "q": 1.0}),
        "loss",
    ),
    (
        Rung("crossed", "crossed", {"n_deps": 3}),
        Rung("nested", "nested", {"n_deps": 3}),
        "impossible",
    ),
    (
        Rung("Dyck 3", "dyck", {"k": 2, "depth": 3, "pairs": 4}),
        Rung("Dyck 2", "dyck", {"k": 2, "depth": 2, "pairs": 4}),
        "impossible",
    ),
]


@pytest.mark.parametrize(
    "source,wrong,expected", WRONG, ids=lambda x: x.name if isinstance(x, Rung) else x
)
def test_a_predictor_with_the_wrong_distribution_is_rejected(source, wrong, expected):
    """must reject: the same criterion fails when the predictor does not describe the generator."""
    n = 4000
    strings = generate.generate_strings(build_generator(source, seed=21), n, **source.sample)
    predictor = exact.predictor_for(build_generator(wrong), **wrong.sample)
    scored = exact.evaluate_on_strings(predictor, strings)
    if expected == "impossible":
        assert scored["impossible"] > 0
    else:
        true_entropy = exact.trial_statistics(
            exact.predictor_for(build_generator(source), **source.sample)
        )["string_entropy"]
        assert scored["log_loss"] - true_entropy > 5 * scored["log_loss_sd"] / math.sqrt(n)


def test_unsupported_generators_are_refused():
    """must reject: cases the predictors do not cover raise instead of returning a number."""
    from symseq.generators import DyckGenerator

    with pytest.raises(NotImplementedError):
        exact.predictor_for(DyckGenerator(k=2, mode="stack", p_open=0.3, n_distractors=0, seed=0))
    with pytest.raises(NotImplementedError):
        exact.predictor_for(generate.two_state_grammar(), length_range=(1, 5))
    with pytest.raises(NotImplementedError):
        exact.predictor_for(object())
