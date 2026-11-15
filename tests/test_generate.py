"""Generated strings are valid strings of their grammar. Every test has a case it must reject."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from tutorial import exact, generate
from tutorial.ladder import Rung, build_generator, default_ladder, sample_strings

N = 300


def is_nested(string, pairs) -> bool:
    """A_i A_j ... B_j B_i: the terminals mirror the starts."""
    half = len(string) // 2
    terminal_of = dict(pairs)
    starts, terminals = string[:half], string[half:]
    return len(set(starts)) == half and [terminal_of.get(s) for s in reversed(starts)] == list(
        terminals
    )


def is_crossed(string, pairs) -> bool:
    """A_i A_j ... B_i B_j: the terminals follow the order of the starts."""
    half = len(string) // 2
    terminal_of = dict(pairs)
    starts, terminals = string[:half], string[half:]
    return len(set(starts)) == half and [terminal_of.get(s) for s in starts] == list(terminals)


def is_dyck(string, close_of, max_depth) -> bool:
    """Brackets balanced and correctly matched, never deeper than ``max_depth``."""
    stack = []
    for symbol in string:
        if symbol in close_of:
            stack.append(symbol)
            if len(stack) > max_depth:
                return False
        elif not stack or close_of[stack.pop()] != symbol:
            return False
    return not stack


@pytest.mark.parametrize(
    "rung",
    [r for r in default_ladder() if r.kind in ("two_state", "regular", "nax")],
    ids=lambda r: r.name,
)
def test_grammar_strings_are_grammatical(rung):
    generator = build_generator(rung, seed=3)
    strings = generate.generate_strings(generator, N)
    assert len(strings) == N
    assert all(generator.is_grammatical(string) for string in strings)
    # an independent route: every string has positive probability under the exact predictor
    assert exact.evaluate_on_strings(exact.predictor_for(generator), strings)["impossible"] == 0

    # must reject: strings that SymSeq itself corrupted, and a string with its first two symbols swapped
    corrupted = [generator.generate_nongrammatical_string() for _ in range(20)]
    assert not any(generator.is_grammatical(string) for string in corrupted)
    swapped = next([s[1], s[0], *s[2:]] for s in strings if s[0] != s[1])
    assert not generator.is_grammatical(swapped)
    assert exact.evaluate_on_strings(exact.predictor_for(generator), [swapped])["impossible"] == 1


def test_nax_strings_have_context_cue_fillers_probe():
    generator = generate.nax_grammar(3, seed=1)
    for string in generate.generate_strings(generator, N):
        context, cue, *fillers, probe = string
        assert context in generator.contexts
        assert cue == generator.cue_map[context]
        assert all(filler in generator.fillers for filler in fillers)
        assert probe in generator.probe_map.values()
    # must reject: a cue that belongs to another context
    assert not generator.is_grammatical(["1", generator.cue_map["2"], generator.probe_map["1"]])


@pytest.mark.parametrize("randomize", [True, False])
def test_non_adjacent_dependency_strings_are_in_the_vocabulary(randomize):
    generator = generate.nad(3, 3, seed=1)
    vocabulary = {
        tuple(s)
        for s in generator.generate_vocabulary(
            filler_len=2, randomize_fillers=randomize, verbose=False
        )
    }
    assert len(vocabulary) == (27 if randomize else 9)
    strings = generate.generate_strings(generator, N, filler_len=2, randomize_fillers=randomize)
    assert all(tuple(string) in vocabulary for string in strings)
    # must reject: SymSeq's own violations (the terminal of another pair)
    violations = [
        generator.generate_string(filler_len=2, randomize_fillers=randomize, violation=True)
        for _ in range(20)
    ]
    assert not any(tuple(string) in vocabulary for string in violations)


def test_nested_and_crossed_strings_follow_their_order():
    nested, crossed = generate.nested_nad(3, seed=1), generate.crossed_nad(3, seed=1)
    nested_strings = generate.generate_strings(nested, N)
    crossed_strings = generate.generate_strings(crossed, N)
    assert all(is_nested(string, nested.dependency_pairs) for string in nested_strings)
    assert all(is_crossed(string, crossed.dependency_pairs) for string in crossed_strings)
    assert {tuple(s) for s in nested_strings} == {
        tuple([*(f"A{i}" for i in order), *(f"B{i}" for i in reversed(order))])
        for order in itertools.permutations(range(3))
    }
    # must reject: each order is invalid for the other language, and so are SymSeq's own violations
    assert not any(is_crossed(string, nested.dependency_pairs) for string in nested_strings)
    assert not any(is_nested(string, crossed.dependency_pairs) for string in crossed_strings)
    assert not any(
        is_nested(nested.generate_string(violation=True), nested.dependency_pairs)
        for _ in range(20)
    )
    assert not any(
        is_crossed(crossed.generate_string(violation=True), crossed.dependency_pairs)
        for _ in range(20)
    )


@pytest.mark.parametrize("depth", [1, 2, 3, None])
def test_dyck_strings_are_balanced_within_the_depth(depth):
    generator = generate.dyck(k=2, depth=depth, pairs=4, seed=1)
    limit = depth if depth is not None else 4
    strings = generate.generate_strings(generator, N)
    assert all(len(string) == 8 for string in strings)
    assert all(is_dyck(string, generator.parentheses, limit) for string in strings)
    # must reject: SymSeq's own corrupted strings, and a string deeper than the limit
    corrupted = generator.generate_nongrammatical_strings(20)
    assert not any(is_dyck(string, generator.parentheses, limit) for string in corrupted)
    if depth is not None and depth < 4:
        too_deep = ["a"] * (depth + 1) + ["A"] * (depth + 1) + ["a", "A"] * (3 - depth)
        assert is_dyck(too_deep, generator.parentheses, 4)
        assert not is_dyck(too_deep, generator.parentheses, limit)


def test_two_state_grammar_declares_no_zero_probability_transition():
    grammar = generate.two_state_grammar(0.5, 1.0)
    assert grammar.is_grammatical(["A", "A", "B"])
    assert not grammar.is_grammatical(["A", "B", "B"])  # B is never repeated when q = 1
    assert generate.two_state_grammar(0.5, 0.6).is_grammatical(["A", "B", "B"])
    for p, q in [(0.0, 1.0), (0.5, 0.0), (1.2, 1.0)]:
        with pytest.raises(ValueError):
            generate.two_state_grammar(p, q)  # must reject


def test_regular_grammar_has_the_requested_size_and_is_unifilar():
    for n_states, alphabet_size in [(4, 4), (6, 4), (6, 3), (8, 4), (5, 2)]:
        grammar = generate.regular_grammar(n_states, alphabet_size, 0.3, seed=0)
        ambiguities, depth = generate.split_states(n_states, alphabet_size)
        assert alphabet_size - ambiguities + ambiguities * depth == n_states
        assert len(grammar.states) - 1 == n_states  # SymSeq counts the end marker as a state
        assert len(grammar.alphabet) == alphabet_size
        symbol_of = {state.tostring(): state.symbol for state in grammar.states}
        for source in symbol_of:
            emitted = [symbol_of[target] for s, target, _ in grammar.transitions if s == source]
            assert len(emitted) == len(set(emitted))  # one transition per symbol
    with pytest.raises(ValueError):
        generate.split_states(3, 4)  # must reject: fewer states than symbols


def test_regular_grammar_exists_over_the_range_of_the_notebook_controls():
    """Alphabet 2 to 6 and any number of states from the alphabet size to twice that size, as notebook A clamps it."""
    for alphabet_size in range(2, 7):
        for n_states in range(alphabet_size, 2 * alphabet_size + 1):
            grammar = generate.regular_grammar(n_states, alphabet_size, 0.3, seed=0)
            assert len(grammar.states) - 1 == n_states
    # must reject: outside that range SymSeq refuses, here because there are more transitions than slots
    with pytest.raises(ValueError, match="Too many edges"):
        generate.regular_grammar(7, 2, 0.3, seed=0)
    source = (
        Path(__file__).resolve().parents[1] / "notebooks" / "a_generate_and_measure.py"
    ).read_text()
    assert (
        "min(max(states_slider.value, alphabet_slider.value), 2 * alphabet_slider.value)" in source
    )


def test_stream_joins_strings_with_or_without_the_end_marker():
    strings = [["A", "B"], ["A", "A", "B"]]
    assert generate.as_stream(strings) == ["A", "B", "#", "A", "A", "B", "#"]
    assert generate.as_stream(strings, delimiter=False) == ["A", "B", "A", "A", "B"]

    def split(stream: list[str]) -> list[list[str]]:
        """The strings of a stream, cut at the end marker."""
        out, current = [], []
        for symbol in stream:
            if symbol == "#":
                out.append(current)
                current = []
            else:
                current.append(symbol)
        return [*out, current] if current else out

    assert split(generate.as_stream(strings)) == strings
    # must reject: without the marker the strings cannot be recovered from the stream
    assert split(generate.as_stream(strings, delimiter=False)) != strings


def test_sampling_is_reproducible_and_depends_on_the_seed():
    for rung in [
        *default_ladder(),
        Rung("Dyck (depth 3)", "dyck", {"k": 2, "depth": 3, "pairs": 4}),
    ]:
        first = sample_strings(rung, 50, seed=7)
        assert first == sample_strings(rung, 50, seed=7)
        assert first != sample_strings(
            rung, 50, seed=8
        )  # must reject: another seed gives other strings
