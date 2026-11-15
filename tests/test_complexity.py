"""Grammars of controlled complexity: the exact predictor, the admission rule, the strings and the measures.

Each test has a case it must reject. Grammar seeds and data seeds are 101 or more.
"""

from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
import pytest
from symseq.generators.ag.artificial_grammar import ArtificialGrammar
from symseq.generators.ag.generator import _check_grammar_validity

from tutorial import complexity as cx
from tutorial import config, exact
from tutorial.generate import EOS

SETTINGS = config.COMPLEXITY

# A small grammar in which the string does not determine the walk: B leads to two states of A.
AMBIGUOUS = cx.GrammarSpec(
    name="ambiguous",
    level=0,
    seed=101,
    target_nats=0.0,
    states=("B", "A(0)", "A(1)", "C"),
    alphabet=("A", "B", "C"),
    edges=(
        ("B", "A(0)"),
        ("B", "A(1)"),
        ("A(0)", "A(0)"),
        ("A(0)", "C"),
        ("A(1)", "B"),
        ("A(1)", "C"),
        ("C", "B"),
    ),
    start="B",
    terminals=("C", "A(1)"),
)

# The same graph with one state per symbol: nothing has to be remembered beyond the last symbol.
PLAIN = cx.GrammarSpec(
    name="plain",
    level=0,
    seed=101,
    target_nats=0.0,
    states=("A", "B", "C", "D"),
    alphabet=("A", "B", "C", "D"),
    edges=(("A", "B"), ("A", "C"), ("B", "B"), ("B", "D"), ("C", "A"), ("C", "D"), ("D", "A")),
    start="A",
    terminals=("D", "C"),
)

# Two states of A that no state can reach both: a string has one walk, and memory matters.
UNIFILAR = cx.GrammarSpec(
    name="unifilar",
    level=0,
    seed=101,
    target_nats=0.0,
    states=("S", "A(0)", "A(1)", "B", "C"),
    alphabet=("A", "B", "C", "S"),
    edges=(
        ("S", "A(0)"),
        ("S", "B"),
        ("B", "A(1)"),
        ("B", "S"),
        ("A(0)", "B"),
        ("A(0)", "C"),
        ("A(1)", "A(1)"),
        ("A(1)", "C"),
        ("C", "S"),
        ("C", "B"),
    ),
    start="S",
    terminals=("C",),
)

# The two states of A lead to different symbols: after A, only what came before says what follows.
MEMORY = cx.GrammarSpec(
    name="memory",
    level=0,
    seed=101,
    target_nats=0.0,
    states=("S", "A(0)", "A(1)", "B", "C", "D"),
    alphabet=("A", "B", "C", "D", "S"),
    edges=(
        ("S", "A(0)"),
        ("S", "D"),
        ("D", "A(1)"),
        ("A(0)", "B"),
        ("A(1)", "C"),
        ("B", "S"),
        ("B", "D"),
        ("C", "S"),
        ("C", "D"),
    ),
    start="S",
    terminals=("B", "C"),
)

# No cycle: a finite language, so no length bound is needed to list every string.
ACYCLIC = cx.GrammarSpec(
    name="acyclic",
    level=0,
    seed=101,
    target_nats=0.0,
    states=("B", "A(0)", "A(1)", "C", "D"),
    alphabet=("A", "B", "C", "D"),
    edges=(("B", "A(0)"), ("B", "A(1)"), ("A(0)", "C"), ("A(0)", "D"), ("A(1)", "D"), ("C", "D")),
    start="B",
    terminals=("D", "A(1)"),
)


def brute_force(spec: cx.GrammarSpec, min_length: int, max_length: int) -> dict:
    """Every string within the bounds with its probability, by listing every walk of the grammar.

    Written without any of the module's code: walks are extended state by state with the
    probabilities of the SymSeq grammar, the walks of one string are added up, strings outside the
    bounds are dropped as the sampler drops them, and the rest is rescaled to sum to one.
    """
    grammar = cx.build_grammar(spec)
    symbol = {state.tostring(): state.symbol for state in grammar.states}
    following: dict[str, list] = {}
    for source, target, probability in grammar.transitions:
        following.setdefault(source, []).append((target, probability))
    table: dict[tuple, float] = {}
    stack = [((name,), probability) for name, probability in grammar.start_probabilities.items()]
    while stack:
        walk, mass = stack.pop()
        for target, probability in following[walk[-1]]:
            if target == EOS:
                if min_length <= len(walk) <= max_length:
                    string = tuple(symbol[state] for state in walk)
                    table[string] = table.get(string, 0.0) + mass * probability
            elif len(walk) < max_length:
                stack.append(((*walk, target), mass * probability))
    total = sum(table.values())
    return {string: mass / total for string, mass in table.items()}


def conditionals(table: dict) -> dict:
    """For every prefix of a table of strings: the distribution of the next outcome."""
    out: dict[tuple, dict] = {}
    for string, mass in table.items():
        for position in range(len(string) + 1):
            outcome = string[position] if position < len(string) else EOS
            row = out.setdefault(string[:position], {})
            row[outcome] = row.get(outcome, 0.0) + mass
    return {
        prefix: {o: m / sum(row.values()) for o, m in row.items()} for prefix, row in out.items()
    }


def largest_difference(predictor, table: dict) -> float:
    """Largest difference between a predictor and a table of strings, over all prefixes and outcomes."""
    worst = 0.0
    for prefix, row in conditionals(table).items():
        state = predictor.initial()
        for symbol in prefix:
            state = predictor.advance(state, symbol)
        found = predictor.distribution(state)
        for outcome in set(row) | set(found):
            worst = max(worst, abs(row.get(outcome, 0.0) - found.get(outcome, 0.0)))
    return worst


def differs(predictor, table: dict) -> bool:
    """True if a predictor is not the one of a table of strings: a probability is off, or a prefix is impossible."""
    try:
        return largest_difference(predictor, table) > 1e-3
    except ValueError:
        return True


# --- the exact predictor --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "bounds"),
    [(AMBIGUOUS, (2, 7)), (AMBIGUOUS, (3, 5)), (UNIFILAR, (2, 8)), (PLAIN, (2, 6))],
)
def test_exact_predictor_equals_the_listing_of_all_walks(spec, bounds):
    table = brute_force(spec, *bounds)
    predictor = cx.BoundedPredictor(cx.build_grammar(spec), *bounds)
    assert len(table) > 5
    assert largest_difference(predictor, table) < 1e-12
    listed = exact.enumerate_strings(predictor)
    assert set(listed) == set(table)
    assert max(abs(listed[string] - table[string]) for string in table) < 1e-12
    assert cx.count_strings(predictor) == len(table)
    assert all(
        abs(cx.string_probability(predictor, string) - mass) < 1e-12
        for string, mass in table.items()
    )
    # must reject: the same grammar with another upper bound, and with another lower bound
    assert differs(cx.BoundedPredictor(cx.build_grammar(spec), bounds[0], 20), table)
    tighter = cx.BoundedPredictor(cx.build_grammar(spec), bounds[0] + 2, bounds[1])
    assert differs(tighter, table)  # the shortest strings can no longer end
    assert min(cx.string_probability(tighter, string) for string in table) == 0.0


def test_hidden_states_are_inferred_and_a_predictor_that_ignores_them_is_wrong():
    """In the ambiguous grammar the symbol A has two states; what follows depends on which one it is."""
    bounds = (2, 7)
    table = brute_force(AMBIGUOUS, *bounds)
    rows = conditionals(table)
    predictor = cx.BoundedPredictor(cx.build_grammar(AMBIGUOUS), *bounds)
    assert not cx.is_unifilar(predictor)
    # after "B A" both states of A are possible: the belief is mixed, and the walks of a string add up
    state = predictor.advance(predictor.advance(predictor.initial(), "B"), "A")
    assert sorted(round(weight, 6) for weight in state[1] if weight > 0) == [0.5, 0.5]
    assert len(table) < sum(1 for _ in walks(AMBIGUOUS, *bounds))
    # must reject: the predictor of the grammar with one state per symbol (the two A merged) on these strings
    merged = cx.GrammarSpec(
        name="merged",
        level=0,
        seed=101,
        target_nats=0.0,
        states=("B", "A", "C"),
        alphabet=("A", "B", "C"),
        edges=(("B", "A"), ("A", "A"), ("A", "C"), ("A", "B"), ("C", "B")),
        start="B",
        terminals=("C", "A"),
    )
    first_order = cx.BoundedPredictor(cx.build_grammar(merged), *bounds)
    differences = []
    for prefix, row in rows.items():
        try:
            state = first_order.initial()
            for symbol in prefix:
                state = first_order.advance(state, symbol)
        except ValueError:
            continue
        found = first_order.distribution(state)
        differences.append(
            max(abs(row.get(o, 0.0) - found.get(o, 0.0)) for o in set(row) | set(found))
        )
    assert max(differences) > 0.05


def walks(spec: cx.GrammarSpec, min_length: int, max_length: int):
    """Every walk from the start state to the marker with a length within the bounds."""
    stack = [(spec.start,)]
    while stack:
        walk = stack.pop()
        if walk[-1] in spec.terminals and min_length <= len(walk) <= max_length:
            yield walk
        if len(walk) < max_length:
            stack.extend((*walk, target) for source, target in spec.edges if source == walk[-1])


def test_the_predictor_of_exact_py_is_exact_with_repeated_symbols_but_not_with_length_bounds():
    """``exact.GrammarPredictor`` filters the hidden state correctly; it knows nothing of length bounds."""
    grammar = cx.build_grammar(ACYCLIC)
    table = brute_force(ACYCLIC, 1, 10)  # the language is finite: these bounds exclude nothing
    assert largest_difference(exact.GrammarPredictor(grammar), table) < 1e-12
    assert largest_difference(cx.BoundedPredictor(grammar, 1, 10), table) < 1e-12
    # must reject: with bounds that exclude strings, the predictor of exact.py is no longer exact
    bounded = brute_force(AMBIGUOUS, 2, 7)
    assert largest_difference(exact.GrammarPredictor(cx.build_grammar(AMBIGUOUS)), bounded) > 1e-3
    with pytest.raises(NotImplementedError):
        exact.predictor_for(cx.build_grammar(AMBIGUOUS), length_range=(2, 7))


@pytest.mark.parametrize(("spec", "bounds"), [(AMBIGUOUS, (2, 7)), (UNIFILAR, (3, 9))])
def test_exact_predictor_describes_the_strings_symseq_draws(spec, bounds):
    """Frequencies of 6000 strings from SymSeq's sampler against the exact probabilities."""
    grammar = cx.build_grammar(spec, seed=101)
    predictor = cx.BoundedPredictor(grammar, *bounds)
    table = exact.enumerate_strings(predictor)
    n = 6000
    drawn = [tuple(grammar.generate_string(length_range=bounds)) for _ in range(n)]
    assert set(drawn) <= set(table)

    def outside(probabilities: dict) -> int:
        """Strings whose frequency is more than five standard errors from a probability."""
        count = 0
        for string, p in probabilities.items():
            frequency = sum(1 for other in drawn if other == string) / n
            count += abs(frequency - p) > 5.0 * math.sqrt(p * (1.0 - p) / n) + 1.0 / n
        return count

    assert outside(table) == 0
    surprisal = [-math.log2(table[string]) for string in drawn]
    entropy = -sum(p * math.log2(p) for p in table.values())
    assert abs(np.mean(surprisal) - entropy) < 5.0 * np.std(surprisal) / math.sqrt(n)
    assert abs(cx.exact_statistics(predictor)["string_entropy"] - entropy) < 1e-9
    assert (
        abs(cx.exact_statistics(predictor)["mean_length"] - np.mean([len(s) for s in drawn])) < 0.1
    )
    # must reject: the probabilities of the same grammar under other bounds
    other = exact.enumerate_strings(cx.BoundedPredictor(grammar, bounds[0], bounds[1] + 3))
    assert outside({string: other[string] for string in table}) > 0


def statistics_from_the_listing(table: dict) -> dict:
    """Entropy, ceiling and the two gaps of a table of strings, per target, with other code than the module's."""
    rows = conditionals(table)
    mass = {
        prefix: sum(m for s, m in table.items() if s[: len(prefix)] == prefix) for prefix in rows
    }
    targets = sum(mass.values())
    entropy = -sum(mass[x] * p * math.log2(p) for x, row in rows.items() for p in row.values())
    ceiling = sum(mass[x] * max(row.values()) for x, row in rows.items())

    def gap(context) -> float:
        joint: dict = {}
        for prefix, row in rows.items():
            for outcome, p in row.items():
                key = context(prefix)
                joint.setdefault(key, {}).setdefault(outcome, 0.0)
                joint[key][outcome] += mass[prefix] * p
        cross = -sum(
            m * math.log2(m / sum(row.values())) for row in joint.values() for m in row.values()
        )
        return (cross - entropy) / targets

    return {
        "mean_length": targets - 1.0,
        "entropy": entropy / targets,
        "ceiling": ceiling / targets,
        "first_order_gap": gap(lambda x: x[-1] if x else EOS),
        "memory_gap": gap(lambda x: (x[-1] if x else EOS, len(x))),
    }


def test_exact_statistics_by_hand_and_against_the_listing():
    """Entropy, ceiling and the two gaps computed from the table of all strings, with other code."""
    bounds = (2, 7)
    found = {}
    for spec in (AMBIGUOUS, UNIFILAR, PLAIN):
        listed = statistics_from_the_listing(brute_force(spec, *bounds))
        found[spec.name] = cx.exact_statistics(cx.BoundedPredictor(cx.build_grammar(spec), *bounds))
        assert found[spec.name]["dropped_mass"] == 0.0
        for key, value in listed.items():
            assert abs(found[spec.name][key] - value) < 1e-12, (spec.name, key)
        assert (
            -1e-12 <= found[spec.name]["memory_gap"] <= found[spec.name]["first_order_gap"] + 1e-12
        )
        assert found[spec.name]["bounds_gap"] > 0.0
    # must reject: the statistics of another grammar are not these
    assert abs(found["plain"]["entropy"] - found["ambiguous"]["entropy"]) > 1e-3
    assert abs(found["unifilar"]["memory_gap"] - found["plain"]["memory_gap"]) > 1e-3


def test_length_bounds_that_exclude_nothing_cost_nothing():
    """The walk without bounds is the exact predictor when no string is rejected, and not otherwise."""
    loose = cx.exact_statistics(cx.BoundedPredictor(cx.build_grammar(ACYCLIC), 1, 10))
    assert loose["bounds_gap"] < 1e-12
    # must reject: with bounds that exclude strings the walk is not the exact predictor
    tight = cx.exact_statistics(cx.BoundedPredictor(cx.build_grammar(ACYCLIC), 2, 3))
    assert tight["bounds_gap"] > 0.01


# --- the admission rule ---------------------------------------------------------------------------


def test_admission_rejects_a_grammar_in_which_every_symbol_has_one_state():
    plain = cx.describe(PLAIN, 4, 8)
    band, threshold = (0.0, 10.0), SETTINGS.min_memory_gap_bits
    # with one state per symbol the previous symbol and the position say all there is to say
    assert abs(plain["memory_gap"]) < 1e-12
    assert cx.admission(plain, band, threshold, 10) == ["memory gap below the minimum"]
    # a predictor that sees only the previous symbol is worse than the exact one by more than the
    # threshold, because of the length bounds alone: that rule would have admitted this grammar
    assert plain["first_order_gap"] > threshold
    for bounds in ((2, 6), (3, 6), (5, 8)):
        assert abs(cx.describe(PLAIN, *bounds)["memory_gap"]) < 1e-12
    # must reject the opposite: two states of one symbol make the gap positive, and the grammar is admitted
    memory = cx.describe(MEMORY, 3, 12)
    assert memory["unifilar"] and memory["memory_gap"] >= threshold
    assert cx.admission(memory, band, threshold, 10) == []


def test_admission_applies_each_rule():
    values = cx.describe(MEMORY, 3, 12)
    entropy, gap = values["topological_entropy_bits"], values["memory_gap"]
    typical = 2.0 ** values["string_entropy"]
    assert cx.admission(values, (entropy - 0.1, entropy + 0.1), 0.05, 10) == []
    # must reject: outside the band (the upper end is excluded), and too small a gap
    assert cx.admission(values, (entropy - 0.2, entropy), 0.05, 10) == [
        "topological entropy outside the band of the level"
    ]
    assert (
        cx.admission(values, (entropy, entropy + 0.2), 0.05, 10) == []
    )  # the lower end is included
    assert cx.admission(values, (0.0, 10.0), gap + 1e-6, 10) == ["memory gap below the minimum"]
    # must reject: fewer typical strings than a run needs, although there are enough strings
    assert typical < 14 <= values["n_strings"]
    assert cx.admission(values, (0.0, 10.0), 0.05, 14) == ["too few typical strings"]
    # the number of typical strings never exceeds the number of strings, so the count never rejects alone
    assert cx.admission(values, (0.0, 10.0), 0.05, values["n_strings"] + 1) == [
        "too few distinct strings",
        "too few typical strings",
    ]
    for spec, bounds in (
        (PLAIN, (4, 8)),
        (UNIFILAR, (2, 8)),
        (AMBIGUOUS, (2, 7)),
        (MEMORY, (3, 12)),
    ):
        found = cx.describe(spec, *bounds)
        assert 2.0 ** found["string_entropy"] <= found["n_strings"] + 1e-9
    assert len(cx.admission(values, (5.0, 6.0), 1.0, 10**9)) == 4


def test_topological_entropy_is_symseqs_value_in_bits_and_the_growth_rate_of_strings():
    grammar = cx.build_grammar(UNIFILAR)
    nats = grammar.topological_entropy(correct_terminal_sink=True)
    bits = cx.topological_entropy_bits(grammar)
    assert abs(bits - nats / math.log(2.0)) < 1e-12
    # the transition from the marker to the start is declared, so SymSeq's open value is the closed one
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert abs(grammar.topological_entropy() - nats) < 1e-12
    # for a unifilar grammar strings are walks: joined strings of total length L grow as 2^(bits L)
    table = np.zeros((6, 6))
    names = [*UNIFILAR.states, EOS]
    for source, target in (
        *UNIFILAR.edges,
        *((t, EOS) for t in UNIFILAR.terminals),
        (EOS, UNIFILAR.start),
    ):
        table[names.index(source), names.index(target)] = 1
    counts = [np.linalg.matrix_power(table, length).sum() for length in (40, 41)]
    assert abs(math.log2(counts[1] / counts[0]) - bits) < 1e-6
    # must reject: the value in nats is not the value in bits
    assert abs(nats - bits) > 0.1


def test_count_of_strings_counts_strings_and_not_walks():
    bounds = (2, 7)
    ambiguous = cx.BoundedPredictor(cx.build_grammar(AMBIGUOUS), *bounds)
    n_walks = sum(1 for _ in walks(AMBIGUOUS, *bounds))
    assert cx.count_strings(ambiguous) == len(brute_force(AMBIGUOUS, *bounds))
    # must reject: for the ambiguous grammar the number of walks is larger
    assert cx.count_strings(ambiguous) < n_walks
    unifilar = cx.BoundedPredictor(cx.build_grammar(UNIFILAR), *bounds)
    assert cx.is_unifilar(unifilar)
    assert cx.count_strings(unifilar) == sum(1 for _ in walks(UNIFILAR, *bounds))


# --- the route from SymSeq's synthesis to a grammar -----------------------------------------------


@pytest.fixture(scope="module")
def candidates():
    """The first grammars of the highest and lowest level from grammar seed 101 on."""
    found = {}
    for index in (0, len(SETTINGS.levels) - 1):
        for seed in range(101, 131):
            spec, _ = cx.candidate(
                index,
                seed,
                SETTINGS.levels[index].target_nats,
                SETTINGS.n_states,
                SETTINGS.alphabet_size,
                SETTINGS.n_terminal_states,
            )
            if spec is not None:
                found[index] = spec
                break
    return found


def test_a_candidate_has_the_declared_size_and_uses_every_state(candidates):
    assert len(candidates) == 2
    for spec in candidates.values():
        assert (
            len(spec.states) == SETTINGS.n_states and len(spec.alphabet) == SETTINGS.alphabet_size
        )
        symbols = [name.split("(")[0] for name in spec.states]
        repeated = [symbol for symbol in spec.alphabet if symbols.count(symbol) == 2]
        assert len(repeated) == SETTINGS.n_states - SETTINGS.alphabet_size
        assert sorted(set(symbols)) == sorted(spec.alphabet)
        assert (
            len(spec.terminals) == SETTINGS.n_terminal_states and spec.start not in spec.terminals
        )
        grammar = cx.build_grammar(spec)
        assert cx.is_unifilar(
            cx.BoundedPredictor(grammar, SETTINGS.min_length, SETTINGS.max_length)
        )
        # SymSeq's own rule for a valid random grammar: every state is reached and reaches the marker
        assert _check_grammar_validity(grammar.graph, [spec.start], 2, EOS, verbose=False)
        # equal probabilities out of every state, as in SymSeq's synthesis route
        out: dict[str, list] = {}
        for source, _, probability in grammar.transitions:
            out.setdefault(source, []).append(probability)
        assert all(max(p) - min(p) < 1e-12 and abs(sum(p) - 1.0) < 1e-12 for p in out.values())
        # the synthesis was aimed at the target of the level
        achieved = cx.describe(spec, SETTINGS.min_length, SETTINGS.max_length)["synthesised_nats"]
        assert abs(achieved - spec.target_nats) < 0.25
    # must reject: a grammar with a state no string passes through is not valid by SymSeq's rule
    spec = candidates[0]
    orphan = cx.GrammarSpec.from_dict(
        {**spec.as_dict(), "edges": [edge for edge in spec.edges if edge[1] != spec.terminals[0]]}
    )
    assert not _check_grammar_validity(
        cx.build_grammar(orphan).graph, [orphan.start], 2, EOS, verbose=False
    )


def test_trim_choices_and_pairings_are_what_they_say():
    table = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]
    )
    # every state lies on a cycle, so any start and any other terminal state use every state
    assert len(cx.trim_choices(table, 1)) == 4 * 3
    # states 1 and 2 are both reached from state 0, so they cannot share a symbol; 1 and 3 can
    pairs = {pair for (pair,) in cx.unifilar_pairings(table, 1)}
    assert (1, 2) not in pairs and (1, 3) in pairs
    assert cx.name_states(4, [(1, 3)], ["X", "Y", "Z"]) == ("Y", "X(0)", "Z", "X(1)")
    # must reject: with state 3 a dead end only state 3 can be terminal, and state 0 must be the start
    dead_end = table.copy()
    dead_end[3, 0] = 0
    assert cx.trim_choices(dead_end, 1) == [(0, (3,))]
    assert cx.trim_choices(np.zeros((3, 3), dtype=int), 1) == []


def test_candidates_are_seeded_and_symseqs_own_route_is_not(candidates):
    index = 0
    arguments = (
        SETTINGS.levels[index].target_nats,
        SETTINGS.n_states,
        SETTINGS.alphabet_size,
        SETTINGS.n_terminal_states,
    )
    spec = candidates[index]
    again, _ = cx.candidate(index, spec.seed, *arguments)
    assert again == spec
    assert cx.GrammarSpec.from_dict(spec.as_dict()) == spec
    # must reject: another grammar seed gives another grammar, and not the same transitions under other names
    others = [
        cx.candidate(index, seed, *arguments)[0] for seed in range(spec.seed + 1, spec.seed + 6)
    ]
    others = [other for other in others if other is not None]
    assert len(others) >= 2 and all(other.edges != spec.edges for other in others)

    def degrees(grammar: cx.GrammarSpec) -> tuple:
        """Transitions out of and into every state, whatever the states are called."""
        out = [sum(source == state for source, _ in grammar.edges) for state in grammar.states]
        into = [sum(target == state for _, target in grammar.edges) for state in grammar.states]
        return tuple(sorted(zip(out, into, strict=True)))

    assert len({degrees(grammar) for grammar in (spec, *others)}) > 1
    # the reason for calling the synthesis directly: SymSeq's from_constraints gives one table for every seed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tables = [
            sorted(
                (source, target)
                for source, target, _ in ArtificialGrammar.from_constraints(
                    alphabet_size=SETTINGS.n_states, target_complexity=0.8, seed=seed
                ).transitions
                if EOS not in (source, target)
            )
            for seed in (101, 102)
        ]
    assert tables[0] == tables[1]
    rng_tables = [
        cx.synthesise(0.8, SETTINGS.n_states, np.random.default_rng(seed)) for seed in (101, 102)
    ]
    assert not np.array_equal(*rng_tables)


# --- the strings of a run -------------------------------------------------------------------------


def drawn(sample: str, seed: int = 101, n_train: int = 60, n_heldout: int = 30) -> cx.Strings:
    return cx.draw_strings(UNIFILAR, seed, n_train, n_heldout, 3, 14, 20_000, sample)


@pytest.mark.parametrize("sample", cx.TRAINING_SAMPLES)
def test_grammatical_strings_are_grammatical_and_ungrammatical_ones_are_not(sample):
    strings = drawn(sample)
    grammar = cx.build_grammar(UNIFILAR)
    predictor = cx.BoundedPredictor(grammar, 3, 14)
    classes = cx.classes_of(UNIFILAR)
    training = strings.training(60, sample)
    assert all(grammar.is_grammatical(list(string)) for string in (*training, *strings.heldout))
    assert not any(grammar.is_grammatical(list(string)) for string in strings.ungrammatical)
    assert all(3 <= len(string) <= 14 for string in (*training, *strings.heldout))
    # one planted violation: same length, one symbol changed
    for good, bad in zip(strings.heldout, strings.ungrammatical, strict=True):
        assert len(good) == len(bad) and sum(a != b for a, b in zip(good, bad, strict=True)) == 1
    # the exact predictor agrees: every held-out string is possible, no ungrammatical one is
    _, possible = cx.exact_distributions(predictor, strings.heldout, classes, 15)
    _, possible_bad = cx.exact_distributions(predictor, strings.ungrammatical, classes, 15)
    assert possible.all() and not possible_bad.any()
    # must reject: a grammatical string among the ungrammatical ones is found possible
    _, mixed = cx.exact_distributions(
        predictor, [strings.heldout[0], strings.ungrammatical[0]], classes, 15
    )
    assert mixed.tolist() == [True, False]


@pytest.mark.parametrize("sample", cx.TRAINING_SAMPLES)
def test_training_validation_and_heldout_strings_are_disjoint(sample):
    strings = drawn(sample)
    training = strings.training(60, sample)
    distinct = list(training)
    gradient, validation = cx.split_training(len(distinct), 0.2, np.random.default_rng(101))
    assert len(validation) == round(0.2 * len(distinct)) and not set(gradient) & set(validation)
    assert sorted(gradient + validation) == list(range(len(distinct)))
    sets = ([distinct[i] for i in gradient], [distinct[i] for i in validation], strings.heldout)
    cx.assert_disjoint(*sets)
    cx.assert_apart(strings.ungrammatical, *sets)
    # two held-out strings may give the same ungrammatical string: a repeat among them is allowed
    cx.assert_apart([strings.ungrammatical[0], *strings.ungrammatical], *sets)
    assert len(set(strings.heldout)) == 30 and not set(strings.heldout) & set(strings.sequence)
    # must reject: an ungrammatical string that is also a gradient, a validation or a held-out string
    for leaked in (sets[0][0], sets[1][0], strings.heldout[3]):
        with pytest.raises(ValueError, match="also a training or a held-out string"):
            cx.assert_apart([*strings.ungrammatical[:5], leaked], *sets)
    # must reject: the strict rule still refuses a repeat among the ungrammatical strings, which
    # is why it is not the rule for them
    with pytest.raises(ValueError, match="more than once"):
        cx.assert_disjoint(*sets, [strings.ungrammatical[0], *strings.ungrammatical])
    # must reject: a held-out string copied into the training strings, and a string repeated in one set
    with pytest.raises(ValueError, match="more than once"):
        cx.assert_disjoint([*sets[0], strings.heldout[3]], sets[1], strings.heldout)
    with pytest.raises(ValueError, match="more than once"):
        cx.assert_disjoint([distinct[0], distinct[0]])
    with pytest.raises(ValueError, match="leaves no strings"):
        cx.split_training(2, 0.2, np.random.default_rng(101))


def test_the_two_training_samples_are_what_they_say():
    distinct, draws = drawn("distinct"), drawn("draws")
    # "distinct": the first n distinct strings, once each; smaller sizes are the first of the larger
    assert list(distinct.training(60, "distinct").values()) == [1] * 60
    assert list(distinct.training(20, "distinct")) == list(distinct.training(60, "distinct"))[:20]
    assert len(distinct.sequence) > 60  # frequent strings were drawn more than once on the way
    # "draws": the first n draws, a string as often as it was drawn
    counts = draws.training(60, "draws")
    assert sum(counts.values()) == 60 == len(draws.sequence) and max(counts.values()) > 1
    assert sum(draws.training(20, "draws").values()) == 20
    # ... and they are the first n draws, so the strings of a smaller size are the beginning of
    # those of every larger one, counted here from the sequence of the draws
    for n in (20, 40, 60):
        first = draws.sequence[:n]
        assert draws.training(n, "draws") == {s: first.count(s) for s in dict.fromkeys(first)}
        assert list(draws.training(n, "draws")) == list(dict.fromkeys(first))
    small, large = draws.training(20, "draws"), draws.training(60, "draws")
    assert all(large[string] >= times for string, times in small.items())
    # must reject: the last n draws, which are other strings
    last = draws.sequence[-20:]
    assert small != {s: last.count(s) for s in dict.fromkeys(last)}
    assert draws.sequence[:20] != last
    # the two samples start from the same draws of SymSeq
    assert distinct.sequence[:60] == draws.sequence
    # must reject: asking for more than was drawn, and an unknown sample
    with pytest.raises(ValueError, match="were drawn"):
        draws.training(61, "draws")
    with pytest.raises(ValueError, match="must be one of"):
        draws.training(10, "weighted")
    with pytest.raises(RuntimeError, match="after 50 draws"):
        cx.draw_strings(UNIFILAR, 101, 60, 30, 3, 14, 50, "distinct")


def test_a_planted_violation_that_is_grammatical_or_a_training_string_is_refused(monkeypatch):
    """The guard of ``draw_strings``: what SymSeq returns as a violation is checked, not trusted."""
    training_string = drawn("draws").sequence[0]
    accepts = ArtificialGrammar.is_grammatical
    # a violation that is none: SymSeq hands the held-out string back unchanged
    monkeypatch.setattr(
        ArtificialGrammar, "add_deviant", lambda self, string, *a, **k: list(string)
    )
    with pytest.raises(ValueError, match="planted violation is grammatical or is a training"):
        drawn("draws")
    # a violation that is a training string, and a grammar that calls it ungrammatical all the
    # same: only the comparison with the training strings can refuse it
    planting = []

    def training_string_back(self, string, *a, **k):
        planting.append(True)
        return list(training_string)

    monkeypatch.setattr(ArtificialGrammar, "add_deviant", training_string_back)
    monkeypatch.setattr(
        ArtificialGrammar,
        "is_grammatical",
        lambda self, string, *a, **k: False if planting else accepts(self, string, *a, **k),
    )
    with pytest.raises(ValueError, match="planted violation is grammatical or is a training"):
        drawn("draws")
    assert planting  # the patched function was reached
    # must reject: with SymSeq's own functions the same call returns its strings
    monkeypatch.undo()
    assert len(drawn("draws").ungrammatical) == 30


def test_strings_are_the_same_for_the_same_seed():
    assert (
        drawn("distinct") == drawn("distinct")
        or drawn("distinct").sequence == drawn("distinct").sequence
    )
    first, again, other = drawn("draws", 101), drawn("draws", 101), drawn("draws", 102)
    assert (first.sequence, first.heldout, first.ungrammatical, first.draws) == (
        again.sequence,
        again.heldout,
        again.ungrammatical,
        again.draws,
    )
    # must reject: another training seed gives other strings
    assert other.sequence != first.sequence and other.heldout != first.heldout


def test_encoding_by_hand():
    classes = ("#", "A", "B", "C")
    inputs, targets = cx.encode([("A", "B"), ("C",)], classes, 4)
    assert inputs.shape == (2, 4, 4)
    assert inputs[0].argmax(axis=-1).tolist()[:3] == [0, 1, 2] and inputs[0, 3].sum() == 0
    assert targets.tolist() == [[1, 2, 0, cx.PAD], [3, 0, cx.PAD, cx.PAD]]
    assert inputs.sum() == 3 + 2  # one row per symbol and one for the marker in front
    # must reject: a string that does not fit
    with pytest.raises(ValueError, match="does not fit"):
        cx.encode([("A", "B", "C", "A")], classes, 4)


# --- the measures ---------------------------------------------------------------------------------


def test_divergence_and_top_choice_by_hand():
    p = np.array([[[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]]])
    targets = np.array([[1, 2]])
    q = np.array([[[0.25, 0.75, 0.0], [0.5, 0.25, 0.25]]])
    # position 1: 0.5 log2(0.5/0.25) + 0.5 log2(0.5/0.75); position 2: 0.25 log2(1) + 0.75 log2(3)
    by_hand = (0.5 * 1.0 + 0.5 * math.log2(2.0 / 3.0) + 0.75 * math.log2(3.0)) / 2.0
    assert abs(cx.divergence_bits(p, cx.log2_of(q), targets) - by_hand) < 1e-12
    # first choices of q: class 1 (true probability 0.5) and class 0 (0.0); the best are 0.5 and 0.75
    assert abs(cx.top_choice_ratio(p, cx.log2_of(q), targets) - 0.5 / 1.25) < 1e-12
    # the exact predictor: divergence zero, ratio one
    assert cx.divergence_bits(p, cx.log2_of(p), targets) == 0.0
    assert cx.top_choice_ratio(p, cx.log2_of(p), targets) == 1.0
    # positions without target do not count
    assert (
        abs(
            cx.divergence_bits(p, cx.log2_of(q), np.array([[1, cx.PAD]]))
            - (0.5 + 0.5 * math.log2(2 / 3))
        )
        < 1e-12
    )
    # must reject: a model that gives probability zero to a possible outcome is infinitely far
    assert (
        cx.divergence_bits(p, cx.log2_of(np.array([[[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]]])), targets)
        == math.inf
    )


def test_divergence_is_zero_for_the_exact_predictor_and_positive_for_a_perturbed_one():
    strings = drawn("distinct")
    predictor = cx.BoundedPredictor(cx.build_grammar(UNIFILAR), 3, 14)
    classes = cx.classes_of(UNIFILAR)
    _, targets = cx.encode(strings.heldout, classes, 15)
    p, _ = cx.exact_distributions(predictor, strings.heldout, classes, 15)
    assert np.abs(p.sum(axis=-1)[targets != cx.PAD] - 1.0).max() < 1e-12
    assert cx.divergence_bits(p, cx.log2_of(p), targets) == 0.0
    rng = np.random.default_rng(101)
    for scale in (0.01, 0.1, 1.0):
        perturbed = p + scale * rng.random(p.shape)
        perturbed /= perturbed.sum(axis=-1, keepdims=True)
        found = cx.divergence_bits(p, cx.log2_of(perturbed), targets)
        assert found > 0.0
        assert cx.top_choice_ratio(p, cx.log2_of(perturbed), targets) <= 1.0
    assert found > 0.05  # the largest perturbation is far
    # must reject: the predictor of the same grammar under other bounds is not at zero
    other, _ = cx.exact_distributions(
        cx.BoundedPredictor(cx.build_grammar(UNIFILAR), 3, 30), strings.heldout, classes, 15
    )
    assert cx.divergence_bits(p, cx.log2_of(other), targets) > 1e-4


def test_uniform_guess_and_prediction_score_by_hand():
    """Eq. (12) and Eq. (13): D0 by its two routes, and the score at its two anchors and beyond one."""
    p = np.array([[[0.5, 0.5, 0.0], [0.0, 0.25, 0.75]]])
    targets = np.array([[1, 2]])
    q = np.array([[[0.25, 0.75, 0.0], [0.5, 0.25, 0.25]]])
    entropies = [1.0, 0.25 * 2.0 + 0.75 * math.log2(4.0 / 3.0)]
    d0 = math.log2(3.0) - sum(entropies) / 2.0  # log2 K minus the mean entropy
    assert abs(cx.mean_entropy_bits(p, targets) - sum(entropies) / 2.0) < 1e-12
    assert abs(cx.uniform_divergence_bits(p, targets) - d0) < 1e-12
    uniform = np.full(p.shape, 1.0 / 3.0)
    assert abs(cx.divergence_bits(p, cx.log2_of(uniform), targets) - d0) < 1e-12
    # the score of q: one minus its divergence (computed by hand in the test above) over D0 ...
    d = (0.5 * 1.0 + 0.5 * math.log2(2.0 / 3.0) + 0.75 * math.log2(3.0)) / 2.0
    score = cx.prediction_score(cx.divergence_bits(p, cx.log2_of(q), targets), d0)
    assert abs(score - (1.0 - d / d0)) < 1e-12
    assert abs(score - (-0.0276699919853973)) < 1e-12  # q is a little worse than the guess
    halfway = cx.divergence_bits(p, cx.log2_of(0.5 * p + 0.5 / 3.0), targets)
    assert 0.0 < cx.prediction_score(halfway, d0) < 1.0  # between the guess and the exact predictor
    # ... which is the share of the gap in expected cross-entropy between the guess and the exact predictor
    x_guess = math.log2(3.0)
    x_model = ((0.5 * 2.0 + 0.5 * math.log2(4.0 / 3.0)) + (0.25 * 2.0 + 0.75 * 2.0)) / 2.0
    x_exact = sum(entropies) / 2.0
    assert abs(score - (x_guess - x_model) / (x_guess - x_exact)) < 1e-12
    # the two anchors, through the function that every run uses
    for model, anchor in ((p, 1.0), (uniform, 0.0)):
        found = cx.measures(cx.log2_of(model), cx.log2_of(model), p, targets, targets)
        assert abs(found["score"] - anchor) < 1e-12
    # a model that is worse than the guess has a negative score
    worse = np.array([[[0.05, 0.05, 0.9], [0.9, 0.05, 0.05]]])
    assert cx.prediction_score(cx.divergence_bits(p, cx.log2_of(worse), targets), d0) < -0.5
    # positions without target do not count
    one_target = np.array([[1, cx.PAD]])
    assert abs(cx.uniform_divergence_bits(p, one_target) - (math.log2(3.0) - 1.0)) < 1e-12
    assert abs(cx.mean_entropy_bits(p, one_target) - 1.0) < 1e-12
    # must reject: a guess that is no further from the exact predictor than the predictor itself
    for refused in (0.0, -0.1, math.nan):
        with pytest.raises(ValueError, match="positive divergence"):
            cx.prediction_score(0.1, refused)
    with pytest.raises(ValueError, match="positive divergence"):
        cx.measures(cx.log2_of(uniform), cx.log2_of(uniform), uniform, targets, targets)


def test_score_is_one_only_for_the_exact_predictor_and_uses_the_targets_of_the_run():
    """On strings of a grammar: no model is above 1, and D0 is not the generator's entropy."""
    predictor = cx.BoundedPredictor(cx.build_grammar(UNIFILAR), 3, 14)
    classes = cx.classes_of(UNIFILAR)
    strings = cx.draw_strings(UNIFILAR, 101, 4000, 60, 3, 14, 100_000, "draws")
    _, targets = cx.encode(strings.heldout, classes, 15)
    p, _ = cx.exact_distributions(predictor, strings.heldout, classes, 15)
    d0 = cx.uniform_divergence_bits(p, targets)
    assert abs(d0 - (math.log2(len(classes)) - cx.mean_entropy_bits(p, targets))) < 1e-12
    assert cx.prediction_score(cx.divergence_bits(p, cx.log2_of(p), targets), d0) == 1.0
    rng = np.random.default_rng(101)
    scores = []
    for scale in (0.01, 0.1, 1.0, 100.0):
        perturbed = p + scale * rng.random(p.shape)
        perturbed /= perturbed.sum(axis=-1, keepdims=True)
        scores.append(
            cx.prediction_score(cx.divergence_bits(p, cx.log2_of(perturbed), targets), d0)
        )
    assert all(score < 1.0 for score in scores) and scores == sorted(scores, reverse=True)
    # must reject: D0 from the entropy of the generator. The held-out strings are the ones that
    # 4000 draws did not produce, so they are not a sample of the generator, and the two differ.
    generator = math.log2(len(classes)) - cx.describe(UNIFILAR, 3, 14)["entropy"]
    assert abs(generator - d0) > 0.01
    d = cx.divergence_bits(p, cx.log2_of(perturbed), targets)
    assert abs(cx.prediction_score(d, generator) - cx.prediction_score(d, d0)) > 1e-3


def test_area_under_the_curve_by_hand_and_at_its_two_ends():
    ungrammatical, grammatical = np.array([3.0, 1.0, 2.0, 2.0]), np.array([2.0, 0.5, 4.0])
    pairs = [(u > g) + 0.5 * (u == g) for u in ungrammatical for g in grammatical]
    assert abs(cx.auc(ungrammatical, grammatical) - sum(pairs) / len(pairs)) < 1e-12
    assert cx.auc(np.array([math.inf, 9.0]), np.array([1.0, 8.9, 3.0])) == 1.0
    # a score that ignores the string is at chance
    assert cx.auc(np.full(7, 2.5), np.full(5, 2.5)) == 0.5
    # must reject: with the two sets exchanged the area is the complement
    assert abs(cx.auc(grammatical, ungrammatical) - (1.0 - sum(pairs) / len(pairs))) < 1e-12
    with pytest.raises(ValueError, match="at least one"):
        cx.auc(np.array([]), grammatical)


def test_the_exact_predictor_separates_every_ungrammatical_string():
    strings = drawn("distinct")
    predictor = cx.BoundedPredictor(cx.build_grammar(UNIFILAR), 3, 14)
    classes = cx.classes_of(UNIFILAR)
    _, targets = cx.encode(strings.heldout, classes, 15)
    _, targets_bad = cx.encode(strings.ungrammatical, classes, 15)
    p, _ = cx.exact_distributions(predictor, strings.heldout, classes, 15)
    p_bad, _ = cx.exact_distributions(predictor, strings.ungrammatical, classes, 15)
    scores = cx.largest_surprisal(cx.log2_of(p), targets)
    scores_bad = cx.largest_surprisal(cx.log2_of(p_bad), targets_bad)
    assert np.isfinite(scores).all() and np.isinf(scores_bad).all()
    found = cx.measures(cx.log2_of(p), cx.log2_of(p_bad), p, targets, targets_bad)
    assert found == {"divergence_bits": 0.0, "score": 1.0, "top_choice_ratio": 1.0, "auc": 1.0}
    # a model that ignores the string (the same distribution everywhere) is at chance when lengths are equal
    flat = np.full(p.shape, 1.0 / len(classes))
    assert (
        cx.auc(
            cx.largest_surprisal(cx.log2_of(flat), targets_bad),
            cx.largest_surprisal(cx.log2_of(flat), targets),
        )
        == 0.5
    )
    # must reject: the largest surprisal by hand for one string, which a mean would not give
    one = cx.log2_of(np.array([[[0.5, 0.5], [0.25, 0.75], [1.0, 0.0]]]))
    assert cx.largest_surprisal(one, np.array([[0, 0, cx.PAD]])).tolist() == [2.0]


def test_first_order_model_by_hand():
    classes = ("#", "A", "B")
    table = 2.0 ** cx.first_order_log2({("A", "B"): 1, ("A",): 2}, classes)
    # after the marker: A three times; after A: B once, the marker twice; after B: the marker once
    assert np.allclose(table[0], [1 / 6, 4 / 6, 1 / 6])
    assert np.allclose(table[1], [3 / 6, 1 / 6, 2 / 6])
    assert np.allclose(table[2], [2 / 4, 1 / 4, 1 / 4])
    assert np.allclose(table.sum(axis=1), 1.0)
    # must reject: the same strings once each give other probabilities
    once = 2.0 ** cx.first_order_log2({("A", "B"): 1, ("A",): 1}, classes)
    assert not np.allclose(once[1], table[1])


def test_references_that_know_the_graph():
    """Transitions counted along the walks of the training strings, and the walk without length bounds."""
    predictor = cx.BoundedPredictor(cx.build_grammar(UNIFILAR), 3, 14)
    names = list(predictor.states)
    assert [names[i] for i in cx.walk_states(predictor, ("S", "B", "A", "A", "C"))] == [
        "S",
        "B",
        "A(1)",
        "A(1)",
        "C",
    ]
    counted = cx.counted_transitions(predictor, {("S", "B", "A", "C"): 3, ("S", "A", "C"): 1})
    row = counted[names.index("S")]  # S -> A(0) once, S -> B three times, one added to each
    assert (
        abs(row[names.index("B")] - 4 / 6) < 1e-12 and abs(row[names.index("A(0)")] - 2 / 6) < 1e-12
    )
    assert np.allclose(counted.sum(axis=1), 1.0)
    assert (
        (counted > 0) == (cx.counted_transitions(predictor) > 0)
    ).all()  # only what the graph allows
    # with many training strings the counted transitions approach the walk's own
    strings = cx.draw_strings(UNIFILAR, 101, 4000, 30, 3, 14, 100_000, "draws")
    learned = cx.counted_transitions(predictor, strings.training(4000, "draws"))
    classes = cx.classes_of(UNIFILAR)
    _, targets = cx.encode(strings.heldout, classes, 15)
    p, _ = cx.exact_distributions(predictor, strings.heldout, classes, 15)

    def divergence(model) -> float:
        return cx.divergence_bits(
            p, cx.state_model_log2(predictor, model, strings.heldout, classes, 15), targets
        )

    walk = divergence(cx.counted_transitions(predictor))
    assert walk > 0.0  # the bounds exclude strings, so the walk is not the exact predictor
    assert divergence(learned) < 2.0 * walk + 0.02
    # must reject: few training strings are further away, and a string without a walk is refused
    few = cx.counted_transitions(predictor, strings.training(5, "draws"))
    assert divergence(few) > divergence(learned)
    with pytest.raises(ValueError, match="exactly one walk"):
        cx.walk_states(predictor, ("S", "C"))
    ambiguous = cx.BoundedPredictor(cx.build_grammar(AMBIGUOUS), 2, 7)
    with pytest.raises(ValueError, match="exactly one walk"):
        cx.walk_states(ambiguous, ("B", "A", "C"))


def test_settings_of_the_experiment_are_declared_once_and_seeds_are_disjoint():
    design, production, pilot = config.COMPLEXITY_DESIGN, config.COMPLEXITY, config.COMPLEXITY_PILOT
    assert design.sizes == (50, 200, 800) and design.n_heldout == 200
    assert design.validation_fraction == 0.2 and design.min_memory_gap_bits == 0.05
    assert design.seeds == (1, 2, 3) and design.first_grammar_seed == 1
    assert design.grammars_per_level == 3 and design.levels == config.COMPLEXITY_LEVELS
    assert len(config.COMPLEXITY_LEVELS) == 5
    assert design.min_strings >= max(design.sizes) + design.n_heldout
    # every kind of run trains on draws; the other sample is for --out only
    assert cx.TRAINING_SAMPLES == ("distinct", "draws")
    for settings in config.COMPLEXITY_SETTINGS.values():
        assert settings.training_sample == "draws"
    assert design.training_sample == "draws"
    bands = [level.band_bits for level in design.levels]
    assert all(low < high for low, high in bands)
    assert all(first[1] <= second[0] for first, second in itertools.pairwise(bands))  # no overlap

    def changed(settings) -> set:
        return {key for key, value in settings.as_dict().items() if value != design.as_dict()[key]}

    # the production runs differ from the design by what the time rule cut, in its order, and by
    # nothing else: two training seeds, and four levels without the one named before the pilot
    assert changed(production) == {"seeds", "levels"}
    assert production.seeds == (1, 2) and len(production.levels) == 4
    dropped = config.COMPLEXITY_LEVELS[config.COMPLEXITY_LEVEL_TO_DROP]
    assert dropped not in production.levels
    assert set(production.levels) | {dropped} == set(design.levels)
    assert len(production.levels) > config.COMPLEXITY_LIVE_LEVEL
    assert config.COMPLEXITY_LIVE_SIZE in production.sizes
    # the pilot differs from the design in its seeds and its number of grammars, and in nothing else
    assert changed(pilot) == {"grammars_per_level", "first_grammar_seed", "seeds"}
    assert min(pilot.seeds) >= 101 and pilot.first_grammar_seed >= 101
    assert config.COMPLEXITY_LIVE_SEED >= 101 and config.COMPLEXITY_LIVE_SEED not in pilot.seeds
    # must reject: production seeds are below every exploration seed, so no grammar or run is shared
    last_production_seed = production.first_grammar_seed + production.max_candidates - 1
    assert last_production_seed < 101 and max(production.seeds) < 101
    assert changed(config.COMPLEXITY_SMOKE) > changed(
        pilot
    )  # the smoke settings are not the pilot's
    # the optimiser, the batch size and the size of the networks are those of the six-grammar runs
    assert (config.PRODUCTION.learning_rate, config.PRODUCTION.batch_size) == (1e-3, 32)
    assert config.PRODUCTION.parameter_budget == 20_000
