"""Regular grammars of controlled complexity: synthesis, admission, exact predictor, data and measures.

The networks notebook compares its two networks on grammars whose complexity is raised step by
step. This module makes the grammars with SymSeq's own synthesis, decides which of them are
admitted, states the exact next-symbol distribution of an admitted grammar, draws the strings of a
run with SymSeq's sampler, and defines the two measures. Nothing here trains a network
(``tutorial.complexity_runs`` does).

## Theory

A grammar is a graph of states. Each state emits one symbol, several states may emit the same
symbol, and a string is a walk from the start state until the end marker ``#`` is reached from a
terminal state. All transitions out of a state are equally probable.

Complexity. The topological entropy of the grammar is the growth rate of the number of walks on the
closed graph, in which the marker is a state that leads back to the start state:

    Eq. (1)  h_top = log2 lambda_max(A)        bits per symbol,

with A the table of allowed transitions, marker included (SymSeq:
``ArtificialGrammar.topological_entropy(correct_terminal_sink=True)``, in nats, divided by ln 2).
The grammars here are unifilar (no state has two successors with the same symbol, one start state),
so a string is one walk and Eq. (1) is also the growth rate of the number of strings.

Bounded length. SymSeq draws a walk and rejects it if its length is outside [L_min, L_max]
(``generate_string(length_range=...)``). The strings that remain follow the walk's distribution
conditioned on acceptance. With T the transition probabilities between states, e_s the probability
of ending from state s, and pi the start distribution, the probability that a walk whose t-th
symbol is emitted by state s is accepted is

    Eq. (2)  beta_t(s) = e_s [L_min <= t <= L_max] + sum_s' T_ss' beta_{t+1}(s'),   beta_{L_max+1} = 0,

and the probability that a walk is accepted is Z = sum_s pi_s beta_1(s).

Exact predictor. The state is hidden where a symbol has several states. It is inferred from the
symbols seen so far by the forward filter

    Eq. (3)  alpha_1(s) = pi_s [sym(s) = x_1],
             alpha_{t+1}(s') = sum_s alpha_t(s) T_ss' [sym(s') = x_{t+1}],

and the distribution of the next outcome given the prefix x_1 ... x_t is

    Eq. (4)  P(x | x_1..x_t) = sum_{s,s'} alpha_t(s) T_ss' [sym(s') = x] beta_{t+1}(s') / (alpha_t . beta_t),
             P(# | x_1..x_t) = sum_s alpha_t(s) e_s [L_min <= t <= L_max] / (alpha_t . beta_t).

Memory. A predictor that sees only the previous symbol can do no better than the distribution of
the next outcome given that symbol, q_1(x | a). Its expected excess cross-entropy over the exact
predictor is the information the earlier symbols carry about the next one,

    Eq. (5)  gap_1 = H(next | previous symbol) - H(next | prefix)        bits per target.

Because of the length bounds the position alone carries some of that information, even when every
symbol has one state. The admission rule therefore uses the stricter

    Eq. (6)  gap = H(next | previous symbol, position) - H(next | prefix)  <=  gap_1,

which is exactly zero for a grammar in which every symbol has one state.

Measures of a model q against the exact predictor p, on held-out grammatical strings (the targets
of a string of L symbols are its L symbols and the marker):

    Eq. (7)  divergence = mean over targets of sum_x p(x | prefix) log2 [p(x | prefix) / q(x | prefix)]
    Eq. (8)  top choice = sum over targets of p(argmax_x q) / sum over targets of max_x p
    Eq. (9)  score(string) = max over its targets of -log2 q(target | prefix)
    Eq. (10) AUC = P(score of an ungrammatical string > score of a grammatical one) + ties / 2

Eq. (7) is a Kullback-Leibler divergence: never negative, zero only where q equals p. Eq. (8)
cannot exceed 1. Under the exact predictor an ungrammatical string contains an outcome of
probability zero, so its score is infinite and the AUC is 1; a score that ignores the string gives 0.5.

Prediction score. The divergence of Eq. (7) has no upper anchor, and what a model that knows
nothing would reach differs from grammar to grammar. That model is the uniform guess u over the K
outcomes of the grammar (its symbols and the marker). On the same targets its divergence is

    Eq. (12) D0 = mean over targets of sum_x p(x | prefix) log2 [p(x | prefix) K]
                = log2 K - mean over targets of H(p(. | prefix)),

which is smaller for a grammar that leaves more open. The prediction score of a model q is the
share of the way from the uniform guess to the exact predictor that q has gone,

    Eq. (13) S(q) = 1 - D(q) / D0 = (X0 - Xq) / (X0 - Xp),

with X0, Xq and Xp the expected cross-entropies of the guess, the model and the exact predictor on
those targets. S is 1 only for the exact predictor, 0 for the uniform guess, never above 1, and
negative for a model that is worse than the guess. D0 is taken on the held-out targets of the run,
not from the entropy of the generator: the held-out strings are not a sample of the generator.

Reference model. A first-order model with add-one smoothing, fitted on the training strings:

    Eq. (11) q(x | a) = (count(a, x) + 1) / (count(a, .) + number of classes).

Ref: Bollt and Jones (2000), "The complexity of artificial grammars", Nonlinear Dynamics, Psychology,
and Life Sciences 4, 153-168 (Eq. 1). Rabiner (1989), "A tutorial on hidden Markov models",
Proc. IEEE 77, 257-286 (Eq. 2 and 3 are its backward and forward variables). Hanley and McNeil
(1982), "The meaning and use of the area under a ROC curve", Radiology 143, 29-36 (Eq. 10).
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import rankdata
from symseq.generators.ag.artificial_grammar import ArtificialGrammar
from symseq.generators.ag.synthesis import generate_grammar_with_target_te_complexity

from tutorial import exact
from tutorial.generate import EOS

LN2 = math.log(2.0)
PAD = -1  # target where there is none
EOS_CLASS = 0  # class of the end marker, as in SeqBench


# --- grammars -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GrammarSpec:
    """Everything that defines one grammar, in plain values that can be saved as JSON.

    ``states`` are named as SymSeq names them: ``"C"`` for the only state of a symbol, ``"A(0)"`` and
    ``"A(1)"`` for two states of one symbol. ``edges`` are the allowed transitions between states.
    """

    name: str
    level: int  # index of the level whose synthesis target was used
    seed: int  # grammar seed
    target_nats: float  # target given to SymSeq's synthesis: ln of the spectral radius of ``edges``
    states: tuple[str, ...]
    alphabet: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    start: str
    terminals: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, saved: dict) -> GrammarSpec:
        return cls(
            name=saved["name"],
            level=int(saved["level"]),
            seed=int(saved["seed"]),
            target_nats=float(saved["target_nats"]),
            states=tuple(saved["states"]),
            alphabet=tuple(saved["alphabet"]),
            edges=tuple((source, target) for source, target in saved["edges"]),
            start=saved["start"],
            terminals=tuple(saved["terminals"]),
        )


def synthesise(target_nats: float, n_states: int, rng: np.random.Generator) -> np.ndarray:
    """Allowed transitions between ``n_states`` states at a target complexity, from SymSeq's synthesis.

    Calls ``symseq.generators.ag.synthesis.generate_grammar_with_target_te_complexity`` and returns
    its table (source x target, 0 or 1). The target is the natural logarithm of the table's spectral
    radius. The function is called directly, with the generator given here, because
    ``ArtificialGrammar.from_constraints(target_complexity=...)`` does not pass its seed on to the
    synthesis: at the pinned commit it returns the same table for every seed.
    """
    with warnings.catch_warnings():
        # the synthesis exponentiates large score differences; the overflow gives probability 0 or 1
        warnings.simplefilter("ignore", RuntimeWarning)
        table = generate_grammar_with_target_te_complexity(
            target_complexity=target_nats, n_nodes=n_states, rng=rng
        )[0]
    return (np.asarray(table) > 0).astype(int)


def reachability(table: np.ndarray) -> np.ndarray:
    """``reach[i, j]``: state j can be reached from state i in zero or more transitions."""
    n = len(table)
    reach = (np.asarray(table) > 0) | np.eye(n, dtype=bool)
    for middle in range(n):
        reach |= reach[:, [middle]] & reach[[middle], :]
    return reach


def trim_choices(table: np.ndarray, n_terminals: int) -> list[tuple[int, tuple[int, ...]]]:
    """Every (start state, terminal states) for which each state lies on a walk from start to end.

    The start state is not a terminal state, so the shortest string has two symbols. SymSeq's
    synthesis route draws the start and terminal states at random and does not check that a string
    can be generated; here they are drawn among the choices this function lists.
    """
    reach = reachability(table)
    n = len(table)
    choices = []
    for start in range(n):
        if not reach[start].all():
            continue
        others = [state for state in range(n) if state != start]
        for terminals in itertools.combinations(others, n_terminals):
            if reach[:, list(terminals)].any(axis=1).all():
                choices.append((start, terminals))
    return choices


def unifilar_pairings(table: np.ndarray, n_pairs: int) -> list[tuple[tuple[int, int], ...]]:
    """Every way to choose ``n_pairs`` disjoint pairs of states that may share a symbol.

    Two states may share a symbol if no state has transitions to both. With such pairs, and one
    start state, a string determines its walk: the grammar is unifilar.
    """
    reaches = np.asarray(table) > 0
    n = len(table)
    allowed = [
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if not (reaches[:, u] & reaches[:, v]).any()
    ]
    return [
        pairs
        for pairs in itertools.combinations(allowed, n_pairs)
        if len({state for pair in pairs for state in pair}) == 2 * n_pairs
    ]


def name_states(
    n_states: int, pairs: Sequence[tuple[int, int]], letters: Sequence[str]
) -> tuple[str, ...]:
    """State names: the first letters go to the pairs (``"A(0)"``, ``"A(1)"``), the others to single states."""
    names: dict[int, str] = {}
    for letter, (first, second) in zip(letters, pairs, strict=False):
        names[first], names[second] = f"{letter}(0)", f"{letter}(1)"
    single = iter(letters[len(pairs) :])
    for state in range(n_states):
        if state not in names:
            names[state] = next(single)
    return tuple(names[state] for state in range(n_states))


def build_grammar(spec: GrammarSpec, seed: int = 0) -> ArtificialGrammar:
    """The SymSeq grammar of a specification, with its sampling of strings seeded.

    Transitions out of a state are equally probable, the marker included for a terminal state, as
    in SymSeq's synthesis route (``symseq.generators.ag.generator.grammar_with_complexity``). As
    there, the transition from the marker to the start state is declared, so the graph is closed.
    """
    transitions: list[tuple[str, str, float]] = []
    for source in spec.states:
        targets = [target for origin, target in spec.edges if origin == source]
        if source in spec.terminals:
            targets.append(EOS)
        transitions.extend((source, target, 1.0 / len(targets)) for target in targets)
    transitions.append((EOS, spec.start, 1.0))
    grammar = ArtificialGrammar(
        label=spec.name,
        states=[*spec.states, EOS],
        alphabet=list(spec.alphabet),
        transitions=transitions,
        start_states=[spec.start],
        terminal_states=list(spec.terminals),
        eos=EOS,
        seed=seed,
    )
    grammar.rng = np.random.default_rng(seed)  # SymSeq keeps only 999 sampling streams otherwise
    return grammar


def topological_entropy_bits(grammar: ArtificialGrammar) -> float:
    """Eq. (1): SymSeq's topological entropy of the closed graph, converted from nats to bits."""
    return float(grammar.topological_entropy(correct_terminal_sink=True)) / LN2


# --- exact predictor ------------------------------------------------------------------------------


class BoundedPredictor(exact.Predictor):
    """Next-outcome distribution of a grammar whose strings are drawn with length bounds.  Eq. (2) to (4)

    The state of the predictor is ``(t, alpha)``: the number of symbols seen and the forward vector
    over the grammar's states, scaled to sum to one. It has the interface of
    ``tutorial.exact.Predictor``, so ``exact.enumerate_strings`` lists every string with its probability.
    """

    def __init__(self, grammar: ArtificialGrammar, min_length: int, max_length: int):
        if not 1 <= min_length <= max_length:
            raise ValueError(f"Need 1 <= min_length <= max_length, got {min_length}, {max_length}.")
        self.eos = grammar.eos
        inner = [state for state in grammar.states if state.tostring() != grammar.eos]
        self.states = tuple(state.tostring() for state in inner)
        self.symbol_of = tuple(state.symbol for state in inner)
        self.alphabet = tuple(grammar.alphabet)
        self.min_length, self.max_length = int(min_length), int(max_length)
        index = {name: position for position, name in enumerate(self.states)}
        n = len(inner)
        self.transition = np.zeros((n, n))  # T[s, s']
        self.exit = np.zeros(n)  # e_s
        for source, target, probability in grammar.transitions:
            if source == self.eos:
                continue  # the way back to the start closes the graph; a string ends at the marker
            if target == self.eos:
                self.exit[index[source]] = probability
            else:
                self.transition[index[source], index[target]] = probability
        self.start = np.zeros(n)  # pi
        for name, probability in grammar.start_probabilities.items():
            self.start[index[name]] = probability
        self.emits = {
            symbol: np.array([emitted == symbol for emitted in self.symbol_of])
            for symbol in self.alphabet
        }
        self.beta = np.zeros((self.max_length + 2, n))  # Eq. (2); row 0 is not used
        for t in range(self.max_length, 0, -1):
            self.beta[t] = self.transition @ self.beta[t + 1]
            if t >= self.min_length:
                self.beta[t] += self.exit
        self.acceptance = float(self.start @ self.beta[1])  # Z
        if self.acceptance <= 0.0:
            raise ValueError("No string of this grammar has a length within the bounds.")

    def initial(self):
        return (0, None)

    def _forward(self, state) -> tuple[np.ndarray, float]:
        """Weight of every state as the next one, and of the marker, before the length bounds."""
        t, alpha = state
        if t == 0:
            return self.start, 0.0
        alpha = np.asarray(alpha)
        stop = float(alpha @ self.exit) if self.min_length <= t <= self.max_length else 0.0
        return alpha @ self.transition, stop

    def distribution(self, state) -> dict[str, float]:
        t = state[0]
        forward, stop = self._forward(state)
        weights = forward * self.beta[t + 1]  # Eq. (4), numerators
        total = stop + float(weights.sum())  # alpha_t . beta_t
        out = {}
        for symbol, emits in self.emits.items():
            mass = float(weights[emits].sum())
            if mass > 0.0:
                out[symbol] = mass / total
        if stop > 0.0:
            out[self.eos] = stop / total
        return out

    def walk_distribution(self, state) -> dict[str, float]:
        """The next outcome of the walk itself, as if no string were ever rejected for its length."""
        t, alpha = state
        forward = self.start if t == 0 else np.asarray(alpha) @ self.transition
        stop = 0.0 if t == 0 else float(np.asarray(alpha) @ self.exit)
        total = stop + float(forward.sum())
        out = {symbol: float(forward[emits].sum()) / total for symbol, emits in self.emits.items()}
        out[self.eos] = stop / total
        return {outcome: mass for outcome, mass in out.items() if mass > 0.0}

    def advance(self, state, symbol: str):
        t = state[0]
        forward, _ = self._forward(state)
        if symbol not in self.emits:
            raise ValueError(f"Symbol {symbol!r} is not in the alphabet.")
        reached = np.where(self.emits[symbol], forward, 0.0)  # Eq. (3)
        if float(reached @ self.beta[t + 1]) <= 0.0:
            raise ValueError(f"Symbol {symbol!r} cannot follow this prefix.")
        return (t + 1, tuple((reached / reached.sum()).tolist()))

    def key(self, state) -> Hashable:
        t, alpha = state
        return (t, None if alpha is None else tuple(round(weight, 12) for weight in alpha))


def is_unifilar(predictor: BoundedPredictor) -> bool:
    """True if a string determines its walk: one start state per symbol, one successor per symbol."""
    allowed = predictor.transition > 0
    for row in (predictor.start > 0, *allowed):
        symbols = [predictor.symbol_of[state] for state in np.flatnonzero(row)]
        if len(symbols) != len(set(symbols)):
            return False
    return True


def count_strings(predictor: BoundedPredictor) -> int:
    """The exact number of distinct strings with a length within the bounds.

    Prefixes are grouped by the set of states they can end in (the subset construction), so strings
    are counted, not walks; the two differ where a string has more than one walk.
    """
    n = len(predictor.states)
    allowed = predictor.transition > 0
    step = [
        {
            symbol: sum(1 << target for target in np.flatnonzero(allowed[source] & emits))
            for symbol, emits in predictor.emits.items()
        }
        for source in range(n)
    ]
    ends = sum(1 << state for state in np.flatnonzero(predictor.exit > 0))
    frontier: dict[int, int] = {}
    for emits in predictor.emits.values():
        subset = sum(1 << state for state in np.flatnonzero((predictor.start > 0) & emits))
        if subset:
            frontier[subset] = frontier.get(subset, 0) + 1
    total = 0
    for length in range(1, predictor.max_length + 1):
        if length >= predictor.min_length:
            total += sum(count for subset, count in frontier.items() if subset & ends)
        following: dict[int, int] = {}
        for subset, count in frontier.items():
            members = [state for state in range(n) if subset >> state & 1]
            for symbol in predictor.emits:
                reached = 0
                for state in members:
                    reached |= step[state][symbol]
                if reached:
                    following[reached] = following.get(reached, 0) + count
        frontier = following
    return total


def _conditional_entropy(joint: dict) -> float:
    """sum over contexts and outcomes of -mass log2 (mass / mass of the context), in bits."""
    total = 0.0
    for row in joint.values():
        context = sum(row.values())
        total -= sum(mass * math.log2(mass / context) for mass in row.values() if mass > 0.0)
    return total


def exact_statistics(predictor: exact.Predictor, tol: float = 1e-12) -> dict[str, float]:
    """Exact expectations over the strings of a predictor, by carrying every prefix class forward.

    Targets are every symbol of a string and its marker. Returned, per target unless said otherwise:
    ``entropy`` (the cross-entropy of the exact predictor, in bits), ``ceiling`` (its accuracy),
    ``first_order_gap`` (Eq. 5), ``memory_gap`` (Eq. 6) and ``bounds_gap`` (the divergence from the
    exact predictor to the walk without length bounds: what ignoring the bounds costs); per string:
    ``mean_length`` and ``string_entropy``. Prefix classes of probability below ``tol`` are dropped
    and their total is returned as ``dropped_mass``.
    """
    start = predictor.initial()
    frontier = {(predictor.eos, predictor.key(start)): (start, 1.0)}
    targets = entropy = hits = dropped = bounds = 0.0
    unbounded = getattr(predictor, "walk_distribution", None)
    by_symbol: dict = {}  # previous symbol -> outcome -> probability
    by_position: dict = {}  # (previous symbol, position) -> outcome -> probability
    position = 0
    while frontier:
        following: dict = {}
        for (previous, _), (state, mass) in frontier.items():
            if mass < tol:
                dropped += mass
                continue
            distribution = predictor.distribution(state)
            targets += mass
            entropy -= mass * sum(p * math.log2(p) for p in distribution.values() if p > 0.0)
            hits += mass * max(distribution.values())
            if unbounded is not None:
                walk = unbounded(state)
                bounds += mass * sum(
                    p * math.log2(p / walk[outcome])
                    for outcome, p in distribution.items()
                    if p > 0.0
                )
            for joint, context in ((by_symbol, previous), (by_position, (previous, position))):
                row = joint.setdefault(context, {})
                for outcome, probability in distribution.items():
                    row[outcome] = row.get(outcome, 0.0) + mass * probability
            for symbol, probability in distribution.items():
                if symbol == predictor.eos or probability <= 0.0:
                    continue
                child = predictor.advance(state, symbol)
                key = (symbol, predictor.key(child))
                following[key] = (child, following.get(key, (None, 0.0))[1] + mass * probability)
        frontier = following
        position += 1
    first_order = _conditional_entropy(by_symbol) / targets
    with_position = _conditional_entropy(by_position) / targets
    return {
        "mean_length": targets - 1.0,
        "string_entropy": entropy,
        "entropy": entropy / targets,
        "ceiling": hits / targets,
        "first_order_cross_entropy": first_order,
        "first_order_gap": first_order - entropy / targets,  # Eq. (5)
        "memory_gap": with_position - entropy / targets,  # Eq. (6)
        "bounds_gap": bounds / targets,
        "dropped_mass": dropped,
    }


# --- admission ------------------------------------------------------------------------------------


def describe(spec: GrammarSpec, min_length: int, max_length: int) -> dict:
    """What the admission rule reads from a grammar, all computed exactly from its definition."""
    grammar = build_grammar(spec)
    predictor = BoundedPredictor(grammar, min_length, max_length)
    statistics = exact_statistics(predictor)
    edges = set(spec.edges)
    table = np.array(
        [[float((source, target) in edges) for target in spec.states] for source in spec.states]
    )
    radius = float(np.max(np.abs(np.linalg.eigvals(table))))
    return {
        "topological_entropy_bits": topological_entropy_bits(grammar),
        "synthesised_nats": math.log(radius) if radius > 0.0 else float("-inf"),
        "n_edges": len(spec.edges),
        "unifilar": is_unifilar(predictor),
        "n_strings": count_strings(predictor),
        "acceptance": predictor.acceptance,
        **statistics,
    }


def admission(
    values: dict, band_bits: tuple[float, float], min_gap_bits: float, min_strings: int
) -> list[str]:
    """The reasons for which a grammar is not admitted; an empty list means it is admitted.

    Four rules, each read from ``describe``: the achieved topological entropy lies in the band of
    the level; memory matters (Eq. 6 at least ``min_gap_bits``); there are at least ``min_strings``
    distinct strings within the length bounds; and at least ``min_strings`` typical strings,
    2^H with H the entropy of a string in bits. The last rule is the one that binds: a grammar can
    hold a million strings and put nearly all probability on a few dozen, and then the strings of
    a run can neither be drawn in reasonable time nor be typical of the grammar.
    """
    reasons = []
    low, high = band_bits
    if not low <= values["topological_entropy_bits"] < high:
        reasons.append("topological entropy outside the band of the level")
    if values["memory_gap"] < min_gap_bits:
        reasons.append("memory gap below the minimum")
    if values["n_strings"] < min_strings:
        reasons.append("too few distinct strings")
    if values["string_entropy"] < math.log2(min_strings):
        reasons.append("too few typical strings")
    return reasons


def candidate(
    level: int,
    seed: int,
    target_nats: float,
    n_states: int,
    alphabet_size: int,
    n_terminals: int,
) -> tuple[GrammarSpec | None, str]:
    """The grammar of one level and grammar seed, or None with the reason why there is none.

    One generator, seeded with the grammar seed and the level, drives every draw in this order:
    SymSeq's synthesis of the transitions; the start and terminal states, among the choices that use
    every state; the pairs of states that share a symbol, among the choices that keep the grammar
    unifilar; the letters of the states.
    """
    rng = np.random.default_rng([seed, level])
    table = synthesise(target_nats, n_states, rng)
    choices = trim_choices(table, n_terminals)
    if not choices:
        return None, "no start and terminal states that use every state"
    start, terminals = choices[int(rng.integers(len(choices)))]
    pairings = unifilar_pairings(table, n_states - alphabet_size)
    if not pairings:
        return None, "no pairs of states that can share a symbol"
    pairs = pairings[int(rng.integers(len(pairings)))]
    alphabet = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:alphabet_size])
    letters = [alphabet[position] for position in rng.permutation(alphabet_size)]
    names = name_states(n_states, pairs, letters)
    spec = GrammarSpec(
        name=f"level {level + 1}, grammar {seed}",
        level=level,
        seed=seed,
        target_nats=float(target_nats),
        states=names,
        alphabet=alphabet,
        edges=tuple(
            sorted((names[source], names[target]) for source, target in np.argwhere(table > 0))
        ),
        start=names[start],
        terminals=tuple(sorted(names[state] for state in terminals)),
    )
    return spec, ""


# --- strings of a run -----------------------------------------------------------------------------


TRAINING_SAMPLES = ("distinct", "draws")


@dataclass(frozen=True, eq=False)
class Strings:
    """The strings of one grammar and training seed.

    ``sequence`` holds the strings drawn for training in the order of the draws, repeats included.
    ``heldout`` are distinct strings drawn afterwards that are not in ``sequence``.
    ``ungrammatical[i]`` is ``heldout[i]`` with one planted violation. Two held-out strings can
    give the same ungrammatical string, so ``ungrammatical`` may hold a string more than once; it
    is kept as it is, one string per held-out string.
    """

    sequence: tuple[tuple[str, ...], ...]
    heldout: tuple[tuple[str, ...], ...]
    ungrammatical: tuple[tuple[str, ...], ...]
    draws: int  # strings drawn from the grammar in all

    def training(self, n: int, sample: str) -> dict[tuple[str, ...], int]:
        """The training strings of size ``n`` and how often each is used, in the order of first appearance.

        ``"distinct"``: the first ``n`` distinct strings, each used once. ``"draws"``: the first
        ``n`` draws, so a string drawn several times is used several times.
        """
        if sample == "distinct":
            first = list(dict.fromkeys(self.sequence))
            if len(first) < n:
                raise ValueError(
                    f"Only {len(first)} distinct training strings were drawn, {n} asked."
                )
            return dict.fromkeys(first[:n], 1)
        if sample == "draws":
            if len(self.sequence) < n:
                raise ValueError(
                    f"Only {len(self.sequence)} training strings were drawn, {n} asked."
                )
            counts: dict[tuple[str, ...], int] = {}
            for string in self.sequence[:n]:
                counts[string] = counts.get(string, 0) + 1
            return counts
        raise ValueError(f"The training sample must be one of {TRAINING_SAMPLES}, got {sample!r}.")


def draw_strings(
    spec: GrammarSpec,
    seed: int,
    n_train: int,
    n_heldout: int,
    min_length: int,
    max_length: int,
    max_draws: int,
    sample: str = "draws",
) -> Strings:
    """Training strings, distinct held-out strings, and one ungrammatical string per held-out string.

    Strings are drawn with SymSeq's ``generate_string(length_range=...)``. With the sample
    ``"distinct"`` the training draws stop when ``n_train`` distinct strings exist, with ``"draws"``
    after ``n_train`` draws. Drawing then goes on until ``n_heldout`` distinct strings exist that
    are not training strings. Each ungrammatical string is made by SymSeq's
    ``add_deviant(..., n_deviants=1)`` from a held-out string. The function raises if this takes
    more than ``max_draws`` draws, if a string that should be grammatical is not, or if a string
    with a planted violation is grammatical or is a training string.
    """
    if sample not in TRAINING_SAMPLES:
        raise ValueError(f"The training sample must be one of {TRAINING_SAMPLES}, got {sample!r}.")
    grammar = build_grammar(spec)
    grammar.rng = np.random.default_rng([seed, spec.seed, spec.level])
    sequence: list[tuple[str, ...]] = []
    known: set[tuple[str, ...]] = set()
    heldout: dict[tuple[str, ...], None] = {}
    draws = 0

    def draw() -> tuple[str, ...]:
        nonlocal draws
        if draws >= max_draws:
            raise RuntimeError(
                f"{spec.name}: {len(known)} distinct training and {len(heldout)} held-out strings "
                f"after {draws} draws; {n_train} and {n_heldout} are needed."
            )
        draws += 1
        return tuple(grammar.generate_string(length_range=(min_length, max_length)))

    while (len(known) if sample == "distinct" else len(sequence)) < n_train:
        string = draw()
        sequence.append(string)
        known.add(string)
    while len(heldout) < n_heldout:
        string = draw()
        if string not in known:
            heldout[string] = None
    if not all(grammar.is_grammatical(list(string)) for string in (*known, *heldout)):
        raise ValueError(f"{spec.name}: SymSeq drew a string that it does not accept.")
    ungrammatical = []
    for string in heldout:
        deviant = tuple(grammar.add_deviant(list(string), n_deviants=1))
        if grammar.is_grammatical(list(deviant)) or deviant in known:
            raise ValueError(
                f"{spec.name}: a string with a planted violation is grammatical or is a training string."
            )
        ungrammatical.append(deviant)
    return Strings(tuple(sequence), tuple(heldout), tuple(ungrammatical), draws)


def split_training(n_strings: int, fraction: float, rng: np.random.Generator) -> tuple[list, list]:
    """Positions of the strings used for the gradient steps and of those kept for validation."""
    n_validation = round(fraction * n_strings)
    if not 0 < n_validation < n_strings:
        raise ValueError(f"A validation share of {fraction} leaves no strings on one side.")
    order = rng.permutation(n_strings)
    return sorted(order[n_validation:].tolist()), sorted(order[:n_validation].tolist())


def assert_disjoint(*sets: Sequence[tuple[str, ...]]) -> None:
    """Raise if a string occurs twice, inside one set or in two of them."""
    seen: set[tuple[str, ...]] = set()
    for strings in sets:
        for string in strings:
            if tuple(string) in seen:
                raise ValueError(f"The string {' '.join(string)} occurs more than once.")
            seen.add(tuple(string))


def assert_apart(strings: Sequence[tuple[str, ...]], *others: Sequence[tuple[str, ...]]) -> None:
    """Raise if a string of ``strings`` is in one of the other sets. Repeats inside ``strings`` are allowed.

    This is the rule for the ungrammatical strings: each held-out string gives one, two held-out
    strings may give the same one, and none of them may be a training or a held-out string.
    """
    elsewhere = {tuple(string) for other in others for string in other}
    for string in strings:
        if tuple(string) in elsewhere:
            raise ValueError(
                f"The string {' '.join(string)} is ungrammatical by construction and is also "
                "a training or a held-out string."
            )


def classes_of(spec: GrammarSpec) -> tuple[str, ...]:
    """The symbol of every class; class 0 is the end marker."""
    return (EOS, *spec.alphabet)


def encode(
    strings: Sequence[Sequence[str]], classes: Sequence[str], width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Strings as network input and targets.

    A string x_1 ... x_L becomes the input row ``# x_1 ... x_L`` (one-hot, shape
    (strings, width, classes)) and the target row ``x_1 ... x_L #`` (class numbers, shape
    (strings, width)); positions after the string are zero rows with the target ``PAD``.
    """
    class_of = {symbol: number for number, symbol in enumerate(classes)}
    inputs = np.zeros((len(strings), width, len(classes)), dtype=np.float32)
    targets = np.full((len(strings), width), PAD, dtype=np.int64)
    for row, string in enumerate(strings):
        if len(string) + 1 > width:
            raise ValueError(
                f"A string of {len(string)} symbols does not fit in {width} positions."
            )
        tokens = [EOS_CLASS, *(class_of[symbol] for symbol in string)]
        inputs[row, np.arange(len(tokens)), tokens] = 1.0
        targets[row, : len(tokens)] = [*tokens[1:], EOS_CLASS]
    return inputs, targets


def exact_distributions(
    predictor: exact.Predictor, strings: Sequence[Sequence[str]], classes: Sequence[str], width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (4) at every target of every string, and whether each string is possible.

    ``p[i, t]`` is the distribution of outcome t + 1 of string i given its first t symbols, over the
    classes. For a string with an outcome of probability zero the rows after that outcome stay
    zero, and ``possible[i]`` is False.
    """
    class_of = {symbol: number for number, symbol in enumerate(classes)}
    p = np.zeros((len(strings), width, len(classes)))
    possible = np.ones(len(strings), dtype=bool)
    known: dict[Hashable, np.ndarray] = {}
    moves: dict[Hashable, object] = {}
    for row, string in enumerate(strings):
        state = predictor.initial()
        for t, outcome in enumerate([*string, predictor.eos]):
            key = predictor.key(state)
            if key not in known:
                vector = np.zeros(len(classes))
                for symbol, probability in predictor.distribution(state).items():
                    vector[class_of[symbol]] = probability
                known[key] = vector
            p[row, t] = known[key]
            if p[row, t, class_of[outcome]] <= 0.0:
                possible[row] = False
                break
            if outcome != predictor.eos:
                if (key, outcome) not in moves:
                    moves[key, outcome] = predictor.advance(state, outcome)
                state = moves[key, outcome]
    return p, possible


def string_probability(predictor: exact.Predictor, string: Sequence[str]) -> float:
    """Probability that a drawn string is this one (zero for a string the grammar cannot produce)."""
    state, probability = predictor.initial(), 1.0
    for outcome in [*string, predictor.eos]:
        step = predictor.distribution(state).get(outcome, 0.0)
        if step <= 0.0:
            return 0.0
        probability *= step
        if outcome != predictor.eos:
            state = predictor.advance(state, outcome)
    return probability


# --- measures -------------------------------------------------------------------------------------


def log2_of(p: np.ndarray) -> np.ndarray:
    """log2 of probabilities, minus infinity where the probability is zero."""
    with np.errstate(divide="ignore"):
        return np.log2(p)


def divergence_bits(p: np.ndarray, log2_q: np.ndarray, targets: np.ndarray) -> float:
    """Eq. (7): mean Kullback-Leibler divergence from the true distribution to the model's, in bits per target."""
    scored = targets != PAD
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, p * (np.log2(p) - log2_q), 0.0)
    return float(terms.sum(axis=-1)[scored].mean())


def mean_entropy_bits(p: np.ndarray, targets: np.ndarray) -> float:
    """Mean entropy of the true distributions over the targets, in bits per target."""
    scored = targets != PAD
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, -p * np.log2(p), 0.0)
    return float(terms.sum(axis=-1)[scored].mean())


def uniform_divergence_bits(p: np.ndarray, targets: np.ndarray) -> float:
    """Eq. (12): D0, the divergence of Eq. (7) for the uniform guess over all outcomes."""
    uniform = np.full(p.shape, -math.log2(p.shape[-1]))
    return divergence_bits(p, uniform, targets)


def prediction_score(divergence: float, uniform_divergence: float) -> float:
    """Eq. (13): 1 for the exact predictor, 0 for the uniform guess, negative for a model worse than it."""
    if not uniform_divergence > 0.0:
        raise ValueError(
            "The uniform guess must lie at a positive divergence from the exact predictor; "
            f"got {uniform_divergence}."
        )
    return 1.0 - divergence / uniform_divergence


def top_choice_ratio(p: np.ndarray, log2_q: np.ndarray, targets: np.ndarray) -> float:
    """Eq. (8): probability that the model's first choice is right, divided by the best achievable value."""
    scored = targets != PAD
    choice = log2_q.argmax(axis=-1)
    right = np.take_along_axis(p, choice[..., None], axis=-1)[..., 0]
    return float(right[scored].sum() / p.max(axis=-1)[scored].sum())


def largest_surprisal(log2_q: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Eq. (9): for every string, the largest surprisal of a target under the model, in bits."""
    scored = targets != PAD
    surprisal = -np.take_along_axis(log2_q, np.where(scored, targets, 0)[..., None], axis=-1)[
        ..., 0
    ]
    return np.where(scored, surprisal, -np.inf).max(axis=-1)


def auc(ungrammatical: np.ndarray, grammatical: np.ndarray) -> float:
    """Eq. (10): area under the ROC curve of a score that should be larger for ungrammatical strings.

    Computed from the ranks of the pooled scores (Mann-Whitney), with ties counted as one half.
    """
    ungrammatical, grammatical = np.asarray(ungrammatical), np.asarray(grammatical)
    if len(ungrammatical) == 0 or len(grammatical) == 0:
        raise ValueError("Both sets of scores must hold at least one string.")
    ranks = rankdata(np.concatenate([ungrammatical, grammatical]))
    n_u, n_g = len(ungrammatical), len(grammatical)
    return float((ranks[:n_u].sum() - n_u * (n_u + 1) / 2.0) / (n_u * n_g))


def first_order_log2(training: dict[tuple[str, ...], int], classes: Sequence[str]) -> np.ndarray:
    """Eq. (11): log2 q(x | a) of a first-order model with add-one smoothing, shape (classes, classes).

    ``training`` gives how often each training string is used. Row a is the previous symbol; row 0,
    the marker, is the context of the first symbol of a string.
    """
    class_of = {symbol: number for number, symbol in enumerate(classes)}
    counts = np.zeros((len(classes), len(classes)))
    for string, times in training.items():
        tokens = [EOS_CLASS, *(class_of[symbol] for symbol in string), EOS_CLASS]
        for previous, following in itertools.pairwise(tokens):
            counts[previous, following] += times
    return np.log2((counts + 1.0) / (counts.sum(axis=1, keepdims=True) + len(classes)))


# --- two references that know the graph -----------------------------------------------------------


def walk_states(predictor: BoundedPredictor, string: Sequence[str]) -> list[int]:
    """The states a string passes through. Only for a unifilar grammar, where there is one such walk."""
    reachable = predictor.start > 0
    walk = []
    for symbol in string:
        states = np.flatnonzero(reachable & predictor.emits[symbol])
        if len(states) != 1:
            raise ValueError(f"The string {' '.join(string)} does not have exactly one walk.")
        walk.append(int(states[0]))
        reachable = predictor.transition[walk[-1]] > 0
    return walk


def counted_transitions(
    predictor: BoundedPredictor, training: dict[tuple[str, ...], int] | None = None
) -> np.ndarray:
    """Probability of every outcome after every state, shape (states, states + 1); the last column is the marker.

    Without ``training``: the probabilities of the walk itself, which ignore the length bounds. With
    ``training``: counted along the walks of the training strings, with one added to every outcome
    the graph allows. This is a learner that is given the graph and has only its probabilities to
    estimate; what separates it from the exact predictor is the training sample, not the grammar.
    """
    walk = np.hstack([predictor.transition, predictor.exit[:, None]])
    if training is None:
        return walk
    counts = (walk > 0).astype(float)
    last = len(predictor.states)
    for string, times in training.items():
        states = walk_states(predictor, string)
        for source, target in itertools.pairwise(states):
            counts[source, target] += times
        counts[states[-1], last] += times
    return counts / counts.sum(axis=1, keepdims=True)


def state_model_log2(
    predictor: BoundedPredictor,
    model: np.ndarray,
    strings: Sequence[Sequence[str]],
    classes: Sequence[str],
    width: int,
) -> np.ndarray:
    """log2-probabilities of a model that predicts from the grammar's state, at every target of every string."""
    class_of = {symbol: number for number, symbol in enumerate(classes)}
    emitted = np.array([class_of[symbol] for symbol in predictor.symbol_of])
    per_state = np.zeros((len(predictor.states), len(classes)))
    np.add.at(per_state.T, emitted, model[:, :-1].T)
    per_state[:, EOS_CLASS] = model[:, -1]
    first = np.zeros(len(classes))
    np.add.at(first, emitted, predictor.start)
    q = np.zeros((len(strings), width, len(classes)))
    for row, string in enumerate(strings):
        q[row, 0] = first
        states = walk_states(predictor, string)
        q[row, 1 : len(states) + 1] = per_state[states]
    return log2_of(q)


def measures(
    log2_q_heldout: np.ndarray,
    log2_q_ungrammatical: np.ndarray,
    p_heldout: np.ndarray,
    targets_heldout: np.ndarray,
    targets_ungrammatical: np.ndarray,
) -> dict[str, float]:
    """The measures of one model from its log-probabilities on the held-out and ungrammatical strings."""
    divergence = divergence_bits(p_heldout, log2_q_heldout, targets_heldout)
    return {
        "divergence_bits": divergence,
        "score": prediction_score(
            divergence, uniform_divergence_bits(p_heldout, targets_heldout)
        ),  # Eq. (13)
        "top_choice_ratio": top_choice_ratio(p_heldout, log2_q_heldout, targets_heldout),
        "auc": auc(
            largest_surprisal(log2_q_ungrammatical, targets_ungrammatical),
            largest_surprisal(log2_q_heldout, targets_heldout),
        ),
    }
