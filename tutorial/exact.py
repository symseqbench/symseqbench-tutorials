"""Exact statistics of a generator: entropy of its strings and the next-symbol accuracy ceiling.

## Theory

A string is drawn symbol by symbol. Given the symbols seen so far (the prefix), the next outcome,
a symbol or the end of the string, has a distribution fixed by the generator. A predictor that knows
the generator cannot do better than name the most probable outcome, so the best achievable
next-symbol accuracy is

    Eq. (1)  ceiling = sum_prefix P(prefix) max_x P(x | prefix) / sum_prefix P(prefix)

over the prefixes at which a prediction is scored. By the chain rule the entropy of a whole string is

    Eq. (2)  H(string) = sum_prefix P(prefix) H(next | prefix)

and, because the strings of a dataset are independent, the sequence of strings joined with the end
marker has entropy rate

    Eq. (3)  h = H(string) / (mean length + 1)   bits per symbol.

Each ``Predictor`` below states the conditional distribution of one SymSeq generator, read from that
generator's sampling code. It does not sample. ``evaluate_on_strings`` scores a predictor on strings
that SymSeq generated, which is how the predictors are checked against the real generators.

Two scoring conventions are returned, matching the two ways SeqBench builds next-symbol targets:
``within`` scores symbols 2..L of a string (task from SymSeq, the end marker is never a target), and
``with_eos`` also scores the end marker after symbol L (task from SeqBench). The first symbol of a
string is never a target in either.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from functools import cache

from symseq.generators import (
    ArtificialGrammar,
    CrossedNonAdjacentDependencies,
    DyckGenerator,
    NestedNonAdjacentDependencies,
    NonAdjacentDependencies,
)

from tutorial.generate import EOS


def _entropy(distribution: dict[str, float]) -> float:
    return -sum(p * math.log2(p) for p in distribution.values() if p > 0.0)


class Predictor:
    """Conditional distribution of the next outcome given the prefix of a string."""

    eos: str = EOS

    def initial(self):
        """State before the first symbol."""
        raise NotImplementedError

    def distribution(self, state) -> dict[str, float]:
        """Probability of every possible next outcome, the end marker included."""
        raise NotImplementedError

    def advance(self, state, symbol: str):
        """State after ``symbol`` has been observed."""
        raise NotImplementedError

    def key(self, state) -> Hashable:
        """States with the same key have the same future statistics and are merged."""
        return state


class GrammarPredictor(Predictor):
    """Any SymSeq ``ArtificialGrammar``, nAX included. The state is the belief over the grammar's states."""

    def __init__(self, grammar: ArtificialGrammar):
        self.eos = grammar.eos
        self._symbol = {state.tostring(): state.symbol for state in grammar.states}
        self._successors: dict[str, list[tuple[str, float]]] = {name: [] for name in self._symbol}
        for source, target, probability in grammar.transitions:
            if probability > 0.0:
                self._successors[source].append((target, float(probability)))
        self._start = dict(grammar.start_probabilities)

    def initial(self):
        return None

    def _next_states(self, belief) -> dict[str, float]:
        if belief is None:
            return dict(self._start)
        out: dict[str, float] = {}
        for state, weight in belief:
            for target, probability in self._successors[state]:
                out[target] = out.get(target, 0.0) + weight * probability
        return out

    def distribution(self, belief) -> dict[str, float]:
        out: dict[str, float] = {}
        for state, probability in self._next_states(belief).items():
            symbol = self._symbol[state]
            out[symbol] = out.get(symbol, 0.0) + probability
        return out

    def advance(self, belief, symbol: str):
        reached = {
            s: p
            for s, p in self._next_states(belief).items()
            if self._symbol[s] == symbol and p > 0.0
        }
        total = sum(reached.values())
        if total <= 0.0:
            raise ValueError(f"Symbol {symbol!r} cannot follow this prefix.")
        return tuple(sorted((state, probability / total) for state, probability in reached.items()))

    def key(self, belief) -> Hashable:
        if belief is None:
            return None
        return tuple((state, round(weight, 12)) for state, weight in belief)


class NADPredictor(Predictor):
    """SymSeq ``NonAdjacentDependencies``: ``A_i``, then the fillers, then ``B_i``."""

    def __init__(
        self,
        generator: NonAdjacentDependencies,
        filler_len: int = 1,
        randomize_fillers: bool = False,
    ):
        self.eos = generator.eos
        self.starts = list(generator.start_symbols)
        self.terminal_of = dict(generator.dependency_pairs)
        self.fillers = list(generator.fillers)
        self.filler_len = filler_len if self.fillers else 0
        self.randomize = randomize_fillers

    def initial(self):
        return ()

    def distribution(self, prefix) -> dict[str, float]:
        position = len(prefix)
        if position == 0:
            return {start: 1.0 / len(self.starts) for start in self.starts}
        if position <= self.filler_len:
            if self.randomize or position == 1:
                return {filler: 1.0 / len(self.fillers) for filler in self.fillers}
            return {prefix[1]: 1.0}  # one filler, repeated
        if position == self.filler_len + 1:
            return {self.terminal_of[prefix[0]]: 1.0}
        return {self.eos: 1.0}

    def advance(self, prefix, symbol: str):
        return (*prefix, symbol)


class OrderedNADPredictor(Predictor):
    """SymSeq nested and crossed dependencies: a uniform permutation of the starts, then the terminals."""

    def __init__(self, generator: NestedNonAdjacentDependencies | CrossedNonAdjacentDependencies):
        self.eos = generator.eos
        self.starts = list(generator.start_symbols)
        self.terminal_of = dict(generator.dependency_pairs)
        self.nested = isinstance(generator, NestedNonAdjacentDependencies)
        self.n = len(self.starts)

    def initial(self):
        return ()

    def distribution(self, prefix) -> dict[str, float]:
        position = len(prefix)
        if position < self.n:
            remaining = [start for start in self.starts if start not in prefix]
            return {start: 1.0 / len(remaining) for start in remaining}
        if position < 2 * self.n:
            opened = prefix[: self.n]
            order = opened[::-1] if self.nested else opened
            return {self.terminal_of[order[position - self.n]]: 1.0}
        return {self.eos: 1.0}

    def advance(self, prefix, symbol: str):
        return (*prefix, symbol)


class DyckPredictor(Predictor):
    """SymSeq ``DyckGenerator`` in uniform mode, without distractors.

    Every bracket structure with ``target_pairs`` pairs and depth at most ``max_depth`` is equally
    likely, and the type of each pair is uniform. The probability of opening is therefore the share
    of the admissible completions that start with an opening bracket.
    """

    def __init__(self, generator: DyckGenerator):
        if generator.mode != "uniform":
            raise NotImplementedError("Only the uniform mode of DyckGenerator is covered.")
        self.eos = EOS
        self.steps = 2 * int(generator.target_pairs)
        self.max_depth = (
            generator.max_depth if generator.max_depth is not None else int(generator.target_pairs)
        )
        self.close_of = dict(generator.parentheses)
        self.opens = list(self.close_of) if generator.uniform_colorize else list(self.close_of)[:1]

        @cache
        def completions(remaining: int, depth: int) -> int:
            """Number of bracket structures that finish a string from this depth in this many symbols."""
            if depth < 0 or depth > self.max_depth or depth > remaining:
                return 0
            if remaining == 0:
                return 1
            return completions(remaining - 1, depth + 1) + completions(remaining - 1, depth - 1)

        self._completions = completions

    def initial(self):
        return (0, ())

    def distribution(self, state) -> dict[str, float]:
        done, stack = state
        remaining = self.steps - done
        if remaining == 0:
            return {self.eos: 1.0}
        total = self._completions(remaining, len(stack))
        p_open = self._completions(remaining - 1, len(stack) + 1) / total
        out = {symbol: p_open / len(self.opens) for symbol in self.opens} if p_open > 0.0 else {}
        if p_open < 1.0:
            out[self.close_of[stack[-1]]] = 1.0 - p_open
        return out

    def advance(self, state, symbol: str):
        done, stack = state
        if symbol in self.close_of:
            return (done + 1, (*stack, symbol))
        if not stack or self.close_of[stack[-1]] != symbol:
            raise ValueError(f"Symbol {symbol!r} cannot follow this prefix.")
        return (done + 1, stack[:-1])


def predictor_for(generator, **sample) -> Predictor:
    """The predictor of a SymSeq generator. ``sample`` are the arguments of its ``generate_string_set``."""
    if isinstance(generator, ArtificialGrammar):
        if sample:
            raise NotImplementedError(
                f"Sampling constraints {sorted(sample)} change the distribution and are not covered."
            )
        return GrammarPredictor(generator)
    if isinstance(generator, NonAdjacentDependencies):
        return NADPredictor(generator, **sample)
    if isinstance(generator, NestedNonAdjacentDependencies | CrossedNonAdjacentDependencies):
        return OrderedNADPredictor(generator)
    if isinstance(generator, DyckGenerator):
        return DyckPredictor(generator)
    raise NotImplementedError(f"No predictor for {type(generator).__name__}.")


def trial_statistics(
    predictor: Predictor, tol: float = 1e-12, max_steps: int = 100_000
) -> dict[str, float]:
    """Eq. (1) to Eq. (3), by carrying the probability of every prefix state forward one symbol at a time.

    Prefix states whose probability falls below ``tol`` are dropped; their total is returned as
    ``dropped_mass``. ``previous_symbol_with_eos`` and ``previous_symbol_within`` are the accuracies
    of the best predictor that sees only the previous symbol: the gap to the ceiling is what memory
    beyond one symbol is worth on this grammar.
    """
    # frontier: (previous symbol, state key) -> (state, probability of being there)
    frontier: dict[Hashable, tuple[object, float]] = {
        (None, predictor.key(predictor.initial())): (predictor.initial(), 1.0)
    }
    string_entropy = mean_length = dropped = 0.0
    hits_eos = n_eos = hits_within = n_within = 0.0
    after: dict[str, dict[str, float]] = {}  # previous symbol -> next outcome -> joint probability
    for step in range(max_steps):
        if not frontier:
            break
        following: dict[Hashable, tuple[object, float]] = {}
        for (previous, _), (state, mass) in frontier.items():
            if mass < tol:
                dropped += mass
                continue
            distribution = predictor.distribution(state)
            p_end = distribution.get(predictor.eos, 0.0)
            string_entropy += mass * _entropy(distribution)  # Eq. (2)
            mean_length += mass * (1.0 - p_end)
            if step >= 1:  # the first symbol of a string is not a target
                hits_eos += mass * max(distribution.values())  # Eq. (1), end marker scored
                n_eos += mass
                symbols_only = [p for symbol, p in distribution.items() if symbol != predictor.eos]
                hits_within += mass * max(symbols_only, default=0.0)  # Eq. (1), symbols only
                n_within += mass * (1.0 - p_end)
                joint = after.setdefault(previous, {})
                for outcome, probability in distribution.items():
                    joint[outcome] = joint.get(outcome, 0.0) + mass * probability
            for symbol, probability in distribution.items():
                if symbol == predictor.eos or probability <= 0.0:
                    continue
                child = predictor.advance(state, symbol)
                key = (symbol, predictor.key(child))
                following[key] = (child, following.get(key, (None, 0.0))[1] + mass * probability)
        frontier = following
    else:
        raise RuntimeError("The strings did not end within max_steps symbols.")
    previous_eos = sum(max(joint.values()) for joint in after.values())
    previous_within = sum(
        max((p for outcome, p in joint.items() if outcome != predictor.eos), default=0.0)
        for joint in after.values()
    )
    return {
        "mean_length": mean_length,
        "string_entropy": string_entropy,
        "entropy_rate": string_entropy / (mean_length + 1.0),  # Eq. (3)
        "ceiling_with_eos": hits_eos / n_eos if n_eos else float("nan"),
        "ceiling_within": hits_within / n_within if n_within else float("nan"),
        "previous_symbol_with_eos": previous_eos / n_eos if n_eos else float("nan"),
        "previous_symbol_within": previous_within / n_within if n_within else float("nan"),
        "dropped_mass": dropped,
    }


def evaluate_on_strings(predictor: Predictor, strings: Sequence[Sequence[str]]) -> dict[str, float]:
    """Score the predictor on generated strings: accuracy of its most probable outcome and its log loss.

    If the predictor states the generator's distribution, the accuracies estimate the ceilings of
    ``trial_statistics`` and ``log_loss`` (bits per string) estimates ``string_entropy``; a predictor
    with a wrong distribution has a larger log loss. ``log_loss_sd`` is the standard deviation of
    the loss across strings. ``impossible`` counts the strings to which the predictor gives
    probability zero.
    """
    hits_eos = n_eos = hits_within = n_within = 0
    losses: list[float] = []
    impossible = 0
    for string in strings:
        state = predictor.initial()
        loss = 0.0
        possible = True
        for position, outcome in enumerate([*string, predictor.eos]):
            distribution = predictor.distribution(state)
            probability = distribution.get(outcome, 0.0)
            if probability <= 0.0:
                possible = False
                break
            loss -= math.log2(probability)
            if position >= 1:
                guess = max(sorted(distribution), key=distribution.__getitem__)
                n_eos += 1
                hits_eos += guess == outcome
                if outcome != predictor.eos:
                    symbols_only = {s: p for s, p in distribution.items() if s != predictor.eos}
                    n_within += 1
                    hits_within += (
                        max(sorted(symbols_only), key=symbols_only.__getitem__) == outcome
                    )
            if outcome != predictor.eos:
                state = predictor.advance(state, outcome)
        if possible:
            losses.append(loss)
        else:
            impossible += 1
    mean_loss = sum(losses) / len(losses) if losses else float("nan")
    spread = (
        math.sqrt(sum((loss - mean_loss) ** 2 for loss in losses) / (len(losses) - 1))
        if len(losses) > 1
        else 0.0
    )
    return {
        "accuracy_with_eos": hits_eos / n_eos if n_eos else float("nan"),
        "accuracy_within": hits_within / n_within if n_within else float("nan"),
        "n_with_eos": n_eos,
        "n_within": n_within,
        "log_loss": mean_loss,
        "log_loss_sd": spread,
        "impossible": impossible,
    }


def enumerate_strings(predictor: Predictor, tol: float = 1e-12) -> dict[tuple[str, ...], float]:
    """Every string with its probability, by following all prefixes. Only for small generators."""
    out: dict[tuple[str, ...], float] = {}
    stack: list[tuple[tuple[str, ...], object, float]] = [((), predictor.initial(), 1.0)]
    while stack:
        prefix, state, mass = stack.pop()
        for symbol, probability in predictor.distribution(state).items():
            reached = mass * probability
            if symbol == predictor.eos:
                out[prefix] = out.get(prefix, 0.0) + reached
            elif reached >= tol:
                stack.append(((*prefix, symbol), predictor.advance(state, symbol), reached))
    return out


def ceiling_from_strings(strings: dict[tuple[str, ...], float], eos: str = EOS) -> dict[str, float]:
    """Eq. (1) computed from an explicit table of strings, without any predictor.

    The strings are grouped by prefix; at every prefix the mass of each continuation is added up and
    the largest is kept. This is the brute-force route against which ``trial_statistics`` is tested.
    """
    continuations: dict[tuple[str, ...], dict[str, float]] = {}
    for string, probability in strings.items():
        for position in range(1, len(string) + 1):
            outcome = string[position] if position < len(string) else eos
            table = continuations.setdefault(string[:position], {})
            table[outcome] = table.get(outcome, 0.0) + probability
    after: dict[str, dict[str, float]] = {}  # previous symbol -> next outcome -> probability
    for prefix, table in continuations.items():
        joint = after.setdefault(prefix[-1], {})
        for outcome, mass in table.items():
            joint[outcome] = joint.get(outcome, 0.0) + mass
    hits_eos = n_eos = hits_within = n_within = 0.0
    for table in continuations.values():
        hits_eos += max(table.values())
        n_eos += sum(table.values())
        symbols_only = [mass for outcome, mass in table.items() if outcome != eos]
        hits_within += max(symbols_only, default=0.0)
        n_within += sum(symbols_only)
    previous_eos = sum(max(joint.values()) for joint in after.values())
    previous_within = sum(
        max((mass for outcome, mass in joint.items() if outcome != eos), default=0.0)
        for joint in after.values()
    )
    return {
        "ceiling_with_eos": hits_eos / n_eos if n_eos else float("nan"),
        "ceiling_within": hits_within / n_within if n_within else float("nan"),
        "previous_symbol_with_eos": previous_eos / n_eos if n_eos else float("nan"),
        "previous_symbol_within": previous_within / n_within if n_within else float("nan"),
    }


def follow(predictor: Predictor, sequence: Sequence[str]) -> list[dict[str, float]]:
    """The generator's distribution of the next outcome after every symbol of a joined sequence.

    ``sequence`` is made of strings joined with the end marker and starts at the first symbol of a
    string, as a SeqBench sample does; it may stop inside a string. Entry ``t`` of the result is the
    distribution of symbol ``t + 1`` given the symbols up to ``t``: after a symbol it is the
    distribution of ``Predictor.distribution`` for the prefix of the current string, and after the
    end marker it is the distribution of the first symbol of a string, because strings are
    independent. A symbol that cannot follow its prefix raises ``ValueError``.
    """
    known: dict[Hashable, dict[str, float]] = {}
    state = predictor.initial()
    out = []
    for symbol in sequence:
        if symbol == predictor.eos:
            state = predictor.initial()
        else:
            if predictor.distribution(state).get(symbol, 0.0) <= 0.0:
                raise ValueError(f"Symbol {symbol!r} cannot follow this prefix.")
            state = predictor.advance(state, symbol)
        key = predictor.key(state)
        if key not in known:
            known[key] = predictor.distribution(state)
        out.append(known[key])
    return out
