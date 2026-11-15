"""Closed forms of the worked example, and brute-force routes to the same numbers.

## Theory

The worked example is the grammar with two symbols A and B and the transitions A -> A, A -> B and
B -> A (B is never repeated). Joining its strings ``A^k B`` without the end marker gives a stationary
first-order Markov chain over {A, B} with transition matrix P (rows are the current symbol):

    Eq. (1)  pi P = pi                       stationary distribution
    Eq. (2)  h   = sum_i pi_i H(P_i.)        entropy rate, bits per symbol
    Eq. (3)  H_1 = H(pi)                     single-symbol entropy, bits
    Eq. (4)  E   = H_1 - h                   excess entropy (memory) of a first-order chain, bits
    Eq. (5)  TE  = log2 lambda_max(P > 0)    topological entropy, bits per symbol

Eq. (4) follows from H_L = H_1 + (L - 1) h for a first-order chain, so H_L - L h = H_1 - h for every
L >= 1. With p = P(A -> B): pi = (1, p) / (1 + p), h(p) = H(p) / (1 + p), and for p = 1/2:
pi = (2/3, 1/3), h = 2/3, H_1 = log2(3) - 2/3, E = log2(3) - 4/3, TE = log2((1 + sqrt(5)) / 2).

Two more closed forms are drawn for the slides. With T the table of allowed transitions (T = P > 0,
here [[1, 1], [1, 0]]) and phi = (1 + sqrt(5)) / 2:

    Eq. (6)  N_L = 1' T^(L-1) 1                number of allowed words of L symbols: 2, 3, 5, 8, ...
    Eq. (7)  N_L ~ phi^(L+2) / sqrt(5)         the line that log N_L follows, of slope TE per symbol
    Eq. (8)  H_L = E + L h                     block entropy of a first-order chain, L >= 1

Eq. (7) is Binet's formula for the Fibonacci number N_L = F_(L+2) without its vanishing term
(-1/phi)^(L+2) / sqrt(5), which is smaller than one half for every L. Eq. (8) is Eq. (4) rearranged.

Ref: Crutchfield and Feldman (2003), doi:10.1063/1.1530990 (entropy rate, excess entropy; the chain
above is their golden-mean process). Ref: SymSeq ``estimate_emc``, whose order-R estimates
``h_R = H(X_{R+1} | X_1..X_R)`` and ``E_R = H_R - R h_R`` equal Eq. (2) and Eq. (4) at R = 1.

All entropies here are in bits. SymSeq returns topological entropy in nats.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

PHI = (1.0 + math.sqrt(5.0)) / 2.0
LN2 = math.log(2.0)


def entropy_bits(probabilities) -> float:
    """Shannon entropy of a probability vector, in bits. Zero entries contribute nothing."""
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p)))


def binary_entropy(p: float) -> float:
    """H(p) = -p log2 p - (1 - p) log2 (1 - p), in bits."""
    return entropy_bits([p, 1.0 - p])


def stream_chain(grammar, delimiter: bool = False) -> tuple[list[str], np.ndarray]:
    """Markov chain of the sequence obtained by joining the strings of a SymSeq ``ArtificialGrammar``.

    Returns the symbols and the row-stochastic matrix P with ``P[i, j] = P(next = j | current = i)``.
    With ``delimiter`` the end marker is a symbol of the chain and is followed by a start state
    (SymSeq's loop-back closure, ``get_transition_table(correct_terminal_sink=True)``). Without it,
    the end of a string is followed directly by the first symbol of the next string.

    The grammar must have one state per symbol; otherwise the symbol sequence is not a first-order
    chain and Eq. (2) to Eq. (4) do not apply.
    """
    symbols = [state.symbol for state in grammar.states]
    if len(set(symbols)) != len(symbols):
        raise ValueError(
            "The grammar has several states for one symbol; the closed forms need one state per symbol."
        )
    closed = np.asarray(
        grammar.get_transition_table(full=True, correct_terminal_sink=True), dtype=float
    )
    P = closed.T  # SymSeq stores target x source
    eos = symbols.index(grammar.eos)
    if not delimiter:
        keep = [i for i in range(len(symbols)) if i != eos]
        P = P[np.ix_(keep, keep)] + np.outer(P[keep, eos], P[eos, keep])
        symbols = [symbols[i] for i in keep]
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("The rows of the chain do not sum to one.")
    return symbols, P


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Eq. (1): the probability vector pi with pi P = pi. Raises if it is not unique."""
    n = P.shape[0]
    A = np.vstack([P.T - np.eye(n), np.ones((1, n))])
    b = np.concatenate([np.zeros(n), [1.0]])
    if np.linalg.matrix_rank(A) < n:
        raise ValueError("The chain has more than one stationary distribution.")
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi


def chain_entropy_rate(P: np.ndarray) -> float:
    """Eq. (2): h = sum_i pi_i H(P_i.), in bits per symbol."""
    pi = stationary_distribution(P)
    return float(sum(pi[i] * entropy_bits(P[i]) for i in range(P.shape[0])))


def chain_symbol_entropy(P: np.ndarray) -> float:
    """Eq. (3): H_1 = H(pi), in bits."""
    return entropy_bits(stationary_distribution(P))


def chain_excess_entropy(P: np.ndarray) -> float:
    """Eq. (4): E = H_1 - h, in bits. Valid for a first-order chain."""
    return chain_symbol_entropy(P) - chain_entropy_rate(P)


def chain_topological_entropy(P: np.ndarray) -> float:
    """Eq. (5): log2 of the largest eigenvalue modulus of the Boolean matrix ``P > 0``, in bits per symbol."""
    eigenvalues = np.linalg.eigvals((P > 0.0).astype(float))
    return float(np.log2(np.max(np.abs(eigenvalues))))


def two_state_closed_forms(
    p: float = 0.5, q: float = 1.0, delimiter: bool = False
) -> dict[str, float]:
    """Closed forms for ``two_state_grammar(p, q)``, written out by hand.

    Without the end marker the sequence is the two-state chain with P(A -> B) = p and P(B -> A) = q.
    With the marker ``#`` it is the three-symbol chain A -> A (1 - p), A -> B (p), B -> B (1 - q),
    B -> # (q), # -> A (1).
    """
    if not delimiter:
        pi = {"A": q / (p + q), "B": p / (p + q)}
        boolean = np.array([[p < 1.0, True], [True, q < 1.0]], dtype=float)
    else:
        c = 1.0 / (1.0 / p + 1.0 / q + 1.0)
        pi = {"A": c / p, "B": c / q, "#": c}
        boolean = np.array(
            [[p < 1.0, True, False], [False, q < 1.0, True], [True, False, False]], dtype=float
        )
    h = pi["A"] * binary_entropy(p) + pi["B"] * binary_entropy(
        q
    )  # Eq. (2); the marker row is deterministic
    h1 = entropy_bits(list(pi.values()))  # Eq. (3)
    te = float(np.log2(np.max(np.abs(np.linalg.eigvals(boolean)))))  # Eq. (5)
    return {
        **{f"pi_{symbol}": value for symbol, value in pi.items()},
        "entropy_rate": h,
        "symbol_entropy": h1,
        "excess_entropy": h1 - h,  # Eq. (4)
        "topological_entropy": te,
    }


def golden_mean_entropy_rate(p):
    """h(p) = H(p) / (1 + p): entropy rate of the worked example as a function of p = P(A -> B)."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    H = np.where((p <= 0.0) | (p >= 1.0), 0.0, H)
    return H / (1.0 + p)


def allowed_word_counts(max_length: int) -> list[int]:
    """Eq. (6): the number of allowed words of 1 to ``max_length`` symbols of the worked example.

    Exact integers, from powers of the table of allowed transitions T = [[1, 1], [1, 0]]: the row
    vector ``reach`` holds, for each last symbol, the number of allowed words that end in it.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be at least 1, got {max_length}.")
    allowed = ((1, 1), (1, 0))  # rows: current symbol A, B; columns: next symbol A, B
    reach = [1, 1]  # words of one symbol: A and B
    counts = []
    for _ in range(max_length):
        counts.append(sum(reach))
        reach = [sum(reach[i] * allowed[i][j] for i in range(2)) for j in range(2)]
    return counts


def word_count_line(length) -> np.ndarray:
    """Eq. (7): phi^(L+2) / sqrt(5), the line the number of allowed words follows."""
    return PHI ** (np.asarray(length, dtype=float) + 2.0) / math.sqrt(5.0)


def two_state_block_entropies(p: float = 0.5, max_length: int = 8) -> list[float]:
    """Eq. (8): H_L = E + L h for L = 1 to ``max_length``, from the hand formulas of the worked example."""
    values = two_state_closed_forms(p, 1.0, delimiter=False)
    return [
        values["excess_entropy"] + length * values["entropy_rate"]
        for length in range(1, max_length + 1)
    ]


# ----------------------------- brute force -----------------------------


def string_distribution(
    grammar, tol: float = 1e-13, max_strings: int = 200_000
) -> dict[tuple[str, ...], float]:
    """Enumerate the strings of a SymSeq ``ArtificialGrammar`` with their probabilities.

    Every path from a start state to the end marker is followed until its probability falls below
    ``tol``. Paths that emit the same symbols are added. The probabilities sum to one minus the mass
    of the paths that were cut.
    """
    names = [state.tostring() for state in grammar.states]
    symbol_of = {state.tostring(): state.symbol for state in grammar.states}
    successors: dict[str, list[tuple[str, float]]] = {name: [] for name in names}
    for source, target, probability in grammar.transitions:
        if probability > 0.0:
            successors[source].append((target, probability))
    out: dict[tuple[str, ...], float] = {}
    stack = [
        ((symbol_of[name],), name, probability)
        for name, probability in grammar.start_probabilities.items()
    ]
    while stack:
        prefix, state, probability = stack.pop()
        for target, step in successors[state]:
            reached = probability * step
            if target == grammar.eos:
                out[prefix] = out.get(prefix, 0.0) + reached
                if len(out) > max_strings:
                    raise RuntimeError(
                        "Too many strings to enumerate; raise tol or use a smaller grammar."
                    )
            elif reached >= tol:
                stack.append(((*prefix, symbol_of[target]), target, reached))
    return out


def renewal_statistics(
    strings: dict[tuple[str, ...], float], delimiter: bool = False, eos: str = "#"
) -> dict[str, float]:
    """Entropy rate and single-symbol entropy of the joined sequence, from the distribution of whole strings.

    The joined sequence is a succession of independent strings, so its entropy rate is the entropy of
    one string divided by the mean number of symbols it contributes (its length, plus one with the
    marker). This holds when the string boundaries can be read off the sequence, which is the case
    with the marker and, without it, for the two-state grammar (a new string starts at every B -> A).
    This route uses neither the stationary distribution nor Eq. (2).
    """
    mass = sum(strings.values())
    probabilities = np.array(list(strings.values())) / mass
    lengths = np.array([len(string) + (1 if delimiter else 0) for string in strings], dtype=float)
    mean_length = float(np.sum(probabilities * lengths))
    counts: dict[str, float] = {}
    for string, probability in zip(strings, probabilities, strict=True):
        for symbol in string:
            counts[symbol] = counts.get(symbol, 0.0) + probability
        if delimiter:
            counts[eos] = counts.get(eos, 0.0) + probability
    frequencies = np.array(list(counts.values())) / mean_length
    return {
        "entropy_rate": entropy_bits(probabilities) / mean_length,
        "symbol_entropy": entropy_bits(frequencies),
        "mean_length": mean_length,
        "string_entropy": entropy_bits(probabilities),
        "enumerated_mass": float(mass),
    }


def block_entropy_enumerated(P: np.ndarray, length: int) -> float:
    """H_L by enumerating every word of ``length`` symbols with its stationary probability, in bits."""
    pi = stationary_distribution(P)
    n = P.shape[0]
    total = 0.0
    for word in itertools.product(range(n), repeat=length):
        probability = pi[word[0]]
        for a, b in itertools.pairwise(word):
            probability *= P[a, b]
        if probability > 0.0:
            total -= probability * math.log2(probability)
    return total


def count_words(P: np.ndarray, length: int) -> int:
    """Number of words of ``length`` symbols the chain can produce, counted one by one."""
    allowed = P > 0.0
    n = P.shape[0]
    return sum(
        all(allowed[a, b] for a, b in itertools.pairwise(word))
        for word in itertools.product(range(n), repeat=length)
    )
