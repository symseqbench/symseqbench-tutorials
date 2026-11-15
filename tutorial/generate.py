"""Thin wrappers around SymSeq's generators.

Every function here builds one SymSeq generator and returns it unchanged, so that the SymSeq object
(its states, transitions and sampling methods) stays visible to the participant. The only logic added
is the mapping from the tutorial's controls (number of states, alphabet size, number of contexts,
depth) to SymSeq's own arguments.

SymSeq calls: ``ArtificialGrammar``, ``ArtificialGrammar.from_constraints``, ``nAX``,
``NonAdjacentDependencies``, ``NestedNonAdjacentDependencies``, ``CrossedNonAdjacentDependencies``,
``DyckGenerator`` and their ``generate_string_set`` methods.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from symseq.generators import (
    ArtificialGrammar,
    CrossedNonAdjacentDependencies,
    DyckGenerator,
    NestedNonAdjacentDependencies,
    NonAdjacentDependencies,
    nAX,
)

EOS = "#"

# Symbols used when an nAX grammar needs more than SymSeq's two default contexts. For n = 2 the
# resulting maps are SymSeq's defaults: cues A, B and probes X, Y.
_NAX_CUES = ("A", "B", "E", "F", "G", "H")
_NAX_PROBES = ("X", "Y", "V", "W", "U", "T")
_NAX_FILLERS = ("C", "D", "Z")


def two_state_grammar(p: float = 0.5, q: float = 1.0, seed: int = 0) -> ArtificialGrammar:
    """Grammar with two states, A (start) and B (terminal). This is the worked example.

    From A the grammar stays in A with probability ``1 - p`` and moves to B with probability ``p``.
    From B it stays in B with probability ``1 - q`` and ends the string with probability ``q``.
    The strings are ``A^k B^m`` with ``k, m >= 1``. Transitions of probability zero are not declared,
    because SymSeq treats every declared transition as allowed when it checks grammaticality.
    """
    if not (0.0 < p <= 1.0 and 0.0 < q <= 1.0):
        raise ValueError(f"p and q must be in (0, 1], got p={p}, q={q}.")
    transitions: list[tuple[str, str, float]] = []
    if p < 1.0:
        transitions.append(("A", "A", 1.0 - p))
    transitions.append(("A", "B", p))
    if q < 1.0:
        transitions.append(("B", "B", 1.0 - q))
    transitions.append(("B", EOS, q))
    return ArtificialGrammar(
        label=f"two-state (p={p:g}, q={q:g})",
        states=["A", "B"],
        alphabet=["A", "B"],
        transitions=transitions,
        start_states=["A"],
        terminal_states=["B"],
        eos=EOS,
        seed=seed,
    )


def split_states(n_states: int, alphabet_size: int) -> tuple[int, int]:
    """Translate (number of states, alphabet size) into SymSeq's (ambiguities, ambiguity_depth).

    SymSeq builds ``alphabet_size - ambiguities + ambiguities * ambiguity_depth`` states: ``ambiguities``
    symbols are each emitted by ``ambiguity_depth`` different states. The split with the most repeated
    symbols is chosen, which spreads the extra states evenly.
    """
    extra = n_states - alphabet_size
    if alphabet_size < 1 or extra < 0:
        raise ValueError(
            f"Need n_states >= alphabet_size >= 1, got n_states={n_states}, alphabet_size={alphabet_size}."
        )
    if extra == 0:
        return 0, 0
    for ambiguities in range(min(alphabet_size, extra), 0, -1):
        if extra % ambiguities == 0:
            return ambiguities, extra // ambiguities + 1
    raise AssertionError("unreachable: ambiguities = 1 always divides")


def regular_grammar(
    n_states: int = 6,
    alphabet_size: int = 4,
    transition_density: float = 0.3,
    seed: int = 0,
    unifilar: bool = True,
) -> ArtificialGrammar:
    """Regular grammar sampled by SymSeq under constraints (``ArtificialGrammar.from_constraints``).

    The constraints are the alphabet size, the number of states, the density of the transition
    table, one start and one terminal state, and strings of at least two symbols. With ``unifilar``
    each state has at most one outgoing transition per symbol, and state copies that generate the
    same strings are rejected.
    """
    ambiguities, depth = split_states(n_states, alphabet_size)
    with warnings.catch_warnings():
        # SymSeq warns when the density leaves fewer than 1.5 transitions per state; the grammar is still valid.
        warnings.simplefilter("ignore", UserWarning)
        return ArtificialGrammar.from_constraints(
            label=f"regular ({n_states} states, {alphabet_size} symbols)",
            alphabet_size=alphabet_size,
            ambiguities=ambiguities,
            ambiguity_depth=depth,
            n_start_states=1,
            n_terminal_states=1,
            transition_density=transition_density,
            min_string_length=2,
            eos=EOS,
            seed=seed,
            enforce_unifilarity=unifilar,
            reject_equivalent_copies=unifilar,
        )


def nax_grammar(n: int = 2, p_target: float = 0.6, seed: int = 0) -> nAX:
    """n-AX grammar: context, cue, fillers, probe. ``n`` is the number of contexts.

    In SymSeq the ``n`` of n-AX counts the contexts, each with its own cue and its own target probe.
    The probe is the target of the trial's context with probability ``p_target`` and one of the other
    probes otherwise. The context has to be remembered across the fillers.
    """
    if not 2 <= n <= len(_NAX_CUES):
        raise ValueError(f"n must be between 2 and {len(_NAX_CUES)}, got {n}.")
    contexts = tuple(str(i + 1) for i in range(n))
    return nAX(
        label=f"{n}-AX",
        contexts=contexts,
        cue_map=dict(zip(contexts, _NAX_CUES, strict=False)),
        probe_map=dict(zip(contexts, _NAX_PROBES, strict=False)),
        fillers=_NAX_FILLERS,
        p_target=p_target,
        eos=EOS,
        seed=seed,
    )


def nad(n_deps: int = 2, n_fillers: int = 3, seed: int = 0) -> NonAdjacentDependencies:
    """Non-adjacent dependencies ``A_i X ... X B_i`` with ``n_deps`` pairs and ``n_fillers`` filler symbols."""
    return NonAdjacentDependencies(
        label=f"NAD ({n_deps} pairs)",
        n_deps=n_deps,
        n_unique_fillers=n_fillers,
        eos=EOS,
        seed=seed,
        verbose=False,
    )


def nested_nad(n_deps: int = 2, seed: int = 0) -> NestedNonAdjacentDependencies:
    """Nested dependencies ``A_i A_j ... B_j B_i``: the last opened pair closes first."""
    return NestedNonAdjacentDependencies(
        label=f"nested ({n_deps} pairs)", n_deps=n_deps, eos=EOS, seed=seed, verbose=False
    )


def crossed_nad(n_deps: int = 2, seed: int = 0) -> CrossedNonAdjacentDependencies:
    """Crossed dependencies ``A_i A_j ... B_i B_j``: the first opened pair closes first."""
    return CrossedNonAdjacentDependencies(
        label=f"crossed ({n_deps} pairs)", n_deps=n_deps, eos=EOS, seed=seed, verbose=False
    )


def dyck(k: int = 2, depth: int | None = 2, pairs: int = 4, seed: int = 0) -> DyckGenerator:
    """Dyck language with ``k`` bracket types, ``pairs`` bracket pairs per string and nesting depth at most ``depth``.

    SymSeq's uniform mode is used: every bracket structure with ``pairs`` pairs and depth at most
    ``depth`` is equally likely, and the type of each pair is drawn uniformly. No distractors.
    """
    return DyckGenerator(
        k=k, mode="uniform", target_pairs=pairs, max_depth=depth, n_distractors=0, seed=seed
    )


def generate_strings(generator, n: int, **kwargs) -> list[list[str]]:
    """Generate ``n`` grammatical strings with the generator's own ``generate_string_set``.

    SymSeq's generators return either the list of strings or a ``(strings, labels)`` pair. This
    wrapper returns the list of strings in both cases. Keyword arguments are forwarded unchanged,
    for example ``filler_len`` and ``randomize_fillers`` for non-adjacent dependencies.
    """
    out = generator.generate_string_set(n, **kwargs)
    if isinstance(out, tuple):
        strings, labels = out
        if not all(labels):
            raise ValueError(
                "The generator returned violations; generate_strings expects grammatical strings only."
            )
    else:
        strings = out
    return [list(string) for string in strings]


def as_stream(
    strings: Sequence[Sequence[str]], delimiter: bool = True, eos: str = EOS
) -> list[str]:
    """Join strings into one long sequence, with the end-of-string marker after each string by default.

    With the marker the sequence is the one SeqBench builds (each trial followed by ``#``) and the one
    on which SymSeq defines the topological entropy of a grammar (every exit loops back to the start).
    """
    stream: list[str] = []
    for string in strings:
        stream.extend(string)
        if delimiter:
            stream.append(eos)
    return stream
