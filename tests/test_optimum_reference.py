"""The reference of a run is the exact predictor of its own grammar.

Added after the independent check of the networks code (2026-11-12): with the optimum of 2-AX
computed from 2-AX at another target probability (same alphabet, so no symbol is impossible) the
whole suite still passed. ``check_against_optimum`` compares a network with the reference it is
given and cannot notice a wrong reference. Two properties tie the reference to the grammar:

1. Agreement with the endless sequence. The optimum on the held-out samples differs from the
   generator's values for an endless sequence only by the cut at 40 symbols and by sampling.
   Largest documented difference at the production size (Dyck, depth 3): 0.019 in accuracy and
   0.051 bit in cross-entropy. Tolerances: 0.03 and 0.08 bit.
2. Calibration. On samples of its own grammar the mean surprisal of the exact predictor
   (``cross_entropy_realised``) equals its mean entropy (``cross_entropy``) up to sampling error.
   With at least 3500 targets and a surprisal standard deviation below 1.5 bit, five standard
   errors are below 0.13 bit. This bound is loose: it detects a reference that is too confident
   (sharper than the data), not one that is too flat.

Property 1 rejects the reference of a sibling grammar in both directions, property 2 in the
direction it can; both cases are tested.
"""

from __future__ import annotations

import pytest

from tutorial import config
from tutorial.datasets import build_rung_data, optimum
from tutorial.training import exact_values

SEED = config.EXPLORATION_SEEDS[0]
WINDOW_ACCURACY = 0.03
WINDOW_BITS = 0.08
CALIBRATION_BITS = 0.13
MIN_TARGETS = 3500


@pytest.fixture(scope="module")
def rung_data():
    return {
        name: build_rung_data(rung, SEED, config.SMOKE) for name, rung in config.CANDIDATES.items()
    }


def window_gap(found: dict, rung) -> tuple[float, float]:
    """Distance of an optimum from the endless-sequence values of ``rung`` (accuracy, bits)."""
    endless = exact_values(rung)
    return (
        abs(found["accuracy_with_eos"] - endless["ceiling_with_eos"]),
        abs(found["cross_entropy"] - endless["conditional_entropy"]),
    )


@pytest.mark.parametrize("name", list(config.CANDIDATES))
def test_optimum_agrees_with_the_endless_sequence_of_its_own_grammar(rung_data, name):
    data = rung_data[name]
    accuracy_gap, bits_gap = window_gap(data.optimum, data.rung)
    assert accuracy_gap <= WINDOW_ACCURACY, (name, accuracy_gap)
    assert bits_gap <= WINDOW_BITS, (name, bits_gap)


@pytest.mark.parametrize("name", list(config.CANDIDATES))
def test_exact_predictor_is_calibrated_on_its_own_grammar(rung_data, name):
    found = rung_data[name].optimum
    assert found["n_targets"] >= MIN_TARGETS
    assert abs(found["cross_entropy_realised"] - found["cross_entropy"]) <= CALIBRATION_BITS, name


@pytest.mark.parametrize(
    ("name", "other"),
    [
        ("2-AX", "2-AX (p_target 0.9)"),
        ("2-AX (p_target 0.9)", "2-AX"),
        ("4-AX", "4-AX (p_target 0.9)"),
    ],
)
def test_the_reference_of_another_grammar_with_the_same_alphabet_is_rejected(
    rung_data, name, other
):
    """Must reject: the optimum computed with the predictor of a sibling grammar."""
    data = rung_data[name]
    wrong = optimum(
        config.CANDIDATES[other], data.symbols, data.heldout
    )  # runs: no impossible symbol
    accuracy_gap, bits_gap = window_gap(wrong, data.rung)
    assert accuracy_gap > WINDOW_ACCURACY or bits_gap > WINDOW_BITS, (accuracy_gap, bits_gap)


@pytest.mark.parametrize(
    ("name", "other"), [("2-AX", "2-AX (p_target 0.9)"), ("4-AX", "4-AX (p_target 0.9)")]
)
def test_a_reference_sharper_than_the_data_is_not_calibrated(rung_data, name, other):
    """Must reject: the predictor of the grammar with the more probable target, on the flatter data."""
    data = rung_data[name]
    wrong = optimum(config.CANDIDATES[other], data.symbols, data.heldout)
    assert wrong["cross_entropy_realised"] - wrong["cross_entropy"] > CALIBRATION_BITS


def test_tolerances_are_tighter_than_the_gap_between_sibling_grammars():
    """The tolerances must separate the two 2-AX rungs; otherwise the tests above say nothing."""
    first, second = (
        exact_values(config.CANDIDATES[name]) for name in ("2-AX", "2-AX (p_target 0.9)")
    )
    assert abs(first["ceiling_with_eos"] - second["ceiling_with_eos"]) > 2 * WINDOW_ACCURACY
