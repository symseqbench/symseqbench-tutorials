"""The training loop: measures, the optimum as a bound, the two-state criterion, seeds, threads.

Every test has a case it must reject. Training runs use the exploration seeds, never the plotted ones.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest
import torch

from tutorial import config, datasets, exact, training
from tutorial.datasets import PAD
from tutorial.ladder import build_generator

SEED = config.EXPLORATION_SEEDS[0]
SMALL = replace(config.SMOKE, n_train=128, n_heldout=64, epochs=2)


class GivenTheTarget(torch.nn.Module):
    """A predictor that is handed the targets: what a leak of the target into the input looks like."""

    def __init__(self, split: datasets.Split):
        super().__init__()
        self.scores = 50.0 * torch.nn.functional.one_hot(
            split.targets.clamp(min=0), split.inputs.shape[-1]
        )

    def forward(self, x):
        return self.scores


class KnowsTheGrammar(torch.nn.Module):
    """The exact predictor as a network: its scores are the log-probabilities of the generator."""

    def __init__(self, data: datasets.RungData):
        super().__init__()
        predictor = exact.predictor_for(build_generator(data.rung), **data.rung.sample)
        rows = []
        for tokens in data.heldout.tokens.tolist():
            distributions = exact.follow(predictor, [data.symbols[token] for token in tokens])
            rows.append(
                [
                    [
                        math.log(d[symbol]) if d.get(symbol, 0.0) > 0.0 else -1e9
                        for symbol in data.symbols
                    ]
                    for d in distributions
                ]
            )
        self.scores = torch.tensor(rows)

    def forward(self, x):
        return self.scores


def history_of(model: torch.nn.Module, split: datasets.Split) -> dict:
    return {key: [value] for key, value in training.evaluate(model, split).items()}


@pytest.fixture(scope="module")
def nested():
    return datasets.build_rung_data(config.CANDIDATES["nested"], SEED, SMALL)


@pytest.fixture(scope="module")
def two_state():
    return datasets.build_rung_data(config.CANDIDATES["two-state"], SEED, SMALL)


def test_measures_on_a_case_computed_by_hand():
    """Three classes (marker, A, B), one sample: A B # A, targets B # none A.

    Scores: position 0 puts 0.5 on B (target B: hit), position 1 puts 0.25 on the marker and 0.5 on
    A (target marker: miss), position 3 puts 0.15 on A, 0.10 on B and 0.75 on the marker (target A:
    a miss when the marker may be named, a hit among symbols).
    """
    inputs = torch.nn.functional.one_hot(torch.tensor([[1, 2, 0, 1]]), 3).float()
    targets = torch.tensor([[2, 0, PAD, 1]])
    probabilities = torch.tensor(
        [[[0.25, 0.25, 0.5], [0.25, 0.5, 0.25], [0.2, 0.3, 0.5], [0.75, 0.15, 0.10]]]
    )

    class Fixed(torch.nn.Module):
        def forward(self, x):
            return probabilities.log()

    measured = training.evaluate(Fixed(), datasets.Split(inputs, targets))
    assert measured["accuracy_with_eos"] == pytest.approx(1 / 3)  # hit, miss, miss
    assert measured["accuracy_within"] == pytest.approx(1.0)  # targets B and A, both hits
    expected_bits = -(math.log2(0.5) + math.log2(0.25) + math.log2(0.15)) / 3
    assert measured["cross_entropy"] == pytest.approx(expected_bits, abs=1e-6)
    # must reject: the position without target is not scored (it would change every number)
    everything = training.evaluate(Fixed(), datasets.Split(inputs, torch.tensor([[2, 0, 2, 1]])))
    assert everything["cross_entropy"] != pytest.approx(expected_bits, abs=1e-3)


@pytest.mark.parametrize("name", ["two-state", "nested", "4-AX (p_target 0.9)", "Dyck (depth 3)"])
def test_the_exact_predictor_scores_the_optimum_and_nothing_beats_it(name):
    data = datasets.build_rung_data(config.CANDIDATES[name], SEED, SMALL)
    optimum = data.optimum
    exact_history = history_of(KnowsTheGrammar(data), data.heldout)
    # two independent routes to the same number: tensors here, a Python loop in datasets.optimum
    assert exact_history["cross_entropy"][0] == pytest.approx(
        optimum["cross_entropy_realised"], abs=1e-5
    )
    checks = training.check_against_optimum(exact_history, optimum)
    assert checks["passed"]
    assert abs(checks["cross_entropy"]["largest_advantage_bits"]) < 0.5
    for convention in ("with_eos", "within"):
        key = f"accuracy_{convention}"
        allowed = checks[key]["allowed"]
        assert allowed == pytest.approx(
            training.SIGMAS
            * math.sqrt(optimum[f"variance_{convention}"])
            / optimum[f"n_{convention}"]
        )
        assert abs(exact_history[key][0] - optimum[key]) <= allowed

    # must reject: a predictor that is given the target beats the optimum on both measures
    leaked = training.check_against_optimum(
        history_of(GivenTheTarget(data.heldout), data.heldout), optimum
    )
    assert not leaked["passed"]
    assert not leaked["cross_entropy"]["passed"]
    assert leaked["cross_entropy"]["largest_advantage_bits"] > 10 * training.BITS_TOLERANCE
    assert not leaked["accuracy_with_eos"]["passed"] and not leaked["accuracy_within"]["passed"]


def test_a_run_scored_against_the_optimum_of_another_grammar_is_refused(nested, two_state):
    """must reject: the exact predictor of the nested grammar against the optimum of the two-state grammar."""
    history = history_of(KnowsTheGrammar(nested), nested.heldout)
    assert training.check_against_optimum(history, nested.optimum)["passed"]
    assert not training.check_against_optimum(history, two_state.optimum)["passed"]


def test_tolerances_are_those_written_in_the_acceptance_criteria():
    assert training.BITS_TOLERANCE == 20.0 and training.SIGMAS == 5.0
    optimum = {
        "n_targets": 1000,
        "cross_entropy_realised": 0.5,
        "n_with_eos": 1000,
        "accuracy_with_eos": 0.7,
        "variance_with_eos": 100.0,
        "n_within": 800,
        "accuracy_within": 0.6,
        "variance_within": 64.0,
    }

    def passed(cross_entropy, with_eos, within):
        history = {
            "cross_entropy": [1.0, cross_entropy],
            "accuracy_with_eos": [0.1, with_eos],
            "accuracy_within": [0.1, within],
        }
        return training.check_against_optimum(history, optimum)["passed"]

    assert passed(0.5 - 0.019, 0.7 + 0.049, 0.6 + 0.049)  # 19 bits, 4.9 sigma, 4.9 sigma
    # must reject: each bound exceeded on its own (21 bits; 5.1 sigma of 10 hits; 5.1 sigma of 8 hits)
    assert not passed(0.5 - 0.021, 0.7, 0.6)
    assert not passed(0.5, 0.7 + 0.051, 0.6)
    assert not passed(0.5, 0.7, 0.6 + 0.051)


def test_two_state_grammar_is_learned_within_five_epochs():
    """Criterion 8: production settings, exploration seed, 5 epochs, both models within the margins."""
    settings = replace(config.PRODUCTION, epochs=5)
    rung = config.CANDIDATES["two-state"]
    data = datasets.build_rung_data(rung, SEED, settings)
    assert (len(data.train) + data.n_train_removed, len(data.heldout)) == (2000, 500)
    for model in config.MODELS:
        run = training.train_run(rung, model, SEED, settings, data=data)
        assert run["history"]["epoch"] == [0, 1, 2, 3, 4, 5]
        assert training.reaches_optimum(run, epoch=5, accuracy_margin=0.02, bits_margin=0.05), (
            model,
            run["final"],
            run["optimum"],
        )
        assert run["checks"]["passed"]
        # must reject: the same network before training
        assert not training.reaches_optimum(run, epoch=0, accuracy_margin=0.02, bits_margin=0.05)


@pytest.mark.parametrize("model", config.MODELS)
def test_identical_seeds_give_identical_runs(model, nested):
    rung = nested.rung
    first = training.train_run(rung, model, SEED, SMALL, data=nested)
    again = training.train_run(rung, model, SEED, SMALL, data=nested)
    assert first["history"] == again["history"]
    assert first["weights_sha256"] == again["weights_sha256"]
    # the data are drawn from the seed too: a run that builds its own data gives the same numbers
    rebuilt = training.train_run(rung, model, SEED, SMALL)
    assert rebuilt["history"] == first["history"]
    assert rebuilt["weights_sha256"] == first["weights_sha256"]
    # must reject: another seed
    other = training.train_run(rung, model, config.EXPLORATION_SEEDS[1], SMALL)
    assert other["history"] != first["history"]
    assert other["weights_sha256"] != first["weights_sha256"]
    assert other["optimum"] == first["optimum"]  # the held-out samples do not depend on the seed


def test_the_thread_limit_is_set_and_recorded(nested):
    before = torch.get_num_threads()
    try:
        for threads in (1, 2):
            run = training.train_run(
                nested.rung, "rnn", SEED, replace(SMALL, epochs=1, threads=threads), data=nested
            )
            assert run["environment"]["threads"] == threads == torch.get_num_threads()
        for wrong in (0, -1, 1.5):
            with pytest.raises(ValueError):
                training.set_threads(wrong)  # must reject
    finally:
        torch.set_num_threads(before)


def test_a_run_holds_what_the_figure_and_the_checks_need(nested):
    calls = []
    run = training.train_run(
        nested.rung, "rnn", SEED, SMALL, data=nested, progress=lambda e, row: calls.append(e)
    )
    assert calls == [0, 1, 2]
    assert run["settings"] == SMALL.as_dict()
    assert run["n_parameters"] == 20_107 and run["model_size"] == {"hidden units": 134}
    assert run["data"]["n_train"] + run["data"]["n_train_removed"] == SMALL.n_train
    assert run["data"]["seqbench_seeds"] == {
        "train": datasets.stream_seed(SEED + 1),
        "heldout": datasets.stream_seed(0),
    }
    assert run["history"]["train_loss"][0] is None and len(run["history"]["cross_entropy"]) == 3
    assert run["final"]["cross_entropy"] == run["history"]["cross_entropy"][-1]
    endless = run["exact"]
    assert endless["entropy_rate"] == pytest.approx(math.log2(6) / 7)
    assert endless["first_symbol_entropy"] == pytest.approx(math.log2(3))
    assert endless["conditional_entropy"] == pytest.approx(1 / 6)  # one coin flip in six targets
    # must reject: the entropy rate is not the best cross-entropy per SeqBench target
    assert abs(endless["entropy_rate"] - endless["conditional_entropy"]) > 0.1
    assert json.loads(json.dumps(run)) == run  # numbers, strings, lists and dictionaries only
    rows = training.history_table({"RNN": run}, every=1)
    assert [row["epoch"] for row in rows] == [0, 1, 2]
    assert rows[-1]["RNN: accuracy"] == run["final"]["accuracy_with_eos"]
