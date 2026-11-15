"""Data built by SeqBench: same grammar and same strings as SymSeq, targets, and no shared sample.

Every test has a case it must reject.
"""

from __future__ import annotations

import importlib.util
import itertools
import logging
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tutorial import config, datasets, exact
from tutorial.datasets import EOS_INDEX, PAD
from tutorial.ladder import build_generator

REPO = Path(__file__).resolve().parents[1]
LENGTH = 40
SMALL = replace(config.SMOKE, n_train=40, n_heldout=30)
RUNGS = list(config.CANDIDATES.values())


def symseq_sample(rung, seed: int, index: int, length: int = LENGTH) -> tuple[list[str], list[str]]:
    """Sample ``index`` as SymSeq alone draws it: strings of ``generate_string`` joined with the marker.

    The generator is the one of notebook A (``build_generator``), seeded as SeqBench seeds the sample.
    Returns the first ``length`` symbols and, for each, the next symbol (a dot after a marker).
    """
    generator = build_generator(rung)
    generator.rng = np.random.default_rng(seed * index)
    stream: list[str] = []
    while len(stream) < length + 1:
        stream += [*generator.generate_string(**rung.sample), "#"]
    targets = [
        "." if symbol == "#" else following
        for symbol, following in itertools.pairwise(stream[: length + 1])
    ]
    return stream[:length], targets


def mismatches(rung, split, symbols, seed: int, reference_rung=None, reference_seed=None) -> int:
    """Number of samples whose symbols or targets differ from what SymSeq draws."""
    wrong = 0
    for row, (tokens, targets) in enumerate(
        zip(split.tokens.tolist(), split.targets.tolist(), strict=True)
    ):
        expected = symseq_sample(reference_rung or rung, reference_seed or seed, row + 1)
        found = (
            [symbols[token] for token in tokens],
            [symbols[t] if t != PAD else "." for t in targets],
        )
        wrong += found != expected
    return wrong


@pytest.mark.parametrize("rung", RUNGS, ids=lambda rung: rung.name)
def test_the_network_receives_the_strings_symseq_generates(rung):
    n = 20
    for stream in (datasets.HELDOUT_STREAM, 102):
        seed = datasets.stream_seed(stream)
        split, symbols, configuration = datasets.build_split(rung, n, seed, LENGTH)
        assert symbols == ("#", *build_generator(rung).alphabet)
        assert (
            split.inputs.shape == (n, LENGTH, len(symbols)) and split.inputs.dtype == torch.float32
        )
        assert split.targets.shape == (n, LENGTH) and split.targets.dtype == torch.int64
        assert configuration["seqbench"]["seed"] == seed
        assert mismatches(rung, split, symbols, seed) == 0

        # must reject: another seed, and one changed symbol
        assert mismatches(rung, split, symbols, seed, reference_seed=seed + 2) == n
        changed = split.inputs.clone()
        changed[3, 5] = changed[3, 5].roll(1)
        assert mismatches(rung, datasets.Split(changed, split.targets), symbols, seed) == 1
    # must reject: the strings of another grammar with the same symbols, where there is one
    twin = {
        "nested": "crossed",
        "crossed": "nested",
        "2-AX": "2-AX (p_target 0.9)",
        "2-AX (p_target 0.9)": "2-AX",
        "4-AX": "4-AX (p_target 0.9)",
        "4-AX (p_target 0.9)": "4-AX",
        "Dyck (depth 2)": "Dyck (depth 3)",
        "Dyck (depth 3)": "Dyck (depth 2)",
    }.get(rung.name)
    if twin is not None:
        assert mismatches(rung, split, symbols, seed, reference_rung=config.CANDIDATES[twin]) > 0


@pytest.mark.parametrize("rung", RUNGS, ids=lambda rung: rung.name)
def test_targets_are_seqbench_labels_and_every_symbol_is_possible(rung):
    data = datasets.build_rung_data(rung, 101, SMALL)
    for split in (data.train, data.heldout):
        tokens, targets = split.tokens, split.targets
        at_marker = tokens == EOS_INDEX
        assert bool((targets[at_marker] == PAD).all()) and bool((targets[~at_marker] != PAD).all())
        inside = ~at_marker[:, :-1]
        assert bool((targets[:, :-1][inside] == tokens[:, 1:][inside]).all())
    # the exact predictor of the rung gives every symbol and target a positive probability
    assert datasets.optimum(rung, data.symbols, data.train)["n_targets"] > 0
    optimum = data.optimum
    assert optimum["n_targets"] == int((data.heldout.targets != PAD).sum())
    assert optimum["n_with_eos"] == optimum["n_targets"]  # SeqBench: no target at a marker
    assert optimum["n_within"] == int(
        ((data.heldout.targets != PAD) & (data.heldout.targets != EOS_INDEX)).sum()
    )
    assert math.isfinite(optimum["cross_entropy_realised"])

    # must reject: a target changed to a symbol that cannot follow
    impossible = data.heldout.targets.clone()
    row, position = next(
        (r, p)
        for r in range(len(data.heldout))
        for p in range(LENGTH)
        if impossible[r, p] == EOS_INDEX
    )
    wrong_symbol = {"two_state": "B", "regular": None, "nax": "1", "dyck": "a"}.get(rung.kind, "A0")
    if wrong_symbol is not None:  # a symbol that can never come where the marker came
        impossible[row, position] = data.symbols.index(wrong_symbol)
        with pytest.raises(ValueError, match="cannot follow"):
            datasets.optimum(rung, data.symbols, datasets.Split(data.heldout.inputs, impossible))


def test_the_predictor_of_another_grammar_finds_impossible_symbols():
    """must reject: nested data scored with the crossed predictor, and the reverse."""
    nested, crossed = config.CANDIDATES["nested"], config.CANDIDATES["crossed"]
    data = datasets.build_rung_data(nested, 101, SMALL)
    assert datasets.optimum(nested, data.symbols, data.heldout) == data.optimum
    with pytest.raises(ValueError, match="cannot follow"):
        datasets.optimum(crossed, data.symbols, data.heldout)
    data = datasets.build_rung_data(crossed, 101, SMALL)
    with pytest.raises(ValueError, match="cannot follow"):
        datasets.optimum(nested, data.symbols, data.heldout)


def test_optimum_on_the_samples_has_the_values_derived_by_hand():
    """Non-adjacent: 40 symbols are 8 whole strings A X X B #, with targets X, X, B, #.

    The two fillers are uniform over 3 symbols and the rest is certain: accuracy (1/3 + 1/3 + 1 + 1) / 4
    with the marker and (1/3 + 1/3 + 1) / 3 without, entropy 2 log2(3) / 4 bits per target.
    Nested: 5 whole strings of 7 symbols and the first 5 symbols of a sixth. Per whole string the
    targets are worth 1/2, 1, 1, 1, 1, 1, and the cut string adds 1/2, 1, 1, 1, 1: 32 / 35.
    """
    settings = replace(config.SMOKE, n_train=40, n_heldout=60)
    nad = datasets.build_rung_data(config.CANDIDATES["non-adjacent"], 101, settings).optimum
    assert nad["accuracy_with_eos"] == pytest.approx(2 / 3, abs=1e-12)
    assert nad["accuracy_within"] == pytest.approx(5 / 9, abs=1e-12)
    assert nad["cross_entropy"] == pytest.approx(math.log2(3) / 2, abs=1e-12)
    assert nad["cross_entropy_realised"] == pytest.approx(math.log2(3) / 2, abs=1e-12)
    assert nad["n_targets"] == 60 * 32
    nested = datasets.build_rung_data(config.CANDIDATES["nested"], 101, settings).optimum
    assert nested["accuracy_with_eos"] == pytest.approx(32 / 35, abs=1e-12)
    assert nested["cross_entropy"] == pytest.approx(6 / 35, abs=1e-12)
    assert nested["variance_with_eos"] == pytest.approx(60 * 6 * 0.25, abs=1e-9)
    # must reject: the ceiling of the endless sequence is another number, because the samples are cut
    endless = exact.trial_statistics(
        exact.predictor_for(build_generator(config.CANDIDATES["nested"]))
    )
    assert abs(endless["ceiling_with_eos"] - nested["accuracy_with_eos"]) > 2e-3


def test_follow_gives_the_distribution_after_every_symbol():
    predictor = exact.predictor_for(build_generator(config.CANDIDATES["two-state"]))
    after = exact.follow(predictor, ["A", "A", "B", "#", "A"])
    assert after[0] == {"A": 0.5, "B": 0.5} and after[1] == {"A": 0.5, "B": 0.5}
    assert after[2] == {"#": 1.0}  # after B the string ends
    assert after[3] == {"A": 1.0}  # after the marker a new string starts with A
    assert after[4] == {"A": 0.5, "B": 0.5}
    with pytest.raises(ValueError, match="cannot follow"):
        exact.follow(predictor, ["A", "B", "B"])  # must reject: B is never repeated


def is_prime(n: int) -> bool:
    """Miller-Rabin with bases 2, 3, 5, 7: exact below 3.2e9. Independent of ``stream_seed``."""
    if n < 2:
        return False
    for small in (2, 3, 5, 7):
        if n % small == 0:
            return n == small
    d, r = n - 1, 0
    while d % 2 == 0:
        d, r = d // 2, r + 1
    for base in (2, 3, 5, 7):
        x = pow(base, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def test_stream_seeds_are_distinct_primes_above_the_largest_dataset():
    seeds = [datasets.stream_seed(index) for index in range(120)]
    assert seeds[0] == 1_000_003 and seeds == sorted(set(seeds))
    assert all(is_prime(seed) and seed > datasets.MAX_SAMPLES for seed in seeds)
    assert datasets.stream_seed(7) == seeds[7]  # the same on every call
    assert not is_prime(1_000_001) and 1_000_001 not in seeds  # must reject: 101 x 9901
    for wrong in (-1, datasets.MAX_STREAMS):
        with pytest.raises(ValueError):
            datasets.stream_seed(wrong)  # must reject


def test_no_two_datasets_draw_a_sample_from_the_same_generator_state():
    heldout = datasets.stream_seed(datasets.HELDOUT_STREAM)
    for seed in (*config.PLOTTED_SEEDS, *config.EXPLORATION_SEEDS, config.LIVE_SEED):
        train = datasets.stream_seed(seed + 1)
        datasets.assert_disjoint_streams(train, 100_000, heldout, 100_000)
    # must reject: one seed for both, and two small seeds with a common multiple (2 * 2 == 4 * 1)
    with pytest.raises(ValueError, match="same generator state"):
        datasets.assert_disjoint_streams(heldout, 10, heldout, 10)
    with pytest.raises(ValueError, match="same generator state"):
        datasets.assert_disjoint_streams(2, 10, 4, 10)


def test_seqbench_splits_of_one_seed_are_the_same_samples():
    """The defect the workaround exists for: must reject the idea that 'train' and 'test' differ."""
    rung = config.CANDIDATES["Dyck (depth 3)"]
    seed = datasets.stream_seed(5)
    train, _, _ = datasets.build_split(rung, 30, seed, LENGTH, split="train")
    test, _, _ = datasets.build_split(rung, 30, seed, LENGTH, split="test")
    assert torch.equal(train.inputs, test.inputs)
    other, _, _ = datasets.build_split(rung, 30, datasets.stream_seed(6), LENGTH, split="test")
    assert int(datasets.shared_rows(other, train).sum()) == 0


@pytest.mark.parametrize("rung", RUNGS, ids=lambda rung: rung.name)
def test_no_heldout_sample_is_a_training_sample(rung):
    data = datasets.build_rung_data(rung, 101, SMALL)
    datasets.assert_disjoint(data.train, data.heldout)
    assert len(data.train) + data.n_train_removed == SMALL.n_train
    assert len(data.heldout) == SMALL.n_heldout
    # must reject: the same data with one held-out sample copied into the training samples
    leaked = datasets.Split(
        torch.cat([data.train.inputs, data.heldout.inputs[4:5]]),
        torch.cat([data.train.targets, data.heldout.targets[4:5]]),
    )
    with pytest.raises(ValueError, match="1 held-out samples are also training samples"):
        datasets.assert_disjoint(leaked, data.heldout)


def test_chance_coincidences_exist_for_a_grammar_with_few_strings_and_are_removed():
    """Nested, 2000 and 500 samples: only 6^6 different samples exist, so about 21 coincide."""
    rung, settings = config.CANDIDATES["nested"], config.PRODUCTION
    train, _, _ = datasets.build_split(
        rung, settings.n_train, datasets.stream_seed(102), LENGTH, "train"
    )
    heldout, _, _ = datasets.build_split(
        rung, settings.n_heldout, datasets.stream_seed(datasets.HELDOUT_STREAM), LENGTH, "test"
    )
    coincidences = int(datasets.shared_rows(train, heldout).sum())
    assert 5 <= coincidences <= 60
    with pytest.raises(ValueError, match="also training samples"):
        datasets.assert_disjoint(train, heldout)  # must reject: the datasets as SeqBench built them
    data = datasets.build_rung_data(rung, 101, settings)
    assert data.n_train_removed == coincidences and len(data.train) == 2000 - coincidences
    assert len(data.heldout) == 500  # the held-out samples are left as drawn
    datasets.assert_disjoint(data.train, data.heldout)


def test_seqbench_builds_the_grammar_of_the_rung():
    rung = config.CANDIDATES["4-AX (p_target 0.9)"]
    ours = datasets.signature(build_generator(rung), rung.sample)
    assert ours["type"] == "nAX" and len(ours["alphabet"]) == 15
    # must reject: the same family with another probability, another depth, another filler length
    other = config.CANDIDATES["4-AX"]
    assert datasets.signature(build_generator(other), other.sample) != ours
    dyck2, dyck3 = config.CANDIDATES["Dyck (depth 2)"], config.CANDIDATES["Dyck (depth 3)"]
    assert datasets.signature(build_generator(dyck2)) != datasets.signature(build_generator(dyck3))
    nad = config.CANDIDATES["non-adjacent"]
    assert datasets.signature(build_generator(nad), nad.sample) != datasets.signature(
        build_generator(nad), {"filler_len": 1, "randomize_fillers": True}
    )
    for wrong in ({"n_samples": 0}, {"targets": "all"}):
        arguments = {"n_samples": 5, "seed": datasets.stream_seed(1), "sample_length": LENGTH}
        with pytest.raises(ValueError):
            datasets.build_split(rung, **{**arguments, **wrong})  # must reject


def test_build_split_refuses_a_grammar_that_is_not_the_rung(monkeypatch):
    """must reject: a configuration that names another grammar than the rung."""
    rung, other = config.CANDIDATES["Dyck (depth 2)"], config.CANDIDATES["Dyck (depth 3)"]
    genuine = datasets.generator_config
    monkeypatch.setattr(datasets, "generator_config", lambda _: genuine(other))
    with pytest.raises(ValueError, match="another grammar"):
        datasets.build_split(rung, 5, datasets.stream_seed(1), LENGTH)


def test_stream_targets_add_the_first_symbol_of_the_next_string():
    rung = config.CANDIDATES["two-state"]
    seed = datasets.stream_seed(3)
    seqbench, symbols, _ = datasets.build_split(rung, 10, seed, LENGTH, targets="seqbench")
    stream, _, _ = datasets.build_split(rung, 10, seed, LENGTH, targets="stream")
    tokens = stream.tokens
    at_marker = tokens == EOS_INDEX
    assert torch.equal(stream.inputs, seqbench.inputs)
    assert torch.equal(stream.targets[~at_marker], seqbench.targets[~at_marker])
    inner = at_marker[:, :-1]
    assert bool(
        (stream.targets[:, :-1][inner] == symbols.index("A")).all()
    )  # a string starts with A
    assert bool(
        (stream.targets[:, -1][at_marker[:, -1]] == PAD).all()
    )  # nothing known past the sample
    # must reject: SeqBench's own labels have no target at a marker
    assert bool((seqbench.targets[at_marker] == PAD).all()) and int(inner.sum()) > 0
    # with these targets the best cross-entropy counts the first symbols too
    settings = replace(SMALL, targets="stream")
    data = datasets.build_rung_data(rung, 101, settings)
    assert data.optimum["n_targets"] > data.optimum["n_with_eos"]


def test_the_preview_shows_symbols_over_targets_and_no_local_path():
    text, yaml_text = datasets.preview(config.CANDIDATES["nested"], config.LIVE_SEED)
    lines = text.splitlines()
    assert lines[0].startswith("input   A") and lines[1].startswith("target  A")
    assert lines[0].split()[1:][1:7] == lines[1].split()[1:][0:6]  # each target is the next symbol
    assert "NestedNonAdjacentDependencies" in yaml_text and "NStepPrediction" in yaml_text
    assert "path: data/seqbench_cache" in yaml_text
    assert "/home/" not in yaml_text and "Users" not in yaml_text  # must reject: a local path


def fresh_python(code: str) -> subprocess.CompletedProcess:
    """Some lines of Python in a new interpreter, in which SeqBench has not been imported yet."""
    environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=environment, capture_output=True, text=True
    )


def test_seqbench_is_imported_without_its_two_warnings_about_torchaudio():
    """The two warnings that torchaudio is absent are dropped where they are logged, and nothing else."""
    quietly = fresh_python(
        "from tutorial import datasets\nprint(datasets.seqbench_dataloader().__name__)\n"
    )
    assert quietly.returncode == 0, quietly.stderr[-2000:]
    assert (quietly.stdout, quietly.stderr) == ("build_dataloader\n", "")
    # the filter is gone when the import is done: the same message logged again is shown
    after = fresh_python(
        "import logging\n"
        "from tutorial import datasets\n"
        "datasets.seqbench_dataloader()\n"
        "for name, message in datasets.TORCHAUDIO_WARNINGS.items():\n"
        "    assert logging.getLogger(name).filters == []\n"
        "    logging.getLogger(name).warning(message)\n"
    )
    assert after.returncode == 0, after.stderr[-2000:]
    assert after.stderr.count("not loaded. torchaudio not installed?") == 2
    assert "LogMel" in after.stderr and "MFCC" in after.stderr and after.stdout == ""

    # the filter drops these two records and no other one
    def kept(name: str, message: str, level: int = logging.WARNING) -> bool:
        record = logging.LogRecord(name, level, __file__, 1, message, None, None)
        return datasets.not_a_torchaudio_warning(record)

    assert len(datasets.TORCHAUDIO_WARNINGS) == 2
    for name, message in datasets.TORCHAUDIO_WARNINGS.items():
        assert not kept(name, message)
        assert kept(name, message + " Install it for the audio transforms.")
        assert kept(name, "An error of another kind", logging.ERROR)
        assert kept("seqbench.experiment", message)  # the same words from another module
    # must reject: the plain import. It writes the two lines to the standard error whenever
    # torchaudio is absent, as it is in the environment of the tutorial
    plain = fresh_python("from seqbench import build_dataloader\n")
    assert plain.returncode == 0, plain.stderr[-2000:]
    absent = importlib.util.find_spec("torchaudio") is None
    for message in datasets.TORCHAUDIO_WARNINGS.values():
        assert (message in plain.stderr) == absent, message
