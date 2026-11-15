"""Training and held-out data of a ladder rung, built with SeqBench.

SeqBench reads one configuration (a dictionary, or the same content as a YAML file), builds the
SymSeq generator named in it, draws strings, joins them with the end marker ``#`` into samples of
a fixed number of symbols, and returns a PyTorch ``DataLoader`` (``seqbench.build_dataloader``).
Every symbol is one row of the input, one-hot over the alphabet plus the marker; class 0 is the
marker. The target at each position is the next symbol of the same string, or the marker after
the last symbol (SeqBench's task ``NStepPrediction`` with ``n = 1``). At a marker there is no
target (label ``-1``): the first symbol of a string is never predicted.

SeqBench assembles every sample symbol by symbol in Python each time the loader is read, which costs
more than training a small network on it. ``build_split`` therefore reads the loader once and
keeps the tensors. They are the tensors SeqBench produced, unchanged.

## The split workaround

SeqBench draws sample ``i`` of a dataset from ``numpy.random.default_rng(seqbench.seed * i)``,
whatever the split. Three consequences:

1. the ``train`` and ``test`` splits of one configuration are the same samples;
2. sample 0 is the same in every dataset, because its generator is seeded with 0;
3. two datasets with seeds ``a`` and ``b`` share every pair of samples with ``a * i == b * j``.

What is done here, and nowhere else:

- training data and held-out data are two SeqBench datasets, each with its own ``seqbench.seed``;
- these seeds are distinct primes above ``MAX_SAMPLES`` (``stream_seed``). For two such primes
  ``p != q`` and ``1 <= i, j <= MAX_SAMPLES``, ``p * i == q * j`` would need ``p`` to divide ``j``,
  which is smaller than ``p``. No two samples of two datasets come from the same generator state
  (``assert_disjoint_streams``);
- sample 0 of every dataset is dropped;
- two samples drawn from different generator states can still hold the same symbols, which happens
  for grammars with few strings. A training sample whose symbols equal those of a held-out sample
  is removed, the number removed is recorded, and ``assert_disjoint`` checks the result. The
  held-out samples are left as drawn.

SeqBench also reseeds the global generators of numpy and torch when it builds a dataset. Code that
needs its own seed must set it after the data are built (``tutorial.training`` does).
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from tutorial import exact
from tutorial.config import PRODUCTION, Hyperparameters
from tutorial.generate import EOS
from tutorial.ladder import Rung, build_generator

PAD = -1  # SeqBench's label where there is no target
EOS_INDEX = 0  # SeqBench's class of the end marker
MAX_SAMPLES = 1_000_000  # largest dataset for which the stream seeds are guaranteed disjoint
MAX_STREAMS = 10_000
HELDOUT_STREAM = 0  # the held-out data of every grammar; training seed s uses stream s + 1
CACHE = Path(__file__).resolve().parents[1] / "data" / "seqbench_cache"

_primes: list[int] = []


def stream_seed(index: int) -> int:
    """The ``seqbench.seed`` of dataset number ``index``: the ``index``-th prime above ``MAX_SAMPLES``."""
    if not 0 <= index < MAX_STREAMS:
        raise ValueError(f"The stream index must be between 0 and {MAX_STREAMS - 1}, got {index}.")
    candidate = _primes[-1] if _primes else MAX_SAMPLES
    while len(_primes) <= index:
        candidate += 1
        if all(candidate % divisor for divisor in range(2, math.isqrt(candidate) + 1)):
            _primes.append(candidate)
    return _primes[index]


def sample_seeds(seed: int, n_samples: int) -> set[int]:
    """The generator seeds SeqBench uses for samples 1 to ``n_samples`` of a dataset (sample 0 is dropped)."""
    return {seed * index for index in range(1, n_samples + 1)}


def assert_disjoint_streams(seed_a: int, n_a: int, seed_b: int, n_b: int) -> None:
    """Raise if a sample of one dataset is drawn from the same generator state as a sample of the other."""
    shared = sample_seeds(seed_a, n_a) & sample_seeds(seed_b, n_b)
    if shared:
        raise ValueError(
            f"{len(shared)} samples of the two datasets are drawn from the same generator state "
            f"(seqbench.seed {seed_a} and {seed_b})."
        )


# --- SeqBench configuration -----------------------------------------------------------------------


def generator_config(rung: Rung) -> dict:
    """The ``symseq.generator`` block of the configuration: SymSeq's registry name and arguments.

    The arguments are read from the generator that ``tutorial.ladder.build_generator`` builds for
    the rung, so that SeqBench builds the same grammar as notebook A.
    """
    generator = build_generator(rung)
    if rung.kind in ("two_state", "regular"):
        return {
            "type": "ArtificialGrammar",
            "mode": "custom",
            "params": {
                "label": generator.label,
                "states": [s.tostring() for s in generator.states if s.tostring() != generator.eos],
                "alphabet": list(generator.alphabet),
                "transitions": [
                    [source, target, float(probability)]
                    for source, target, probability in generator.transitions
                ],
                "start_states": [s.tostring() for s in generator.start_states],
                "terminal_states": [s.tostring() for s in generator.terminal_states],
            },
        }
    if rung.kind == "nax":
        return {
            "type": "nAX",
            "params": {
                "label": generator.label,
                "contexts": list(generator.contexts),
                "cue_map": dict(generator.cue_map),
                "probe_map": dict(generator.probe_map),
                "fillers": list(generator.fillers),
                "p_target": float(generator.p_target),
            },
        }
    if rung.kind == "nad":
        return {
            "type": "NonAdjacentDependencies",
            "params": {
                "label": generator.label,
                "n_deps": generator.n_deps,
                "n_unique_fillers": generator.n_unique_fillers,
                "verbose": False,
            },
            "trial_params": dict(rung.sample),
        }
    if rung.kind in ("nested", "crossed"):
        return {
            "type": type(generator).__name__,
            "params": {"label": generator.label, "n_deps": generator.n_deps, "verbose": False},
        }
    if rung.kind == "dyck":
        return {
            "type": "Dyck",
            "params": {
                "k": generator.k,
                "mode": generator.mode,
                "target_pairs": generator.target_pairs,
                "max_depth": generator.max_depth,
                "n_distractors": 0,
            },
        }
    raise KeyError(f"No SeqBench configuration for kind {rung.kind!r}.")


def seqbench_config(
    rung: Rung, n_samples: int, seed: int, sample_length: int, storage: str | Path = CACHE
) -> dict:
    """The SeqBench configuration of one dataset: generator, composition of the samples, input, task."""
    generator = build_generator(rung)
    if rung.sample and rung.kind != "nad":
        raise NotImplementedError(f"Sampling arguments {sorted(rung.sample)} are not passed on.")
    return {
        "run": {"seed": 0},
        "symbol_space": {
            "alphabet": {"size": len(generator.alphabet), "symbols": list(generator.alphabet)},
            "eos": EOS,
        },
        "symseq": {"generator": generator_config(rung)},
        "seqbench": {
            "seed": seed,
            "mode": "file",
            "splits": {"train": n_samples, "test": n_samples},
            "time_grid": {"dt": 1.0},
            "storage": {"path": str(storage), "force_rebuild": True},
            "composition": {"combine_sequences": True, "sample_length": sample_length},
            "input_mapping": {
                "base": "one_hot",
                "base_params": {"duration": 1.0},
                "transforms": [],
            },
            "task": {
                "source": "seqbench",
                "id": "next_symbol",
                "type": "NStepPrediction",
                "params": {"n": 1},
            },
        },
    }


def config_yaml(config: dict) -> str:
    """The configuration as the YAML text SeqBench reads from a file."""
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=None, width=100)


def signature(generator, sample: dict | None = None) -> dict:
    """What fixes the distribution of a generator's strings, for comparing two generators."""
    described = {"type": type(generator).__name__, "alphabet": list(generator.alphabet)}
    if hasattr(generator, "transitions"):
        described["transitions"] = sorted(
            (source, target, round(float(probability), 12))
            for source, target, probability in generator.transitions
        )
        described["start"] = sorted(
            (state, round(float(p), 12)) for state, p in generator.start_probabilities.items()
        )
    if hasattr(generator, "dependency_pairs"):
        described["pairs"] = [tuple(pair) for pair in generator.dependency_pairs]
        described["fillers"] = list(getattr(generator, "fillers", ()))
    if hasattr(generator, "parentheses"):
        described["brackets"] = dict(generator.parentheses)
        described["dyck"] = (generator.mode, generator.target_pairs, generator.max_depth)
        described["distractors"] = generator.n_distractors
    described["sample"] = dict(sample or {})
    return described


# --- importing SeqBench ---------------------------------------------------------------------------

# SeqBench imports its audio transforms with everything else, and each of the two logs one warning
# when torchaudio is absent. The tutorial uses no audio and does not install torchaudio.
TORCHAUDIO_WARNINGS = {
    "seqbench.transforms.audio.mel": "LogMel not loaded. torchaudio not installed?",
    "seqbench.transforms.audio.mfcc": "MFCC not loaded. torchaudio not installed?",
}


def not_a_torchaudio_warning(record: logging.LogRecord) -> bool:
    """A logging filter: false for the two records of ``TORCHAUDIO_WARNINGS``, true for every other."""
    return record.getMessage() != TORCHAUDIO_WARNINGS.get(record.name)


def seqbench_dataloader():
    """SeqBench's ``build_dataloader``, imported without its two warnings that torchaudio is absent.

    The two records are dropped at the loggers that emit them, and only while SeqBench is being
    imported. Nothing else is silenced: the warnings go through ``logging`` to the standard error,
    and redirecting that stream around the import would also hide every other warning or message
    the import may give. Neither stream is redirected here.
    """
    loggers = [logging.getLogger(name) for name in TORCHAUDIO_WARNINGS]
    for logger in loggers:
        logger.addFilter(not_a_torchaudio_warning)
    try:
        from seqbench import build_dataloader
    finally:
        for logger in loggers:
            logger.removeFilter(not_a_torchaudio_warning)
    return build_dataloader


# --- reading the loader once ----------------------------------------------------------------------


def quiet(function):
    """Call a function with SeqBench's progress output discarded."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return function()


@dataclass(frozen=True, eq=False)
class Split:
    """The tensors of one dataset.

    ``inputs``: (samples, sample length, classes), one-hot, as SeqBench built them.
    ``targets``: (samples, sample length), class of the symbol to predict, ``PAD`` where none.
    """

    inputs: torch.Tensor
    targets: torch.Tensor

    @property
    def tokens(self) -> torch.Tensor:
        """Class of every input symbol, (samples, sample length)."""
        return self.inputs.argmax(dim=-1)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def keep(self, rows: torch.Tensor) -> Split:
        return Split(self.inputs[rows], self.targets[rows])


def stream_targets(tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """SeqBench's labels plus, at every end marker, the first symbol of the next string."""
    targets = labels.clone()
    at_marker = tokens[:, :-1] == EOS_INDEX
    targets[:, :-1][at_marker] = tokens[:, 1:][at_marker]
    return targets


def build_split(
    rung: Rung,
    n_samples: int,
    seed: int,
    sample_length: int,
    split: str = "train",
    targets: str = "seqbench",
) -> tuple[Split, tuple[str, ...], dict]:
    """One SeqBench dataset of ``n_samples`` samples, read once from ``build_dataloader``.

    Returns the tensors, the symbol of every class (class 0 is the end marker) and the
    configuration. Sample 0 of the SeqBench dataset is dropped (see the module docstring), so
    ``n_samples + 1`` samples are requested. The function raises if SeqBench built another grammar
    than ``tutorial.ladder.build_generator(rung)``, if an input row is not one-hot, or if a label is
    not the next symbol.
    """
    build_dataloader = seqbench_dataloader()
    if not 1 <= n_samples <= MAX_SAMPLES:
        raise ValueError(f"n_samples must be between 1 and {MAX_SAMPLES}, got {n_samples}.")
    if targets not in ("seqbench", "stream"):
        raise ValueError(f"targets must be 'seqbench' or 'stream', got {targets!r}.")
    # one folder per call, removed afterwards: concurrent processes never share files
    storage = CACHE / uuid.uuid4().hex
    config = seqbench_config(rung, n_samples + 1, seed, sample_length, storage)
    try:
        loader = quiet(lambda: build_dataloader(config, split=split, batch_size=256, shuffle=False))
        batches = quiet(lambda: list(loader))
    finally:
        shutil.rmtree(storage, ignore_errors=True)

    source = getattr(loader.dataset.generator, "source", loader.dataset.generator)
    expected = signature(build_generator(rung), rung.sample)
    if signature(source, rung.sample) != expected:
        raise ValueError(f"SeqBench built another grammar than the rung {rung.name!r}.")
    symbols = (EOS, *source.alphabet)

    inputs = torch.cat([batch["data"] for batch in batches])[1:]
    labels = torch.cat([batch["labels"] for batch in batches])[1:]
    if inputs.shape != (n_samples, sample_length, len(symbols)):
        raise ValueError(f"Unexpected shape of the SeqBench inputs: {tuple(inputs.shape)}.")
    if not bool(((inputs == 0) | (inputs == 1)).all() and (inputs.sum(dim=-1) == 1).all()):
        raise ValueError("A row of the SeqBench inputs is not one-hot.")
    tokens = inputs.argmax(dim=-1)
    at_marker = tokens == EOS_INDEX
    if not bool((labels[at_marker] == PAD).all() and (labels[~at_marker] != PAD).all()):
        raise ValueError(
            "SeqBench's labels are not: a target after every symbol, none at a marker."
        )
    inside = ~at_marker[:, :-1]
    if not bool((labels[:, :-1][inside] == tokens[:, 1:][inside]).all()):
        raise ValueError("A SeqBench label is not the next symbol of the sample.")
    if targets == "stream":
        labels = stream_targets(tokens, labels)
    return Split(inputs, labels), symbols, config


# --- training and held-out data of a rung ---------------------------------------------------------


def shared_rows(split: Split, other: Split) -> torch.Tensor:
    """For every sample of ``split``: True if its symbols equal those of a sample of ``other``."""
    known = {row.tobytes() for row in other.tokens.numpy()}
    return torch.tensor([row.tobytes() in known for row in split.tokens.numpy()], dtype=torch.bool)


def assert_disjoint(train: Split, heldout: Split) -> None:
    """Raise if a held-out sample holds the same symbols as a training sample."""
    shared = int(shared_rows(heldout, train).sum())
    if shared:
        raise ValueError(f"{shared} held-out samples are also training samples.")


def optimum(rung: Rung, symbols: tuple[str, ...], split: Split) -> dict:
    """What a predictor that knows the grammar scores on the targets of these samples.

    At every target the generator's distribution of the next symbol given the symbols so far is
    read from ``tutorial.exact`` (``follow``). The best prediction is its most probable outcome.

    - ``accuracy_with_eos``: mean of the largest probability over all targets that follow a symbol.
      ``accuracy_within``: over the targets that are symbols, the largest probability among symbols
      given that the string goes on. These are the expected accuracies of the best predictor on these
      samples; ``variance_*`` is the variance of its number of correct predictions.
    - ``cross_entropy``: mean entropy of the distribution, in bits per target. No predictor has a
      smaller expected cross-entropy. ``cross_entropy_realised`` is the mean of ``-log2`` of the
      probability of the symbol that came, the cross-entropy this predictor obtains on these samples.

    The function raises if a symbol of the samples has probability zero under the rung's generator.
    """
    predictor = exact.predictor_for(build_generator(rung), **rung.sample)
    sums = dict.fromkeys(
        (
            "n_targets",
            "entropy",
            "bits",
            "n_with_eos",
            "hits_with_eos",
            "variance_with_eos",
            "n_within",
            "hits_within",
            "variance_within",
        ),
        0.0,
    )
    for tokens, targets in zip(split.tokens.tolist(), split.targets.tolist(), strict=True):
        distributions = exact.follow(predictor, [symbols[token] for token in tokens])
        for token, target, distribution in zip(tokens, targets, distributions, strict=True):
            if target == PAD:
                continue
            probability = distribution.get(symbols[target], 0.0)
            if probability <= 0.0:
                raise ValueError(f"Target {symbols[target]!r} cannot follow its prefix.")
            sums["n_targets"] += 1
            sums["entropy"] -= sum(p * math.log2(p) for p in distribution.values() if p > 0.0)
            sums["bits"] -= math.log2(probability)
            if token == EOS_INDEX:
                continue  # first symbol of a string: a target only with targets="stream"
            best = max(distribution.values())
            sums["n_with_eos"] += 1
            sums["hits_with_eos"] += best
            sums["variance_with_eos"] += best * (1.0 - best)
            if target != EOS_INDEX:
                goes_on = 1.0 - distribution.get(predictor.eos, 0.0)
                best = max(p for s, p in distribution.items() if s != predictor.eos) / goes_on
                sums["n_within"] += 1
                sums["hits_within"] += best
                sums["variance_within"] += best * (1.0 - best)
    return {
        "n_targets": int(sums["n_targets"]),
        "cross_entropy": sums["entropy"] / sums["n_targets"],
        "cross_entropy_realised": sums["bits"] / sums["n_targets"],
        "n_with_eos": int(sums["n_with_eos"]),
        "accuracy_with_eos": sums["hits_with_eos"] / sums["n_with_eos"],
        "variance_with_eos": sums["variance_with_eos"],
        "n_within": int(sums["n_within"]),
        "accuracy_within": sums["hits_within"] / sums["n_within"],
        "variance_within": sums["variance_within"],
    }


@dataclass(frozen=True, eq=False)
class RungData:
    """Training and held-out data of one rung for one training seed."""

    rung: Rung
    symbols: tuple[str, ...]  # symbol of every class; class 0 is the end marker
    train: Split
    heldout: Split
    optimum: dict  # the exact predictor on the held-out targets, see ``optimum``
    n_train_removed: int  # training samples removed because a held-out sample has the same symbols
    seeds: dict  # the two seqbench.seed values
    configs: dict  # the two SeqBench configurations
    seconds: float  # time to build

    def decode(self, tokens) -> list[str]:
        """Classes back to symbols; ``PAD`` becomes a dot."""
        return [self.symbols[token] if token != PAD else "." for token in tokens]


def build_rung_data(rung: Rung, seed: int, settings: Hyperparameters = PRODUCTION) -> RungData:
    """Training data for one training seed and the held-out data of the rung, with no sample in common."""
    start = time.perf_counter()
    seeds = {"train": stream_seed(seed + 1), "heldout": stream_seed(HELDOUT_STREAM)}
    assert_disjoint_streams(seeds["train"], settings.n_train, seeds["heldout"], settings.n_heldout)
    train, symbols, train_config = build_split(
        rung, settings.n_train, seeds["train"], settings.sample_length, "train", settings.targets
    )
    heldout, heldout_symbols, heldout_config = build_split(
        rung, settings.n_heldout, seeds["heldout"], settings.sample_length, "test", settings.targets
    )
    if heldout_symbols != symbols:
        raise ValueError("The two datasets do not share one alphabet.")
    duplicated = shared_rows(train, heldout)
    train = train.keep(~duplicated)
    assert_disjoint(train, heldout)
    return RungData(
        rung=rung,
        symbols=symbols,
        train=train,
        heldout=heldout,
        optimum=optimum(rung, symbols, heldout),
        n_train_removed=int(duplicated.sum()),
        seeds=seeds,
        configs={"train": train_config, "heldout": heldout_config},
        seconds=time.perf_counter() - start,
    )


def preview(rung: Rung, seed: int, n_samples: int = 2, sample_length: int = 40) -> tuple[str, str]:
    """The first training samples of a seed as text, and the configuration that builds them as YAML.

    Each sample is shown as two lines: the symbols the network reads, and under each symbol the
    target at that position (a dot where there is none).
    """
    split, symbols, config = build_split(rung, n_samples, stream_seed(seed + 1), sample_length)
    width = max(len(symbol) for symbol in symbols)
    lines = []
    for tokens, targets in zip(split.tokens.tolist(), split.targets.tolist(), strict=True):
        lines.append("input   " + " ".join(symbols[token].ljust(width) for token in tokens))
        lines.append(
            "target  " + " ".join((symbols[t] if t != PAD else ".").ljust(width) for t in targets)
        )
        lines.append("")
    config["seqbench"]["storage"]["path"] = "data/seqbench_cache"
    return "\n".join(lines).rstrip(), config_yaml(config)
