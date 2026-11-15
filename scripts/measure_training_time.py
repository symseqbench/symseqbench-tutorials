"""Measure the wall time to train a small tanh RNN for next-symbol prediction on one SeqBench dataset.

One measurement for the planned networks notebook, with every setting fixed before the run and no
tuning: the regular grammar of the ladder (6 states, 4 symbols), datasets built with SeqBench's
``build_dataloader`` (file mode, one-hot input, one row per symbol, samples of 40 symbols made of
joined strings), a ``torch.nn.RNN`` with tanh units and about 20 000 parameters, Adam with learning
rate 1e-3, batches of 32, 5 epochs, 2 threads. The time is split into the data pipeline (iterating
the SeqBench loader) and the network (forward, backward, update), and repeated for several dataset
sizes. Nothing else of the networks notebook is built here.

The accuracy on held-out samples is reported next to the exact ceiling of those samples, as a check
that the timed loop trains. SeqBench generates sample i of every split from ``default_rng(seed * i)``,
so the test split is built with another ``seqbench.seed`` and its sample 0 (seed 0 in every split)
is dropped; see README.

Usage:  uv run python scripts/measure_training_time.py --out scripts/output/training_time.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import time
import timeit
from pathlib import Path

# Two threads at most. The limits must be set before numpy and torch are imported.
for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "2"

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tutorial import exact, generate  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TRAIN_SEED = 1
TEST_SEED = (
    1_000_003  # larger than any training set, so seed * i never meets a training seed for i >= 1
)


def speed_probe() -> float:
    """Microseconds for sum(range(1000)); about 10 on an idle core of a recent laptop."""
    return timeit.timeit(lambda: sum(range(1000)), number=2000) / 2000 * 1e6


def machine_state() -> dict:
    with open("/proc/loadavg") as handle:
        load = handle.read().split()[:3]
    frequencies = []
    for cpu in sorted(os.sched_getaffinity(0)):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
        if path.exists():
            frequencies.append(int(path.read_text()) / 1000.0)
    return {
        "speed_probe_us": round(speed_probe(), 1),
        "loadavg": [float(value) for value in load],
        "mean_cpu_mhz": round(float(np.mean(frequencies)), 0) if frequencies else None,
        "cpus_allowed": sorted(os.sched_getaffinity(0)),
    }


def seqbench_config(
    grammar, n_train: int, n_test: int, sample_length: int, seed: int, cache: Path
) -> dict:
    """SeqBench configuration for an explicit SymSeq grammar, next-symbol prediction, one row per symbol."""
    return {
        "run": {"seed": 0},
        "symbol_space": {
            "alphabet": {"size": len(grammar.alphabet), "symbols": list(grammar.alphabet)},
            "eos": "#",
        },
        "symseq": {
            "generator": {
                "type": "ArtificialGrammar",
                "mode": "custom",
                "params": {
                    "label": grammar.label,
                    "states": [s.tostring() for s in grammar.states if s.tostring() != grammar.eos],
                    "alphabet": list(grammar.alphabet),
                    "transitions": [list(transition) for transition in grammar.transitions],
                    "start_states": [s.tostring() for s in grammar.start_states],
                    "terminal_states": [s.tostring() for s in grammar.terminal_states],
                },
            },
        },
        "seqbench": {
            "seed": seed,
            "mode": "file",
            "splits": {"train": n_train, "test": n_test},
            "time_grid": {"dt": 1.0},
            "storage": {"path": str(cache), "force_rebuild": True},
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


def quiet(function):
    """Call a function with SeqBench's progress output discarded."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return function()


class TanhRNN(torch.nn.Module):
    def __init__(self, n_classes: int, hidden: int):
        super().__init__()
        self.rnn = torch.nn.RNN(n_classes, hidden, nonlinearity="tanh", batch_first=True)
        self.readout = torch.nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.readout(self.rnn(x)[0])


def dataset_ceiling(grammar, class_sequences, labels, pad: int = -1) -> float:
    """Exact ceiling on the scored positions of these samples: mean of the largest next-outcome probability."""
    predictor = exact.GrammarPredictor(grammar)
    symbol_of = {
        0: grammar.eos,
        **{index + 1: symbol for index, symbol in enumerate(grammar.alphabet)},
    }
    total = count = 0.0
    for classes, targets in zip(class_sequences, labels, strict=True):
        state = predictor.initial()
        for class_id, target in zip(classes, targets, strict=True):
            symbol = symbol_of[int(class_id)]
            if symbol == grammar.eos:
                state = predictor.initial()
                continue
            state = predictor.advance(state, symbol)
            if int(target) != pad:
                total += max(predictor.distribution(state).values())
                count += 1
    return total / count


def measure(grammar, n_train: int, arguments) -> dict:
    from seqbench import build_dataloader

    cache = REPO / "data" / "seqbench_cache"
    state_before = machine_state()

    start = time.perf_counter()
    train_config = seqbench_config(
        grammar, n_train, arguments.n_test, arguments.sample_length, TRAIN_SEED, cache
    )
    train_loader = quiet(
        lambda: build_dataloader(
            train_config, split="train", batch_size=arguments.batch_size, shuffle=True
        )
    )
    build_s = time.perf_counter() - start
    test_config = seqbench_config(
        grammar, n_train, arguments.n_test, arguments.sample_length, TEST_SEED, cache
    )
    test_loader = quiet(
        lambda: build_dataloader(
            test_config, split="test", batch_size=arguments.batch_size, shuffle=False
        )
    )

    n_classes = train_loader.dataset.num_classes
    torch.manual_seed(arguments.seed)
    model = TanhRNN(n_classes, arguments.hidden)
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

    epochs = []
    for _ in range(arguments.epochs):
        data_s = network_s = 0.0
        epoch_start = time.perf_counter()
        cpu_start = time.process_time()
        iterator = iter(train_loader)
        while True:
            tick = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                data_s += time.perf_counter() - tick
                break
            tock = time.perf_counter()
            data_s += tock - tick
            logits = model(batch["data"])
            loss = loss_function(logits.reshape(-1, n_classes), batch["labels"].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            network_s += time.perf_counter() - tock
        epochs.append(
            {
                "wall_s": time.perf_counter() - epoch_start,
                "cpu_s": time.process_time() - cpu_start,
                "data_s": data_s,
                "network_s": network_s,
                "last_loss": float(loss.detach()),
            }
        )

    # held-out accuracy, with sample 0 of the test split dropped (it has seed 0, like training sample 0)
    train_samples = {
        tuple(int(c) for c in sample.class_seq) for sample in train_loader.dataset.buffered_samples
    }
    hits = scored = duplicates = 0
    classes, targets = [], []
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            keep = slice(1, None) if index == 0 else slice(None)
            data, labels = batch["data"][keep], batch["labels"][keep]
            prediction = model(data).argmax(-1)
            mask = labels != -1
            hits += int((prediction[mask] == labels[mask]).sum())
            scored += int(mask.sum())
            for row, label_row in zip(data.argmax(-1).tolist(), labels.tolist(), strict=True):
                duplicates += tuple(row) in train_samples
                classes.append(row)
                targets.append(label_row)
    ceiling = dataset_ceiling(grammar, classes, targets)

    walls = [epoch["wall_s"] for epoch in epochs]
    return {
        "n_train": n_train,
        "n_parameters": n_parameters,
        "n_classes": n_classes,
        "build_train_s": build_s,
        "epochs": epochs,
        "epoch_wall_s_mean": float(np.mean(walls)),
        "epoch_data_s_mean": float(np.mean([epoch["data_s"] for epoch in epochs])),
        "epoch_network_s_mean": float(np.mean([epoch["network_s"] for epoch in epochs])),
        "training_wall_s": float(np.sum(walls)),
        "data_ms_per_sample": 1000
        * float(np.mean([epoch["data_s"] for epoch in epochs]))
        / n_train,
        "network_ms_per_sample": 1000
        * float(np.mean([epoch["network_s"] for epoch in epochs]))
        / n_train,
        "test_accuracy": hits / scored,
        "test_ceiling": ceiling,
        "test_positions": scored,
        "test_samples_also_in_train": duplicates,
        "machine_before": state_before,
        "machine_after": machine_state(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 500, 1000, 2000])
    parser.add_argument("--n-test", type=int, default=201)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sample-length", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=136)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=REPO / "scripts" / "output" / "training_time.json"
    )
    arguments = parser.parse_args()

    torch.set_num_threads(2)
    grammar = generate.regular_grammar(n_states=6, alphabet_size=4, transition_density=0.3, seed=0)
    record = {
        "settings": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(arguments).items()
        },
        "grammar": {"label": grammar.label, "transitions": [list(t) for t in grammar.transitions]},
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "torch_threads": torch.get_num_threads(),
        "processor": platform.processor() or platform.machine(),
        "results": [],
    }
    for n_train in arguments.sizes:
        result = measure(grammar, n_train, arguments)
        record["results"].append(result)
        print(
            f"n_train={n_train:5d}  build {result['build_train_s']:6.1f} s  epoch {result['epoch_wall_s_mean']:6.1f} s "
            f"(data {result['epoch_data_s_mean']:6.1f} s, network {result['epoch_network_s_mean']:5.1f} s)  "
            f"{arguments.epochs} epochs {result['training_wall_s']:6.1f} s  accuracy {result['test_accuracy']:.3f} "
            f"(ceiling {result['test_ceiling']:.3f})  probe {result['machine_before']['speed_probe_us']:.0f} us",
            flush=True,
        )
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(record, indent=2))
    sizes = np.array([r["n_train"] for r in record["results"]], dtype=float)
    walls = np.array([r["epoch_wall_s_mean"] for r in record["results"]])
    if len(sizes) >= 2:
        slope, intercept = np.polyfit(sizes, walls, 1)
        record["scaling"] = {
            "epoch_wall_s_per_sample": float(slope),
            "epoch_wall_s_intercept": float(intercept),
        }
        print(f"epoch wall time = {intercept:.2f} s + {1000 * slope:.2f} ms per training sample")
    arguments.out.write_text(json.dumps(record, indent=2))
    print(f"written {arguments.out}")


if __name__ == "__main__":
    main()
