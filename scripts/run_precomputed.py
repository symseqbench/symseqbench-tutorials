"""Train both models on every candidate grammar and save one file per run.

Every setting comes from ``tutorial/config.py``: the grammars (the six of the figure and the five
other candidates), the two models, the seeds and the hyperparameters. One job is one grammar and
one seed: its data are built once and both models are trained on them. A run that is already saved
with the same settings is not repeated, so the script can be stopped and started again.

    uv run python scripts/run_precomputed.py --smoke            short runs -> data/precomputed_smoke
    uv run python scripts/run_precomputed.py --production       the runs of the figure -> data/precomputed
    uv run python scripts/run_precomputed.py --production --shard 2/4     the second of four parts
    uv run python scripts/run_precomputed.py --production --collect       summary and manifest only

Smoke runs use the exploration seeds and a small budget; their results are placeholders. Several
parts can run side by side, one process each, one thread each; a part does not write the summary
and the manifest, ``--collect`` does, after all parts are done.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# One thread per process. The limits must be set before numpy and torch are imported.
THREADS = 1
for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = str(THREADS)

from tutorial import config, datasets, results, training  # noqa: E402


def parse_shard(text: str) -> tuple[int, int]:
    part, _, parts = text.partition("/")
    if not (part.isdigit() and parts.isdigit() and 1 <= int(part) <= int(parts)):
        raise argparse.ArgumentTypeError(
            f"expected PART/PARTS with 1 <= PART <= PARTS, got {text!r}"
        )
    return int(part), int(parts)


def jobs(kind: str, grammars: list[str] | None) -> list[tuple[str, int]]:
    """Every (grammar, seed) of the kind, in a fixed order."""
    names = [*config.LADDER, *config.ALTERNATIVES]
    if grammars is not None:
        unknown = [name for name in grammars if name not in names]
        if unknown:
            raise SystemExit(f"Unknown grammars {unknown}. Known: {names}")
        names = [name for name in names if name in grammars]
    return [(name, seed) for name in names for seed in config.SEEDS[kind]]


def run_job(name: str, seed: int, kind: str, folder: Path) -> list[dict]:
    """Train the models of one grammar and seed that are not saved yet. Returns the new runs."""
    settings = config.SETTINGS[kind]
    todo = [
        model
        for model in config.MODELS
        if results.saved_run(folder, kind, name, model, seed, settings.as_dict()) is None
    ]
    if not todo:
        return []
    rung = config.CANDIDATES[name]
    data = datasets.build_rung_data(rung, seed, settings)
    new = []
    for model in todo:
        run = training.train_run(rung, model, seed, settings, data=data)
        results.write_run(run, folder, kind)
        new.append(run)
        final, optimum = run["final"], run["optimum"]
        print(
            f"{name:22s} seed {seed:3d}  {model:18s} {run['n_parameters']:6d} parameters  "
            f"accuracy {final['accuracy_with_eos']:.3f} (best {optimum['accuracy_with_eos']:.3f})  "
            f"cross-entropy {final['cross_entropy']:.3f} (best {optimum['cross_entropy']:.3f})  "
            f"data {run['seconds']['data']:5.1f} s  training {run['seconds']['training']:6.1f} s  "
            f"checks {'passed' if run['checks']['passed'] else 'FAILED'}",
            flush=True,
        )
    return new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        allow_abbrev=False,
    )
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--smoke", action="store_true", help="short placeholder runs")
    kind.add_argument("--production", action="store_true", help="the runs of the figure")
    parser.add_argument("--shard", type=parse_shard, default=None, metavar="PART/PARTS")
    parser.add_argument("--collect", action="store_true", help="only write summary and manifest")
    parser.add_argument("--grammars", nargs="+", default=None, help="only these grammars")
    parser.add_argument("--out", type=Path, default=None, help="results folder (default: by kind)")
    arguments = parser.parse_args(argv)

    kind_name = "smoke" if arguments.smoke else "production"
    folder = arguments.out if arguments.out is not None else results.FOLDERS[kind_name]
    settings = config.SETTINGS[kind_name]
    if settings.threads != THREADS:
        raise SystemExit(f"config asks for {settings.threads} threads; this script sets {THREADS}.")

    if not arguments.collect:
        todo = jobs(kind_name, arguments.grammars)
        if arguments.shard is not None:
            part, parts = arguments.shard
            todo = todo[part - 1 :: parts]
        start = time.perf_counter()
        new = []
        for name, seed in todo:
            new.extend(run_job(name, seed, kind_name, folder))
        print(
            f"{len(new)} runs computed in {time.perf_counter() - start:.0f} s "
            f"({len(todo) * len(config.MODELS) - len(new)} already saved); folder {folder}"
        )
        if arguments.shard is not None:
            print("This was one part. Run with --collect when all parts are done.")
            return 0

    manifest = results.collect(folder, kind_name)
    print(
        f"{manifest['present']} of {manifest['expected']} runs present, "
        f"{len(manifest['missing'])} missing, {len(manifest['checks_failed'])} with failed checks; "
        f"summary and manifest written in {folder}"
    )
    return 0 if manifest["complete"] and not manifest["checks_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
