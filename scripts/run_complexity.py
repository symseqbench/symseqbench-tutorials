"""Draw grammars of controlled complexity, train both networks on them and save one file per run.

Every setting comes from ``tutorial/config.py`` (``COMPLEXITY`` and the settings derived from it).
A run is one grammar, one network, one number of training strings and one training seed. One job
is one grammar and one training seed: its strings are drawn once and all its runs use them. A run
that is already saved with the same settings is not repeated, so the script can be stopped and
started again, and identical seeds give identical results.

    uv run python scripts/run_complexity.py --pilot                 pilot runs -> data/complexity_pilot
    uv run python scripts/run_complexity.py --production --draw     draw and admit the grammars only
    uv run python scripts/run_complexity.py --production --shard 1/2   the first of two parts
    uv run python scripts/run_complexity.py --production --collect  summary and manifest only
    uv run python scripts/run_complexity.py --smoke --out FOLDER    a few short runs that exercise the code
    uv run python scripts/run_complexity.py --pilot --training-sample draws --out FOLDER
                                                                    the pilot with the other training sample

Without ``--draw``, ``--shard`` or ``--collect`` the script does all three steps in one process.
A job that raises is recorded in ``failed/`` of the folder and the next job is run; the part then
ends with exit code 1, and the collect step reports the folder as incomplete, with the job, the
text of its error and the runs that are missing.
The pilot uses grammar seeds and training seeds from 101 on; its results are placeholders. The
production grammars are drawn with seeds from 1 on when ``--production`` is first run, and their
admission is decided by the written rule alone. Several parts can run side by side, one process
each, one thread each; they need the grammars to be drawn first, and ``--collect`` afterwards.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
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

from tutorial import complexity_runs as runs  # noqa: E402
from tutorial import config  # noqa: E402
from tutorial.complexity import TRAINING_SAMPLES  # noqa: E402


def parse_shard(text: str) -> tuple[int, int]:
    part, _, parts = text.partition("/")
    if not (part.isdigit() and parts.isdigit() and 1 <= int(part) <= int(parts)):
        raise argparse.ArgumentTypeError(
            f"expected PART/PARTS with 1 <= PART <= PARTS, got {text!r}"
        )
    return int(part), int(parts)


def jobs(drawn: dict, settings: config.ComplexitySettings) -> list[tuple]:
    """Every (grammar, values of its admission, training seed), in a fixed order.

    The order goes through the levels up and then down in turn, so that every part of a sharded
    run holds grammars of every level and the parts take a similar time.
    """
    by_level: dict[int, list] = {}
    for spec, values in runs.admitted_grammars(drawn):
        by_level.setdefault(spec.level, []).append((spec, values))
    ordered, group = [], 0
    for seed in settings.seeds:
        for rank in range(max((len(entries) for entries in by_level.values()), default=0)):
            levels = sorted(by_level, reverse=bool(group % 2))
            ordered.extend(
                (*by_level[level][rank], seed) for level in levels if rank < len(by_level[level])
            )
            group += 1
    return ordered


def run_job(spec, values, seed: int, kind: str, folder: Path, settings) -> list[dict]:
    """Train the runs of one grammar and seed that are not saved yet. Returns the new runs.

    A job that raises does not end the part. Which job it was and the text of the error are
    printed and written into ``failed/`` of the folder (``complexity_runs.record_failure``), the
    runs it did not reach stay missing, and the next job is run. A job that runs through removes
    its record.
    """
    todo = [
        (model, n_train)
        for n_train in settings.sizes
        for model in config.MODELS
        if runs.saved_run(folder, kind, spec, model, n_train, seed, settings) is None
    ]
    new = []
    try:
        data = runs.prepare(spec, values, seed, settings) if todo else None
        for model, n_train in todo:
            run = runs.train_run(data, model, n_train, settings)
            runs.write_run(run, folder, kind)
            new.append(run)
            found, first = run["measures"], run["first_order"]
            print(
                f"{spec.name:22s} seed {seed:3d}  {n_train:4d} strings  {model:18s} "
                f"score {found['score']:.3f}  area {found['auc']:.3f} "
                f"(first-order {first['score']:.3f}, {first['auc']:.3f})  "
                f"divergence {found['divergence_bits']:.4f}  step {run['best_step']:5d}  "
                f"training {run['seconds']['training']:6.1f} s  "
                f"probe {max(run['environment']['speed_probe_us']):5.1f} us  "
                f"checks {'passed' if run['checks']['passed'] else 'FAILED'}",
                flush=True,
            )
    except Exception as error:  # whatever it is: recorded, and the part goes on
        path = runs.record_failure(folder, kind, spec, seed, error)
        print(
            f"{spec.name:22s} seed {seed:3d}  JOB FAILED: {type(error).__name__}: {error} "
            f"(recorded in {path}; {len(todo) - len(new)} of its runs are missing)",
            flush=True,
        )
        return new
    runs.clear_failure(folder, spec, seed)
    return new


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--pilot", action="store_true", help="pilot runs on seeds from 101 on")
    kind.add_argument("--production", action="store_true", help="the runs of the figure")
    kind.add_argument("--smoke", action="store_true", help="a few short runs; needs --out")
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--draw", action="store_true", help="only draw and admit the grammars")
    step.add_argument("--shard", type=parse_shard, default=None, metavar="PART/PARTS")
    step.add_argument("--collect", action="store_true", help="only write summary and manifest")
    parser.add_argument("--out", type=Path, default=None, help="results folder (default: by kind)")
    parser.add_argument(
        "--training-sample",
        choices=TRAINING_SAMPLES,
        default=None,
        help="use this training sample instead of the one in config; needs --out",
    )
    arguments = parser.parse_args(argv)

    kind_name = "pilot" if arguments.pilot else "production" if arguments.production else "smoke"
    settings = config.COMPLEXITY_SETTINGS[kind_name]
    if arguments.out is None and kind_name not in runs.FOLDERS:
        raise SystemExit("--smoke writes nothing into the repository's data: give --out FOLDER.")
    if arguments.training_sample is not None:
        if arguments.out is None:
            raise SystemExit(
                "--training-sample departs from tutorial/config.py: give --out FOLDER, so that the "
                "folders of the repository hold only runs made with the configured settings."
            )
        settings = replace(settings, training_sample=arguments.training_sample)
    folder = arguments.out if arguments.out is not None else runs.FOLDERS[kind_name]
    if config.PRODUCTION.threads != THREADS:
        raise SystemExit(
            f"config asks for {config.PRODUCTION.threads} threads; this script sets {THREADS}."
        )

    everything = not (arguments.draw or arguments.shard or arguments.collect)
    grammars_saved = (Path(folder) / "grammars.json").exists()
    if arguments.draw or (everything and not grammars_saved):
        if grammars_saved:
            runs.load_grammars(folder, kind_name, settings)  # raises if drawn with other settings
            print(f"The grammars in {folder} are already drawn with these settings; nothing drawn.")
        else:
            start = time.perf_counter()
            drawn = runs.write_grammars(
                folder, kind_name, settings, report=lambda line: print(line, flush=True)
            )
            print(runs.admission_table(drawn).to_string())
            print(f"grammars drawn in {time.perf_counter() - start:.0f} s; folder {folder}")
        if arguments.draw:
            return 0

    drawn = runs.load_grammars(folder, kind_name, settings)
    if not arguments.collect:
        todo = jobs(drawn, settings)
        if arguments.shard is not None:
            part, parts = arguments.shard
            todo = todo[part - 1 :: parts]
        start = time.perf_counter()
        new = []
        for spec, values, seed in todo:
            new.extend(run_job(spec, values, seed, kind_name, folder, settings))
        per_job = len(settings.sizes) * len(config.MODELS)
        print(
            f"{len(new)} runs computed in {time.perf_counter() - start:.0f} s "
            f"({len(todo) * per_job - len(new)} already saved); folder {folder}"
        )
        names = {runs.job_name(spec, seed) for spec, _, seed in todo}
        failed = [record for record in runs.failed_jobs(folder) if record["job"] in names]
        for record in failed:
            print(f"FAILED JOB {record['job']}: {record['error']}")
        if arguments.shard is not None:
            print("This was one part. Run with --collect when all parts are done.")
            return 1 if failed else 0

    manifest = runs.collect(folder, kind_name, settings)
    print(
        f"{manifest['present']} of {manifest['expected']} runs present, "
        f"{len(manifest['missing'])} missing, {len(manifest['checks_failed'])} with failed checks, "
        f"{len(manifest['failed_jobs'])} failed jobs, "
        f"levels short of grammars: {manifest['levels_short_of_grammars'] or 'none'}; "
        f"summary and manifest written in {folder}"
    )
    for record in manifest["failed_jobs"]:
        print(f"FAILED JOB {record['job']}: {record['error']}")
    if manifest["missing"]:
        print("missing runs: " + ", ".join(manifest["missing"]))
    return 0 if manifest["complete"] and not manifest["checks_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
