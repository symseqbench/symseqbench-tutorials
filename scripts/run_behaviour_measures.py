"""Compute the measures of notebook B for the four datasets and their surrogates, and save them.

Writes one JSON file per dataset to ``data/behaviour_measures/`` and ``manifest.json`` with the
SHA-256 of each. Every setting and every seed is in ``tutorial.behaviour_measures.SETTINGS``, so two
runs give the same bytes. A dataset whose result exists, computed from the same table with the same
settings, is not computed again. Wall times and the speed probe go to ``timings.json``, which is not
part of the results.

Usage:  uv run python scripts/run_behaviour_measures.py
        uv run python scripts/run_behaviour_measures.py --datasets seal turtle --out some/folder
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread, set before numpy is loaded
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
import timeit
from pathlib import Path

from tutorial import behaviour
from tutorial import behaviour_measures as bm


def speed_probe() -> float:
    """Microseconds for ``sum(range(1000))``: about 10 on an idle core of the build workstation."""
    return timeit.timeit(lambda: sum(range(1000)), number=2000) / 2000 * 1e6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=bm.RESULTS)
    parser.add_argument(
        "--datasets", nargs="+", choices=list(behaviour.DATASETS), default=list(behaviour.DATASETS)
    )
    parser.add_argument(
        "--force", action="store_true", help="compute again even if a result is saved"
    )
    arguments = parser.parse_args()
    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)
    behaviour.check_folder()
    tables = behaviour.read_manifest()
    timings_path = out / bm.TIMINGS
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}

    for dataset in arguments.datasets:
        digest = tables[f"{dataset}.csv.gz"]
        if not arguments.force and bm.saved_result(dataset, digest, bm.SETTINGS, out) is not None:
            print(f"{dataset}: saved result kept")
            continue
        strings = behaviour.load_strings(dataset).strings
        before, start = speed_probe(), time.perf_counter()
        result = bm.measure_dataset(strings, bm.SETTINGS)
        seconds = time.perf_counter() - start
        bm.write_result(dataset, result, digest, out)
        timings[dataset] = {
            "seconds": round(seconds, 1),
            "speed_probe_us": [round(before, 1), round(speed_probe(), 1)],
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        timings_path.write_text(json.dumps(timings, indent=1))
        print(
            f"{dataset}: {result['n_strings']} strings, {result['n_tokens']} symbols, {seconds:.0f} s, "
            f"checks {'passed' if result['checks']['passed'] else 'FAILED'}"
        )

    manifest = bm.collect(out, bm.SETTINGS)
    print(f"written {out}: {len(manifest['files'])} of {len(manifest['datasets'])} datasets")
    if manifest["missing"]:
        print(f"missing: {manifest['missing']}")
    if manifest["checks_failed"]:
        print(f"known answers of the surrogates NOT met for: {manifest['checks_failed']}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
