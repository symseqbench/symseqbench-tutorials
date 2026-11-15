"""Build the shipped behaviour tables in ``data/behaviour/`` from the public source files.

Reads the raw files of four open datasets (never copies them), turns each into strings of bouts
and writes one table per dataset, ``SOURCES.md`` and ``MANIFEST.sha256``. The preprocessing follows
``comparative_analysis_v9.py`` of the analysis repository: strings cut at rest, at least ten bouts
per string, at most 500 strings, seed 42. The differences are listed in ``SOURCES.md``.

What a bout is depends on how the source was annotated. The turtles and the seals are annotated
observation by observation or sample by sample, so consecutive equal classes are merged, which is
what turns them into bouts. A zebrafish swim bout and a finch syllable are discrete events of the
source: two in a row of the same type are two bouts, and they are never merged.

The zebrafish file is HDF5, so this script (and nothing else in the repository) needs h5py:

    uv run --with h5py python scripts/build_behaviour_sequences.py --raw-root /path/to/animal_behavior

Expected below the raw root:
    zebrafish/Datasets/JM_data/filtered_jmpool_kin.h5 and classnames_jm.npy
    BEBE/raw_jeantet_turtles/Behaviors_*.csv
    bengalese_finch/<bird>/**/*.not.mat
    BEBE/raw_ladds_seals/<animal>/<dd-mm-yy>.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from tutorial import behaviour

REPO = Path(__file__).resolve().parents[1]

REST = "resting"
UNKNOWN = "unknown"  # class of the benchmark for labels it does not assign; kept as a symbol

# Number of zebrafish kept. Fixed before any measure was computed, by this rule: the largest
# multiple of five whose table stays under 2 MB and whose live computation fits the time budget of
# the notebook on a synthetic set of the same shape. The number was kept when the bouts were no
# longer merged; if the budget is exceeded, the number is reduced by the same rule.
ZEBRAFISH_N_FISH = 20
ZEBRAFISH_PAD = 15.0  # value that fills the unused part of each fish's row in the source file

# Classes of the Bio-logger Ethogram Benchmark for the turtle labels, copied from its public
# formatting notebook (github.com/earthspecies/BEBE-datasets, format_jeantet_turtles.ipynb).
TURTLE_CLASSES = {
    "Swimming ascent": "swimming",
    "Swimming descent": "swimming",
    "Swimming horizontally": "swimming",
    "Swimming on the bottom": "swimming",
    "Gliding ascent": "gliding",
    "Right U-turn": "swimming",
    "Stay in surface": "staying at surface",
    "Resting": "resting",
    "Left U-turn": "swimming",
    "Gliding descent": "gliding",
    "Breathing": "breathing",
    "Grabbing stationary": "feeding",
    "Swimming 1 ascent": "swimming",
    "Chewing stationary": "feeding",
    "Resting in flow": "resting",
    "Swimming 1 horizontally": "swimming",
    "Swimming 1 descent": "swimming",
    "Shaking": "unknown",
    "Foraging": "unknown",
    "Prospection": "swimming",
    "Chewing on movement": "feeding",
    "Grabbing on movement": "feeding",
    "Scratching": "scratching",
    "Shaking head": "unknown",
    "Resting watching": "resting",
    "Obstacle": "unknown",
    "Watching": "unknown",
    "Scratching head": "scratching",
    "Sand": "unknown",
    "Flipper beat": "unknown",
    "Interaction": "unknown",
    "Swimming fast descent": "swimming",
    "Catching": "feeding",
    "Landing": "unknown",
    "Resting active": "resting",
    "New video": "unknown",
    "Swimming fast ascent": "swimming",
    "Escape": "unknown",
    "Scratching camera": "scratching",
    "Catching jellyfish": "feeding",
    "Swimming in place": "swimming",
    "Swimming fast horizontally": "swimming",
    "Pursuit": "unknown",
    "Grabbing the wall": "feeding",
    "Stepping back": "swimming",
    "Time mark": "unknown",
    "Regurgitating": "unknown",
    "Chewing jellyfish": "feeding",
    "Hunting jellyfish": "unknown",
}

# Animal of each turtle recording, from the same notebook (its integer codes 0 to 13).
TURTLE_ANIMALS = {
    "CC-07-107_11-02-2019": 0,
    "CC-07-108_11-02-2019": 1,
    "CC-07-115_13-05-2019": 2,
    "CC-07-115_14-05-2019": 3,
    "CC-07-47_14-02-2018": 4,
    "CC-07-48_04-10-2018": 5,
    "CC-07-48_06-10-2018": 6,
    "CC-07-48_08-04-2019_1": 7,
    "CC-07-48_08-04-2019_2": 7,
    "CC-07-48_08-04-2019_3": 7,
    "CC-07-48_08-10-2018": 8,
    "CC-07-48_10-10-2018": 9,
    "CC-07-48_18-02-2019": 10,
    "CC-07-48_22-05-2018": 11,
    "CC-07-48_26-05-2018": 12,
    "CC-07-59_20-05-2018": 13,
}

# Classes of the benchmark for the seal labels (format_ladds_seals.ipynb of the same repository).
SEAL_CLASSES = {
    "Travelling": "travelling",
    "Foraging": "feeding",
    "Other": "unknown",
    "Resting": "resting",
    "Grooming": "grooming",
}
SEAL_RATE = 25.0  # samples per second
FINCH_BIRDS = ("bl26lb16", "gr41rd51", "gy6or6", "or60yw70")


# A string is a dict: individual, recording, bouts. A bout is (behaviour, parts); a part is
# (raw label, duration in seconds). Datasets without raw labels have one part with the label "".


def merge_runs(labels: list[str], durations: list[float]) -> list[tuple[str, float]]:
    """Merge consecutive equal labels and add their durations (NaN if a duration is missing)."""
    runs: list[list] = []
    for label, duration in zip(labels, durations, strict=True):
        if runs and runs[-1][0] == label:
            runs[-1][1] += duration
        else:
            runs.append([label, duration])
    return [(label, duration) for label, duration in runs]


def bouts_of(classes: list[str], raw: list[str], durations: list[float]) -> list[tuple[str, list]]:
    """Bouts of one recording: consecutive rows of one class, each split into runs of one raw label."""
    bouts: list[tuple[str, list]] = []
    start = 0
    for position in range(1, len(classes) + 1):
        if position == len(classes) or classes[position] != classes[start]:
            parts = merge_runs(raw[start:position], durations[start:position])
            bouts.append((classes[start], parts))
            start = position
    return bouts


def cut_at_rest(bouts: list[tuple[str, list]], rest: str | None) -> list[list[tuple[str, list]]]:
    """Cut a recording into strings at every bout of the rest class, which is removed."""
    if rest is None:
        return [bouts]
    strings, current = [], []
    for bout in bouts:
        if bout[0] == rest:
            if current:
                strings.append(current)
            current = []
        else:
            current.append(bout)
    if current:
        strings.append(current)
    return strings


def keep_and_cap(strings: list[dict]) -> list[dict]:
    """Remove strings of fewer than ten bouts; if more than 500 remain, keep a seeded random 500."""
    strings = [s for s in strings if len(s["bouts"]) >= behaviour.MIN_STRING_LENGTH]
    if len(strings) > behaviour.MAX_STRINGS:
        rng = np.random.default_rng(behaviour.SUBSAMPLE_SEED)
        chosen = sorted(rng.choice(len(strings), size=behaviour.MAX_STRINGS, replace=False))
        strings = [strings[i] for i in chosen]
    return strings


def zebrafish(root: Path) -> tuple[list[dict], dict]:
    import h5py  # only this script needs it

    folder = root / "zebrafish" / "Datasets" / "JM_data"
    names = [str(name) for name in np.load(folder / "classnames_jm.npy", allow_pickle=False)]
    with h5py.File(folder / "filtered_jmpool_kin.h5", "r") as file:
        bout_types = file["bout_types"][:]
        times = file["times_bouts"][:]
        rate = float(file["MetaData/frameRate"][0])
    strings = []
    for fish in range(bout_types.shape[0]):
        valid = bout_types[fish] != ZEBRAFISH_PAD
        labels = [names[int(code) - 1] for code in bout_types[fish][valid]]
        frames = times[fish][valid]
        durations = ((frames[:, 1] - frames[:, 0]) / rate).astype(float).tolist()
        # a swim bout is a discrete event: every bout is one token, with its own duration, and
        # consecutive bouts of the same type are not merged
        strings.append(
            {
                "individual": f"fish_{fish:03d}",
                "recording": f"fish_{fish:03d}",
                "bouts": [
                    (label, [("", duration)])
                    for label, duration in zip(labels, durations, strict=True)
                ],
            }
        )
    eligible = [s for s in strings if len(s["bouts"]) >= behaviour.MIN_STRING_LENGTH]
    rng = np.random.default_rng(behaviour.SUBSAMPLE_SEED)
    chosen = sorted(rng.choice(len(eligible), size=ZEBRAFISH_N_FISH, replace=False))
    info = {
        "source_animals": bout_types.shape[0],
        "source_bouts": int((bout_types != ZEBRAFISH_PAD).sum()),
        "eligible_strings": len(eligible),
        "frame_rate": rate,
    }
    return [eligible[i] for i in chosen], info


def turtle(root: Path) -> tuple[list[dict], dict]:
    folder = root / "BEBE" / "raw_jeantet_turtles"
    strings, n_rows, n_bad = [], 0, 0
    for recording, animal in TURTLE_ANIMALS.items():
        table = pd.read_csv(folder / f"Behaviors_{recording}.csv", delimiter=";")
        labels = table["Behavior"].tolist()
        classes = [TURTLE_CLASSES[label] for label in labels]  # an unknown label stops the build
        durations = (table["Stop (s)"] - table["Start (s)"]).tolist()
        durations = [duration if duration > 0 else math.nan for duration in durations]
        n_rows += len(labels)
        n_bad += sum(math.isnan(duration) for duration in durations)
        for bouts in cut_at_rest(bouts_of(classes, labels, durations), REST):
            strings.append(
                {"individual": f"turtle_{animal:02d}", "recording": recording, "bouts": bouts}
            )
    info = {
        "source_animals": len(set(TURTLE_ANIMALS.values())),
        "source_recordings": len(TURTLE_ANIMALS),
        "source_rows": n_rows,
        "rows_without_duration": n_bad,
        "strings_before_filter": len(strings),
    }
    return keep_and_cap(strings), info


def seal(root: Path) -> tuple[list[dict], dict]:
    folder = root / "BEBE" / "raw_ladds_seals"
    files = sorted(folder.glob("*/*.csv"))
    strings, n_rows = [], 0
    for path in files:
        table = pd.read_csv(path, usecols=["behaviour", "type"], dtype=str, keep_default_na=False)
        classes = [SEAL_CLASSES[label] for label in table["type"]]
        raw = table["behaviour"].tolist()
        n_rows += len(raw)
        durations = [1.0 / SEAL_RATE] * len(raw)
        animal = path.parent.name
        for bouts in cut_at_rest(bouts_of(classes, raw, durations), REST):
            strings.append(
                {"individual": animal, "recording": f"{animal}/{path.stem}", "bouts": bouts}
            )
    info = {
        "source_animals": len({path.parent.name for path in files}),
        "source_recordings": len(files),
        "source_rows": n_rows,
        "strings_before_filter": len(strings),
    }
    return keep_and_cap(strings), info


def finch(root: Path) -> tuple[list[dict], dict]:
    folder = root / "bengalese_finch"
    strings, annotated = [], {}
    for bird in FINCH_BIRDS:
        songs = []
        files = sorted((folder / bird).rglob("*.not.mat"))
        annotated[bird] = 0
        for path in files:
            mat = loadmat(path)
            labels = list(str(mat["labels"].flat[0])) if mat["labels"].size else []
            onsets, offsets = mat["onsets"].flatten(), mat["offsets"].flatten()
            if not (len(onsets) == len(offsets) == len(labels)):
                raise ValueError(f"{path.name}: labels, onsets and offsets differ in length.")
            if labels:
                annotated[bird] += 1
            durations = ((offsets - onsets) / 1000.0).tolist()  # milliseconds to seconds
            songs.append(
                {
                    "individual": bird,
                    "recording": path.name.removesuffix(".cbin.not.mat"),
                    "bouts": [
                        (f"syllable {label}", [("", duration)])
                        for label, duration in zip(labels, durations, strict=True)
                    ],
                }
            )
        strings.extend(keep_and_cap(songs))  # each bird is its own set of strings
    most = max(annotated, key=annotated.get)
    if most != behaviour.FINCH_BIRD:
        raise ValueError(
            f"The bird with the most annotated songs is {most}, not {behaviour.FINCH_BIRD}: "
            "change tutorial.behaviour.FINCH_BIRD."
        )
    info = {
        "source_animals": len(FINCH_BIRDS),
        "source_recordings": sum(annotated.values()),
        "annotated_songs": annotated,
    }
    return strings, info


BUILDERS = {"zebrafish": zebrafish, "turtle": turtle, "finch": finch, "seal": seal}


def alphabet(dataset: str, strings: list[dict]) -> dict[tuple[str, str], str]:
    """One alphabet for the whole dataset: (alphabet group, behaviour) -> symbol.

    Behaviours are sorted by name and given the letters A, B, C and so on. For the finch the
    syllable labels of the source are single characters already and are used as the symbols; they
    are specific to each bird, so the bird is the alphabet group.
    """
    symbols: dict[tuple[str, str], str] = {}
    if dataset == "finch":
        for string in strings:
            for name, _ in string["bouts"]:
                label = name.removeprefix("syllable ")
                if len(label) != 1 or label.isspace() or label in (behaviour.EOS, "_"):
                    raise ValueError(f"Syllable label {label!r} cannot be used as a symbol.")
                symbols[(string["individual"], name)] = label
        return symbols
    names = sorted({name for string in strings for name, _ in string["bouts"]})
    for name, letter in zip(names, behaviour.letters(len(names)), strict=True):
        symbols[("", name)] = letter
    return symbols


def rows_of(dataset: str, strings: list[dict]) -> list[tuple]:
    symbols = alphabet(dataset, strings)
    rows = []
    for index, string in enumerate(strings):
        group = string["individual"] if dataset == "finch" else ""
        for bout, (name, parts) in enumerate(string["bouts"]):
            for raw_label, duration in parts:
                rows.append(
                    (
                        string["individual"],
                        string["recording"],
                        index,
                        bout,
                        symbols[(group, name)],
                        name,
                        raw_label,
                        "" if math.isnan(duration) else f"{duration:.4f}",
                    )
                )
    return rows


def write_table(rows: list[tuple], path: Path) -> None:
    """Write a table as gzip-compressed CSV; the same rows always give the same bytes."""
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(behaviour.COLUMNS)
    writer.writerows(rows)
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
        out.write(text.getvalue().encode("utf-8"))


def summary(dataset: str, strings: list[dict], info: dict, path: Path) -> dict:
    lengths = [len(string["bouts"]) for string in strings]
    return {
        **info,
        "strings": len(strings),
        "animals": len({string["individual"] for string in strings}),
        "recordings": len({string["recording"] for string in strings}),
        "bouts": int(sum(lengths)),
        "rows": sum(len(parts) for string in strings for _, parts in string["bouts"]),
        "shortest": int(min(lengths)),
        "median": float(np.median(lengths)),
        "longest": int(max(lengths)),
        "behaviours": len({name for string in strings for name, _ in string["bouts"]}),
        # bouts that have the behaviour of the bout before them in the same string
        "repeats": sum(
            a[0] == b[0] for string in strings for a, b in itertools.pairwise(string["bouts"])
        ),
        "unknown": sum(name == UNKNOWN for string in strings for name, _ in string["bouts"]),
        "per_animal": dict(sorted(Counter(string["individual"] for string in strings).items())),
        "labels_per_animal": {
            animal: len(
                {name for s in strings if s["individual"] == animal for name, _ in s["bouts"]}
            )
            for animal in sorted({string["individual"] for string in strings})
        },
        "bytes": path.stat().st_size,
    }


DIFFERENCES = [
    (
        "Symbols",
        "Letters are assigned string by string, from the behaviours present in that string.",
        "One alphabet per dataset, built once over all its strings. The name of each behaviour in the source is kept beside its symbol.",
    ),
    (
        "Finch birds",
        "The songs of the four birds form one set, from which 500 songs are drawn.",
        "Each bird is its own set of strings, with at most 500 songs, because syllable labels are specific to each bird. The notebook uses the bird with the most annotated songs.",
    ),
    (
        "Zebrafish",
        "All 463 fish.",
        "A random subset of {n_fish} fish (seed 42). The number was set by the file-size and run-time limits of the tutorial before any measure was computed.",
    ),
    (
        "Consecutive zebrafish bouts of one type",
        "Merged into one bout, with their durations added.",
        "Kept as they are: a swim bout is a discrete event, so two bouts of one type in a row are two symbols, each with its own duration. In the table {repeats} of {bouts} bouts ({repeat_share}) have the type of the bout before them. Consecutive equal classes are merged only for the turtles and the seals, where the source is annotated observation by observation or sample by sample and merging is what makes a bout.",
    ),
    (
        "Behaviour classes of the turtles and the seals",
        "The labels are grouped by the script's own table, with a finer numbered level for the transition graph.",
        "The classes are those published with the benchmark. The label of the source is kept in the column raw_label. The measures use the classes.",
    ),
    (
        "Mouse open-field dataset",
        "Included.",
        "Not used.",
    ),
    (
        "Recordings of one animal",
        "The recordings of one animal are joined and their bouts are ordered by the time inside each recording.",
        "Each recording is kept apart, with its bouts in the order of the source file. A string never spans two recordings.",
    ),
    (
        "Durations",
        "Seals: last time stamp minus first time stamp of the bout. Turtles: stop minus start, also where that is not positive.",
        "Seals: number of samples divided by 25 per second. Turtles: stop minus start, left empty where that is not positive.",
    ),
    (
        "Random draws",
        "One random generator, seeded with 42, serves all datasets in turn.",
        "Each draw uses its own generator seeded with 42.",
    ),
]


# Choices that hold for all tables, stated once.
CHOICES = [
    "A string never spans two recordings, and the rows of a table are in the order of the source file.",
    "A zebrafish swim bout and a finch syllable are events: each is one symbol, and two in a row of the same type are two symbols. For the turtles and the seals consecutive equal classes are one bout.",
    "The class unknown of the turtles and the seals (for the seals the label Other of the source) stays a symbol, and strings are not cut there. Its share of the bouts is given with each table above.",
    f"The four finches are four separate sets of strings and are never pooled. The notebook uses the bird with the most annotated songs, {behaviour.FINCH_BIRD}.",
    "The shuffled surrogate of the notebook permutes all bouts of a dataset (for the finch, of one bird) and cuts them into strings of the original lengths. It does not shuffle inside each string.",
]


def share(part: int, whole: int) -> str:
    """A share as a percentage with one decimal, for example 12.3 %."""
    return f"{100 * part / whole:.1f} %"


def contents(dataset: str, s: dict) -> str:
    """The number of behaviours of a table; for the finch, songs and syllable labels of each bird."""
    if dataset != "finch":
        return f"{s['behaviours']} behaviours"
    birds = [
        f"{bird} {s['per_animal'][bird]} songs with {s['labels_per_animal'][bird]} syllable labels"
        for bird in s["per_animal"]
    ]
    return "a bout being one syllable (" + ", ".join(birds) + ")"


def sources_markdown(summaries: dict[str, dict]) -> str:
    """The text of SOURCES.md, with the counts of the tables just written."""
    lines = [
        "# Sources of the behaviour tables",
        "",
        "This folder holds four open datasets as sequences of behaviour labels, one table per dataset. Each table was derived from the public files of its source by `scripts/build_behaviour_sequences.py`. No raw file is in this repository. If you use a table, cite its source as given below.",
        "",
        "Every table has one row per bout and the columns `individual`, `recording`, `string` (number of the string), `bout` (number of the bout inside the string), `symbol`, `behaviour` (name in the source), `raw_label` and `duration_s` (seconds). For the turtles and the seals a bout of one class can hold several labels of the source; it then has one row per label, all with the same bout number.",
        "",
    ]
    for name, meta in behaviour.DATASETS.items():
        s = summaries[name]
        lines += [
            f"## {meta['label'].capitalize()} (`{name}.csv.gz`)",
            "",
            f"- What it is: {meta['animal']}. In the words of the source: {meta['source_words']}",
            f"- Dataset: {meta['citation']} https://doi.org/{meta['doi']}",
        ]
        lines += [f"- Article: {paper}" for paper in meta["papers"]]
        lines += [
            f"- Licence: {meta['licence']}, {meta['licence_url']}",
            f"- In the table: {s['strings']} strings ({meta['string_is']}) from {s['animals']} animals, {s['bouts']} bouts, {contents(name, s)}; strings of {s['shortest']} to {s['longest']} bouts, median {s['median']:g}.",
            "- Changes made: " + " ".join(meta["changes"]),
        ]
        if s["unknown"]:
            lines.append(
                f"- The class unknown: kept as a symbol, {s['unknown']} of {s['bouts']} bouts ({share(s['unknown'], s['bouts'])}). Strings are not cut there."
            )
        lines.append("")
    lines += [
        "## Choices that hold for every table",
        "",
        *(f"- {choice}" for choice in CHOICES),
        "",
        "## Differences from the analysis of the SymSeqBench preprint",
        "",
        "The preprocessing follows the script of that analysis: strings cut at rest, at least ten bouts per string, at most 500 strings, seed 42, and for the turtles and the seals bouts by merging consecutive equal classes. It differs in these points.",
        "",
        "| Step | In that analysis | Here |",
        "|---|---|---|",
    ]
    fish = summaries["zebrafish"]
    numbers = {
        "n_fish": fish["animals"],
        "repeats": fish["repeats"],
        "bouts": fish["bouts"],
        "repeat_share": share(fish["repeats"], fish["bouts"]),
    }
    lines += [
        f"| {step} | {there} | {here.format(**numbers)} |" for step, there, here in DIFFERENCES
    ]
    lines += [
        "",
        "The values computed from these tables are therefore not expected to equal those of the preprint, which also used an earlier version of the library.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument(
        "--raw-root", type=Path, required=True, help="folder that holds the four raw datasets"
    )
    parser.add_argument("--out", type=Path, default=REPO / "data" / "behaviour")
    arguments = parser.parse_args()
    out = arguments.out
    out.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for name, build in BUILDERS.items():
        strings, info = build(arguments.raw_root)
        path = out / f"{name}.csv.gz"
        write_table(rows_of(name, strings), path)
        summaries[name] = summary(name, strings, info, path)
        if name in ("turtle", "seal") and summaries[name]["repeats"]:
            raise ValueError(
                f"{path.name}: consecutive equal classes were not merged into one bout."
            )
        if summaries[name]["bytes"] >= 2_000_000:
            raise ValueError(
                f"{path.name} has {summaries[name]['bytes']} bytes; the limit is 2 MB."
            )
        print(name, {k: v for k, v in summaries[name].items() if "per_animal" not in k})
    (out / "SOURCES.md").write_text(sources_markdown(summaries), encoding="utf-8")
    names = [*(f"{name}.csv.gz" for name in BUILDERS), "SOURCES.md"]
    manifest = "".join(
        f"{hashlib.sha256((out / name).read_bytes()).hexdigest()}  {name}\n" for name in names
    )
    (out / behaviour.MANIFEST).write_text(manifest)
    behaviour.check_folder(out)
    print(f"written {out}")


if __name__ == "__main__":
    main()
