"""The shipped behaviour tables agree with a second route from the raw files. Each test has a case it must reject.

The second route is written with other code than ``scripts/build_behaviour_sequences.py``: for the
turtles and the seals whole columns are run-length encoded with pandas and numpy, where the build
script walks the rows. Zebrafish bouts and finch syllables are events of the source and are taken
one by one, without any merging. The tests need the raw files (and h5py for the zebrafish) and are
skipped where these are absent:

    uv run --with h5py python -m pytest tests/test_behaviour_data.py

The folder of the raw files is ``/media/neuro/Data/animal_behavior`` or the value of the
environment variable ``BEHAVIOUR_RAW_ROOT``.
"""

from __future__ import annotations

import importlib.util
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tutorial import behaviour

REPO = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("BEHAVIOUR_RAW_ROOT", "/media/neuro/Data/animal_behavior"))
needs_raw = pytest.mark.skipif(not RAW.is_dir(), reason="the raw files are not on this machine")


def build_script():
    """The build script as a module, for its constants only (label maps, number of fish)."""
    spec = importlib.util.spec_from_file_location(
        "build_behaviour_sequences", REPO / "scripts" / "build_behaviour_sequences.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def runs(labels: np.ndarray, durations: np.ndarray) -> pd.DataFrame:
    """Run-length encoding of a column of labels: one row per run, with its summed duration."""
    labels = np.asarray(labels, dtype=object)
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    return pd.DataFrame(
        {"behaviour": labels[starts], "duration": np.add.reduceat(durations, starts)}
    )


def strings_between_rests(bouts: pd.DataFrame, rest: str) -> list[pd.DataFrame]:
    """Cut the bouts of a recording at every rest and keep the stretches of at least ten bouts."""
    is_rest = bouts["behaviour"] == rest
    group = is_rest.cumsum()[~is_rest]
    kept = bouts[~is_rest]
    return [
        part.reset_index(drop=True)
        for _, part in kept.groupby(group, sort=True)
        if len(part) >= behaviour.MIN_STRING_LENGTH
    ]


def second_route(dataset: str, merge_events: bool = False, seed: int = 42) -> list[dict]:
    """The strings of a dataset derived again from the raw files: animal, behaviours, durations.

    ``merge_events`` merges consecutive zebrafish bouts of one type, which the table must not do;
    ``seed`` draws the subset of fish. Both exist for the cases the test must reject.
    """
    build = build_script()
    strings = []
    if dataset == "zebrafish":
        h5py = pytest.importorskip("h5py")
        folder = RAW / "zebrafish" / "Datasets" / "JM_data"
        names = np.load(folder / "classnames_jm.npy", allow_pickle=False)
        with h5py.File(folder / "filtered_jmpool_kin.h5", "r") as file:
            types, times = file["bout_types"][:], file["times_bouts"][:]
            rate = float(file["MetaData/frameRate"][0])
        chosen = np.sort(
            np.random.default_rng(seed).choice(
                types.shape[0], size=build.ZEBRAFISH_N_FISH, replace=False
            )
        )
        for fish in chosen:
            valid = types[fish] != 15
            labels = names[types[fish][valid].astype(int) - 1]
            frames = times[fish][valid].astype(float)
            durations = (frames[:, 1] - frames[:, 0]) / rate
            if merge_events:
                bouts = runs(labels, durations)
            else:  # every bout of the source is one bout of the table
                bouts = pd.DataFrame({"behaviour": labels.astype(object), "duration": durations})
            strings.append({"individual": f"fish_{fish:03d}", **bouts.to_dict("list")})
    elif dataset == "turtle":
        for recording, animal in build.TURTLE_ANIMALS.items():
            table = pd.read_csv(
                RAW / "BEBE" / "raw_jeantet_turtles" / f"Behaviors_{recording}.csv", sep=";"
            )
            duration = (table["Stop (s)"] - table["Start (s)"]).where(lambda d: d > 0)
            bouts = runs(
                table["Behavior"].map(build.TURTLE_CLASSES).to_numpy(), duration.to_numpy()
            )
            for part in strings_between_rests(bouts, "resting"):
                strings.append({"individual": f"turtle_{animal:02d}", **part.to_dict("list")})
    elif dataset == "seal":
        for path in sorted((RAW / "BEBE" / "raw_ladds_seals").glob("*/*.csv")):
            classes = pd.read_csv(path, usecols=["type"])["type"].map(build.SEAL_CLASSES).to_numpy()
            bouts = runs(classes, np.full(len(classes), 0.04))
            for part in strings_between_rests(bouts, "resting"):
                strings.append({"individual": path.parent.name, **part.to_dict("list")})
    elif dataset == "finch":
        from scipy.io import loadmat

        for bird in sorted(p.name for p in (RAW / "bengalese_finch").iterdir() if p.is_dir()):
            songs = []
            for path in sorted((RAW / "bengalese_finch" / bird).rglob("*.not.mat")):
                mat = loadmat(path, squeeze_me=True)
                labels = [f"syllable {c}" for c in str(mat["labels"])]
                duration = (np.atleast_1d(mat["offsets"]) - np.atleast_1d(mat["onsets"])) / 1000.0
                if len(labels) >= behaviour.MIN_STRING_LENGTH:
                    songs.append(
                        {"individual": bird, "behaviour": labels, "duration": list(duration)}
                    )
            if len(songs) > behaviour.MAX_STRINGS:
                keep = np.sort(
                    np.random.default_rng(42).choice(len(songs), size=500, replace=False)
                )
                songs = [songs[i] for i in keep]
            strings.extend(songs)
    return strings


def shipped(table: pd.DataFrame) -> list[dict]:
    """The strings of a shipped table at the level of bouts: animal, behaviours, durations, rows per bout."""
    bouts = table.groupby(["string", "bout"], sort=True).agg(
        individual=("individual", "first"),
        behaviour=("behaviour", "first"),
        duration=("duration_s", lambda values: values.sum(skipna=False)),
        rows=("duration_s", "size"),
    )
    return [
        {
            "individual": part["individual"].iloc[0],
            "behaviour": part["behaviour"].tolist(),
            "duration": part["duration"].tolist(),
            "rows": part["rows"].tolist(),
        }
        for _, part in bouts.groupby(level="string", sort=True)
    ]


def totals(strings: list[dict], key: str) -> pd.DataFrame:
    """Number of bouts, of table rows and total duration per behaviour (``key="behaviour"``) or per animal."""
    rows = [
        {
            "key": name if key == "behaviour" else string["individual"],
            "duration": duration,
            "rows": n_rows,
        }
        for string in strings
        for name, duration, n_rows in zip(
            string["behaviour"],
            string["duration"],
            string.get("rows", [1] * len(string["behaviour"])),
            strict=True,
        )
    ]
    frame = pd.DataFrame(rows)
    return frame.groupby("key").agg(
        bouts=("duration", "size"), seconds=("duration", "sum"), rows=("rows", "sum")
    )


def differences(ours: list[dict], theirs: list[dict]) -> list[str]:
    """What differs between the strings of the table and those of the second route (empty if nothing).

    Every row of the table is rounded to 0.0001 s, so a duration may differ by 0.00005 s per row.
    """
    if len(ours) != len(theirs):
        return [f"{len(ours)} strings in the table, {len(theirs)} by the second route"]
    found = set()
    for a, b in zip(ours, theirs, strict=True):
        if a["individual"] != b["individual"] or a["behaviour"] != b["behaviour"]:
            found.add("a string differs in its animal or its behaviours")
            continue
        table, route = np.array(a["duration"]), np.array(b["duration"], dtype=float)
        allowed = 0.5e-4 * np.array(a["rows"]) + 1e-9
        if (
            not np.array_equal(np.isnan(table), np.isnan(route))
            or np.nanmax(np.abs(table - route) - allowed) > 0
        ):
            found.add("the duration of a bout differs")
    for key in ("behaviour", "individual"):
        a, b = totals(ours, key), totals(theirs, key)
        if not a.index.equals(b.index) or not (a["bouts"] == b["bouts"]).all():
            found.add(f"bouts per {key} differ")
        elif ((a["seconds"] - b["seconds"]).abs() > 0.5e-4 * a["rows"] + 1e-9).any():
            found.add(f"total duration per {key} differs")
    return sorted(found)


@needs_raw
@pytest.mark.parametrize("dataset", list(behaviour.DATASETS))
def test_table_equals_the_second_route_from_the_raw_files(dataset):
    table = behaviour.load_table(dataset)
    ours, theirs = shipped(table), second_route(dataset)
    assert differences(ours, theirs) == []
    # must reject: one bout removed from the table, one behaviour renamed, one bout a second longer
    removed = table[~((table["string"] == 0) & (table["bout"] == 3))]
    assert differences(shipped(removed), theirs) != []
    renamed = table.copy()
    renamed.loc[renamed.index[0], "behaviour"] = "something else"
    assert "a string differs in its animal or its behaviours" in differences(
        shipped(renamed), theirs
    )
    longer = table.copy()
    longer.loc[longer.index[7], "duration_s"] += 1.0
    assert "the duration of a bout differs" in differences(shipped(longer), theirs)


@needs_raw
def test_zebrafish_bouts_are_events_and_are_not_merged():
    """Every swim bout of the source is one row of the table; two bouts of one type in a row stay two."""
    ours = shipped(behaviour.load_table("zebrafish"))
    unmerged = second_route("zebrafish")
    assert differences(ours, unmerged) == []
    n_bouts = sum(len(string["behaviour"]) for string in ours)
    repeats = sum(a == b for string in ours for a, b in itertools.pairwise(string["behaviour"]))
    assert repeats > 0.2 * n_bouts  # self-transitions are a large part of the sequence
    # must reject: the route that merges consecutive bouts of one type gives other strings, with
    # exactly the repeated bouts missing
    merged = second_route("zebrafish", merge_events=True)
    assert "a string differs in its animal or its behaviours" in differences(ours, merged)
    assert sum(len(string["behaviour"]) for string in merged) == n_bouts - repeats
    # must reject: another seed draws other fish
    other_fish = [string["individual"] for string in second_route("zebrafish", seed=43)]
    assert other_fish != [string["individual"] for string in ours]


@needs_raw
def test_the_finch_of_the_notebook_has_the_most_annotated_songs():
    counts = {
        bird.name: sum(1 for _ in bird.rglob("*.not.mat"))
        for bird in (RAW / "bengalese_finch").iterdir()
        if bird.is_dir()
    }
    assert max(counts, key=counts.get) == behaviour.FINCH_BIRD
    # must reject: any other bird has fewer annotation files
    assert all(
        n < counts[behaviour.FINCH_BIRD]
        for bird, n in counts.items()
        if bird != behaviour.FINCH_BIRD
    )


@needs_raw
def test_durations_follow_the_sources():
    """Spot checks against numbers read from the raw files by hand (first rows of two files)."""
    turtle = behaviour.load_table("turtle")
    first = turtle.iloc[0]
    assert (first["raw_label"], first["behaviour"]) == ("Swimming fast horizontally", "swimming")
    assert first["duration_s"] == pytest.approx(
        517.9 - 493.3, abs=1e-4
    )  # Stop (s) - Start (s) of row 1
    seal = behaviour.load_table("seal")
    assert seal["duration_s"].min() >= 0.04 - 1e-9  # one sample at 25 per second
    assert np.allclose((seal["duration_s"] * 25).round(), seal["duration_s"] * 25, atol=1e-6)
    # must reject: a duration that is not a whole number of samples
    assert not np.isclose(round(0.05 * 25), 0.05 * 25, atol=1e-6)
