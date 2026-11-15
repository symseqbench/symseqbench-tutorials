"""The shipped behaviour tables, the two surrogates and the check of a participant's sequence. Each test has a case it must reject."""

from __future__ import annotations

import itertools
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tutorial import behaviour
from tutorial import behaviour_measures as bm

REPO = Path(__file__).resolve().parents[1]
PARTICIPANT_FILES = (
    "README.md",
    "data/behaviour/SOURCES.md",
    "notebooks/b_behaviour.py",
    "tutorial/behaviour.py",
    "tutorial/behaviour_measures.py",
)


def chain(n_strings: int, length: int, n_symbols: int, seed: int) -> list[list[str]]:
    """Strings of a first-order chain without immediate repeats, with a random transition table."""
    rng = np.random.default_rng(seed)
    table = rng.dirichlet(np.full(n_symbols, 0.7), size=n_symbols)
    np.fill_diagonal(table, 0.0)
    cumulative = np.cumsum(table / table.sum(axis=1, keepdims=True), axis=1)
    symbols = behaviour.letters(n_symbols)
    strings = []
    for _ in range(n_strings):
        state = int(rng.integers(n_symbols))
        states = [state]
        for draw in rng.random(max(2, int(rng.poisson(length))) - 1):
            state = min(int(np.searchsorted(cumulative[state], draw, side="right")), n_symbols - 1)
            states.append(state)
        strings.append([symbols[s] for s in states])
    return strings


def plug_in_rate(strings: list[list[str]]) -> float:
    """h_1 of a set computed here from its table of pairs (no SymSeq): H(pair) - H(first of pair), bits."""
    tokens, previous, k = bm.pair_tables(strings)
    table = np.bincount(tokens[previous] * k + tokens[previous + 1], minlength=k * k).reshape(k, k)
    joint = table / table.sum()
    rows = joint.sum(axis=1)

    def entropy(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    return entropy(joint.ravel()) - entropy(rows)


# ------------------------------------------------------------------ the shipped tables


def alphabet_faults(table: pd.DataFrame, per_animal: bool) -> list[str]:
    """Symbols that stand for two behaviours, behaviours with two symbols, and forbidden symbols."""
    faults = []
    groups = table.groupby("individual") if per_animal else [("all", table)]
    for name, rows in groups:
        pairs = rows[["symbol", "behaviour"]].drop_duplicates()
        if pairs["symbol"].duplicated().any():
            faults.append(f"{name}: one symbol for two behaviours")
        if pairs["behaviour"].duplicated().any():
            faults.append(f"{name}: one behaviour with two symbols")
    if (table["symbol"] == behaviour.EOS).any() or table["symbol"].str.contains(r"[\s_]").any():
        faults.append("a symbol is the end marker or holds a space or an underscore")
    return faults


@pytest.mark.parametrize("dataset", list(behaviour.DATASETS))
def test_one_symbol_never_stands_for_two_behaviours(dataset):
    table = behaviour.load_table(dataset)
    assert alphabet_faults(table, per_animal=dataset == "finch") == []
    # must reject: one row with another behaviour, and a symbol that is the end marker
    changed = table.copy()
    changed.loc[changed.index[5], "behaviour"] = "another behaviour"
    assert alphabet_faults(changed, per_animal=dataset == "finch") != []
    marked = table.copy()
    marked.loc[marked.index[5], "symbol"] = behaviour.EOS
    assert alphabet_faults(marked, per_animal=dataset == "finch") != []


def test_finch_birds_have_their_own_alphabets_and_are_never_pooled():
    table = behaviour.load_table("finch")
    assert sorted(table["individual"].unique()) == ["bl26lb16", "gr41rd51", "gy6or6", "or60yw70"]
    default = behaviour.load_strings("finch")
    assert default.name == f"finch ({behaviour.FINCH_BIRD})"
    assert set(default.individuals) == {behaviour.FINCH_BIRD}
    other = behaviour.load_strings("finch", individual="gy6or6")
    assert set(other.individuals) == {"gy6or6"} and other.strings != default.strings
    # must reject: the whole table as one alphabet has more syllables than any single bird
    pooled = table["behaviour"].nunique()
    assert all(pooled > rows["behaviour"].nunique() for _, rows in table.groupby("individual"))
    with pytest.raises(KeyError):
        behaviour.load_strings("finch", individual="no such bird")


# Datasets whose bouts are events of the source (a swim bout, a syllable): never merged. In the
# other two a bout is made by merging consecutive equal classes.
EVENTS = ("zebrafish", "finch")
REPEATED = "a symbol is repeated in direct succession"
MERGED = "no bout has the type of the bout before it: the events were merged"


def merge_repeats(table: pd.DataFrame) -> pd.DataFrame:
    """A table of events with every bout removed that has the symbol of the bout before it in its string."""
    repeat = (table["string"] == table["string"].shift()) & (
        table["symbol"] == table["symbol"].shift()
    )
    return table[~repeat]


def shape_faults(table: pd.DataFrame, dataset: str) -> list[str]:
    faults = []
    bouts = (
        table.groupby(["individual", "string", "bout"], sort=False)["symbol"].first().reset_index()
    )
    lengths = bouts.groupby("string").size()
    if (lengths < behaviour.MIN_STRING_LENGTH).any():
        faults.append("a string has fewer than ten bouts")
    per_set = bouts.groupby("individual" if dataset == "finch" else lambda _: 0)["string"].nunique()
    if (per_set > behaviour.MAX_STRINGS).any():
        faults.append("a set has more than 500 strings")
    same_string = bouts["string"].to_numpy()[1:] == bouts["string"].to_numpy()[:-1]
    same_symbol = bouts["symbol"].to_numpy()[1:] == bouts["symbol"].to_numpy()[:-1]
    repeats = (same_string & same_symbol).any()
    if dataset not in EVENTS and repeats:
        faults.append(REPEATED)
    if dataset in EVENTS and not repeats:
        faults.append(MERGED)
    if table["behaviour"].isin(["resting", "rest"]).any():
        faults.append("resting is a behaviour of the table")
    if tuple(table.columns) != behaviour.COLUMNS:
        faults.append("the columns are not those declared")
    return faults


@pytest.mark.parametrize("dataset", list(behaviour.DATASETS))
def test_tables_have_the_declared_shape(dataset):
    table = behaviour.load_table(dataset)
    assert shape_faults(table, dataset) == []
    assert (behaviour.DATA / f"{dataset}.csv.gz").stat().st_size < 2_000_000
    assert (table["duration_s"].dropna() > 0).all()
    # must reject: a string of nine bouts, a bout named resting
    short = table[(table["string"] != 0) | (table["bout"] < 9)]
    assert "a string has fewer than ten bouts" in shape_faults(short, dataset)
    if dataset in EVENTS:
        # must reject: the events merged where two of one type follow each other
        assert MERGED in shape_faults(merge_repeats(table), dataset)
        assert len(merge_repeats(table)) < len(table)
    else:
        # must reject: a repeated symbol where bouts are made by merging
        repeated = table.copy()
        first = repeated.index[(repeated["string"] == 0) & (repeated["bout"] == 0)]
        second = repeated.index[(repeated["string"] == 0) & (repeated["bout"] == 1)]
        repeated.loc[second, "symbol"] = repeated.loc[first[0], "symbol"]
        assert REPEATED in shape_faults(repeated, dataset)
    resting = table.copy()
    resting.loc[resting.index[0], "behaviour"] = "resting"
    assert "resting is a behaviour of the table" in shape_faults(resting, dataset)


def test_strings_are_what_the_table_holds():
    for dataset in behaviour.DATASETS:
        table = behaviour.load_table(dataset)
        if dataset == "finch":
            table = table[table["individual"] == behaviour.FINCH_BIRD]
        strings = behaviour.load_strings(dataset)
        assert len(strings.strings) == table["string"].nunique()
        assert strings.n_tokens == len(table[["string", "bout"]].drop_duplicates())
        assert set(strings.names) == set(table["symbol"]) and strings.alphabet == sorted(
            strings.names
        )
        assert [len(d) for d in strings.durations] == [len(s) for s in strings.strings]
    turtle_raw = behaviour.load_strings("turtle", level="raw")
    assert turtle_raw.n_tokens == len(behaviour.load_table("turtle"))  # one symbol per row
    assert len(turtle_raw.alphabet) > len(behaviour.load_strings("turtle").alphabet)
    # must reject: raw labels where there are none, an unknown level, an unknown dataset
    with pytest.raises(ValueError, match="no raw labels"):
        behaviour.load_strings("zebrafish", level="raw")
    with pytest.raises(ValueError, match="level"):
        behaviour.load_strings("turtle", level="fine")
    with pytest.raises(KeyError):
        behaviour.load_table("mouse")


def test_folder_holds_the_declared_datasets_only_and_matches_its_manifest(work_dir):
    manifest = behaviour.check_folder()
    assert sorted(manifest) == sorted(
        [*(f"{name}.csv.gz" for name in behaviour.DATASETS), "SOURCES.md"]
    )
    assert sorted(behaviour.DATASETS) == ["finch", "seal", "turtle", "zebrafish"]
    copy = work_dir / "behaviour"
    shutil.copytree(behaviour.DATA, copy)
    assert behaviour.check_folder(copy) == manifest
    # must reject: one more file, a changed table, a missing table
    (copy / "mouse.csv.gz").write_bytes(b"x")
    with pytest.raises(behaviour.ManifestError, match="no declared dataset"):
        behaviour.check_folder(copy)
    (copy / "mouse.csv.gz").unlink()
    content = bytearray((copy / "seal.csv.gz").read_bytes())
    content[-1] ^= 1
    (copy / "seal.csv.gz").write_bytes(bytes(content))
    with pytest.raises(behaviour.ManifestError, match="differs"):
        behaviour.check_folder(copy)
    with pytest.raises(behaviour.ManifestError):
        behaviour.load_table("seal", folder=copy)
    (copy / "seal.csv.gz").unlink()
    with pytest.raises(behaviour.ManifestError, match="Missing"):
        behaviour.check_folder(copy)


def names_private_data(text: str) -> bool:
    return "maternal" in text.lower() or "social" in text.lower()


def mouse_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if "mouse" in line.lower()]


def test_no_participant_file_names_data_that_is_not_shipped():
    for name in PARTICIPANT_FILES:
        text = (REPO / name).read_text()
        assert not names_private_data(text), name
        # the mouse open-field data are named once, in the table of differences, as not used
        for line in mouse_lines(text):
            assert "Not used." in line or "not used" in line.lower(), (name, line)
    # must reject: a sentence on private data, and a line that offers the mouse data
    assert names_private_data("the maternal behaviour tables")
    assert mouse_lines("load the mouse open-field strings") == ["load the mouse open-field strings"]
    assert "not used" not in mouse_lines("load the mouse open-field strings")[0].lower()


def test_sources_file_gives_citation_doi_licence_and_changes():
    text = (behaviour.DATA / "SOURCES.md").read_text()
    for meta in behaviour.DATASETS.values():
        assert f"https://doi.org/{meta['doi']}" in text
        assert meta["licence_url"] in text and meta["citation"] in text
        assert all(change in text for change in meta["changes"])
    assert "## Differences from the analysis of the SymSeqBench preprint" in text
    assert text.count("\n| ") >= 10  # header and at least nine differences
    assert broken_paragraphs(text) == []
    # must reject: a paragraph broken over two lines, and an entry without its licence
    assert broken_paragraphs("A sentence that goes\non in the next line.\n") != []
    without = text.replace(behaviour.DATASETS["turtle"]["licence_url"], "")
    assert behaviour.DATASETS["turtle"]["licence_url"] not in without


def bout_counts(dataset: str) -> dict:
    """Bouts of a shipped table, those that repeat the bout before them, and those of the class unknown."""
    table = behaviour.load_table(dataset)
    bouts = table.groupby(["string", "bout"], sort=True)["behaviour"].first().reset_index()
    repeat = (bouts["string"] == bouts["string"].shift()) & (
        bouts["behaviour"] == bouts["behaviour"].shift()
    )
    return {
        "bouts": len(bouts),
        "repeats": int(repeat.sum()),
        "unknown": int((bouts["behaviour"] == "unknown").sum()),
    }


def stated(part: int, whole: int) -> str:
    return f"{part} of {whole} bouts ({100 * part / whole:.1f} %)"


def test_sources_file_states_the_choices_with_the_counts_of_the_tables():
    text = (behaviour.DATA / "SOURCES.md").read_text()
    # zebrafish bouts are events: the row of the differences, and no sentence on merging them
    fish = bout_counts("zebrafish")
    row = next(
        line for line in text.splitlines() if line.startswith("| Consecutive zebrafish bouts")
    )
    assert stated(fish["repeats"], fish["bouts"]) in row and "two symbols" in row
    assert "Consecutive bouts of the same type were merged" not in text
    assert "not for the finch" not in text
    # the class unknown stays a symbol, with its share of the bouts of the turtles and the seals
    for dataset in ("turtle", "seal"):
        counts = bout_counts(dataset)
        assert counts["unknown"] > 0 and counts["repeats"] == 0
        assert f"kept as a symbol, {stated(counts['unknown'], counts['bouts'])}" in text
    assert bout_counts("finch")["unknown"] == 0 and fish["unknown"] == 0
    for choice in (
        "A string never spans two recordings, and the rows of a table are in the order of the source file.",
        "stays a symbol, and strings are not cut there",
        "four separate sets of strings and are never pooled",
        f"the bird with the most annotated songs, {behaviour.FINCH_BIRD}",
        "permutes all bouts of a dataset",
        "It does not shuffle inside each string.",
    ):
        assert choice in text, choice
    # must reject: a share that is not the count of the table, and the file before this row existed
    assert stated(fish["repeats"] + 1, fish["bouts"]) not in text
    assert stated(bout_counts("turtle")["unknown"], bout_counts("seal")["bouts"]) not in text
    without_row = text.replace(row, "")
    assert not any(
        line.startswith("| Consecutive zebrafish bouts") for line in without_row.splitlines()
    )


def broken_paragraphs(text: str) -> list[str]:
    """Lines of prose that directly follow another line of prose (a paragraph must be one line)."""
    lines = text.splitlines()

    def prose(line: str) -> bool:
        return bool(line.strip()) and not line.lstrip().startswith(("#", "-", "|", "```", ">"))

    return [b for a, b in itertools.pairwise(lines) if prose(a) and prose(b)]


# ------------------------------------------------------------------ the surrogates


def test_shuffled_keeps_counts_and_lengths_and_destroys_the_order():
    strings = chain(40, 30, 5, seed=101)
    surrogate = behaviour.shuffled(strings, seed=102)
    assert [len(s) for s in surrogate] == [len(s) for s in strings]
    assert Counter(x for s in surrogate for x in s) == Counter(x for s in strings for x in s)
    assert surrogate != strings
    assert any(a == b for s in surrogate for a, b in itertools.pairwise(s))  # repeats appear
    assert behaviour.shuffled(strings, seed=102) == surrogate
    # must reject: another seed gives another order, and the data have no repeat
    assert behaviour.shuffled(strings, seed=103) != surrogate
    assert not any(a == b for s in strings for a, b in itertools.pairwise(s))


def test_first_order_keeps_lengths_and_first_symbols_and_the_allowed_pairs():
    strings = chain(40, 30, 5, seed=101)
    surrogate = behaviour.first_order(strings, seed=102)
    assert [len(s) for s in surrogate] == [len(s) for s in strings]
    assert [s[0] for s in surrogate] == [s[0] for s in strings]
    seen = {(a, b) for s in strings for a, b in itertools.pairwise(s)}
    assert {(a, b) for s in surrogate for a, b in itertools.pairwise(s)} <= seen
    assert behaviour.first_order(strings, seed=102) == surrogate
    assert behaviour.surrogate(strings, "data", 0) == strings
    # must reject: another seed, a shuffle (which makes pairs the data never had), an unknown kind
    assert behaviour.first_order(strings, seed=103) != surrogate
    shuffled_pairs = {
        (a, b) for s in behaviour.shuffled(strings, 102) for a, b in itertools.pairwise(s)
    }
    assert not shuffled_pairs <= seen
    with pytest.raises(ValueError):
        behaviour.surrogate(strings, "second-order", 0)


def test_transition_table_is_counted_inside_strings_only():
    strings = [list("ABAB"), list("CAC"), list("B")]
    symbols, probabilities, counts = behaviour.transition_matrix(strings)
    assert symbols == ["A", "B", "C"]
    by_hand = np.array([[0, 2, 1], [1, 0, 0], [1, 0, 0]])  # AB AB AC / BA / CA
    assert np.array_equal(counts, by_hand)
    assert np.allclose(probabilities, by_hand / by_hand.sum(axis=1, keepdims=True))  # Eq. (1)
    # Eq. (2): a symbol that is never followed gets the distribution of first symbols
    _, with_dead_end, _ = behaviour.transition_matrix([list("AB"), list("AB"), list("CB")])
    assert np.allclose(with_dead_end[1], [2 / 3, 0, 1 / 3])
    # must reject: the table of the joined sequence, which also counts B->C and C->B across strings
    joined = np.zeros((3, 3))
    stream = [x for s in strings for x in s]
    for a, b in itertools.pairwise(stream):
        joined["ABC".index(a), "ABC".index(b)] += 1
    assert not np.array_equal(joined, counts) and joined[1, 2] == 1 and joined[2, 1] == 1


def test_closed_form_of_the_first_order_surrogate():
    strings = chain(60, 20, 4, seed=101)
    known = behaviour.first_order_entropy_rate(strings)
    estimates = np.array(
        [plug_in_rate(behaviour.first_order(strings, seed)) for seed in range(1000, 1200)]
    )
    standard_error = estimates.std(ddof=1) / math.sqrt(len(estimates))
    # first-order bias of the plug-in estimate: (possible pairs - symbols) / (2 n ln 2)
    possible = np.count_nonzero(behaviour.transition_matrix(strings)[1])
    bias = (possible - 4) / (2 * known["n_transitions"] * math.log(2))
    assert abs(estimates.mean() + bias - known["entropy_rate"]) < 3 * standard_error
    assert known["n_transitions"] == sum(len(s) - 1 for s in strings)
    # the spread of one estimate is what the tolerance of the check assumes, within a factor of two
    predicted = known["surprisal_sd"] / math.sqrt(known["n_transitions"])
    assert 0.5 * predicted < estimates.std(ddof=1) < 2.0 * predicted
    # one long string: Eq. (4) is the entropy rate under the stationary distribution
    long = chain(1, 20_000, 4, seed=104)
    known_long = behaviour.first_order_entropy_rate(long)
    assert known_long["entropy_rate"] == pytest.approx(known_long["stationary"], abs=1e-3)
    # must reject: the closed form of another chain, and the entropy of single symbols
    other = behaviour.first_order_entropy_rate(chain(60, 20, 4, seed=105))["entropy_rate"]
    assert abs(estimates.mean() + bias - other) > 3 * standard_error
    tokens = [x for s in strings for x in s]
    single = -sum(c / len(tokens) * math.log2(c / len(tokens)) for c in Counter(tokens).values())
    assert abs(estimates.mean() - single) > 0.1


def test_expected_visits_by_hand():
    # two symbols, A always followed by B and B by A: a string of length 4 from A visits A, B, A
    probabilities = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert np.allclose(behaviour.expected_visits(probabilities, [0], [4]), [2, 1])
    assert np.allclose(behaviour.expected_visits(probabilities, [0, 1], [4, 2]), [2, 2])
    assert np.allclose(behaviour.expected_visits(probabilities, [0], [1]), [0, 0])  # no transition
    assert np.allclose(behaviour.row_entropies(np.array([[0.5, 0.5], [1.0, 0.0]])), [1.0, 0.0])
    # must reject: the visits of a string that starts in B
    assert not np.allclose(behaviour.expected_visits(probabilities, [1], [4]), [2, 1])


# ------------------------------------------------------------------ a participant's own sequence


def test_pasted_text_and_csv_give_the_same_strings(work_dir):
    text = "A B C A B\nA,C,B\n\nABCA\n" + "A B " * 50
    strings = behaviour.parse_text(text)
    assert strings[:3] == [list("ABCAB"), list("ACB"), list("ABCA")]
    path = work_dir / "mine.csv"
    rows = [(i, symbol, 0.5) for i, string in enumerate(strings) for symbol in string]
    pd.DataFrame(rows, columns=["string", "symbol", "duration"]).to_csv(path, index=False)
    from_csv, durations = behaviour.read_sequence_csv(path)
    assert from_csv == strings and [len(d) for d in durations] == [len(s) for s in strings]
    own, report = behaviour.own_string_set(text=text)
    assert own.strings == strings and report["n_tokens"] == sum(len(s) for s in strings)
    via_path, _ = behaviour.own_string_set(text="ignored", path=str(path))
    assert via_path.strings == strings and via_path.durations is not None
    assert behaviour.own_string_set() is None and behaviour.own_string_set(text="  \n") is None
    assert "fewer than 10 symbols" in " ".join(report["remarks"])
    assert sorted(p.name for p in work_dir.iterdir()) == ["mine.csv"]  # nothing was written
    # must reject: a file that is absent, without the columns, with a negative or missing duration
    with pytest.raises(behaviour.OwnSequenceError, match="No file"):
        behaviour.read_sequence_csv(work_dir / "absent.csv")
    pd.DataFrame({"sequence": [1], "label": ["A"]}).to_csv(path, index=False)
    with pytest.raises(behaviour.OwnSequenceError, match="columns"):
        behaviour.read_sequence_csv(path)
    for bad in (-1.0, "long"):
        pd.DataFrame({"string": [1, 1], "symbol": ["A", "B"], "duration": [1.0, bad]}).to_csv(
            path, index=False
        )
        with pytest.raises(behaviour.OwnSequenceError, match="duration"):
            behaviour.read_sequence_csv(path)


@pytest.mark.parametrize(
    ("strings", "reason"),
    [
        ([], "No string"),
        ([["A", "B"] * 60, []], "empty"),
        ([["A"] * 200], "one symbol only"),
        ([["A", "B"] * 20], "at least 100"),
        ([["A", "B"] * 20_000], "at most 30000"),
        ([["A", "B c"] * 60], "space or an underscore"),
        ([["A", "B_c"] * 60], "space or an underscore"),
        ([["A", "#"] * 60], "reserved"),
        ([["A"], ["B"]] * 60, "no transition"),
        ([["A", ""] * 60], "empty or has spaces"),
    ],
)
def test_own_sequences_that_cannot_be_measured_are_refused_with_the_reason(strings, reason):
    """must reject: each of these inputs, with a sentence that says why."""
    with pytest.raises(behaviour.OwnSequenceError, match=reason):
        behaviour.validate_strings(strings)


def test_a_valid_own_sequence_is_described():
    report = behaviour.validate_strings([["A", "B", "C"] * 40, ["B", "A"] * 30])
    assert (report["n_strings"], report["n_tokens"], report["n_symbols"]) == (2, 180, 3)
    assert report["remarks"] == []
    few = behaviour.validate_strings([["A", "B", "C", "D", "E"] * 21])
    assert any("unreliable" in remark for remark in few["remarks"])


def test_ethogram_strip_has_one_block_per_bout_as_wide_as_it_lasts():
    strings = behaviour.StringSet(
        "test", [list("ABAC")], {"A": "walk", "B": "sit", "C": "eat"}, [[1.0, 3.0, 2.0, 0.5]]
    )
    strip = behaviour.ethogram_html(strings)
    assert strip.count("flex:") == 4 and "flex:3 1 0" in strip and "flex:0.5 1 0" in strip
    assert strip.count("A: walk") == 1 and "B: sit" in strip and "C: eat" in strip
    colours = {symbol: behaviour.SYMBOL_COLORS[i] for i, symbol in enumerate("ABC")}
    assert all(f"background:{colour}" in strip for colour in colours.values())
    assert len(set(behaviour.SYMBOL_COLORS)) == 20
    no_durations = behaviour.StringSet("test", [list("ABAC")], {}, None)
    assert behaviour.ethogram_html(no_durations).count("flex:1 1 0") == 4
    table = behaviour.symbol_table(strings)
    assert table.loc["A", "bouts"] == 2 and table.loc["A", "total time (s)"] == 3.0
    assert table["frequency"].sum() == pytest.approx(1.0)
    # must reject: a string without any positive duration
    with pytest.raises(ValueError, match="positive duration"):
        behaviour.ethogram_html(behaviour.StringSet("test", [list("AB")], {}, [[0.0, math.nan]]))
