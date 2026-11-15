"""The measures of notebook B: wrappers, known answers on the surrogates, saved results. Each test has a case it must reject."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from symseq.metrics.grammar import (
    markov_order_selection,
    mi_decay_analysis,
    topological_entropy,
    vlmc_fit,
)
from symseq.metrics.string import (
    compressibility,
    entropy,
    estimate_emc,
    lzw_complexity,
    permutation_entropy,
)
from symseq.metrics.stringset import pairwise_distances, string_set_entropy
from symseq.metrics.token import token_frequency
from test_behaviour import chain

from tutorial import behaviour
from tutorial import behaviour_measures as bm

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_behaviour_measures.py"
SMALL = replace(bm.SETTINGS, mi_shuffles=8, mi_max_distance=6, band_shuffles=100, n_compared=12)


def same_numbers(a, b, tolerance: float = 1e-9) -> bool:
    """True if two results hold the same structure and numbers (NaN equals NaN)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_numbers(a[key], b[key], tolerance) for key in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(
            same_numbers(x, y, tolerance) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, float | int) and isinstance(b, float | int) and not isinstance(a, bool):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))
    return a == b


@pytest.fixture(scope="module")
def strings():
    return chain(40, 30, 5, seed=101)


@pytest.fixture(scope="module")
def measured(strings):
    return bm.measure_dataset(strings, SMALL)


# ------------------------------------------------------------------ wrappers against SymSeq


def test_token_and_string_levels_are_symseq_values(strings):
    stream = [x for s in strings for x in s]
    assert bm.joined(strings) == stream
    token = bm.token_level(strings)
    assert token["frequencies"] == pytest.approx(token_frequency(stream))
    assert (token["types"], token["n_tokens"]) == (5, len(stream))
    level = bm.string_level(strings, SMALL)
    kept = [s for s in strings if len(s) >= 10]
    assert level["n_strings"] == len(kept)
    block = min(10, min(len(s) for s in kept) // 2)
    assert level["max_block_size"] == block
    per = level["per_string"]
    assert per["symbol_entropy"] == pytest.approx([entropy(s, bias_correction=True) for s in kept])
    assert per["gzip"] == pytest.approx([compressibility(s) for s in kept])
    assert per["lzw"] == pytest.approx([lzw_complexity(s) for s in kept])
    single = [estimate_emc(s, order="bic", max_order=block - 1) for s in kept]
    rates = [math.nan if r["entropy_rate"] is None else r["entropy_rate"] for r in single]
    assert per["entropy_rate"] == pytest.approx(rates, nan_ok=True)
    assert level["summary"]["entropy_rate"]["n"] == sum(not math.isnan(r) for r in rates)
    assert level["summary"]["entropy_rate"]["of"] == len(kept)
    assert level["summary"]["gzip"]["mean"] == pytest.approx(np.mean(per["gzip"]))
    # must reject: the entropy without the correction, and strings below the minimum length
    assert per["symbol_entropy"] != pytest.approx([entropy(s, bias_correction=False) for s in kept])
    assert bm.string_level([list("ABAB")], SMALL)["n_strings"] == 0


def test_string_set_level_is_symseq_values(strings):
    level = bm.stringset_level(strings, SMALL)
    compared = bm.compared_strings(strings, SMALL)
    assert len(compared) == 12 and all(len(s) <= SMALL.compared_length for s in compared)
    upper = np.triu_indices(len(compared), k=1)
    assert level["edit_distance"]["mean"] == pytest.approx(
        pairwise_distances(compared, metric="edit")[upper].mean()
    )
    assert level["edit_distance"]["n"] == 12 * 11 // 2
    assert level["set_entropy"] == pytest.approx(string_set_entropy(strings, use_frequencies=True))
    assert level["mean_length"] == pytest.approx(np.mean([len(s) for s in strings]))
    # the compared strings are a seeded draw: must reject another seed
    assert bm.compared_strings(strings, replace(SMALL, compared_seed=7)) != compared
    assert bm.compared_strings(strings, SMALL) == compared


def test_grammar_level_is_symseq_values(strings):
    level = bm.grammar_level(strings, SMALL)
    fit = markov_order_selection(strings, max_order=5, criterion="BIC", eos=None)
    n = fit["n_observations"]
    assert level["markov_order"] == fit["optimal_order"] and level["n_targets"] == n
    by_order = [-value / (n * math.log(2)) for value in fit["log_likelihoods"]]
    assert level["entropy_rate_by_order"] == pytest.approx(by_order)
    vlmc = vlmc_fit(strings, max_depth=5, eos=None)
    assert (level["vlmc_contexts"], level["vlmc_depth"]) == (
        vlmc["num_terminal_contexts"],
        vlmc["max_depth_used"],
    )
    stream = bm.joined(strings)
    direct = mi_decay_analysis(
        stream, max_distance=6, n_shuffles=8, confidence_level=0.95, random_state=42, units="bits"
    )
    assert level["mi"]["mi"] == pytest.approx(list(direct["mi_values"]))
    assert level["mi"]["upper"] == pytest.approx(list(direct["mi_ci_upper"]))
    assert level["mi"]["distances"] == [1, 2, 3, 4, 5, 6]
    _, lift, values = topological_entropy(
        sequence=stream, method="lift", max_lift=level["lift_limit"], verbose=False
    )
    assert level["topological_lift"] == pytest.approx(float(np.real(lift)) / math.log(2), abs=1e-8)
    assert len(level["lift_values"]) == len(values)
    assert level["density"] == pytest.approx(level["transitions"] / level["states"] ** 2)
    # must reject: another number of shuffles gives another band, and nats are not bits
    other = mi_decay_analysis(
        stream, max_distance=6, n_shuffles=9, confidence_level=0.95, random_state=42, units="bits"
    )
    assert level["mi"]["upper"] != pytest.approx(list(other["mi_ci_upper"]))
    nats = mi_decay_analysis(
        stream, max_distance=6, n_shuffles=8, confidence_level=0.95, random_state=42
    )
    assert level["mi"]["mi"] != pytest.approx(list(nats["mi_values"]))


def test_entropy_rate_by_order_is_the_conditional_entropy_of_one_string():
    """Eq. (1) on one string: order 0 is the entropy of the symbols, order 1 is SymSeq's h_1."""
    string = chain(1, 3000, 4, seed=103)[0]
    rates = bm.entropy_rate_by_order([string], 2)["entropy_rate_by_order"]
    assert rates[0] == pytest.approx(entropy(string[1:], bias_correction=False))
    assert rates[1] == pytest.approx(estimate_emc(string, order=1)["entropy_rate"])
    assert (
        rates[0] > rates[1] > rates[2] > 0
    )  # a plug-in conditional entropy never rises with the order
    # must reject: the rate at order 1 is not the entropy of the symbols
    assert rates[1] != pytest.approx(rates[0], abs=0.05)


def test_topological_entropy_is_in_bits_and_bounded_by_the_number_of_symbols():
    # the chain of notebook A: A->A, A->B, B->A, with log2 of the golden ratio
    graph = bm.observed_graph([list("AABABAAB")])
    assert (graph["states"], graph["transitions"], graph["density"]) == (2, 3, 0.75)
    assert graph["topological_direct"] == pytest.approx(0.694242, abs=1e-6)
    assert graph["topological_direct"] <= graph["topological_bound"] == 1.0
    # must reject: SymSeq's own value, which is in nats
    nats = topological_entropy(
        transitions=np.array([[1.0, 1.0], [1.0, 0.0]]), method="direct", verbose=False
    )
    assert nats == pytest.approx(0.481212, abs=1e-6) and nats != pytest.approx(
        graph["topological_direct"], abs=0.1
    )
    # a pair across two strings is no transition: B->B is never seen inside a string
    assert bm.observed_graph([list("AB"), list("BA")])["transitions"] == 2
    # all transitions seen: the bound is reached
    full = bm.observed_graph([list("AABBA")])
    assert full["topological_direct"] == pytest.approx(full["topological_bound"])


def test_topological_entropy_grows_with_the_amount_of_data(strings):
    rows = bm.against_tokens(strings, replace(SMALL, convergence_tokens=(50, 200, 800)))
    assert rows["n_tokens"] == [50, 200, 800, sum(len(s) for s in strings)]
    assert rows["transitions"] == sorted(rows["transitions"])
    assert rows["topological_direct"] == sorted(rows["topological_direct"])
    assert all(
        value <= bound + 1e-12
        for value, bound in zip(rows["topological_direct"], rows["topological_bound"], strict=True)
    )
    assert bm.first_tokens([list("ABC"), list("DE"), list("F")], 4) == [list("ABC"), ["D"]]
    # the surrogate has no order beyond the first, so its h_2 below its h_1 is bias, largest with few data
    gaps = [h1 - h2 for h1, h2 in zip(rows["h1_first_order"], rows["h2_first_order"], strict=True)]
    assert gaps[0] > gaps[-1] > 0
    # must reject: with all the data the graph has more transitions than with the first 50 symbols
    assert rows["transitions"][0] < rows["transitions"][-1]


def test_lift_estimate_is_limited_and_reports_what_it_tried(strings):
    stream = bm.joined(strings)
    free = bm.lift_estimate(stream, SMALL)
    assert free["lift_limit"] == bm.lift_limit(stream, SMALL) >= 2
    assert len(free["lift_values"]) <= free["lift_limit"] and not math.isnan(
        free["topological_lift"]
    )
    capped = bm.lift_estimate(stream, replace(SMALL, lift_max_blocks=10))
    assert capped["lift_limit"] == 1 and len(capped["lift_values"]) == 1
    assert bm.lift_limit(stream, bm.PREPRINT) == min(15, max(2, len(stream) // 1000))
    # must reject: a sequence below the minimum length gets no value, and says why
    short = bm.lift_estimate(stream[:100], SMALL)
    assert math.isnan(short["topological_lift"]) and "fewer than 500" in short["lift_note"]


def test_settings_are_the_preprint_values_except_the_declared_ones():
    ours, theirs = bm.SETTINGS.as_dict(), bm.PREPRINT.as_dict()
    changed = {key for key in ours if ours[key] != theirs[key]}
    assert changed == set(bm.CHEAPER)
    assert (
        theirs["n_compared"],
        theirs["compared_length"],
        theirs["mi_max_distance"],
        theirs["mi_shuffles"],
    ) == (50, 100, 50, 100)
    assert (
        theirs["min_string_length"],
        theirs["max_block_size"],
        theirs["markov_max_order"],
        theirs["lift_max"],
    ) == (10, 10, 5, 15)
    assert theirs["compared_seed"] == theirs["mi_seed"] == 42
    table = bm.settings_table()
    assert list(table.index[table["why changed"] != ""]) == [
        key for key in ours if key in bm.CHEAPER
    ]
    keys = [key for key, *_ in bm.MEASURES]
    assert len(keys) == len(set(keys)) and {level for _, level, *_ in bm.MEASURES} == set(bm.LEVELS)
    # must reject: a setting changed without a stated reason
    assert {
        key for key in ours if replace(bm.SETTINGS, vlmc_max_depth=3).as_dict()[key] != theirs[key]
    } != set(bm.CHEAPER)


# ------------------------------------------------------------------ two measures that are left out


def renamed(strings: list[list[str]], names: dict[str, str]) -> list[list[str]]:
    return [[names[symbol] for symbol in string] for string in strings]


def test_permutation_entropy_changes_when_the_labels_are_renamed(strings):
    """Why permutation entropy is not a measure of notebook B: it reads the alphabetical order of the labels.

    By hand, for the sequence A B A C repeated, with the order patterns of three successive
    symbols (ties keep their order in time): ABA, BAC, ACA, CAB give the patterns 021, 102, 021,
    120, with frequencies 1/2, 1/4, 1/4: 1.5 bit. With A and B exchanged the sequence is B A B C
    repeated: BAB, ABC, BCB, CBA give 102, 012, 021, 210, four patterns of frequency 1/4: 2 bit.
    """
    sequence = list("ABAC") * 50
    exchanged = renamed([sequence], {"A": "B", "B": "A", "C": "C"})[0]
    assert permutation_entropy(sequence) == pytest.approx(1.5, abs=0.02)
    assert permutation_entropy(exchanged) == pytest.approx(2.0, abs=0.02)
    # nothing about the sequence itself changed: the entropy rate at every order is the same
    rates = bm.entropy_rate_by_order([sequence], 3)["entropy_rate_by_order"]
    assert bm.entropy_rate_by_order([exchanged], 3)["entropy_rate_by_order"] == pytest.approx(
        rates, abs=1e-12
    )
    # the same on a seeded set of strings with five symbols: two of the labels exchanged
    swap = {"A": "E", "E": "A", "B": "B", "C": "C", "D": "D"}
    before = np.array([permutation_entropy(s) for s in strings])
    after = np.array([permutation_entropy(s) for s in renamed(strings, swap)])
    assert np.abs(after - before).max() > 0.05 and not np.allclose(after.mean(), before.mean())
    assert bm.entropy_rate_by_order(renamed(strings, swap), 2)[
        "entropy_rate_by_order"
    ] == pytest.approx(bm.entropy_rate_by_order(strings, 2)["entropy_rate_by_order"], abs=1e-12)
    # must reject: new names in the same alphabetical order change nothing, so the change above
    # comes from the order of the labels and not from the renaming as such
    same_order = renamed(strings, {"A": "P", "B": "Q", "C": "R", "D": "S", "E": "T"})
    assert np.array_equal(np.array([permutation_entropy(s) for s in same_order]), before)
    assert permutation_entropy(renamed([sequence], {"A": "X", "B": "Y", "C": "Z"})[0]) == (
        permutation_entropy(sequence)
    )


def keys_of(value) -> set[str]:
    """Every key of a nested result, at any depth."""
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in keys_of(item)}
    if isinstance(value, list):
        return {key for item in value for key in keys_of(item)}
    return set()


LEFT_OUT = {"permutation_entropy", "pairwise_mi"}


def test_the_two_measures_left_out_are_nowhere_and_edit_distance_stays(measured):
    assert not LEFT_OUT & {key for key, *_ in bm.MEASURES}
    assert not any(
        "permutation" in call or "mutual_information_strings" in call
        for _, _, call, _ in bm.MEASURES
    )
    assert ("edit_distance", "string set") in {(key, level) for key, level, *_ in bm.MEASURES}
    assert not LEFT_OUT & keys_of(bm.plain(measured))
    columns = {column for level in bm.LEVELS for column in bm.level_table(measured, level).columns}
    assert not LEFT_OUT & columns and "edit_distance" in columns
    assert measured["sets"]["data"]["string set"]["edit_distance"]["n"] == 12 * 11 // 2
    results, _ = bm.load_results()
    for name, result in results.items():  # the saved results of the four datasets
        assert not LEFT_OUT & keys_of(result), name
        assert result["sets"]["data"]["string set"]["edit_distance"]["n"] > 0, name
    assert not LEFT_OUT & keys_of(bm.figure_data(results))
    # must reject: a result that holds one of the two, at any depth
    assert (
        keys_of({"sets": {"data": {"string": {"per_string": [{"permutation_entropy": 1.0}]}}}})
        & LEFT_OUT
    )
    assert keys_of({"string set": {"pairwise_mi": {"mean": 0.1}}}) & LEFT_OUT


# ------------------------------------------------------------------ known answers on the surrogates


def test_memory_of_shuffles_is_the_first_order_memory(strings):
    tokens, previous, k = bm.pair_tables(strings)
    table = np.bincount(tokens[previous] * k + tokens[previous + 1], minlength=k * k).reshape(k, k)
    rates = bm.entropy_rate_by_order(strings, 1)["entropy_rate_by_order"]
    assert bm.pair_information(table) == pytest.approx(
        rates[0] - rates[1], abs=1e-9
    )  # numpy against SymSeq
    band = bm.memory_of_shuffles(strings, 50, seed=104)
    assert len(band) == 50 and band.max() < 0.1 * (rates[0] - rates[1])
    assert np.array_equal(band, bm.memory_of_shuffles(strings, 50, seed=104))
    # must reject: another seed gives other shuffles; the data are far above every shuffle
    assert not np.array_equal(band, bm.memory_of_shuffles(strings, 50, seed=105))
    assert rates[0] - rates[1] > band.max()


def test_surrogates_recover_their_known_answers(strings, measured):
    checks = measured["checks"]
    assert checks["passed"] and all(
        check["passed"] for key, check in checks.items() if key != "passed"
    )
    assert checks["first_order_entropy_rate"]["tolerance"] == pytest.approx(
        max(
            0.02,
            5 * measured["known"]["surprisal_sd"] / math.sqrt(measured["known"]["n_transitions"]),
        )
    )
    assert bm.allowed_outside(20) == 6 and bm.allowed_outside(6) < 6
    sets, known = measured["sets"], measured["known"]
    band = bm.memory_of_shuffles(strings, SMALL.band_shuffles, SMALL.band_seed)
    assert bm.surrogate_checks(sets, known, band) == checks
    # must reject: the data in the place of the shuffled surrogate
    as_shuffled = bm.surrogate_checks({**sets, "shuffled": sets["data"]}, known, band)
    assert not as_shuffled["passed"]
    assert (
        not as_shuffled["shuffled_memory"]["passed"]
        and not as_shuffled["shuffled_selects_order_0"]["passed"]
    )
    # must reject: the shuffled surrogate in the place of the first-order one
    as_first = bm.surrogate_checks({**sets, "first-order": sets["shuffled"]}, known, band)
    assert (
        not as_first["first_order_entropy_rate"]["passed"]
        and not as_first["first_order_selects_order_1"]["passed"]
    )
    # must reject: a sequence with slow order is outside the shuffle band at more distances than allowed
    sticky = np.random.default_rng(106).random(4000) < 0.03
    slow = ["AB"[state] for state in np.cumsum(sticky) % 2]
    decay = bm.mi_decay(slow, replace(SMALL, mi_max_distance=20))
    assert bm.outside_band(decay) > bm.allowed_outside(20)


def test_measures_are_seeded(strings, measured):
    again = bm.measure_dataset(strings, SMALL)
    assert same_numbers(bm.plain(again), bm.plain(measured))
    # must reject: another seed of a surrogate changes the result
    other = bm.measure_dataset(strings, replace(SMALL, first_order_seed=7))
    assert not same_numbers(bm.plain(other), bm.plain(measured))
    assert same_numbers(other["sets"]["data"], measured["sets"]["data"])
    assert not same_numbers({"a": [1.0, math.nan]}, {"a": [1.0, 2.0]}) and same_numbers(
        {"a": math.nan}, {"a": math.nan}
    )


def test_tables_of_a_result(measured):
    for level in bm.LEVELS:
        table = bm.level_table(measured, level)
        assert list(table.index) == ["data", "first-order surrogate", "shuffled"]
    assert list(bm.order_table(measured).index) == [0, 1, 2, 3, 4, 5]
    assert bm.checks_table(measured)["holds"].all()
    assert bm.tokens_table(measured).index.name == "symbols used"
    assert "observed first-order graph" in bm.lift_table(measured).columns
    data = bm.figure_data({"seal": measured}, highlight="seal", own=measured)
    assert data["datasets"] == ["seal", "own"] and data["highlight"] == ["seal", "own"]
    assert (
        data["entropy_rate"]["seal"]["data"]
        == measured["sets"]["data"]["grammar"]["entropy_rate_by_order"]
    )
    with pytest.raises(KeyError):
        bm.level_table(measured, "sentence")  # must reject an unknown level


# ------------------------------------------------------------------ saved results


def test_saved_results_match_their_manifest_and_the_shipped_tables(work_dir):
    results, manifest = bm.load_results()
    assert (
        list(results) == list(behaviour.DATASETS)
        and manifest["complete"]
        and not manifest["checks_failed"]
    )
    assert manifest["settings"] == bm.plain(bm.SETTINGS.as_dict())
    bm.check_against_tables(results)
    copy = work_dir / "results"
    shutil.copytree(bm.RESULTS, copy)
    assert bm.load_results(copy)[0].keys() == results.keys()
    # must reject: a result file that is not listed, a changed file, a result from another table
    (copy / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="not in the manifest"):
        bm.load_results(copy)
    (copy / "extra.json").unlink()
    changed = json.loads((copy / "seal.json").read_text())
    changed["n_tokens"] += 1
    (copy / "seal.json").write_text(json.dumps(changed, indent=1))
    with pytest.raises(ValueError, match="changed after"):
        bm.load_results(copy)
    other_table = {
        **results,
        "seal": {**results["seal"], "table_sha256": results["turtle"]["table_sha256"]},
    }
    with pytest.raises(ValueError, match="another table"):
        bm.check_against_tables(other_table)
    (copy / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        bm.load_results(copy)


def test_every_dataset_meets_the_known_answers_and_the_checks_can_say_no():
    results, _ = bm.load_results()
    for name, result in results.items():
        checks = result["checks"]
        assert checks["passed"], name
        # must reject: the data themselves are far above the memory band of the shuffles ...
        assert (
            checks["shuffled_memory"]["memory_of_the_data"] > checks["shuffled_memory"]["threshold"]
        ), name
        # ... and the shuffled surrogate is far from the closed form of the first-order one
        shuffled_rate = result["sets"]["shuffled"]["grammar"]["entropy_rate_by_order"][1]
        first = checks["first_order_entropy_rate"]
        assert abs(shuffled_rate - first["closed_form"]) > first["tolerance"], name
        assert result["sets"]["data"]["token"]["n_tokens"] == result["n_tokens"]
        for kind in behaviour.SURROGATES:  # the surrogates have the size of the data
            assert result["sets"][kind]["token"]["n_tokens"] == result["n_tokens"]


def run_script(*arguments: str) -> subprocess.CompletedProcess:
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_identical_seeds_give_identical_files_and_the_script_resumes(work_dir):
    first, second = work_dir / "first", work_dir / "second"
    for folder in (first, second):
        done = run_script("--datasets", "seal", "--out", str(folder))
        assert done.returncode == 0, done.stderr[-2000:]
    digest = bm.sha256(first / "seal.json")
    assert digest == bm.sha256(second / "seal.json")
    assert digest == bm.sha256(bm.RESULTS / "seal.json")  # and the saved result of the repository
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["files"] == {"seal.json": digest} and not manifest["complete"]
    assert manifest["missing"] == [
        "zebrafish",
        "turtle",
        "finch",
    ]  # absent results are named, not filled
    # a second call computes nothing and rewrites no result
    modified = (first / "seal.json").stat().st_mtime_ns
    again = run_script("--datasets", "seal", "--out", str(first))
    assert (
        "saved result kept" in again.stdout and (first / "seal.json").stat().st_mtime_ns == modified
    )
    # a deleted result is computed again, with the same bytes
    (first / "seal.json").unlink()
    recomputed = run_script("--datasets", "seal", "--out", str(first))
    assert "checks passed" in recomputed.stdout and bm.sha256(first / "seal.json") == digest
    # must reject: another seed of a surrogate gives another result, and the loader refuses a partial folder's extras
    seal = behaviour.load_strings("seal").strings
    other = bm.measure_dataset(seal, replace(bm.SETTINGS, shuffled_seed=7))
    saved = json.loads((first / "seal.json").read_text())
    assert not same_numbers(bm.plain(other)["sets"]["shuffled"], saved["sets"]["shuffled"])
    assert same_numbers(bm.plain(other)["sets"]["data"], saved["sets"]["data"])


def test_live_computation_equals_the_saved_result():
    results, _ = bm.load_results()
    live = bm.plain(bm.measure_dataset(behaviour.load_strings("seal").strings, bm.SETTINGS))
    saved = {
        key: value
        for key, value in results["seal"].items()
        if key not in ("dataset", "table_sha256")
    }
    assert same_numbers(live, saved)
    # must reject: the saved result of another dataset
    other = {
        key: value
        for key, value in results["turtle"].items()
        if key not in ("dataset", "table_sha256")
    }
    assert not same_numbers(live, other)
