"""Open behavioural datasets as sets of strings, two surrogates of a string set, and a participant's own sequence.

Each dataset is shipped as one table, ``data/behaviour/<dataset>.csv.gz``, written by
``scripts/build_behaviour_sequences.py`` from the public source files. A row is one bout; for the
turtles and the seals a bout of one benchmark class can hold several raw labels, and then it has
one row per raw label, all with the same ``bout`` number. Sources, licences and the changes made are
in ``data/behaviour/SOURCES.md`` and in ``DATASETS`` below.

## Theory

Two surrogates are built from a set of strings s_1..s_T over K symbols.

Shuffled. All tokens of the set are permuted once and cut into strings of the original lengths.
The token counts are kept exactly; every dependence between positions is destroyed.

First-order. With N(i, j) the number of times symbol j follows symbol i inside a string,

    P[i, j] = N(i, j) / sum_j N(i, j)                                            Eq. (1)

A symbol that is never followed by another symbol gets the distribution of first symbols as its
row (Eq. 2). Surrogate string t starts with the first symbol of s_t, has the length of s_t, and
draws each next symbol from the row of P of the current symbol. Transition probabilities, first
symbols and string lengths are those of the data; everything else is what a first-order chain
gives.

Known answer for the first-order surrogate. The expected number of transitions out of each symbol
is

    w = sum_t sum_{n=0}^{L_t - 2} e(s_t[0]) P^n                                  Eq. (3)

(e(i) the unit row vector of symbol i, L_t the length of string t), and the entropy rate is

    h = sum_i w_i H(P[i, :]) / sum_i w_i,   H(p) = -sum_j p_j log2 p_j           Eq. (4)

in bits per symbol. For long strings w / sum(w) tends to the stationary distribution of P and h to
the entropy rate of the chain. Ref: Cover and Thomas, Elements of Information Theory, 2nd ed.,
Theorem 4.2.4.
"""

from __future__ import annotations

import hashlib
import html
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from symseq.metrics.grammar import context_counts

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "behaviour"
MANIFEST = "MANIFEST.sha256"
COLUMNS = (
    "individual",
    "recording",
    "string",
    "bout",
    "symbol",
    "behaviour",
    "raw_label",
    "duration_s",
)
EOS = "#"

# Choices of the analysis that the preprocessing follows (comparative_analysis_v9.py).
MIN_STRING_LENGTH = 10
MAX_STRINGS = 500
SUBSAMPLE_SEED = 42

# The finch with the most annotated songs in the source files. The build script recomputes this
# from the raw files and stops if it finds another bird.
FINCH_BIRD = "gr41rd51"

SURROGATES = ("data", "first-order", "shuffled")
SURROGATE_LABELS = {
    "data": "data",
    "first-order": "first-order surrogate",
    "shuffled": "shuffled",
}

# Limits for a participant's own sequence: below the first nothing can be estimated, above the
# second the measures take more than about a minute on a laptop.
OWN_MIN_TOKENS = 100
OWN_MAX_TOKENS = 30_000

DATASETS: dict[str, dict] = {
    "zebrafish": {
        "label": "zebrafish",
        "short": "zebrafish",
        "animal": "larval zebrafish (Danio rerio), swimming bouts",
        "title": "Dataset for Uncovering multiscale structure in the variability of larval zebrafish navigation V2",
        "source_words": "463 freely swimming larval zebrafish were recorded from 30 min to 3 h (from the article of the dataset).",
        "citation": (
            "Sridhar G, Costa A (2024). Dataset for Uncovering multiscale structure in the variability of larval "
            "zebrafish navigation V2. Zenodo."
        ),
        "doi": "10.5281/zenodo.13605471",
        "papers": [
            "Sridhar G, Vergassola M, Marques JC, Orger MB, Costa AC, Wyart C (2024). Uncovering multiscale "
            "structure in the variability of larval zebrafish navigation. Proceedings of the National Academy "
            "of Sciences 121(47), e2410254121. https://doi.org/10.1073/pnas.2410254121",
            "Marques JC, Lackner S, Félix R, Orger MB (2018). Structure of the zebrafish locomotor repertoire "
            "revealed with unsupervised behavioral clustering. Current Biology 28(2), 181-195. "
            "https://doi.org/10.1016/j.cub.2017.12.002 (the thirteen bout types)",
        ],
        "licence": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "string_is": "all bouts of one fish",
        "changes": [
            "Only the bout type and the start and end frame of each bout were read. The thirteen bout types "
            "carry the names of the source file (for example RT, Slow1, J_turn); they are described in "
            "Marques et al. (2018).",
            "Bout durations are end frame minus start frame, divided by the frame rate of 700 per second.",
            "Every bout is one symbol with its own duration; consecutive bouts of the same type were not merged.",
            "A seeded random subset of the fish was kept.",
        ],
    },
    "turtle": {
        "label": "green turtle",
        "short": "green turtle",
        "animal": "free-ranging immature green turtles (Chelonia mydas), Martinique",
        "title": (
            "Raw acceleration, gyroscope and depth profiles associated with the observed behaviours of "
            "free-ranging immature green turtles in Martinique"
        ),
        "source_words": (
            "The collected dataset contains the acceleration, angular velocity and depth profile of thirteen "
            "free-ranging immature green turtles. The video footage and the visualisation of the multi-sensors "
            "data allowed us to identify 46 behaviours with a specific signal (from the README of the dataset)."
        ),
        "citation": (
            "Jeantet L, Planas-Bielsa V, Benhamou S, Geiger S, Martin J, Siegwalt F, Lelong P, Gresser J, "
            "Etienne D, Hiélard G, Arque A, Regis S, Lecerf N, Frouin C, Benhalilou A, Murgale C, Maillet T, "
            "Andreani L, Campistron G, Delvaux H, Guyon C, Richard S, Lefebvre F, Aubert N, Habold C, "
            "Le Maho Y, Chevallier D (2020). Raw acceleration, gyroscope and depth profiles associated with "
            "the observed behaviours of free-ranging immature green turtles in Martinique. Dryad."
        ),
        "doi": "10.5061/dryad.hhmgqnkd9",
        "papers": [
            "Jeantet L et al. (2020). Behavioural inference from signal processing using animal-borne "
            "multi-sensor loggers: a novel solution to extend the knowledge of sea turtle ecology. Royal "
            "Society Open Science 7(5), 200139. https://doi.org/10.1098/rsos.200139",
            "Hoffman B et al. (2024). A benchmark for computational analysis of animal behavior, using "
            "animal-borne tags. Movement Ecology 12, 78. https://doi.org/10.1186/s40462-024-00511-8 (the "
            "behaviour classes; the files were read from the copy in this benchmark, "
            "https://doi.org/10.5281/zenodo.7947104)",
        ],
        "licence": "CC0 1.0",
        "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "string_is": "the bouts between two rests in one recording",
        "changes": [
            "Only the tables of observed behaviours were read, in the order of their rows.",
            "Each observed label was mapped to its class in the benchmark (breathing, feeding, gliding, "
            "resting, scratching, staying at surface, swimming, unknown); the label itself is kept in the "
            "column raw_label.",
            "Consecutive observations of the same class were merged into one bout.",
            "Resting was removed and used to cut each recording into strings.",
            "Strings of fewer than ten bouts were removed.",
            "A duration is the stop time minus the start time of the source; where that is not positive it "
            "is left empty.",
            "The animal of each recording is the number given in the benchmark, which counts 14 animals in "
            "the 16 recordings where the source describes thirteen turtles.",
        ],
    },
    "finch": {
        "label": "Bengalese finch",
        "short": "Bengalese finch",
        "animal": "adult male Bengalese finches (Lonchura striata domestica), song syllables",
        "title": "Bengalese Finch song repository",
        "source_words": (
            "Bengalese Finch song repository (title of the dataset). Syllable labels are specific to each "
            "bird: the same letter in two birds does not name the same syllable."
        ),
        "citation": "Nicholson D, Queen JE, Sober SJ (2017). Bengalese Finch song repository. figshare.",
        "doi": "10.6084/m9.figshare.4805749",
        "papers": [],
        "licence": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "string_is": "the syllables of one song file",
        "changes": [
            "Only the annotation files were read: the syllable labels and their onset and offset times.",
            "A duration is offset minus onset.",
            "Songs of fewer than ten syllables were removed.",
            "At most 500 songs per bird were kept, chosen at random with a fixed seed.",
            "The birds are kept apart. The notebook uses the bird with the most annotated songs.",
        ],
    },
    "seal": {
        "label": "fur seals and sea lions",
        "short": "seals and sea lions",
        "animal": "captive fur seals and sea lions (four otariid species)",
        "title": "Seals dataset",
        "source_words": (
            "Seals dataset (title in the licence file of the dataset), from the study Seeing It All: Evaluating "
            "Supervised Machine Learning Methods for the Classification of Diverse Otariid Behaviours."
        ),
        "citation": (
            "Hoffman B, Cusimano M, Baglione V, Canestrari D, Chevallier D, DeSantis DL, Jeantet L, Ladds MA, "
            "Maekawa T, Mata-Silva V, Moreno-González V, Trapote E, Vainio O, Vehkaoja A, Yoda K, Zacarian K, "
            "Friedlaender A, Rutz C (2023). Bio-logger Ethogram Benchmark: A benchmark for computational "
            "analysis of animal behavior, using animal-borne tags (1.0.0). Zenodo. Seals dataset by Ladds MA, "
            "Thompson AP, Slip DJ, Hocking DP, Harcourt RG."
        ),
        "doi": "10.5281/zenodo.7947104",
        "papers": [
            "Ladds MA, Thompson AP, Slip DJ, Hocking DP, Harcourt RG (2016). Seeing it all: evaluating "
            "supervised machine learning methods for the classification of diverse otariid behaviours. "
            "PLOS ONE 11(12), e0166898. https://doi.org/10.1371/journal.pone.0166898",
            "Hoffman B et al. (2024). A benchmark for computational analysis of animal behavior, using "
            "animal-borne tags. Movement Ecology 12, 78. https://doi.org/10.1186/s40462-024-00511-8",
        ],
        "licence": "CC BY 4.0",
        "licence_url": "https://creativecommons.org/licenses/by/4.0/",
        "string_is": "the bouts between two rests in one recording",
        "changes": [
            "Only the behaviour columns of each recording were read, sample by sample.",
            "Each sample was given its class in the benchmark (feeding, grooming, resting, travelling, "
            "unknown); the finer label of the source is kept in the column raw_label.",
            "Consecutive samples of the same class were merged into one bout.",
            "Resting was removed and used to cut each recording into strings.",
            "Strings of fewer than ten bouts were removed.",
            "A duration is the number of samples divided by the sampling rate of 25 per second.",
        ],
    },
}

# Colours of the symbols in the ethogram strip: the ten strong colours of matplotlib's tab20, then
# the ten light ones; repeated beyond twenty symbols.
SYMBOL_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
)


class ManifestError(ValueError):
    """A shipped table is absent from the manifest or differs from it."""


class OwnSequenceError(ValueError):
    """A participant's sequence cannot be measured; the message says why."""


@dataclass(frozen=True)
class StringSet:
    """A set of strings with the names of its symbols."""

    name: str
    strings: list[list[str]]
    names: dict[str, str]  # symbol -> name of the behaviour in the source
    durations: list[list[float]] | None = None  # seconds, one per symbol; None if unknown
    individuals: list[str] = field(default_factory=list)  # one per string
    level: str = "class"

    @property
    def n_tokens(self) -> int:
        return sum(len(string) for string in self.strings)

    @property
    def alphabet(self) -> list[str]:
        return sorted({symbol for string in self.strings for symbol in string})


# --------------------------------------------------------------------------------------------
# Shipped tables
# --------------------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_manifest(folder: Path = DATA) -> dict[str, str]:
    """File name -> SHA-256, from ``MANIFEST.sha256`` (the format of the ``sha256sum`` tool)."""
    entries = {}
    for line in (Path(folder) / MANIFEST).read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            entries[name.strip().lstrip("*")] = digest
    return entries


def check_folder(folder: Path = DATA) -> dict[str, str]:
    """Check that the folder holds the declared datasets and nothing else, and that every hash matches.

    Raises ``ManifestError`` for a file that is not declared, for a declared file that is absent
    and for a file that differs from the manifest.
    """
    folder = Path(folder)
    declared = {f"{name}.csv.gz" for name in DATASETS} | {"SOURCES.md"}
    present = {path.name for path in folder.iterdir()} - {MANIFEST}
    if present - declared:
        raise ManifestError(
            f"Files of no declared dataset in {folder}: {sorted(present - declared)}."
        )
    if declared - present:
        raise ManifestError(f"Missing from {folder}: {sorted(declared - present)}.")
    manifest = read_manifest(folder)
    if set(manifest) != declared:
        raise ManifestError(f"The manifest lists {sorted(manifest)}, expected {sorted(declared)}.")
    for name, digest in manifest.items():
        if sha256(folder / name) != digest:
            raise ManifestError(f"{name} differs from the manifest.")
    return manifest


def load_table(dataset: str, folder: Path = DATA, verify: bool = True) -> pd.DataFrame:
    """The shipped table of a dataset, after checking its SHA-256 against the manifest."""
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset!r}. Choose one of {sorted(DATASETS)}.")
    path = Path(folder) / f"{dataset}.csv.gz"
    if verify:
        digest = read_manifest(folder).get(path.name)
        if digest is None or sha256(path) != digest:
            raise ManifestError(f"{path.name} is absent from the manifest or differs from it.")
    table = pd.read_csv(
        path,
        dtype={
            "individual": str,
            "recording": str,
            "string": int,
            "bout": int,
            "symbol": str,
            "behaviour": str,
            "raw_label": str,
        },
        keep_default_na=False,
        na_values={"duration_s": [""]},
    )
    if tuple(table.columns) != COLUMNS:
        raise ValueError(
            f"{path.name} has columns {list(table.columns)}, expected {list(COLUMNS)}."
        )
    return table


def letters(n: int) -> list[str]:
    """The first n symbols: A to Z, then AA, AB and so on."""
    if n <= 26:
        return [chr(65 + i) for i in range(n)]
    return [chr(65 + i // 26) + chr(65 + i % 26) for i in range(n)]


def load_strings(
    dataset: str, individual: str | None = None, level: str = "class", folder: Path = DATA
) -> StringSet:
    """A dataset as a set of strings.

    Parameters
    ----------
    dataset : str
        One of ``DATASETS``.
    individual : str, optional
        Keep the strings of one animal. For the finch the default is the bird with the most
        annotated songs, because syllable labels are specific to each bird and birds are never
        pooled.
    level : str
        ``"class"`` (default): one symbol per bout. ``"raw"``: one symbol per raw label, for the
        datasets that have one (turtle, seal); the symbols are then assigned here, in the
        alphabetical order of the labels.
    """
    table = load_table(dataset, folder)
    if dataset == "finch" and individual is None:
        individual = FINCH_BIRD
    if individual is not None:
        table = table[table["individual"] == individual]
        if table.empty:
            raise KeyError(f"No animal {individual!r} in the dataset {dataset!r}.")
    if level == "class":
        bouts = table.groupby(["string", "bout"], sort=True).agg(
            individual=("individual", "first"),
            symbol=("symbol", "first"),
            behaviour=("behaviour", "first"),
            duration_s=("duration_s", lambda values: values.sum(skipna=False)),
        )
        names = dict(zip(bouts["symbol"], bouts["behaviour"], strict=True))
    elif level == "raw":
        if (table["raw_label"] == "").all():
            raise ValueError(f"The dataset {dataset!r} has no raw labels.")
        labels = sorted(table["raw_label"].unique())
        symbol_of = dict(zip(labels, letters(len(labels)), strict=True))
        bouts = table.set_index(["string", "bout"]).assign(
            symbol=table["raw_label"].map(symbol_of).to_numpy()
        )
        names = {symbol: label for label, symbol in symbol_of.items()}
    else:
        raise ValueError(f"level must be 'class' or 'raw', got {level!r}.")
    strings, durations, individuals = [], [], []
    for _, rows in bouts.groupby(level="string", sort=True):
        strings.append(rows["symbol"].tolist())
        durations.append(rows["duration_s"].tolist())
        individuals.append(rows["individual"].iloc[0])
    name = dataset if individual is None or dataset != "finch" else f"{dataset} ({individual})"
    return StringSet(name, strings, dict(sorted(names.items())), durations, individuals, level)


def symbol_table(string_set: StringSet) -> pd.DataFrame:
    """One row per symbol: its name, how often it occurs and how long its bouts last."""
    counts = Counter(symbol for string in string_set.strings for symbol in string)
    total = sum(counts.values())
    per_symbol: dict[str, list[float]] = {symbol: [] for symbol in counts}
    if string_set.durations is not None:
        for string, durations in zip(string_set.strings, string_set.durations, strict=True):
            for symbol, duration in zip(string, durations, strict=True):
                if duration == duration:  # not NaN
                    per_symbol[symbol].append(duration)
    rows = []
    for symbol in sorted(counts):
        durations = per_symbol[symbol]
        rows.append(
            {
                "symbol": symbol,
                "behaviour": string_set.names.get(symbol, symbol),
                "bouts": counts[symbol],
                "frequency": counts[symbol] / total,
                "median duration (s)": float(np.median(durations)) if durations else float("nan"),
                "total time (s)": float(np.sum(durations)) if durations else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("symbol")


def ethogram_html(string_set: StringSet, index: int = 0, max_bouts: int = 150) -> str:
    """An ethogram strip of one string as HTML: one coloured block per bout, as wide as the bout lasts.

    Without durations every block has the same width. Only the first ``max_bouts`` bouts are drawn.
    """
    string = string_set.strings[index][:max_bouts]
    if string_set.durations is None:
        widths = [1.0] * len(string)
    else:
        widths = [
            duration if duration == duration and duration > 0 else 0.0
            for duration in string_set.durations[index][:max_bouts]
        ]
    if not string or sum(widths) <= 0:
        raise ValueError("The string has no bout with a positive duration.")
    colour = {
        symbol: SYMBOL_COLORS[i % len(SYMBOL_COLORS)]
        for i, symbol in enumerate(string_set.alphabet)
    }
    blocks = "".join(
        f'<span title="{html.escape(string_set.names.get(symbol, symbol))}" '
        f'style="flex:{width:.6g} 1 0;background:{colour[symbol]};min-width:0"></span>'
        for symbol, width in zip(string, widths, strict=True)
    )
    legend = "".join(
        f'<span style="white-space:nowrap;margin-right:1em"><span style="display:inline-block;width:0.9em;'
        f'height:0.9em;background:{colour[symbol]};margin-right:0.3em"></span>'
        f"{html.escape(symbol)}: {html.escape(string_set.names.get(symbol, symbol))}</span>"
        for symbol in sorted(set(string))
    )
    return (
        f'<div style="display:flex;height:2.2em;width:100%;border:1px solid #888">{blocks}</div>'
        f'<div style="font-size:0.85em;margin-top:0.4em;line-height:1.6">{legend}</div>'
    )


# --------------------------------------------------------------------------------------------
# Surrogates and their known answers
# --------------------------------------------------------------------------------------------


def transition_matrix(
    strings: Sequence[Sequence[str]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Symbols (sorted), the transition probabilities of Eq. (1) and (2), and the transition counts.

    The counts come from SymSeq's ``context_counts(strings, max_depth=1)``, which never counts a
    transition from the last symbol of one string to the first symbol of the next.
    """
    strings = [list(string) for string in strings]
    symbols = sorted({symbol for string in strings for symbol in string})
    index = {symbol: i for i, symbol in enumerate(symbols)}
    counts = np.zeros((len(symbols), len(symbols)))
    if any(len(string) > 1 for string in strings):
        counted = context_counts(strings, max_depth=1)["counts"]
        for context, following in counted.items():
            if len(context) == 1:
                for symbol, count in following.items():
                    counts[index[context[0]], index[symbol]] = count
    starts = np.zeros(len(symbols))
    for string in strings:
        starts[index[string[0]]] += 1
    probabilities = np.zeros_like(counts)
    for i, row in enumerate(counts):
        # Eq. (1); Eq. (2) for a symbol that is never followed by another one
        probabilities[i] = row / row.sum() if row.sum() > 0 else starts / starts.sum()
    return symbols, probabilities, counts


def shuffled(strings: Sequence[Sequence[str]], seed: int) -> list[list[str]]:
    """All tokens of the set in random order, cut into strings of the original lengths."""
    tokens = [symbol for string in strings for symbol in string]
    order = np.random.default_rng(seed).permutation(len(tokens))
    surrogate, position = [], 0
    for string in strings:
        surrogate.append([tokens[i] for i in order[position : position + len(string)]])
        position += len(string)
    return surrogate


def first_order(strings: Sequence[Sequence[str]], seed: int) -> list[list[str]]:
    """Strings of a first-order Markov chain with the transition probabilities of the data.

    Each surrogate string has the first symbol and the length of the string it replaces.
    """
    symbols, probabilities, _ = transition_matrix(strings)
    index = {symbol: i for i, symbol in enumerate(symbols)}
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0  # guard against rounding in the last column
    rng = np.random.default_rng(seed)
    surrogate = []
    for string in strings:
        draws = rng.random(len(string) - 1)
        state = index[string[0]]
        states = [state]
        for draw in draws:
            state = int(np.searchsorted(cumulative[state], draw, side="right"))
            states.append(state)
        surrogate.append([symbols[state] for state in states])
    return surrogate


def surrogate(strings: Sequence[Sequence[str]], kind: str, seed: int) -> list[list[str]]:
    """The string set itself (``"data"``) or one of its two surrogates."""
    if kind == "data":
        return [list(string) for string in strings]
    if kind == "shuffled":
        return shuffled(strings, seed)
    if kind == "first-order":
        return first_order(strings, seed)
    raise ValueError(f"Unknown surrogate {kind!r}. Choose one of {SURROGATES}.")


def row_entropies(probabilities: np.ndarray) -> np.ndarray:
    """Entropy of each row of a transition matrix, in bits."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0, -probabilities * np.log2(probabilities), 0.0)
    return terms.sum(axis=1)


def expected_visits(
    probabilities: np.ndarray, first_states: Sequence[int], lengths: Sequence[int]
) -> np.ndarray:
    """Eq. (3): expected number of transitions out of each symbol, for chains of given starts and lengths."""
    n_states = probabilities.shape[0]
    visits = np.zeros(n_states)
    by_start: dict[int, Counter] = {}
    for state, length in zip(first_states, lengths, strict=True):
        by_start.setdefault(state, Counter())[length] += 1
    for state, length_counts in by_start.items():
        longest = max(length_counts)
        # number of strings from this start that still have a transition at step n
        alive = np.zeros(max(longest, 1))
        for length, count in length_counts.items():
            alive[: max(length - 1, 0)] += count
        distribution = np.zeros(n_states)
        distribution[state] = 1.0
        for step in range(longest - 1):
            visits += alive[step] * distribution
            distribution = distribution @ probabilities
    return visits


def first_order_entropy_rate(strings: Sequence[Sequence[str]]) -> dict:
    """Known answer for the first-order surrogate of ``strings``: Eq. (4), with what it is made of.

    Returns the entropy rate in bits per symbol (``entropy_rate``), the standard deviation of the
    surprisal of one transition (``surprisal_sd``), the number of transitions (``n_transitions``),
    and the entropy rate under the stationary distribution of the chain (``stationary``), which
    Eq. (4) approaches for long strings.
    """
    symbols, probabilities, _ = transition_matrix(strings)
    index = {symbol: i for i, symbol in enumerate(symbols)}
    visits = expected_visits(
        probabilities, [index[string[0]] for string in strings], [len(s) for s in strings]
    )
    weights = visits / visits.sum()
    entropies = row_entropies(probabilities)
    rate = float(weights @ entropies)
    with np.errstate(divide="ignore", invalid="ignore"):
        surprisal = np.where(probabilities > 0, -np.log2(probabilities), 0.0)
    variance = float(weights @ (probabilities * (surprisal - rate) ** 2).sum(axis=1))
    return {
        "entropy_rate": rate,
        "surprisal_sd": float(np.sqrt(variance)),
        "n_transitions": int(sum(len(string) - 1 for string in strings)),
        "stationary": stationary_entropy_rate(probabilities),
        "symbols": symbols,
    }


def stationary_entropy_rate(probabilities: np.ndarray) -> float:
    """Entropy rate of a chain under its stationary distribution, bits per symbol (NaN if not unique)."""
    values, vectors = np.linalg.eig(probabilities.T)
    close = np.isclose(values, 1.0, atol=1e-9)
    if close.sum() != 1:
        return float("nan")
    stationary = np.real(vectors[:, close][:, 0])
    stationary = stationary / stationary.sum()
    return float(stationary @ row_entropies(probabilities))


# --------------------------------------------------------------------------------------------
# A participant's own sequence
# --------------------------------------------------------------------------------------------


def parse_text(text: str) -> list[list[str]]:
    """Strings from pasted text, one string per line.

    Symbols are separated by spaces, tabs, commas or semicolons. A line without any separator is
    read one character per symbol. Empty lines are skipped.
    """
    strings = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part for part in re.split(r"[\s,;]+", line) if part]
        strings.append(parts if len(parts) > 1 else list(line))
    return strings


def read_sequence_csv(path: str | Path) -> tuple[list[list[str]], list[list[float]] | None]:
    """Strings from a CSV file with the columns ``string`` and ``symbol``, and optionally ``duration``.

    Rows are kept in the order of the file; rows with the same value of ``string`` form one string.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise OwnSequenceError(f"No file at {path}.")
    try:
        table = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as error:
        raise OwnSequenceError(f"{path.name} could not be read as CSV: {error}") from error
    missing = {"string", "symbol"} - set(table.columns)
    if missing:
        raise OwnSequenceError(
            f"{path.name} needs the columns 'string' and 'symbol'; missing: {sorted(missing)}."
        )
    has_duration = "duration" in table.columns
    if has_duration:
        durations = pd.to_numeric(table["duration"], errors="coerce")
        if durations.isna().any() or (durations < 0).any():
            raise OwnSequenceError("The column 'duration' must hold numbers that are not negative.")
        table = table.assign(duration=durations)
    strings, all_durations = [], []
    for _, rows in table.groupby("string", sort=False):
        strings.append(rows["symbol"].tolist())
        if has_duration:
            all_durations.append(rows["duration"].tolist())
    return strings, (all_durations if has_duration else None)


def validate_strings(strings: Sequence[Sequence[str]]) -> dict:
    """Check that a set of strings can be measured, and describe it.

    Raises ``OwnSequenceError`` with the reason. Returns the numbers of strings, tokens and symbols
    and a list of remarks on what the measures will and will not see.
    """
    if not strings:
        raise OwnSequenceError("No string was found.")
    for string in strings:
        if len(string) == 0:
            raise OwnSequenceError("One string is empty.")
        for symbol in string:
            if not isinstance(symbol, str) or not symbol or symbol != symbol.strip():
                raise OwnSequenceError(f"Symbol {symbol!r} is empty or has spaces around it.")
            if any(character.isspace() for character in symbol) or "_" in symbol:
                raise OwnSequenceError(
                    f"Symbol {symbol!r} contains a space or an underscore; use letters or short names."
                )
            if symbol == EOS:
                raise OwnSequenceError(f"The symbol {EOS!r} is reserved for the end of a string.")
    n_tokens = sum(len(string) for string in strings)
    symbols = {symbol for string in strings for symbol in string}
    if len(symbols) < 2:
        raise OwnSequenceError("The sequence has one symbol only; there is nothing to measure.")
    if n_tokens < OWN_MIN_TOKENS:
        raise OwnSequenceError(
            f"The sequence has {n_tokens} symbols; at least {OWN_MIN_TOKENS} are needed."
        )
    if n_tokens > OWN_MAX_TOKENS:
        raise OwnSequenceError(
            f"The sequence has {n_tokens} symbols; the notebook measures at most {OWN_MAX_TOKENS}. "
            "Keep a part of it."
        )
    if all(len(string) < 2 for string in strings):
        raise OwnSequenceError("Every string has one symbol; no transition can be counted.")
    remarks = []
    short = sum(len(string) < MIN_STRING_LENGTH for string in strings)
    if short:
        remarks.append(
            f"{short} of {len(strings)} strings have fewer than {MIN_STRING_LENGTH} symbols; "
            "the measures taken string by string leave them out."
        )
    if n_tokens < 20 * len(symbols) ** 2:
        remarks.append(
            f"{n_tokens} symbols for {len(symbols)} different ones: fewer than 20 per possible pair, "
            "so estimates beyond the first order are unreliable."
        )
    return {
        "n_strings": len(strings),
        "n_tokens": n_tokens,
        "n_symbols": len(symbols),
        "remarks": remarks,
    }


def own_string_set(
    text: str = "", path: str = "", name: str = "your sequence"
) -> tuple[StringSet, dict] | None:
    """A participant's sequence as a string set with its description, or None if nothing was given.

    A path takes precedence over pasted text. Nothing is written and nothing leaves the computer.
    """
    path = path.strip()
    if path:
        strings, durations = read_sequence_csv(path)
    elif text.strip():
        strings, durations = parse_text(text), None
    else:
        return None
    report = validate_strings(strings)
    symbols = sorted({symbol for string in strings for symbol in string})
    return StringSet(name, strings, {symbol: symbol for symbol in symbols}, durations), report
