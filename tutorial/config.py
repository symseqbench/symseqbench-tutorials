"""Every setting of the networks notebook in one place: grammars, seeds and hyperparameters.

The notebook ``notebooks/c_networks.py`` and the script ``scripts/run_precomputed.py`` read their
settings from here and from nowhere else. The values were fixed before any plotted run and are not
tuned on the plotted seeds. Each named setting below is one decision; changing a decision means
changing one name.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from tutorial.ladder import Rung

# --- Grammars -------------------------------------------------------------------------------------

# Every candidate of scripts/output/ladder_proposal.csv, with the same names and parameters.
CANDIDATES: dict[str, Rung] = {
    rung.name: rung
    for rung in (
        Rung("two-state", "two_state", {"p": 0.5, "q": 1.0}),
        Rung(
            "regular (6 states)",
            "regular",
            {"n_states": 6, "alphabet_size": 4, "transition_density": 0.3, "seed": 0},
        ),
        Rung("2-AX", "nax", {"n": 2, "p_target": 0.6}),
        Rung("2-AX (p_target 0.9)", "nax", {"n": 2, "p_target": 0.9}),
        Rung("4-AX", "nax", {"n": 4, "p_target": 0.6}),
        Rung("4-AX (p_target 0.9)", "nax", {"n": 4, "p_target": 0.9}),
        Rung(
            "non-adjacent",
            "nad",
            {"n_deps": 3, "n_fillers": 3},
            {"filler_len": 2, "randomize_fillers": True},
        ),
        Rung("nested", "nested", {"n_deps": 3}),
        Rung("crossed", "crossed", {"n_deps": 3}),
        Rung("Dyck (depth 2)", "dyck", {"k": 2, "depth": 2, "pairs": 4}),
        Rung("Dyck (depth 3)", "dyck", {"k": 2, "depth": 3, "pairs": 4}),
    )
}

# The grammars of the figure, in the order of the horizontal axis.
LADDER: tuple[str, ...] = (
    "two-state",
    "4-AX (p_target 0.9)",
    "non-adjacent",
    "nested",
    "crossed",
    "Dyck (depth 3)",
)

# The other candidates. They are run as well, so that another ladder can be plotted without new runs.
ALTERNATIVES: tuple[str, ...] = tuple(name for name in CANDIDATES if name not in LADDER)

# The grammar that the notebook trains live.
LIVE_RUNG = "two-state"

# --- Models ---------------------------------------------------------------------------------------

MODELS: tuple[str, ...] = ("rnn", "linear transformer")
MODEL_LABELS = {"rnn": "tanh RNN", "linear transformer": "linear transformer"}

# --- Seeds ----------------------------------------------------------------------------------------

# One seed is one complete repetition: it draws the training samples, the initial weights and the
# order of the batches. The held-out samples of a grammar are the same for every seed and model.
PLOTTED_SEEDS: tuple[int, ...] = (1, 2, 3)  # the runs of the figure; nothing is tuned on them
EXPLORATION_SEEDS: tuple[int, ...] = (101, 102, 103)  # smoke runs, tests, timing, any trial run
LIVE_SEED = 104  # the run a participant trains in the notebook; not one of the plotted seeds

# --- Targets and measures -------------------------------------------------------------------------

# "seqbench": the targets are the labels SeqBench emits (its NStepPrediction task with n = 1): every
#   next symbol inside a string and the end marker after the last symbol. The first symbol of a
#   string is never a target.
# "stream": as above, and the first symbol of the next string is a target at the end marker. The
#   best achievable cross-entropy is then the entropy rate of the joined sequence.
TARGETS = "seqbench"

# Which accuracy the figure shows first. "with_eos": every SeqBench target, the end marker included.
# "within": the targets that are symbols, with the prediction restricted to symbols.
PLOT_CONVENTION = "with_eos"
CONVENTION_LABELS = {"with_eos": "with #", "within": "symbols only"}


# --- Hyperparameters ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hyperparameters:
    """Everything that defines one training run, apart from the grammar, the model and the seed."""

    sample_length: int = 40  # symbols per sample (strings joined with the end marker)
    n_train: int = 2000  # training samples
    n_heldout: int = 500  # held-out samples
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3  # Adam, default betas, no weight decay, no schedule, no clipping
    parameter_budget: int = (
        20_000  # trainable parameters of each model, as close as its size allows
    )
    transformer_width: int = 32
    transformer_heads: int = 4
    transformer_layers: int = 2
    targets: str = TARGETS
    threads: int = 1

    def as_dict(self) -> dict:
        return asdict(self)


PRODUCTION = Hyperparameters()

# A short run that exercises every step. Its results are placeholders and are labelled as such.
SMOKE = replace(PRODUCTION, n_train=256, n_heldout=128, epochs=3)

SETTINGS = {"production": PRODUCTION, "smoke": SMOKE}
SEEDS = {"production": PLOTTED_SEEDS, "smoke": EXPLORATION_SEEDS}

if set(PLOTTED_SEEDS) & (set(EXPLORATION_SEEDS) | {LIVE_SEED}):
    raise ValueError("The plotted seeds must not be used for anything else.")


# --- The complexity experiment --------------------------------------------------------------------

# Regular grammars made with SymSeq's synthesis for a target complexity (``tutorial.complexity``),
# both networks trained on N strings of each, and measured against the exact predictor of the
# grammar. The optimiser, the batch size and the size of the networks are those of PRODUCTION.
# Every value below was fixed before any pilot or production run; the levels and the admission
# rule were fixed from grammars drawn with exploration seeds (101 and above), without any network.


@dataclass(frozen=True)
class Level:
    """One level of complexity: the target given to SymSeq's synthesis and the band of achieved values."""

    # target of the synthesis: ln of the spectral radius of the transitions, without the marker
    target_nats: float
    # achieved topological entropy of the closed graph, in bits per symbol: low <= value < high
    band_bits: tuple[float, float]


# The five levels of the design. Bands do not overlap; each target was chosen so that most grammars
# synthesised for it fall into its band.
COMPLEXITY_LEVELS: tuple[Level, ...] = (
    Level(0.8, (1.26, 1.42)),
    Level(1.0, (1.45, 1.60)),
    Level(1.1, (1.63, 1.77)),
    Level(1.3, (1.86, 1.98)),
    Level(1.4, (2.00, 2.12)),
)


@dataclass(frozen=True)
class ComplexitySettings:
    """Everything that defines the complexity experiment, apart from the settings of the networks."""

    # the grammars: the same size at every level
    n_states: int = 8
    alphabet_size: int = 6  # so two symbols are emitted by two states each
    n_terminal_states: int = 3  # states from which a string may end; one start state
    min_length: int = 2  # symbols per string; the start state is never a terminal state
    max_length: int = 40  # SymSeq rejects longer walks; few walks are that long
    levels: tuple[Level, ...] = COMPLEXITY_LEVELS
    grammars_per_level: int = 3
    first_grammar_seed: int = 1  # grammar seeds are tried in order from here
    max_candidates: int = 80  # grammar seeds tried per level before the level is left short
    # the admission rule, next to the band of the level
    # excess cross-entropy of a predictor without memory, in bits per target
    min_memory_gap_bits: float = 0.05
    # distinct strings, and typical strings 2^H: what a run needs (800 + 200)
    min_strings: int = 1000
    # the data of a run
    sizes: tuple[int, ...] = (50, 200, 800)  # training strings N
    # "draws": the first N draws from the generator, repeats kept (the design of the experiment);
    # "distinct": N distinct strings, each once (only for runs with --out)
    training_sample: str = "draws"
    validation_fraction: float = 0.2  # share of the training strings that only chooses the weights
    n_heldout: int = 200  # held-out strings, and as many ungrammatical ones made from them
    max_draws: int = 100_000  # strings drawn from a grammar per job before the job is given up
    # training: the same for both networks and every N
    # gradient steps; the six-grammar runs made 1890 (30 epochs of 63 batches)
    max_steps: int = 2000
    eval_every: int = 50  # steps between two measurements of the validation loss
    # training seeds: strings, validation split, weights, batch order
    seeds: tuple[int, ...] = (1, 2, 3)

    def as_dict(self) -> dict:
        return asdict(self)


# The design as it was written before the pilot: five levels, three training seeds.
COMPLEXITY_DESIGN = ComplexitySettings()

# The level that is left out if the production runs have to be cut to four levels: the fourth,
# which lies closest to a neighbour. Fixed before the pilot.
COMPLEXITY_LEVEL_TO_DROP = 3

# The production runs. The time rule was applied after the pilot (13 November 2026): the design was
# projected to take 6.7 hours in two processes in the slow state of the workstation, above the four
# hours allowed, so the training seeds went from three to two (4.5 hours) and the levels from five
# to four (3.6 hours). Nothing else differs from the design.
COMPLEXITY = replace(
    COMPLEXITY_DESIGN,
    seeds=(1, 2),
    levels=tuple(
        level for index, level in enumerate(COMPLEXITY_LEVELS) if index != COMPLEXITY_LEVEL_TO_DROP
    ),
)

# The pilot: the five levels of the design, one grammar per level and one training seed, on seeds
# no production run uses. Its results are placeholders and are labelled as such.
COMPLEXITY_PILOT = replace(
    COMPLEXITY_DESIGN, grammars_per_level=1, first_grammar_seed=101, seeds=(101,)
)

# A few short runs that exercise every step; they are written only where ``--out`` says.
COMPLEXITY_SMOKE = replace(
    COMPLEXITY_PILOT,
    levels=(COMPLEXITY_LEVELS[1], COMPLEXITY_LEVELS[-1]),
    sizes=(50,),
    n_heldout=40,
    max_steps=60,
    eval_every=20,
)

COMPLEXITY_SETTINGS = {
    "production": COMPLEXITY,
    "pilot": COMPLEXITY_PILOT,
    "smoke": COMPLEXITY_SMOKE,
}

# The run a participant trains in the notebook: first grammar of this level, this many strings.
COMPLEXITY_LIVE_LEVEL = 2
COMPLEXITY_LIVE_SIZE = 200
COMPLEXITY_LIVE_SEED = LIVE_SEED

# The test of the expectation (``tutorial.complexity_runs.trend_table``).
TREND_ALPHA = 0.05  # one-sided p-value at or below which a trend counts as supported
# random orderings when the grammars are too many to list all orderings
TREND_PERMUTATIONS = 100_000
TREND_SEED = 20261119  # seed of those orderings
# A cell of the trend table is at ceiling if every grammar has at least this prediction score ...
CEILING_SCORE = 0.99
# ... or at least this area under the curve.
CEILING_AUC = 0.99

_last_production_seed = COMPLEXITY.first_grammar_seed + COMPLEXITY.max_candidates - 1
if max(_last_production_seed, *COMPLEXITY.seeds) >= min(
    COMPLEXITY_PILOT.first_grammar_seed, *COMPLEXITY_PILOT.seeds, COMPLEXITY_LIVE_SEED
):
    raise ValueError(
        "Production seeds of the complexity experiment must lie below every other seed."
    )
