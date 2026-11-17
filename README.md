# Generating, measuring and learning structured sequences with SymSeqBench

Material for the hands-on tutorial at the Champalimaud Foundation, 19 November 2026. It uses two open-source libraries: SymSeq, which generates and analyses rule-based symbolic sequences, and SeqBench, which turns them into datasets for network training. Everything runs on a laptop CPU.

The repository holds three notebooks, A (generate and measure), B (behaviour) and C (networks), a helper package and their tests. Notebooks B and C read saved results by default and compute them on request. Every notebook has a folder in `fallback/` with a static export, for a participant without a working installation. Notebook C has two parts: the six grammars of notebook A, whose runs are saved, and grammars of controlled complexity, whose production runs are saved as well. The pilot runs that came before them stay in the repository; where they are shown, the notebook and its figure say that they are placeholders.

## Install

You need [uv](https://docs.astral.sh/uv/) (0.6.6 or later), git, and a network connection to GitHub, PyPI and download.pytorch.org. uv fetches Python 3.11 if it is missing.

```bash
git clone https://github.com/symseqbench/symseqbench-tutorials.git
cd symseqbench-tutorials
uv sync
uv run marimo edit notebooks/a_generate_and_measure.py
```

The first cell of the notebook is an install check. Notebooks B and C are opened the same way: `uv run marimo edit notebooks/b_behaviour.py` and `uv run marimo edit notebooks/c_networks.py`. Without network, follow [OFFLINE.md](OFFLINE.md).

Supported platforms, checked against `uv.lock` by `scripts/check_lock_platforms.py`: Linux x86_64 (glibc 2.28 or later), macOS on Apple silicon (macOS 12 or later), macOS on Intel (10.15 or later), Windows x86_64.

## Running the notebooks

The notebooks are marimo notebooks: plain Python files that open in the browser. A notebook runs from top to bottom by itself when it is opened, and after a change every cell that depends on it runs again. If you know Jupyter and have not used marimo, read [docs/RUNNING-THE-NOTEBOOKS.md](docs/RUNNING-THE-NOTEBOOKS.md) first: it shows the screen, the three ways to run a notebook, the keys, how to export to Jupyter, and what to do when something goes wrong.

## What is pinned

| Component | Version | Note |
|---|---|---|
| SymSeq | github.com/symseqbench/SymSeq at `6c3d7236a276deebf8b3f26cbbaf92687f97de5b` | upstream `main`, 30 October 2026 |
| SeqBench | github.com/symseqbench/SeqBench at `c3ce976a72f0c6d73dff3334bb5ecac3838efe40` | upstream `main`, 17 October 2026 |
| Python | 3.11 | |
| torch | 2.11.0, CPU only | 2.2.2 on Intel Macs, the last release with wheels for them |
| numpy | 1.26.4 | the same on every platform, so seeded sequences are identical on every laptop |
| marimo, scipy, pandas, matplotlib | 0.25.2, 1.17.1, 3.0.6, 3.11.2 | |

torch 2.11.0 is the last release whose Apple-silicon wheel runs on macOS 11 to 13; later releases need macOS 14. On Linux and Windows torch comes from the CPU index of PyTorch, so no CUDA package is installed. The two commits are also in `pyproject.toml`, `uv.lock` and `tutorial/env.py`; a test checks that the four places agree.

## Layout

- `notebooks/a_generate_and_measure.py`, `notebooks/b_behaviour.py` and `notebooks/c_networks.py`: notebooks A, B and C, marimo notebooks (plain Python files).
- `tutorial/`: the helper package. `generate.py` and `measures.py` are thin wrappers around SymSeq's generators and measures; `ladder.py` defines the optional complexity ladder; `prescribed.py` synthesises regular grammars at a chosen topological entropy; `closed_forms.py` and `worked_example.py` hold the worked example; `exact.py` computes the exact entropy rate and the next-symbol accuracy ceiling of each generator; `plotting.py` is the plot style and the figures; `notebook_views.py` is how the notebooks show things around their results (a table, a figure at the full width of the column, the names of the options of a control, a short listing), kept out of the cells so that a code cell stays short; `env.py` is the install check. For notebook C: `config.py` holds every setting (grammars, seeds, hyperparameters); `datasets.py` builds training and held-out data with SeqBench; `models.py` has the tanh RNN and the causal linear transformer; `training.py` is the training and evaluation loop; `results.py` reads and writes saved runs and gives each network's distance from the exact predictor; `complexity.py` makes grammars of controlled complexity with SymSeq's synthesis, decides their admission, states their exact predictor and defines the two measures; `complexity_runs.py` trains on them, saves the runs and tests the expectation.
- `tests/`: the tests. Each one has a case it must reject.
- `scripts/`: `check_lock_platforms.py`, `build_wheelhouse.py`, `make_fallback.py`, `measure_training_time.py`, `ladder_proposal.py`, `run_precomputed.py`, `run_complexity.py`, `build_behaviour_sequences.py`, `run_behaviour_measures.py`. The results of the timing and ladder scripts are in `scripts/output/`; the others write to `data/`, `fallback/` or `wheelhouse/`.
- `docs/RUNNING-THE-NOTEBOOKS.md`: how to start, read and change the marimo notebooks, for readers who know Jupyter; `docs/img/` holds its picture.
- `fallback/`: one folder per notebook, written by `scripts/make_fallback.py` (`--notebook a`, `b` or `c`). Each holds a static HTML export of the notebook, its final figure (PDF, PNG, SVG, and a PNG on white) and the numbers of the figure (JSON). `fallback/a_generate_and_measure/` also holds the four tables of notebook A and the two exact panels of the worked example (`worked_example_words` and `worked_example_block_entropy`, in the same five files each). `fallback/b_behaviour/` is made from the saved measures. `fallback/c_networks/` is made from the saved runs and holds the two figures of notebook C, `figure_networks` and `figure_complexity`, in the same five files each.
- `data/precomputed/`: the six-grammar runs of notebook C (66 runs, a summary table, a manifest with the SHA-256 of every run file). `data/precomputed_smoke/` holds short smoke runs in the same form; they are placeholders that only exercise the code, and the tests use them. `data/complexity_pilot/` holds the pilot runs of the complexity experiment (one grammar per level, one training seed, seeds from 101 on), with the grammars and the list of every candidate in `grammars.json`; they are placeholders. The pilot has the five levels of the design and the production runs have four, numbered from 1 in each folder, so the same number can name two different bands: a level is identified by its band of topological entropy, not by its number. `data/complexity/` holds the production runs of that experiment in the same form: 144 runs, from four levels with three grammars each, two training seeds, three numbers of training strings and two networks. `data/behaviour/` holds four open behavioural datasets as tables of bouts, with their sources and licences in `data/behaviour/SOURCES.md`; `data/behaviour_measures/` holds the saved measures of notebook B. No private data enters this repository.

## Notebook A: generate and measure

Notebook A begins with the exact two-state worked example, then synthesises a regular grammar at a chosen topological entropy. Its Glauber/MCMC sampler changes one directed edge at a time until the spectral radius approaches the target; the notebook shows the convergence trace, six distinct generated strings, and the target, achieved open-graph and closed-graph values. The former seven-grammar ladder remains as an optional comparison, not the main exercise. `tutorial/prescribed.py` holds the synthesis and figure-data helpers, and `tests/test_prescribed.py` checks the target, trace and grammatical strings.

## Notebook B: behaviour

Notebook B takes the measures of notebook A to four open datasets: swimming bouts of larval zebrafish, behaviours of green turtles, song syllables of a Bengalese finch, and behaviours of captive fur seals and sea lions. Each dataset is compared with two surrogates built from it, a shuffled one and a first-order Markov one, to show what a table of transition probabilities misses; a participant can add a sequence of their own, which stays on the laptop. A zebrafish swim bout and a finch syllable are events, so each is one symbol and two in a row of the same type stay two; for the turtles and the seals, which are annotated observation by observation, consecutive equal classes form one bout. The tables in `data/behaviour/` are written by `scripts/build_behaviour_sequences.py` from the public files of the sources (`uv run --with h5py python scripts/build_behaviour_sequences.py --raw-root <folder of the raw files>`; only this script needs h5py), and the saved measures by `uv run python scripts/run_behaviour_measures.py`, which gives the same bytes on every run. The helper code is in `tutorial/behaviour.py` (tables, surrogates, check of a participant's sequence) and `tutorial/behaviour_measures.py` (the measures, every setting, the saved results); `uv run python scripts/make_fallback.py --notebook b` writes `fallback/b_behaviour/`. The tests in `tests/test_behaviour*.py` and `tests/test_notebook_b.py` cover: the tables against a second route from the raw files (`uv run --with h5py python -m pytest tests/test_behaviour_data.py`, skipped where the raw files are absent), one alphabet per dataset, the manifests, the wrapped measures against SymSeq's own values, the two surrogates recovering their known answers, identical results for identical seeds, and the notebook running headless.

## The runs of notebook C

Every setting is in `tutorial/config.py`. Both scripts skip runs that are already saved with the same settings, write one JSON file per run, and give identical results for identical seeds. `--shard 2/4` runs the second of four parts, so that several processes can work side by side, each on one thread; `--collect` then writes the summary and the manifest.

The six grammars:

```bash
uv run python scripts/run_precomputed.py --smoke         # short placeholder runs, about 40 s
uv run python scripts/run_precomputed.py --production    # the runs of the first figure
```

The script trains both networks on eleven grammars (the six of the figure and five other candidates) for three seeds: 66 runs. The notebook reads `data/precomputed` when it is complete and `data/precomputed_smoke` otherwise, and says which. The runs of the figure were made on 12 November 2026 and are in `data/precomputed/`.

Grammars of controlled complexity:

```bash
uv run python scripts/run_complexity.py --pilot                    # one grammar per level, seeds from 101 on
uv run python scripts/run_complexity.py --production --draw        # draw and admit the grammars
uv run python scripts/run_complexity.py --production --shard 1/2   # one of two parts; then 2/2
uv run python scripts/run_complexity.py --production --collect     # summary and manifest
uv run python scripts/run_complexity.py --smoke --out FOLDER       # four short runs that exercise the code
```

The script draws regular grammars at several levels of topological entropy with SymSeq's synthesis (five in the pilot, four in the production runs, after the time allowed for them cut one), admits a grammar by a written rule, and trains both networks on the first 50, 200 and 800 strings drawn from each, a string as often as it was drawn. `grammars.json` lists every candidate grammar with the values of the rule and the reason if it was turned away. The notebook reads `data/complexity` when it is complete and `data/complexity_pilot` otherwise, and says which. Both folders exist and the production runs are complete, so they are what the notebook shows. Pilot runs are marked as placeholders in words and in the figure wherever they are shown.

`uv run python scripts/make_fallback.py --notebook c` writes `fallback/c_networks/` from the saved runs. It refuses placeholder results; `--allow-placeholder` exports the pilot runs of the complexity experiment, marked as placeholders in the figure and in its data.

## Tests

```bash
uv run pytest
```

The tests cover: the closed forms of the worked example against two brute-force enumerations and against SymSeq's estimators; generated strings being valid strings of their grammar; prescribed-complexity synthesis reaching different targets with a consistent trace and grammatical examples; the wrapped measures reproducing SymSeq's own values on a fixed seed; the exact ceilings against brute force and against SymSeq's samples; the pinned environment and the wheels of the four platforms; the plot style; notebook A running headless within 180 s. On the workstation where this was built a headless run takes 6 s at full speed and 37 to 45 s when the processor is throttled to about a tenth of its speed, which happens often there.

For notebook C they cover: linear attention against a quadratic implementation of the same attention, which must differ once the causal mask is removed; both networks being causal and having about 20 000 parameters; the samples that reach the network being the strings SymSeq draws for the grammar; no held-out sample being a training sample; no run beating the exact optimum; both networks learning the two-state grammar in five epochs; identical runs for identical seeds; the run script resuming and the loader refusing changed files; notebook C running headless within 180 s, with and without its live training.

For the complexity experiment they cover: the exact predictor against a listing of every walk of small grammars, one of them with a string that has several walks, and against the frequencies of strings SymSeq draws; the divergence being zero for the exact predictor and positive for a perturbed one; the divergence of the uniform guess by two routes, and the prediction score being 1 for the exact predictor, 0 for the uniform guess and never above 1; the area under the curve being 1 for the exact predictor and 0.5 for a score that ignores the string; every ungrammatical string being ungrammatical and every grammatical one grammatical; training, validation and held-out strings being disjoint; the admission rule turning away a grammar in which every symbol has one state; the weights that are measured being those of the lowest validation loss; identical runs for identical seeds; the run files, the manifest and the loader; the trend test against values computed by hand.

For how the notebooks read in the editor they cover: every cell that only shows text hiding its code, and no other cell doing so; no visible cell being longer than 15 lines, with a median of at most 8; every table and every figure being shown through `tutorial/notebook_views.py`, whose table computes no column summaries and whose figure fills the width of the column without being taller than the editor shows; each notebook printing nothing when it is run as a script; SeqBench being imported without its two warnings about torchaudio, with nothing else silenced.

For the fallback folders they cover: each folder holding the files listed above; the numbers of every figure being those of the saved results, recomputed in the test; the two panels of the worked example against a listing of all words and against the block entropies of the stationary chain.

## Conventions

- Entropies are in bits. SymSeq returns topological entropy in nats; the wrappers convert it.
- In the worked example the strings are joined without the end marker `#`. The sequence is then the two-symbol chain A → A, A → B, B → A, with entropy rate 2/3 bit per symbol and memory log2(3) − 4/3 ≈ 0.2516 bit for p(A → B) = 1/2.
- In the ladder the strings are joined with `#`, as SeqBench does. The marker counts as a symbol. For the grammar of the worked example this gives entropy rate 1/2 and memory 1 bit.
- Plots use the sizes and colours of the talk's figure style, with Computer Modern from matplotlib's own math rendering. `tutorial.plotting.apply_style(use_latex=True)` renders through LaTeX where it is installed.
- In notebook C the targets are SeqBench's labels: every next symbol inside a string, and the end marker after its last symbol. The first symbol of a string is never a target. Accuracy is given under two conventions, with the marker as a target ("with #") and without ("symbols only"). Cross-entropy is in bits per target.
- A network is never compared across grammars by its raw accuracy or raw cross-entropy, because grammars differ in how much of a string is random by construction. What is shown is its distance from the exact predictor: the predictor that knows the grammar, computed from `tutorial/exact.py` and `tutorial/complexity.py`. For the six grammars the distance is the network's cross-entropy on the held-out samples minus that of the exact predictor on the same symbols, in bits per target. It estimates a Kullback-Leibler divergence, so it can be slightly negative by chance; its resolution is drawn in the figure. A sample has 40 symbols and usually stops inside a string, and with SeqBench's targets the first symbol of each string is not predicted; `tutorial.config.TARGETS = "stream"` adds that target.
- In the complexity experiment a string x_1 ... x_L is read as `# x_1 ... x_L` and every symbol and the end marker are targets. Strings have 2 to 40 symbols, and the exact predictor knows that longer walks are rejected. The training strings are the first N strings drawn from the grammar, with their repeats; the held-out strings occur in no training set, so they are the less probable ones. Prediction is measured by the Kullback-Leibler divergence from the exact predictor to the network, computed at every target of the held-out strings and never negative. This divergence is not compared across grammars in bits, because a guess that knows only the alphabet is itself closer to the exact predictor in a grammar that leaves more open. The prediction score is one minus the divergence divided by that of the uniform guess on the same targets: 0 for the uniform guess, 1 for the exact predictor. Grammaticality judgement is measured by the area under the ROC curve of the largest surprisal of a string: 0.5 at chance, 1 for the exact predictor. The final figure has one column for each number of training strings, the prediction score above and the area below, with a first-order model as reference in both. Lines join the means of the levels, and the part of a line that crosses a band of the design that was not run (1.86 to 1.98 bits per symbol in the production runs) is dashed: no grammar was drawn in that range. The complexity of a grammar is SymSeq's topological entropy of the closed graph, in bits per symbol. Pilot runs, tests and the live cell use seeds from 101 on; the production runs use grammar seeds from 1 and training seeds 1 and 2.
- One seed is one repetition: it draws the training samples, the initial weights and the order of the batches. The held-out samples of a grammar are the same for every seed and model. The figure uses seeds 1, 2, 3; trial runs, tests and smoke runs use 101, 102, 103.

## Notes on the two libraries

Nothing in SymSeq or SeqBench was changed. These are the points where the installed code behaves differently from what one would expect, and where the workaround is.

SymSeq:

- `ArtificialGrammar.topological_entropy()` returns 0 with a warning for an open grammar. The wrappers call it with `correct_terminal_sink=True` (`tutorial/measures.py`). The end marker is then a state of the closed graph: for the grammar of the worked example this gives 0.5515 bit, not log2(φ) = 0.6942 bit, which is the value for the two-symbol chain and is obtained with `symseq.metrics.grammar.topological_entropy(transitions=..., method="direct")` on the 2 × 2 table.
- The lift method of `topological_entropy` stops at the first lift that changes the value by less than 0.05 nats. On a Dyck sequence (two bracket types, depth 3, four pairs) it returns 1.85 bit where the exact value is 0.86 bit. The ladder table uses the direct method, which exists only for `ArtificialGrammar`; the lift estimate is kept as `tutorial.measures.lift_topological_entropy` and is not in the table.
- `estimate_emc` returns the entropy rate at a finite Markov order, which is an upper bound. For grammars with long dependencies the estimate is above the true value and the excess entropy is below it. Notebook A shows the exact entropy rate next to the estimate (`tutorial/exact.py`).
- `nAX.generate_trial` ignores `n_fillers` and `length_range`; both work only through `generate_string`. The number of fillers of n-AX trials is therefore geometric in any SeqBench dataset.
- `ArtificialGrammar` reseeds its sampling from an integer in [1, 1000), so there are 999 distinct sampling streams. `tutorial.ladder.build_generator` replaces the generator's `rng` after construction.
- `chomsky_classification` labels the sequence of a six-state regular grammar "Unrestricted (Type 0)" with `reliable = True`. It is not used; the `family` column of the ladder is declared in `tutorial/ladder.py`.
- `pairwise_distances` always prints a progress bar; the wrapper calls `edit_distance` pair by pair. `mi_decay_analysis` takes too long for a table cell and is not in the table.
- `ArtificialGrammar.from_constraints(target_complexity=...)` returns the same transitions for every seed: the seed is not passed on to the synthesis, which seeds itself. It also does not check that a string can be generated, so string generation can run for minutes and fail, and it does not keep a grammar with repeated symbols unifilar. `tutorial.complexity.candidate` calls the synthesis function `generate_grammar_with_target_te_complexity` with its own generator and assigns start state, terminal states and symbols itself.
- For a grammar with repeated symbols `topological_entropy` is the growth rate of the number of walks, which is the growth rate of the number of strings only if the grammar is unifilar. The grammars of the complexity experiment are unifilar.
- `generate_string(length_range=...)` rejects walks outside the length range, so the strings follow the walk's distribution conditioned on acceptance. `tutorial.complexity.BoundedPredictor` states that distribution; `tutorial.exact.GrammarPredictor` does not cover length bounds.
- `plot_grammar(grammar, ax=...)` fails at the pinned commit; `tutorial.plotting.draw_grammar` calls the class behind it, `symseq.viz.mc_graph.MarkovChain`.

SeqBench:

- Sample i of every split is generated from `default_rng(seed * i)`. The `train` and `test` splits of one configuration are therefore identical sample by sample, sample 0 is the same for every seed, and with `seed = 0` all samples of a dataset are identical. The workaround is in one place, `tutorial/datasets.py`: training and held-out data are two datasets whose seeds are distinct primes above one million, sample 0 is dropped, and a training sample with the same symbols as a held-out sample is removed (for the nested grammar, which has only 46 656 different samples, 25 of 2000).
- With combined sequences the task `NStepPrediction` is applied to each string before the strings are joined. The end marker therefore has no target: the first symbol of a string is never predicted. The last position of a sample carries the true next symbol, which lies outside the sample.
- Building a dataset reseeds the global generators of numpy and torch with `seqbench.seed`. `tutorial/training.py` sets its own seed after the data are built.
- On the first build of a file-mode dataset `target_probs` has twice the number of class columns (the transitions are parsed twice). It is not used here.
- One pass over a `build_dataloader` loader costs far more than the network's computation for a small network, because every sample is assembled symbol by symbol in Python. Measured: about 1.3 ms per sample of 40 symbols for the pipeline against 0.2 ms for a tanh network of 20 000 parameters; the numbers are in `scripts/output/training_time.json` and `training_time_run2.json`.
- Importing SeqBench logs two warnings when torchaudio is absent, "LogMel not loaded. torchaudio not installed?" and the same for MFCC. They are harmless, because the tutorial uses no audio. `tutorial.datasets.seqbench_dataloader` drops these two records where it imports SeqBench and nothing else, so a notebook run as a script prints nothing.
- The complexity experiment does not use SeqBench. Its measures are taken string by string, on strings that must not occur in training, and SeqBench joins strings into samples of a fixed number of symbols; the strings are drawn from SymSeq directly (`tutorial.complexity.draw_strings`).

## Licence

The code and the text of this repository are under the MIT licence, see [LICENSE](LICENSE). The four behavioural datasets in `data/behaviour/` are derived from open datasets and keep the licences of their sources (CC BY 4.0 and CC0), which are stated with the citations in [data/behaviour/SOURCES.md](data/behaviour/SOURCES.md).
