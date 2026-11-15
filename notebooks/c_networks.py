import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="C. Networks")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # C. Networks

    This notebook compares two small networks, a tanh RNN and a causal linear transformer, on sequences drawn from grammars. A grammar states how probable every next symbol is, so a network is not scored by its accuracy but by its distance from the predictor that knows the grammar. The notebook shows that distance for the six grammars of notebook A, then raises the complexity of a grammar step by step and asks how each network follows, in prediction and in judging whether a string is grammatical.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | How this notebook works

    It is a marimo notebook, and it ran from top to bottom when you opened it.

    - Text is shown as text; double-click it to see its source. A code cell shows its result below the code.
    - Move a control, or change a cell and press `Ctrl+Enter` (`Cmd+Enter` on a Mac): every cell that depends on it runs again by itself.
    - A variable is defined in one cell only; a name that starts with an underscore belongs to its cell.
    - The round button at the bottom right runs whatever is out of date, and the square above it stops a computation. It is yellow after opening because marimo draws text without running it; nothing is missing.

    The full guide is `docs/RUNNING-THE-NOTEBOOKS.md`.

    ///
    """)
    return


@app.cell
def _(mo):
    load_saved = mo.ui.switch(value=True, label="load saved results")
    load_saved
    return (load_saved,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the switch on, nothing is trained and every result is read from the repository. With the switch off, section 3 trains both networks on one grammar, which takes one to two minutes, depending on the laptop. All other results are always read from the repository.
    """)
    return


@app.cell
def _():
    from tutorial import complexity, complexity_runs, config, datasets, env, ladder, models, plotting, results
    from tutorial import notebook_views as views

    if not env.environment_ok():
        raise RuntimeError("The install check failed. Run notebook A for the details, or see OFFLINE.md.")
    return complexity, complexity_runs, config, datasets, ladder, models, plotting, results, views


@app.cell(hide_code=True)
def _(config, ladder, mo):
    _ceilings = ladder.exact_table([config.CANDIDATES[_name] for _name in ("non-adjacent", "nested")])
    _open = _ceilings.loc["non-adjacent", "ceiling_with_eos"]
    _closed = _ceilings.loc["nested", "ceiling_with_eos"]
    mo.md(
        rf"""
    ## 1. The measure

    A network reads a string one symbol at a time and, after each symbol, gives a probability to every possible next symbol. Most grammars leave the next symbol partly open, so even a predictor that knows the grammar is often wrong, and how often depends on the grammar. In the non-adjacent grammar a string is A X X B with two random fillers, and the best achievable accuracy is {_open:.2f}. In the nested grammar with three pairs there are six strings and one open symbol, and the best achievable accuracy is {_closed:.2f}. Both numbers count the targets of the runs of section 2: every symbol after the first one and the end marker, because the first symbol of a string is not a target there. An accuracy of 0.80 would be impossible on the first and poor on the second. Raw accuracy, and raw cross-entropy for the same reason, compare the generators and not the networks.

    What is compared here is the distance of a network from the exact predictor: the predictor that knows the grammar and gives every next symbol its true probability. With $p$ the true distribution and $q$ the network's, the distance at one position is the Kullback-Leibler divergence $\sum_x p(x) \log_2 [p(x) / q(x)]$, in bits. It is zero only when the network gives the true probabilities, it is never negative, and zero means the same on every grammar: the network knows what can be known. Nothing is said about which grammar is harder: the question is how far each network is from what can be known.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Six grammars: distance from the exact predictor over training

    Both networks were trained on the six grammars of notebook A. SeqBench builds the data: it draws strings from the SymSeq generator, joins them with `#` into samples of 40 symbols, and gives the next symbol as the target at every position. The settings were fixed before the runs and are the same for both networks and all grammars: Adam with learning rate 0.001, batches of 32 samples, 30 epochs, 2000 training samples of 40 symbols, 500 held-out samples, no weight decay, no learning-rate schedule, no gradient clipping. One seed is one complete repetition. The figure uses seeds 1, 2 and 3. Nothing was tuned on them; trial runs used seeds 101 to 103.

    **Tanh RNN** (`torch.nn.RNN`): $h_t = \tanh(W x_t + U h_{t-1} + b)$, followed by a linear readout. **Causal linear transformer**: two blocks of attention and a small two-layer network, with a learned embedding of the position. Its attention replaces the exponential of softmax attention by a product of features, so the sums over the past become running sums and the cost grows linearly with the length of the sequence. Both networks have close to 20 000 trainable parameters on every grammar.
    """)
    return


@app.cell
def _(config, datasets, mo, models, views):
    _text, _settings = datasets.preview(config.CANDIDATES["non-adjacent"], config.LIVE_SEED)
    _attention = (models.feature_map, models.linear_attention, models.CausalLinearAttention)
    mo.accordion(
        {
            "Two samples of the non-adjacent grammar as SeqBench builds them": views.samples(_text, _settings),
            "The attention of the linear transformer (tutorial/models.py)": views.source_of(*_attention),
        }
    )
    return


@app.cell
def _(results):
    saved_runs, manifest = results.load_runs(results.default_folder())
    placeholder = manifest["placeholder"]
    return manifest, placeholder, saved_runs


@app.cell(hide_code=True)
def _(manifest, mo, placeholder):
    (
        mo.callout(
            mo.md(
                f"**The saved results of the six grammars are placeholders.** They come from short smoke runs "
                f"({manifest['settings']['n_train']} training samples, {manifest['settings']['epochs']} epochs, "
                f"seeds {manifest['seeds']}) that only exercise the code. The networks are far from trained, "
                "and no number of this section says anything about the grammars or the networks."
            ),
            kind="warn",
        )
        if placeholder
        else mo.md(
            f"Saved results: {manifest['present']} runs, seeds {manifest['seeds']}, {manifest['settings']['epochs']} epochs."
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After every epoch the cross-entropy of the network on the held-out samples is compared with the cross-entropy of the exact predictor on the same symbols. The difference, in bits per target, estimates the divergence of section 1. It is an estimate from the symbols that happened to be drawn, so it can be slightly negative by chance, and below the resolution of the held-out samples it cannot be told from zero. The table gives the distance after the last epoch.
    """)
    return


@app.cell
def _(config, manifest, results, saved_runs, views):
    distance = results.distance_data(saved_runs, config.LADDER, manifest["kind"])
    views.table(results.distance_table(distance), 4)
    return (distance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Each panel of the figure is one grammar, and all panels share one logarithmic scale. Thick lines: mean over seeds. Thin lines: single seeds. Dashed grey line: the resolution of the held-out samples. The panels compare the two networks within a grammar. The grammars differ in how much of a string is random by construction, and the figure says nothing about which grammar is harder.
    """)
    return


@app.cell(hide_code=True)
def _(mo, placeholder):
    (
        mo.callout(
            mo.md("**Placeholder figure.** The saved results are short smoke runs. The figure shows its layout, not a result."),
            kind="warn",
        )
        if placeholder
        else None
    )
    return


@app.cell(hide_code=True)
def _(config, distance, mo, placeholder):
    def _reading():
        final = {
            (grammar, model): [curve[-1] for curve in values["distance"][index]]
            for index, grammar in enumerate(distance["grammars"])
            for model, values in distance["models"].items()
        }
        largest = max(sum(seeds) / len(seeds) for seeds in final.values())
        gaps = {}
        for grammar in distance["grammars"]:
            means = {model: sum(final[grammar, model]) / len(final[grammar, model]) for model in config.MODELS}
            far = max(means, key=means.get)
            near = min(means, key=means.get)
            gaps[grammar] = (means[far] / means[near], far, near, means[far], means[near])
        grammar = max(gaps, key=lambda name: gaps[name][0])
        ratio, far, near, far_mean, near_mean = gaps[grammar]
        every_seed = min(final[grammar, far]) > max(final[grammar, near])
        epochs = distance["epochs"][0][-1]
        return mo.md(
            f"""
    **Reading the figure.** After {epochs} epochs both networks are within {largest:.3f} bit per target of the exact predictor on all six grammars. The two networks differ most on {grammar}: the {config.MODEL_LABELS[far]} stays {ratio:.1f} times further from the exact predictor than the {config.MODEL_LABELS[near]} ({far_mean:.3f} against {near_mean:.3f} bit){", in every seed" if every_seed else ", but not in every seed"}.
    """
        )

    None if placeholder else _reading()
    return


@app.cell
def _(distance, plotting, views):
    plotting.apply_style()
    figure_six = plotting.figure_networks(distance)
    views.full_width(figure_six)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Grammars of controlled complexity

    The six grammars differ in kind. Here the kind is fixed and the complexity is raised. SymSeq synthesises the transitions of a regular grammar for a target complexity (`symseq.generators.ag.synthesis`). Every grammar below has 8 states and 6 symbols, so two symbols are emitted by two states each. After such a symbol the state of the grammar is not known from the symbol alone, and what may follow depends on what came before. The complexity of a grammar is its topological entropy, the growth rate of the number of allowed strings in bits per symbol (notebook A, section 1). It is computed by SymSeq on the closed graph, in which the end marker leads back to the start state.

    A grammar is admitted only if memory matters in it: the best predictor without memory must be at least 0.05 bit per target worse than the exact predictor. It must also hold enough strings for training and testing with different strings. The table lists the admitted grammars; the levels are bands of the achieved topological entropy.
    """)
    return


@app.cell
def _(complexity_runs):
    complexity_saved, complexity_manifest, grammars = complexity_runs.load_runs(complexity_runs.default_folder())
    complexity_placeholder = complexity_manifest["placeholder"]
    return complexity_manifest, complexity_placeholder, complexity_saved, grammars


@app.cell(hide_code=True)
def _(complexity_manifest, complexity_placeholder, config, mo):
    (
        mo.callout(
            mo.md(
                f"**The saved results of the complexity experiment are placeholders.** They come from pilot runs: "
                f"{complexity_manifest['settings']['grammars_per_level']} grammar per level and training seeds "
                f"{complexity_manifest['seeds']}, made to check the code and the time. "
                "No number of this section is a result. "
                f"The pilot has {len(complexity_manifest['settings']['levels'])} levels and the production runs have "
                f"{len(config.COMPLEXITY.levels)}, numbered from 1 in each, so a level is identified by its band of "
                "topological entropy and not by its number."
            ),
            kind="warn",
        )
        if complexity_placeholder
        else mo.md(
            f"Saved results of the complexity experiment: {complexity_manifest['present']} runs, "
            f"{complexity_manifest['settings']['grammars_per_level']} grammars per level, training seeds {complexity_manifest['seeds']}."
        )
    )
    return


@app.cell
def _(complexity_runs, grammars, mo, views):
    _turned_away = views.table(complexity_runs.admission_table(grammars))
    mo.vstack(
        [
            views.table(complexity_runs.grammar_table(grammars), 3),
            mo.accordion({"How many candidate grammars each rule turned away": _turned_away}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Choose a level and a number of training strings. The first grammar of the level is drawn below, and it is the grammar that the next cells train.
    """)
    return


@app.cell
def _(complexity_runs, config, grammars, mo, views):
    _admitted = complexity_runs.admitted_grammars(grammars)
    level_choice = views.level_dropdown(_admitted, config.COMPLEXITY_LIVE_LEVEL, label="level")
    size_choice = views.size_radio(config.COMPLEXITY.sizes, config.COMPLEXITY_LIVE_SIZE, label="training strings")
    mo.hstack([level_choice, size_choice], justify="start", gap=2)
    return level_choice, size_choice


@app.cell
def _(complexity_runs, config, grammars, level_choice, views):
    live_spec, live_values = views.first_grammar(complexity_runs.admitted_grammars(grammars), level_choice.value)
    live_settings = config.COMPLEXITY
    return live_settings, live_spec, live_values


@app.cell
def _(complexity, config, live_settings, live_spec, live_values, plotting, views):
    _lengths = (live_settings.min_length, live_settings.max_length)
    _grammar = complexity.build_grammar(live_spec)
    # six different strings to look at, and each of them once more with one planted violation
    _strings = complexity.draw_strings(live_spec, config.COMPLEXITY_LIVE_SEED, 6, 6, *_lengths, 10_000, "distinct")
    _predictor = complexity.BoundedPredictor(_grammar, *_lengths)  # the exact predictor of the grammar
    plotting.apply_style()
    # beside the drawing: each string with its largest surprisal under the exact predictor
    views.grammar_card(plotting.draw_grammar(_grammar), live_spec, live_values, _strings, _predictor)
    return


@app.cell(hide_code=True)
def _(complexity_runs, complexity_saved, config, live_settings, mo):
    _sizes = ", ".join(str(_size) for _size in live_settings.sizes)
    _jobs = complexity_runs.job_table(complexity_saved)  # the strings of the saved runs, by level and N
    _largest = int(_jobs.index.get_level_values("training strings").max())
    _of_largest = _jobs.xs(_largest, level="training strings")
    _distinct = _of_largest["distinct among them"]
    _mass = 100 * _of_largest["probability of the training strings"]
    _heldout, _generator = _of_largest["mean length: held-out"], _of_largest["mean length: generator"]
    _training = {
        "draws": f"the $N$ training strings ($N$ = {_sizes}) are the first $N$ strings drawn from the grammar, and a string that is drawn several times is used several times. In the runs shown, {_distinct.min():.0f} to {_distinct.max():.0f} of the {_largest} training strings are distinct.",
        "distinct": f"$N$ distinct strings are drawn for training ($N$ = {_sizes}), and each is used once.",
    }[live_settings.training_sample]
    _share = round(100 * live_settings.validation_fraction)
    _parameters = f"{config.PRODUCTION.parameter_budget:,}".replace(",", " ")
    mo.md(
        rf"""
    **Data.** For every grammar and training seed, {_training} {_share} % of the distinct training strings, with all their repeats, only choose the weights. {live_settings.n_heldout} held-out strings are grammatical strings that occur in no training set. The probable strings occur in training, so the held-out strings are the less probable and longer ones: in the runs shown the {_largest} training strings hold {_mass.min():.0f} to {_mass.max():.0f} % of the probability of all strings, and a held-out string has {_heldout.min():.1f} to {_heldout.max():.1f} symbols on average, against {_generator.min():.1f} to {_generator.max():.1f} for the grammar itself. Each held-out string gives one ungrammatical string, made by SymSeq with one planted violation (`add_deviant`) and confirmed with `is_grammatical`. A string $x_1 \dots x_L$ is read as `# x_1 ... x_L`, and the targets are $x_1 \dots x_L$ and `#`.

    **Training.** Both networks as in section 2, with the same optimiser and close to {_parameters} parameters, for {live_settings.max_steps} gradient steps whatever $N$. The weights that are measured are those of the step with the lowest loss on the validation strings.
    """
    )
    return


@app.cell(hide_code=True)
def _(complexity, complexity_runs, complexity_saved, live_spec, mo):
    _jobs = complexity_runs.job_table(complexity_saved)  # the strings of the saved runs, by level and N
    _largest = int(_jobs.index.get_level_values("training strings").max())
    _of_largest = _jobs.xs(_largest, level="training strings")
    _d0 = _of_largest["uniform guess: divergence (bits)"]  # one value per level, lowest level first
    _outcomes = len(complexity.classes_of(live_spec))
    mo.md(
        rf"""
    **Prediction.** The distance of a model $q$ from the exact predictor is the Kullback-Leibler divergence of section 1, averaged over all targets of the held-out strings: $D(q)$, in bits per target. The exact predictor knows the length limits of the strings and infers the hidden state of the grammar from the symbols seen so far. Across grammars $D(q)$ cannot be compared as it is. A guess that knows only the alphabet gives the same probability to each of the {_outcomes} outcomes, and its own divergence $D_0$ depends on the grammar: the more a grammar leaves open, the closer the guess already is. In the runs shown $D_0$ is {_d0.iloc[0]:.2f} bit per target at the lowest level and {_d0.iloc[-1]:.2f} at the highest. The prediction score is the share of the way from this guess to the exact predictor that a model has gone:

    $$S(q) = 1 - \frac{{D(q)}}{{D_0}}$$

    The score is 1 only for the exact predictor and 0 for the uniform guess. It cannot exceed 1, and it is negative for a model that is worse than the guess. The tables also give $D(q)$ in bits and how often the first choice of the network is right, divided by the best achievable value. Neither of the two is compared across levels.
    """
    )
    return


@app.cell(hide_code=True)
def _(complexity_runs, complexity_saved, mo):
    _jobs = complexity_runs.job_table(complexity_saved)  # the strings of the saved runs, by level and N
    _smallest = int(_jobs.index.get_level_values("training strings").min())
    _of_smallest = _jobs.xs(_smallest, level="training strings")
    _gradient, _choice = _of_smallest["used for gradient steps"], _of_smallest["used to choose the weights"]
    _choice_distinct = _of_smallest["distinct among those"]
    mo.md(
        rf"""
    **Grammaticality judgement.** The score of a string is the largest surprisal the network has at any of its positions, the end marker included. The measure is the area under the ROC curve of this score, ungrammatical against grammatical held-out strings: 0.5 is chance and 1 is what the exact predictor reaches, because a violation has probability zero.

    **Reference.** A first-order model with add-one smoothing, fitted on the same training strings, is measured in the same two ways: what familiarity with pairs of symbols alone achieves. Every model gets the same $N$ strings and uses them as it needs. The first-order model counts on all $N$. A network takes its gradient steps on about four fifths of them and chooses its weights with the rest: in the runs shown with {_smallest} training strings, {_gradient.min():.0f} to {_gradient.max():.0f} strings give gradient steps and {_choice.min():.0f} to {_choice.max():.0f} choose the weights, which are {_choice_distinct.min():.0f} to {_choice_distinct.max():.0f} distinct strings. With so few strings the choice of the weights is noisy.
    """
    )
    return


@app.cell
def _(complexity_runs, complexity_saved, config, live_settings, live_spec, live_values, load_saved, mo, size_choice, views):
    trained_here = not load_saved.value
    if load_saved.value:
        trained = views.runs_of(complexity_saved, live_spec.name, size_choice.value, config.MODELS)
    else:
        _data = complexity_runs.prepare(live_spec, live_values, config.COMPLEXITY_LIVE_SEED, live_settings)
        trained = {}
        for _model in config.MODELS:
            with mo.status.progress_bar(**views.bar_settings(config.MODEL_LABELS[_model], live_settings)) as _bar:
                trained[_model] = complexity_runs.train_run(
                    _data, _model, size_choice.value, live_settings, progress=views.advance(_bar)
                )
    return trained, trained_here


@app.cell(hide_code=True)
def _(complexity_placeholder, config, live_spec, mo, size_choice, trained, trained_here):
    _source = (
        f"trained here in {sum(_run['seconds']['training'] for _run in trained.values()):.0f} s, training seed {config.COMPLEXITY_LIVE_SEED}"
        if trained_here
        else f"saved run, training seed {next(iter(trained.values()))['seed']}"
        + (", a placeholder" if complexity_placeholder else "")
    )
    mo.md(f"**{live_spec.name}, {size_choice.value} training strings** ({_source})")
    return


@app.cell
def _(complexity_runs, config, mo, trained, views):
    _named = {config.MODEL_LABELS[_model]: _run for _model, _run in trained.items()}
    _history = views.table(complexity_runs.history_table(_named), 3)
    mo.vstack(
        [
            views.table(complexity_runs.measures_table(_named), 3),
            mo.accordion({"Validation loss, prediction score and divergence during training": _history}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### All grammars

    The expectation was fixed before the production runs: at a fixed number of training strings the prediction score falls and the area under the curve falls as the topological entropy rises, for both networks, and both rise with more training strings. The unit is the grammar, with its mean over training seeds. The first table gives Spearman's rank correlation across the grammars and a one-sided permutation p-value for every network, number of training strings and measure. The second counts the grammars in which more training strings are better. The divergence in bits carries no expectation; its correlation is listed further down, without a verdict.
    """)
    return


@app.cell
def _(complexity_manifest, complexity_runs, complexity_saved, grammars, views):
    complexity_data = complexity_runs.figure_data(complexity_saved, complexity_manifest, grammars)
    all_trends = complexity_runs.trend_table(complexity_data)
    trends = all_trends[all_trends["expected"] != "none"]  # the cells of the expectation: score and area
    views.table(trends, 4)
    return all_trends, complexity_data, trends


@app.cell
def _(complexity_data, complexity_runs, views):
    sizes = complexity_runs.size_table(complexity_data)
    views.table(sizes, 4)
    return (sizes,)


@app.cell
def _(all_trends, complexity_data, complexity_runs, complexity_saved, mo, views):
    mo.accordion(
        {
            "Means of the levels": views.table(complexity_runs.mean_table(complexity_data), 3),
            "The divergence in bits: no expectation": views.table(all_trends[all_trends["expected"] == "none"], 4),
            "The strings of the runs": views.table(complexity_runs.job_table(complexity_saved), 3),
        }
    )
    return


@app.cell(hide_code=True)
def _(complexity_runs, complexity_saved, mo):
    _jobs = complexity_runs.job_table(complexity_saved)
    _smallest = int(_jobs.index.get_level_values("training strings").min())
    _choice = _jobs.xs(_smallest, level="training strings")["used to choose the weights"]
    mo.md(
        f"""
    One column for each number of training strings. A, upper row: the prediction score. B, lower row: the area under the ROC curve of the grammaticality judgement. Every panel shows the two networks and, in grey, the first-order model against the topological entropy of the grammar. Points are grammars, and lines join the means of the levels. The dotted lines are the anchors of a row: the exact predictor at 1, the uniform guess at 0 in A, and chance at 0.5 in B. Stars: the run trained above, if the switch was off.

    The grey line is not a network without memory under the same conditions. The first-order model counts on all the training strings, while a network keeps about a fifth of them to choose its weights: {_choice.min():.0f} to {_choice.max():.0f} strings of the {_smallest} in the first column.
    """
    )
    return


@app.cell(hide_code=True)
def _(complexity_placeholder, mo):
    (
        mo.callout(
            mo.md("**Placeholder figure.** The saved results are pilot runs. The figure shows its layout, not a result."),
            kind="warn",
        )
        if complexity_placeholder
        else None
    )
    return


@app.cell(hide_code=True)
def _(complexity_data, complexity_placeholder, mo, sizes, trends):
    def _reading():
        parts = []
        for name in ("prediction score", "area under the curve"):
            rows = trends[trends["measure"] == name]
            parts.append(
                f"The {name} falls with the topological entropy in {int((rows['verdict'] == 'supported').sum())} of {len(rows)} "
                f"combinations of network and number of training strings, and is at ceiling in {int(rows['at ceiling'].sum())}."
            )
        ends = sizes[sizes["sizes"] == f"{complexity_data['sizes'][0]} to {complexity_data['sizes'][-1]}"]
        better = ", ".join(
            f"{row['better with more strings']} of {row['grammars']} grammars ({row['network']}, "
            f"{'score' if row['measure'] == 'prediction score' else 'area'})"
            for _, row in ends.iterrows()
        )
        return mo.md(
            f"""
    **Reading the figure.** {" ".join(parts)} From the smallest to the largest number of training strings the measure is better in {better}. A combination counts as supporting the expectation if Spearman's correlation is negative and its one-sided p-value is at most 0.05.
    """
        )

    None if complexity_placeholder else _reading()
    return


@app.cell
def _(complexity_data, live_values, plotting, size_choice, trained, trained_here, views):
    plotting.apply_style()
    _star = views.live_point(live_values, size_choice.value, trained) if trained_here else None
    figure = plotting.figure_complexity(complexity_data, live=_star)
    views.full_width(figure)
    return (figure,)


if __name__ == "__main__":
    app.run()
