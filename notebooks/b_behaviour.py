import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="B. Behaviour")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # B. Behaviour

    This notebook will take the measures of notebook A to real behaviour. Four open datasets are read as sequences of symbols, one symbol per bout: swimming bouts of larval zebrafish, behaviours of green turtles, song syllables of a Bengalese finch, and behaviours of captive fur seals and sea lions. The notebook asks one question: what does a first-order description of a behavioural sequence miss? A first-order description is a table of transition probabilities, which says how often each behaviour follows each other behaviour.

    To answer, each dataset is compared with two surrogates built from it. The *shuffled* surrogate has the same symbols in random order. The *first-order* surrogate is drawn from the transition table of the data. What the data show beyond the first-order surrogate is what the table misses.

    No class of the Chomsky hierarchy is assigned to any animal here: the class of a language cannot be identified from examples of its strings alone, and any finite set of observed strings is also produced by a finite-state grammar.
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
    With the switch on, the measures are read from the repository. With the switch off, section 4 computes them for the chosen dataset and its two surrogates, which takes between a few seconds and about a minute on a laptop. The results of the other datasets are always read from the repository.
    """)
    return


@app.cell
def _():
    import pandas as pd

    from tutorial import behaviour, env, plotting
    from tutorial import behaviour_measures as bm
    from tutorial import notebook_views as views

    if not env.environment_ok():
        raise RuntimeError("The install check failed. Run notebook A for the details, or see OFFLINE.md.")
    behaviour.check_folder()  # the four tables are those of the manifest, and nothing else is in the folder
    saved_results, saved_manifest = bm.load_results()
    bm.check_against_tables(saved_results)  # each saved result was computed from the shipped table
    return behaviour, bm, pd, plotting, saved_results, views


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Four open datasets

    Each dataset is a table in `data/behaviour/`, derived from the public files of its source. The sources, their licences and every change made are listed in `data/behaviour/SOURCES.md`. A *string* is one stretch of behaviour: all bouts of one fish, one song, or the bouts between two rests.
    """)
    return


@app.cell
def _(behaviour, pd, saved_results, views):
    _rows = [
        {
            "dataset": name,
            "animals": meta["animal"],
            "a string is": meta["string_is"],
            "strings": saved_results[name]["n_strings"],
            "symbols in all": saved_results[name]["n_tokens"],
            "licence": meta["licence"],
        }
        for name, meta in behaviour.DATASETS.items()
    ]
    views.table(pd.DataFrame(_rows).set_index("dataset"))
    return


@app.cell
def _(behaviour, mo):
    dataset_choice = mo.ui.dropdown(options=list(behaviour.DATASETS), value="zebrafish", label="dataset")
    dataset_choice
    return (dataset_choice,)


@app.cell
def _(behaviour, dataset_choice):
    chosen = behaviour.load_strings(dataset_choice.value)
    return (chosen,)


@app.cell(hide_code=True)
def _(behaviour, chosen, dataset_choice, mo):
    _meta = behaviour.DATASETS[dataset_choice.value]
    mo.md(
        f"**{_meta['label'].capitalize()}**: {_meta['animal']}. {len(chosen.strings)} strings, {chosen.n_tokens} symbols. "
        f"Source: {_meta['citation']} https://doi.org/{_meta['doi']}. Licence: [{_meta['licence']}]({_meta['licence_url']})."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. What the symbols are

    One alphabet is built for the whole dataset. The table gives the name of each symbol in the source, how often it occurs and how long its bouts last. For the turtles and the seals the symbols are the behaviour classes of a published benchmark; `behaviour.load_strings("turtle", level="raw")` gives the finer labels of the source instead. For the finch the symbols are the syllable labels of one bird; labels of different birds do not name the same syllable, so birds are never pooled.
    """)
    return


@app.cell
def _(behaviour, chosen, views):
    views.table(behaviour.symbol_table(chosen), 3)
    return


@app.cell
def _(chosen, mo):
    string_slider = mo.ui.slider(0, len(chosen.strings) - 1, value=0, label="string", show_value=True)
    string_slider
    return (string_slider,)


@app.cell(hide_code=True)
def _(chosen, mo, string_slider):
    _string = chosen.strings[string_slider.value]
    mo.md(
        f"String {string_slider.value} of {chosen.individuals[string_slider.value]}: {len(_string)} symbols. "
        "The strip shows its first 150 bouts, each as wide as it lasts."
    )
    return


@app.cell
def _(behaviour, chosen, mo, string_slider):
    mo.Html(behaviour.ethogram_html(chosen, string_slider.value))
    return


@app.cell(hide_code=True)
def _(chosen, mo, string_slider):
    mo.md(f"""
    `{' '.join(chosen.strings[string_slider.value][:60])}`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Two surrogates

    The shuffled surrogate permutes all symbols of the dataset and cuts them into strings of the original lengths. It keeps how often each symbol occurs and nothing else. The first-order surrogate counts, inside the strings, how often each symbol follows each other symbol, and draws new strings from these transition probabilities; each new string starts with the first symbol of a real string and has its length. Both use fixed seeds. The first 40 symbols of the string chosen above, and of the two strings that replace it:
    """)
    return


@app.cell
def _(bm, chosen, pd, string_slider, views):
    _sets = bm.surrogate_sets(chosen.strings)
    _first = {kind: " ".join(strings[string_slider.value][:40]) for kind, strings in _sets.items()}
    views.table(pd.DataFrame({"first 40 symbols": _first}))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The measures, with the surrogates beside the data

    The measures are those of notebook A, at the same four levels, with the settings of the behaviour figure of the SymSeqBench preprint where a laptop allows it. The next cell reads the saved results, or computes them if the switch at the top is off.
    """)
    return


@app.cell
def _(bm, chosen, dataset_choice, load_saved, mo, saved_results):
    if load_saved.value:
        result = saved_results[dataset_choice.value]
        computed_here = False
    else:
        with mo.status.spinner(title="Measuring the data and its two surrogates") as _spinner:
            result = bm.plain(
                bm.measure_dataset(chosen.strings, progress=lambda part: _spinner.update(subtitle=part))
            )
        computed_here = True
    return computed_here, result


@app.cell(hide_code=True)
def _(computed_here, mo):
    mo.md("computed here" if computed_here else "saved results")
    return


@app.cell
def _(bm, result, views):
    _strings = "**Strings** (mean over strings ± standard error; n is the number of strings with an estimate)"
    views.titled_tables(
        {
            "**Tokens**": bm.level_table(result, "token").round(3),
            _strings: bm.level_table(result, "string"),
        }
    )
    return


@app.cell
def _(bm, result, views):
    views.titled_tables(
        {
            "**String sets**": bm.level_table(result, "string set").round(3),
            "**Grammars**": bm.level_table(result, "grammar").round(3),
        }
    )
    return


@app.cell
def _(bm, mo, views):
    _settings = mo.vstack([views.table(bm.settings_table()), mo.md(bm.SEEDS_NOTE)])
    mo.accordion(
        {
            "Which SymSeq call each column comes from": views.table(bm.measures_legend()),
            "Settings, next to those of the preprint's figure": _settings,
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read each table from the bottom row up. The shuffled row is what the symbol frequencies alone give. The first-order row adds the transition table. Where the data row differs from the first-order row, the data hold something that the table does not: strings that resemble each other more or less than chance, contexts longer than one symbol, or dependence over many symbols.

    Two cautions. The measures taken string by string need long strings: for short strings SymSeq returns no estimate, or an estimate at order 0, which is why n can be far below the number of strings. A single transition table also ignores that animals, days and songs differ from one another, so part of what it misses is this variety.

    Permutation entropy is not among the measures: it is defined for values that have an order, and behaviour labels have none, so its value would change if the labels were renamed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. What the estimates can and cannot say

    **Entropy rate.** The next table gives $h_k$, the entropy of the next symbol given the $k$ symbols before it, in bits per symbol. Every symbol after the first of a string is predicted; where fewer than $k$ symbols precede it, the context is the ones that exist. The true value can only fall as $k$ grows, so every $h_k$ is an upper bound of the entropy rate. The estimate falls for a second reason: with long contexts most contexts are seen once, and a context seen once predicts its own continuation perfectly. The surrogates show how large this is. The first-order surrogate has no structure beyond $k = 1$, and its exact value is in the last column. The shuffled surrogate has no structure at all. Whatever they lose at larger $k$ is bias, and only a larger loss in the data is structure.
    """)
    return


@app.cell
def _(bm, result, views):
    views.table(bm.order_table(result), 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Estimates against the amount of data.** The same estimates on the first symbols of the dataset only. The topological entropy of the observed transition graph can never exceed $\log_2$ of the number of symbols, and it grows with the amount of data, because rare transitions appear late. The entropy rate at order 2 of the first-order surrogate approaches its value at order 1 as data are added: the distance between the two is the bias.
    """)
    return


@app.cell
def _(bm, result, views):
    views.table(bm.tokens_table(result), 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Topological entropy from blocks.** SymSeq's lift method estimates the topological entropy of a sequence from blocks of growing length and stops when the value changes by less than 0.05 nats, or at the largest lift it is allowed. The table gives the value at every lift, in bits. For the first-order surrogate the answer is known: it is the value of the transition graph it was drawn from, in the last column of the data row. The estimate does not settle on it: it falls with every lift, because long blocks are too rare to be seen in all their variants, and it stops where the rule or the limit says, not where the answer is. The first lift can lie above the last column, because the joined sequence also contains the junctions between strings, which no animal produced.
    """)
    return


@app.cell
def _(bm, result, views):
    views.table(bm.lift_table(result), 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Known answers.** The surrogates have answers that are known in advance, and the pipeline must recover them: order 1 and the exact entropy rate for the first-order surrogate; order 0, no memory, and mutual information that leaves the 95 % band of shuffled sequences at no more distances than chance allows for the shuffled one.
    """)
    return


@app.cell
def _(bm, result, views):
    views.table(bm.checks_table(result))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Your own sequence

    Paste a sequence below, one string per line, with the symbols separated by spaces or commas (a line without separators is read one character per symbol). Or give the path of a CSV file with the columns `string` and `symbol`, and optionally `duration`. The sequence stays on this computer: nothing is written and nothing is sent. It needs at least 100 symbols and may have at most 30 000.
    """)
    return


@app.cell
def _(mo):
    own_text = mo.ui.text_area(placeholder="A B C A B C A C ...", label="paste a sequence", rows=4, full_width=True)
    own_path = mo.ui.text(placeholder="/path/to/my_sequence.csv", label="or the path of a CSV file", full_width=True)
    mo.vstack([own_text, own_path])
    return own_path, own_text


@app.cell
def _(behaviour, bm, mo, own_path, own_text):
    own_result, own_report, own_message = None, None, ""
    try:
        _own = behaviour.own_string_set(text=own_text.value, path=own_path.value)
        if _own is not None:
            own_report = _own[1]
            with mo.status.spinner(title="Measuring your sequence and its two surrogates"):
                own_result = bm.plain(bm.measure_dataset(_own[0].strings))
    except behaviour.OwnSequenceError as _error:
        own_message = str(_error)
    return own_message, own_report, own_result


@app.cell(hide_code=True)
def _(mo, own_message, own_report, own_result):
    (
        mo.callout(mo.md(f"This sequence cannot be measured. {own_message}"), kind="warn")
        if own_message
        else mo.md("No sequence given.")
        if own_result is None
        else mo.md(
            f"**Your sequence**: {own_report['n_strings']} strings, {own_report['n_tokens']} symbols, "
            f"{own_report['n_symbols']} different ones. " + " ".join(own_report["remarks"])
        )
    )
    return


@app.cell
def _(bm, own_result, views):
    views.titled_tables(
        {
            "**Strings**": bm.level_table(own_result, "string"),
            "**String sets**": bm.level_table(own_result, "string set").round(3),
        }
    ) if own_result is not None else None
    return


@app.cell
def _(bm, own_result, views):
    views.titled_tables(
        {
            "**Grammars**": bm.level_table(own_result, "grammar").round(3),
            "**Entropy rate at order k**": bm.order_table(own_result).round(3),
        }
    ) if own_result is not None else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. One figure

    Each column is one dataset; the dataset chosen above has a heavy frame, and your own sequence gets a column of its own. A: entropy of the next symbol given the $k$ symbols before it. Blue: data. Orange: first-order surrogate. Dashed grey: shuffled. Grey line: exact value of the first-order surrogate; where the orange points fall below it, the estimate is biased, and the blue points are biased in the same way. The order chosen by BIC for the data is written in each panel. B: mutual information between two symbols against their distance, on a logarithmic axis; the grey band is what shuffled sequences give.

    What a first-order description misses is the distance between blue and orange: contexts longer than one symbol in A, and dependence that outlasts the transition table in B.
    """)
    return


@app.cell
def _(bm, dataset_choice, own_result, plotting, result, saved_results, views):
    plotting.apply_style()
    figure_data = bm.figure_data(
        {**saved_results, dataset_choice.value: result}, highlight=dataset_choice.value, own=own_result
    )
    figure = plotting.figure_behaviour(figure_data)
    views.full_width(figure)
    return figure, figure_data


if __name__ == "__main__":
    app.run()
