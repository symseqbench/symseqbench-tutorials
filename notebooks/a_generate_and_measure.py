import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="A. Generate and measure")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A. Generate and measure

    This notebook will first check the installation. It will then take one small grammar, compute its entropy rate and its memory (excess entropy) in closed form, draw two exact pictures of it, and compare the closed forms with estimates from generated sequences. After that it will generate strings from a ladder of grammars of increasing structural complexity and measure them at four levels: tokens, strings, string sets and grammars. It ends with one figure.

    The generators and the measures are those of SymSeq. The functions imported from `tutorial` are short wrappers around them.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Install check

    The table lists the packages of the tutorial. The two libraries must be at the commits recorded in the README.
    """)
    return


@app.cell
def _():
    import pandas as pd

    from tutorial import env
    from tutorial import notebook_views as views

    install_rows = env.environment_report()
    views.table(pd.DataFrame(install_rows))
    return env, install_rows, pd, views


@app.cell
def _(env, install_rows):
    if not env.environment_ok(install_rows):
        raise RuntimeError(
            "The install check failed for: "
            + ", ".join(row["component"] for row in install_rows if not row["ok"])
            + ". Run `uv sync` in the repository, or see OFFLINE.md."
        )
    install_ok = True
    return (install_ok,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Worked example: a grammar with two symbols

    The grammar has two symbols, A and B. After A the next symbol is B with probability $p$ and A otherwise. After B the next symbol is always A, so B is never repeated. SymSeq writes this grammar with A as the start state and B as the terminal state. Its strings are A...AB, and a long sequence is obtained by joining strings.

    The sequence is a chain with stationary distribution $\pi = (1, p)/(1+p)$. Its closed forms are

    $$h = \frac{H(p)}{1+p}, \qquad H_1 = H(\pi_A), \qquad E = H_1 - h, \qquad h_{\mathrm{top}} = \log_2 \frac{1+\sqrt{5}}{2},$$

    where $H(p)$ is the binary entropy, $h$ the entropy rate, $H_1$ the entropy of single symbols, $E$ the memory (excess entropy) and $h_{\mathrm{top}}$ the topological entropy. For $p = 1/2$: $h = 2/3$ bit per symbol and $E = \log_2 3 - 4/3 \approx 0.2516$ bit.
    """)
    return


@app.cell
def _(mo):
    p_slider = mo.ui.slider(0.05, 0.95, step=0.05, value=0.5, label="p(A → B)", show_value=True)
    length_slider = mo.ui.slider(3000, 60000, step=3000, value=18000, label="sequence length (symbols)", show_value=True)
    seed_box = mo.ui.number(0, 9999, value=0, label="seed")
    mo.hstack([p_slider, length_slider, seed_box], justify="start")
    return length_slider, p_slider, seed_box


@app.cell
def _(install_ok, length_slider, p_slider, seed_box):
    from tutorial import plotting, worked_example
    from tutorial.generate import two_state_grammar

    assert install_ok
    plotting.apply_style()
    p = p_slider.value
    grammar = two_state_grammar(p)
    # the control is the length of the sequence, so the work is the same for every p
    _n_strings = worked_example.strings_for_length(p, length_slider.value)
    strings, sequence = worked_example.joined_sequence(p, _n_strings, seed_box.value)
    return grammar, p, plotting, sequence, strings, worked_example


@app.cell(hide_code=True)
def _(grammar, mo, sequence, strings):
    _first = "  ".join("".join(string) for string in strings[:12])
    _start = "".join(sequence[:60])
    mo.md(f"""
    Transitions of the SymSeq grammar (source, target, probability; `#` ends a string): `{grammar.transitions}`

    First strings: `{_first}`

    Joined sequence, {len(strings)} strings and {len(sequence)} symbols: `{_start}...`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Two exact pictures of this grammar come first, before anything is estimated.

    The first counts the words that the grammar allows. A word is allowed if it never has two B in a row. There are 2, 3, 5, 8, 13 allowed words of 1, 2, 3, 4, 5 symbols, and so on. On a logarithmic axis the counts follow a straight line. Its slope is the topological entropy, $\log_2 \varphi \approx 0.694$ bit per symbol, where $\varphi$ is the golden ratio. SymSeq reports this value in nats: 0.481. The counts depend only on which transitions exist, not on $p$. The graph beside the plot is drawn with SymSeq's own class for Markov chains, for the two symbols without the end marker.
    """)
    return


@app.cell
def _(plotting, views, worked_example):
    figure_words = plotting.figure_worked_example_words(worked_example.words_panel())
    views.full_width(figure_words)
    return (figure_words,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The second picture is the block entropy $H(L)$: the entropy of the words of $L$ symbols, each with its probability, for the value of $p$ set above. For this grammar the values lie on a straight line from $L = 1$ on. The slope of the line is the entropy rate $h$. The line meets $L = 0$ at the memory $E$.
    """)
    return


@app.cell
def _(p, plotting, views, worked_example):
    figure_block_entropy = plotting.figure_worked_example_block_entropy(worked_example.block_entropy_panel(p))
    views.full_width(figure_block_entropy)
    return (figure_block_entropy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The next table will put the closed forms next to SymSeq's estimates on the joined sequence. The entropy rate and the memory come from `estimate_emc`: it selects a Markov order $R$ by BIC and returns the conditional entropy $h_R = H(X_{R+1} \mid X_1 \dots X_R)$ and $E_R = H_R - R\,h_R$, both computed from the observed block frequencies. For this grammar $R = 1$. The topological entropy comes from `topological_entropy` with the lift method, which SymSeq returns in nats and which is converted to bits here.
    """)
    return


@app.cell
def _(p, sequence, views, worked_example):
    comparison = worked_example.comparison(p, sequence)
    views.table(comparison, 4)
    return (comparison,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The estimates will now be repeated on the first $N$ symbols of the sequence, for increasing $N$. The entropy rate will also be estimated at several values of $p$: it is largest at $p = 1/\varphi^2 \approx 0.382$, where it reaches the topological entropy.
    """)
    return


@app.cell
def _(p, pd, seed_box, sequence, views, worked_example):
    convergence = worked_example.convergence(p, sequence)
    curve = worked_example.entropy_rate_curve(seed=seed_box.value)
    views.table(pd.DataFrame({key: convergence[key] for key in ("n", "entropy_rate", "excess_entropy")}), 4)
    return convergence, curve


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A ladder of grammars

    The next cells will generate strings from seven grammars: the worked example, a regular grammar sampled under constraints, an n-AX grammar, non-adjacent dependencies (plain, nested and crossed), and a Dyck language. The controls set the number of states and the alphabet size of the regular grammar, the number of contexts $n$ of n-AX, the number of fillers between the two elements of a non-adjacent dependency, and the nesting depth of the Dyck language. The number of states is kept between the alphabet size and twice the alphabet size.

    From here on the strings are joined with the end marker `#`, as SeqBench does when it builds a dataset. The marker counts as a symbol, so the values of the worked example change: $h = 1/2$ and $E = 1$ for $p = 1/2$.
    """)
    return


@app.cell
def _(mo):
    alphabet_slider = mo.ui.slider(2, 6, value=4, label="regular: alphabet size", show_value=True)
    states_slider = mo.ui.slider(2, 10, value=6, label="regular: number of states", show_value=True)
    contexts_slider = mo.ui.slider(2, 4, value=2, label="n-AX: contexts n", show_value=True)
    mo.hstack([alphabet_slider, states_slider, contexts_slider], justify="start")
    return alphabet_slider, contexts_slider, states_slider


@app.cell
def _(mo):
    fillers_slider = mo.ui.slider(0, 4, value=2, label="non-adjacent: fillers", show_value=True)
    depth_slider = mo.ui.slider(1, 4, value=2, label="Dyck: depth", show_value=True)
    per_rung_slider = mo.ui.slider(200, 3000, step=200, value=600, label="strings per grammar", show_value=True)
    mo.hstack([fillers_slider, depth_slider, per_rung_slider], justify="start")
    return depth_slider, fillers_slider, per_rung_slider


@app.cell
def _():
    from tutorial.ladder import Rung

    # Rungs added here appear in every table and in the figure below. Example:
    # my_rungs = [Rung("Dyck, three bracket types", "dyck", {"k": 3, "depth": 2, "pairs": 4})]
    my_rungs = []
    return Rung, my_rungs


@app.cell
def _(
    alphabet_slider,
    contexts_slider,
    depth_slider,
    fillers_slider,
    install_ok,
    my_rungs,
    states_slider,
):
    from tutorial import ladder

    assert install_ok
    rungs = [
        *ladder.default_ladder(
            # SymSeq finds such a grammar for every number of states from the alphabet size to twice that size
            n_states=min(max(states_slider.value, alphabet_slider.value), 2 * alphabet_slider.value),
            alphabet_size=alphabet_slider.value,
            n_contexts=contexts_slider.value,
            dependency_length=fillers_slider.value,
            depth=depth_slider.value,
        ),
        *my_rungs,
    ]
    return ladder, rungs


@app.cell
def _(ladder, pd, rungs, seed_box, views):
    examples = pd.DataFrame(
        [
            {
                "rung": rung.name,
                "family": rung.family,
                "three strings": "   ".join(" ".join(string) for string in ladder.sample_strings(rung, 3, seed_box.value)),
            }
            for rung in rungs
        ]
    ).set_index("rung")
    views.table(examples)
    return (examples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The column `family` names the class of the unbounded language family. Each grammar generated here has a bounded number of pairs or a bounded depth, and is therefore finite-state.

    The next table will hold the measures at the four levels. The token and string measures are computed on the joined sequence, the string-set and grammar measures on the set of strings.
    """)
    return


@app.cell
def _(ladder, per_rung_slider, rungs, seed_box):
    table = ladder.ladder_table(rungs, n_strings=per_rung_slider.value, seed=seed_box.value)
    return (table,)


@app.cell
def _(table, views):
    views.titled_tables({"**Tokens**": table[["", "token"]].droplevel(0, axis=1)})
    return


@app.cell
def _(table, views):
    views.titled_tables({"**Strings**": table["string"].round(3)})
    return


@app.cell
def _(table, views):
    views.titled_tables({"**String sets**": table["string set"].round(3)})
    return


@app.cell
def _(table, views):
    views.titled_tables({"**Grammars**": table["grammar"].round(3)})
    return


@app.cell
def _(ladder, mo, views):
    mo.accordion({"Which SymSeq call each column comes from": views.table(ladder.measures_legend())})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The entropy rate in the table is estimated at the Markov order shown next to it (`emc_order`). A finite order gives an upper bound: for grammars with long dependencies the estimate lies above the true value, and the memory is underestimated. The next table will show the entropy rate computed from each generator, without sampling, next to the estimate.
    """)
    return


@app.cell
def _(ladder, rungs, table, views):
    exact_values = ladder.exact_table(rungs)
    entropy_rates = exact_values[["mean_length", "entropy_rate"]].rename(columns={"entropy_rate": "entropy rate, exact"})
    entropy_rates.insert(2, "entropy rate, estimate", table[("string", "entropy_rate")])
    entropy_rates.insert(3, "order of the estimate", table[("string", "emc_order")])
    views.table(entropy_rates, 3)
    return (exact_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. One figure

    The figure will collect the results. A: estimates of the entropy rate and of the memory of the worked example against the length of the sequence, with the closed forms as grey lines. B: entropy rate of the worked example against $p$, with the topological entropy as the dashed line. C: entropy rate of each grammar of the ladder, exact (open circles) and estimated (filled circles). D: estimated memory of each grammar.
    """)
    return


@app.cell
def _(convergence, curve, exact_values, ladder, plotting, table, views):
    plotting.apply_style()
    figure = plotting.figure_generate_and_measure(convergence, curve, ladder.figure_rungs(table, exact_values))
    views.full_width(figure)
    return (figure,)


if __name__ == "__main__":
    app.run()
