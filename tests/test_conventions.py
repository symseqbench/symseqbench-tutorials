"""The notebook conventions of 13 November, the guide and the licence lines. Each test has a case it must reject.

The conventions: a notebook runs when it is opened (one table in ``pyproject.toml``), every cell that
only shows text hides its code and no other cell does, no visible code cell is longer than 15
lines with a median of at most 8, each notebook has the box "How this notebook works" directly
after its title, and a notebook run as a script on saved results prints nothing. The guide ``docs/RUNNING-THE-NOTEBOOKS.md`` may name only
files that exist and options that the installed tools have. ``data/behaviour/SOURCES.md`` must state
the DOI and the licence of each dataset as they were checked at the source on 12 November 2026.
"""

from __future__ import annotations

import ast
import os
import re
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("a_generate_and_measure", "b_behaviour", "c_networks")
GUIDE = REPO / "docs" / "RUNNING-THE-NOTEBOOKS.md"
BOX = "/// admonition | How this notebook works"

# DOI and licence of each dataset, as read at the source (DataCite) on 12 November 2026. They are
# written here a second time on purpose: a change of the licence text in the repository must fail.
CHECKED_AT_SOURCE = {
    "Zebrafish": ("10.5281/zenodo.13605471", "CC BY 4.0"),
    "Green turtle": ("10.5061/dryad.hhmgqnkd9", "CC0 1.0"),
    "Bengalese finch": ("10.6084/m9.figshare.4805749", "CC BY 4.0"),
    "Fur seals and sea lions": ("10.5281/zenodo.7947104", "CC BY 4.0"),
}


# ------------------------------------------------------------------ the notebooks


TEXT_BUILDERS = ("mo.md", "mo.callout")
LONGEST_VISIBLE_CELL = 15  # lines of code that the editor shows for one cell
LARGEST_MEDIAN = 8  # median of those lengths over the visible cells of a notebook


def builds_text(node: ast.AST, local: dict[str, ast.FunctionDef]) -> bool:
    """Is this expression text built by ``mo.md`` or ``mo.callout``?

    Directly, through a conditional of which one side may be nothing, or through a function that
    is defined in the same cell and returns such text.
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.IfExp):
        sides = (node.body, node.orelse)
        nothing_at_all = all(isinstance(side, ast.Constant) for side in sides)
        return not nothing_at_all and all(builds_text(side, local) for side in sides)
    if isinstance(node, ast.Call):
        name = ast.unparse(node.func)
        if name in TEXT_BUILDERS:
            return True
        if name in local:
            returned = [n.value for n in ast.walk(local[name]) if isinstance(n, ast.Return)]
            return bool(returned) and all(
                value is not None and builds_text(value, local) for value in returned
            )
    return False


def cell_facts(source: str) -> list[dict]:
    """What the rules need to know of every cell of a marimo notebook.

    ``shows_text``: what the cell displays (its last expression) is text in the sense of
    ``builds_text``. ``returns``: the names it hands to other cells. ``only_text``: it shows text
    and returns nothing. ``hidden``: its code is hidden. ``length``: the lines of code the editor
    shows, from the first line of the first statement to the last line of the last one, without
    the ``return`` that marimo writes; ``length_with_return`` counts that as well.
    """
    found = []
    for node in ast.parse(source).body:
        if not (isinstance(node, ast.FunctionDef) and node.decorator_list):
            continue
        decorator = node.decorator_list[0]
        hidden = isinstance(decorator, ast.Call) and any(
            keyword.arg == "hide_code" and getattr(keyword.value, "value", None) is True
            for keyword in decorator.keywords
        )
        last = node.body[-1]
        shown = node.body[:-1] if isinstance(last, ast.Return) else node.body
        returned = last.value if isinstance(last, ast.Return) else None
        names = (
            []
            if returned is None
            else [ast.unparse(element) for element in returned.elts]
            if isinstance(returned, ast.Tuple)
            else [ast.unparse(returned)]
        )
        local = {n.name: n for n in shown if isinstance(n, ast.FunctionDef)}
        display = shown[-1].value if shown and isinstance(shown[-1], ast.Expr) else None
        shows_text = display is not None and builds_text(display, local)
        found.append(
            {
                "shows_text": shows_text,
                "returns": names,
                "only_text": shows_text and not names,
                "hidden": hidden,
                "length": shown[-1].end_lineno - shown[0].lineno + 1 if shown else 0,
                "length_with_return": last.end_lineno - node.body[0].lineno + 1,
                "source": ast.get_source_segment(source, node) or "",
            }
        )
    return found


def cells(source: str) -> list[tuple[bool, bool, str]]:
    """For every cell of a marimo notebook: does it only show text, is its code hidden, and its source."""
    return [(cell["only_text"], cell["hidden"], cell["source"]) for cell in cell_facts(source)]


def wrongly_shown_or_hidden(source: str) -> list[int]:
    """Numbers of the cells that break the rule: a cell that only shows text hides its code, and only such a cell does.

    A cell only shows text if it returns nothing and what it displays is built by ``mo.md`` or
    ``mo.callout``; it may compute the numbers of its text in local variables first.
    """
    return [i for i, (only_text, hidden, _) in enumerate(cells(source)) if only_text != hidden]


def mixing_cells(source: str) -> list[int]:
    """Numbers of the cells that show text and also hand a name to other cells: they are to be split."""
    return [
        i for i, cell in enumerate(cell_facts(source)) if cell["shows_text"] and cell["returns"]
    ]


def visible_lengths(source: str, with_return: bool = False) -> list[int]:
    """The length of every cell that shows its code, in lines of code."""
    key = "length_with_return" if with_return else "length"
    return [cell[key] for cell in cell_facts(source) if not cell["hidden"]]


def too_long(source: str) -> list[str]:
    """What breaks the limits on the visible code of a notebook; empty if nothing does."""
    found = []
    lengths = visible_lengths(source)
    if max(lengths) > LONGEST_VISIBLE_CELL:
        found.append(f"a visible cell of {max(lengths)} lines")
    if max(visible_lengths(source, with_return=True)) > LONGEST_VISIBLE_CELL:
        found.append("a visible cell that is too long with its return statement")
    if statistics.median(lengths) > LARGEST_MEDIAN:
        found.append(f"a median of {statistics.median(lengths):g} lines")
    return found


def as_cell(body: str, hidden: bool = False, returns: str = "") -> str:
    """The source of one marimo cell with this body, for the cases a rule must reject."""
    decorator = "@app.cell(hide_code=True)" if hidden else "@app.cell"
    lines = "".join(f"    {line}\n" for line in body.splitlines())
    return (
        f"\n\n{decorator}\ndef _(mo, value):\n{lines}    return{' ' + returns if returns else ''}\n"
    )


def box_follows_title(source: str) -> bool:
    """The first text cell is the title, and the box is the text cell directly after it."""
    texts = [(i, text) for i, (only_text, _, text) in enumerate(cells(source)) if only_text]
    if len(texts) < 2:
        return False
    (first, title), (second, box) = texts[0], texts[1]
    return "\n    # " in title and second == first + 1 and BOX in box and source.count(BOX) == 1


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_text_cells_hide_their_code_and_only_they_do(name):
    source = (REPO / "notebooks" / f"{name}.py").read_text()
    assert wrongly_shown_or_hidden(source) == []
    assert mixing_cells(source) == []
    assert sum(only_text for only_text, _, _ in cells(source)) >= 8  # the rule is not empty
    # the rule holds for text that computes its numbers first, for a conditional and for a
    # function of the cell: each of these is a text cell, and is right only when hidden
    computed = '_number = 2 * value\nmo.md(f"The number is {_number}.")'
    conditional = (
        '(\n    mo.callout(mo.md("A warning."), kind="warn")\n    if value\n    else None\n)'
    )
    function = (
        'def _reading():\n    return mo.md(f"It is {value}.")\n\nNone if value else _reading()'
    )
    for body in (computed, conditional, function):
        assert wrongly_shown_or_hidden(source + as_cell(body, hidden=True)) == []
        # must reject: a text-building cell that shows its code
        assert len(wrongly_shown_or_hidden(source + as_cell(body))) == 1
    # must reject: a text-building cell of the notebook that shows its code ...
    assert (
        wrongly_shown_or_hidden(source.replace("@app.cell(hide_code=True)", "@app.cell", 1)) != []
    )
    # ... and a cell that defines a used name and hides its code: the first cell of the notebook,
    # a control, a table, and a figure
    assert (
        wrongly_shown_or_hidden(source.replace("@app.cell\n", "@app.cell(hide_code=True)\n", 1))
        != []
    )
    for body, returns in (
        ('choice = mo.ui.slider(1, 5, label="a control")\nchoice', "(choice,)"),
        ("views.table(value)", ""),
        ("figure = plotting.figure_networks(value)\nviews.full_width(figure)", "(figure,)"),
        ('mo.vstack([mo.md("A title"), views.table(value)])', ""),
    ):
        assert wrongly_shown_or_hidden(source + as_cell(body, returns=returns)) == []
        assert len(wrongly_shown_or_hidden(source + as_cell(body, True, returns))) == 1
    # must reject: a cell that builds text and also hands a name to other cells
    mixed = 'chosen = value + 1\nmo.md(f"The chosen value is {chosen}.")'
    assert len(mixing_cells(source + as_cell(mixed, returns="(chosen,)"))) == 1
    assert mixing_cells(source + as_cell(computed, hidden=True)) == []


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_visible_code_cells_are_short(name):
    """No visible cell is longer than 15 lines, and the median over the visible cells is at most 8."""
    source = (REPO / "notebooks" / f"{name}.py").read_text()
    assert too_long(source) == []
    assert len(visible_lengths(source)) >= 10  # the rule is not empty
    assert LONGEST_VISIBLE_CELL == 15 and LARGEST_MEDIAN == 8
    # a cell that only shows text may be as long as its text needs: its code is hidden
    long_body = "\n".join(f"_part{i} = {i}" for i in range(30))
    long_text = long_body + '\nmo.md(f"Thirty parts, the last one {_part29}.")'
    assert too_long(source + as_cell(long_text, hidden=True)) == []
    # must reject: one visible cell of 16 lines, and one that has 15 lines only without its return
    sixteen = "\n".join(f"part{i} = {i}" for i in range(16))
    assert too_long(source + as_cell(sixteen)) == [
        "a visible cell of 16 lines",
        "a visible cell that is too long with its return statement",
    ]
    fifteen = "\n".join(f"part{i} = {i}" for i in range(15))
    assert too_long(source + as_cell(fifteen, returns="(part0,)")) == [
        "a visible cell that is too long with its return statement"
    ]
    assert too_long(source + as_cell("\n".join(f"part{i} = {i}" for i in range(14)))) == []
    # must reject: a notebook whose visible cells have a median of 9 lines
    nine = "\n".join(f"part{i} = {i}" for i in range(9))
    many = "".join(as_cell(nine) for _ in range(3 * len(visible_lengths(source))))
    assert too_long(source + many) == ["a median of 9 lines"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_the_box_follows_the_title(name):
    source = (REPO / "notebooks" / f"{name}.py").read_text()
    assert box_follows_title(source)
    assert "docs/RUNNING-THE-NOTEBOOKS.md" in source and GUIDE.exists()
    # must reject: a notebook without the box, and one that has it twice
    assert not box_follows_title(source.replace(BOX, "/// admonition | Something else"))
    assert not box_follows_title(source + f"\n# {BOX}\n")


def run_as_script(path: Path) -> subprocess.CompletedProcess:
    """A plain Python run of a notebook, as the guide gives it, in one thread. Both streams are kept."""
    environment = dict(
        os.environ,
        MPLBACKEND="Agg",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    return subprocess.run(
        [sys.executable, str(path)], cwd=REPO, env=environment, capture_output=True, text=True
    )


def printed(done: subprocess.CompletedProcess) -> list[str]:
    """What a run wrote, one entry for each stream it wrote to. Empty when it wrote nothing."""
    streams = (("standard output", done.stdout), ("standard error", done.stderr))
    return [f"{name}: {text[:400]!r}" for name, text in streams if text]


ONE_CELL = """import marimo

app = marimo.App()


@app.cell
def _():
    {line}
    return


if __name__ == "__main__":
    app.run()
"""


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_a_notebook_run_as_a_script_prints_nothing(name):
    """With every control at its default the notebooks read saved results: not one byte is written."""
    done = run_as_script(REPO / "notebooks" / f"{name}.py")
    assert done.returncode == 0, done.stderr[-2000:]
    assert printed(done) == []


def test_what_a_cell_writes_is_seen_by_the_check(work_dir):
    def of_cell(line: str) -> list[str]:
        path = work_dir / "one_cell.py"
        path.write_text(ONE_CELL.format(line=line))
        done = run_as_script(path)
        assert done.returncode == 0, done.stderr[-2000:]
        return printed(done)

    assert of_cell("value = 1") == []
    # must reject: one printed character, a warning, and a line that a library logs
    assert of_cell('print("x", end="")') == ["standard output: 'x'"]
    warned = of_cell('__import__("warnings").warn("a warning of a cell")')
    assert len(warned) == 1 and "a warning of a cell" in warned[0]
    logged = of_cell('__import__("logging").getLogger("library").warning("not loaded")')
    assert len(logged) == 1 and "not loaded" in logged[0]


def runs_on_opening(project: dict) -> bool:
    return (
        project.get("tool", {}).get("marimo", {}).get("runtime", {}).get("auto_instantiate") is True
    )


def test_a_notebook_runs_when_it_is_opened():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert runs_on_opening(project)
    # must reject: the table absent, and the value switched off
    assert not runs_on_opening({"tool": {"ruff": {}}})
    assert not runs_on_opening({"tool": {"marimo": {"runtime": {"auto_instantiate": False}}}})


# ------------------------------------------------------------------ the guide


def named_files(text: str) -> list[str]:
    """Every path of the repository that the guide names: in a sentence, in a command, in a table or as a picture."""
    names = re.findall(
        r"(?<![\w/.-])((?:notebooks|fallback|docs|tutorial|data|scripts)/[\w./-]+\.\w+)", text
    )
    names += [f"notebooks/{name}" for name in re.findall(r"`([abc]_\w+\.py)`", text)]
    names += re.findall(r"`([A-Z]+\.md|pyproject\.toml)`", text)
    names += [f"docs/{target}" for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)]
    return sorted(set(names) - {"notebooks/NAME.py"})


def missing_files(text: str) -> list[str]:
    return [name for name in named_files(text) if not (REPO / name).exists()]


def options_of(*command: str) -> set[str]:
    done = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True)
    return set(re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", done.stdout))


def known_options() -> set[str]:
    marimo = (sys.executable, "-m", "marimo")
    known = {"--with"}  # an option of uv, not of marimo
    for sub in (("edit",), ("run",), ("export", "ipynb"), ("export", "html"), ("export", "script")):
        known |= options_of(*marimo, *sub, "--help")
    return known


def unknown_options(text: str, known: set[str]) -> list[str]:
    return sorted(set(re.findall(r"(?<![\w-])(--[a-z][a-z-]+)", text)) - known)


def test_the_guide_names_files_that_exist():
    text = GUIDE.read_text()
    assert len(named_files(text)) >= 8 and missing_files(text) == []
    for name in NOTEBOOKS:
        assert f"notebooks/{name}.py" in text
    # must reject: a misspelt file name, and a picture that is not there
    misspelt = text.replace("notebooks/b_behaviour.py", "notebooks/b_behavior.py")
    assert missing_files(misspelt) == ["notebooks/b_behavior.py"]
    assert missing_files(text.replace("`c_networks.py`", "`c_network.py`")) == [
        "notebooks/c_network.py"
    ]
    assert missing_files(text.replace("img/marimo-editor.png", "img/no-such-picture.png")) != []


def test_the_guide_names_options_and_tutorials_that_exist():
    text = GUIDE.read_text()
    known = known_options()
    assert unknown_options(text, known) == []
    tutorials = set(re.findall(r"marimo tutorial ([a-z-]+)", text))
    offered = subprocess.run(
        (sys.executable, "-m", "marimo", "tutorial", "--help"),
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout
    assert len(tutorials) >= 3 and all(
        re.search(rf"[{{|]{name}[|}}]", offered) for name in tutorials
    )
    # must reject: an option that marimo does not have, and a tutorial it does not offer
    assert unknown_options(text.replace("--headless", "--no-browser"), known) == ["--no-browser"]
    assert not re.search(r"[{|]jupyter-users[|}]", offered)


# ------------------------------------------------------------------ the licences


def sections(text: str) -> dict[str, str]:
    """The text of SOURCES.md under each second-level heading, by the heading without its file name."""
    parts = re.split(r"(?m)^## ", text)[1:]
    return {part.split("\n", 1)[0].split(" (`")[0]: part for part in parts}


def licence_faults(text: str) -> list[str]:
    faults = []
    found = sections(text)
    for dataset, (doi, licence) in CHECKED_AT_SOURCE.items():
        section = found.get(dataset, "")
        lines = [line for line in section.splitlines() if line.startswith("- Licence:")]
        if f"https://doi.org/{doi}" not in section:
            faults.append(f"{dataset}: DOI")
        if len(lines) != 1 or not lines[0].startswith(f"- Licence: {licence},"):
            faults.append(f"{dataset}: licence")
    return faults


def test_sources_state_the_doi_and_the_licence_checked_at_the_source():
    from tutorial import behaviour

    text = (REPO / "data" / "behaviour" / "SOURCES.md").read_text()
    assert licence_faults(text) == []
    stated = {(entry["doi"], entry["licence"]) for entry in behaviour.DATASETS.values()}
    assert stated == set(CHECKED_AT_SOURCE.values())
    # must reject: a licence made stricter or looser, a changed DOI, a missing section
    assert (
        licence_faults(text.replace("- Licence: CC BY 4.0,", "- Licence: CC BY-NC 4.0,", 1)) != []
    )
    assert licence_faults(text.replace("- Licence: CC0 1.0,", "- Licence: CC BY 4.0,")) == [
        "Green turtle: licence"
    ]
    assert (
        licence_faults(text.replace("10.6084/m9.figshare.4805749", "10.6084/m9.figshare.0000000"))
        != []
    )
    assert licence_faults(text.replace("## Zebrafish", "## Danio")) == [
        "Zebrafish: DOI",
        "Zebrafish: licence",
    ]
