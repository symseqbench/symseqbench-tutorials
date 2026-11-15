"""README.md and OFFLINE.md describe the three notebooks and their fallback folders as they are. Each test has a case it must reject."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from test_behaviour import broken_paragraphs

REPO = Path(__file__).resolve().parents[1]
FOLDERS = ("fallback/a_generate_and_measure/", "fallback/b_behaviour/", "fallback/c_networks/")
NOTEBOOKS = (
    "notebooks/a_generate_and_measure.py",
    "notebooks/b_behaviour.py",
    "notebooks/c_networks.py",
)

# Sentences that were true before the runs of notebook C existed, before its fallback was made and
# before zebrafish bouts stopped being merged. None may come back.
OUTDATED = (
    "are still placeholders",
    "have not been made",
    "will be made after its runs",
    "does not exist yet",
    "will hold the runs",
    "Every notebook has saved results in `fallback/`: the tables, the figure",
    "bouts of the same type were merged",
)


def outdated(text: str) -> list[str]:
    return [sentence for sentence in OUTDATED if sentence in text]


def prose_only(text: str) -> str:
    """A Markdown text without its fenced code blocks, whose lines are not paragraphs."""
    kept, inside = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            inside = not inside
            kept.append("")
        elif not inside:
            kept.append(line)
    return "\n".join(kept)


@pytest.mark.parametrize("name", ["README.md", "OFFLINE.md"])
def test_documents_name_the_three_notebooks_and_their_fallback_folders(name):
    text = (REPO / name).read_text()
    for folder in FOLDERS:
        assert folder in text, folder
        assert (REPO / folder).is_dir()
        assert any(path.suffix == ".html" for path in (REPO / folder).iterdir()), folder
    for notebook in NOTEBOOKS[1:]:
        assert notebook in text and (REPO / notebook).is_file()
    assert outdated(text) == []
    assert broken_paragraphs(prose_only(text)) == []
    # must reject: a document that names two of the three folders, and one with a broken paragraph
    assert FOLDERS[2] not in text.replace(FOLDERS[2], "")
    assert broken_paragraphs("A sentence that goes\non in the next line.\n") != []


def test_sentences_of_the_earlier_documents_are_refused():
    """must reject: the sentences these files held before the runs and the fallback of notebook C existed."""
    earlier = (
        "The results that notebook C shows for the grammars it does not train are still placeholders "
        "from short smoke runs; the runs of its figure have not been made. The fallback of notebook C "
        "will be made after its runs. `data/precomputed/` will hold the runs of the figure; it does not "
        "exist yet. Every notebook has saved results in `fallback/`: the tables, the figure and a "
        "static HTML export. Consecutive bouts of the same type were merged into one."
    )
    assert sorted(outdated(earlier)) == sorted(OUTDATED)
    # the sentences that describe the present state are in the README
    readme = (REPO / "README.md").read_text()
    assert "`data/precomputed/`: the six-grammar runs of notebook C (66 runs" in readme
    assert "--notebook c" in readme and "refuses placeholder results" in readme
    assert "worked_example_words" in readme and "worked_example_block_entropy" in readme
    assert (REPO / "data" / "precomputed" / "manifest.json").is_file()


def numbered_items_as_list(text: str) -> str:
    """Numbered list items written as bulleted ones: each is one line of a list, not a paragraph."""
    return "\n".join(re.sub(r"^\s*\d+\. ", "- ", line) for line in text.splitlines())


def notebook_c_row(text: str) -> str:
    """The row of notebook C in the table of the running guide."""
    (row,) = (line for line in text.splitlines() if line.startswith("| C, `c_networks.py`"))
    return row


def test_documents_describe_the_two_parts_of_notebook_c():
    """README, OFFLINE and the running guide say what notebook C shows and controls, as it is now."""
    import json

    readme = (REPO / "README.md").read_text()
    for named in (
        "scripts/run_complexity.py",
        "`data/complexity_pilot/`",
        "`data/complexity/`",
        "`figure_networks` and `figure_complexity`",
        "--allow-placeholder",
        "distance from the exact predictor",
        "`complexity.py`",
        "`complexity_runs.py`",
        "prediction score",
        "0 for the uniform guess, 1 for the exact predictor",
        "the first 50, 200 and 800 strings drawn from each",
        "one column for each number of training strings",
        "a level is identified by its band of topological entropy, not by its number",
    ):
        assert named in readme, named
    # must reject: the sentences of the first design, in which the divergence in bits was the
    # measure across grammars and the training strings were distinct
    for earlier in (
        "trains both networks on 50, 200 and 800 strings of each",
        "never negative; grammaticality judgement by the area",
    ):
        assert earlier not in readme, earlier
    assert (REPO / "scripts" / "run_complexity.py").is_file()
    assert (REPO / "data" / "complexity_pilot" / "manifest.json").is_file()
    # the README says that only pilot runs exist exactly as long as that is so
    production = REPO / "data" / "complexity" / "manifest.json"
    complete = production.is_file() and json.loads(production.read_text())["complete"]
    assert ("Only the pilot runs exist" in readme) != complete
    assert "data/complexity" in (REPO / "OFFLINE.md").read_text()
    # the row of the running guide names the controls the notebook has
    guide = (REPO / "docs" / "RUNNING-THE-NOTEBOOKS.md").read_text()
    row = notebook_c_row(guide)
    notebook = (REPO / "notebooks" / "c_networks.py").read_text()
    for label in ("load saved results", "level", "training strings"):
        assert f'"{label}"' in row and f'label="{label}"' in notebook, label
    assert "ends in the second" in guide and "Each notebook ends in one figure" not in guide
    assert broken_paragraphs(numbered_items_as_list(prose_only(guide))) == []
    # must reject: a paragraph of the guide broken in two, with or without numbered items around it
    assert broken_paragraphs(numbered_items_as_list("1. One item.\n2. Another item.\n")) == []
    assert broken_paragraphs(
        numbered_items_as_list("1. One item.\n\nA sentence that goes\non.\n")
    ) == ["on."]
    # must reject: the row as it was, which names controls the notebook no longer has
    earlier = (
        '| C, `c_networks.py` | Switch "load saved results" (on), dropdown "grammar", choice of the '
        "targets that count for the accuracy | With the switch off, two networks are trained. |"
    )
    assert 'label="grammar"' not in notebook and '"level"' not in notebook_c_row(earlier)
