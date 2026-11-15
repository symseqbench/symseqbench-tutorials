# Installing without network

The tutorial can be installed from a local folder of wheels (a wheelhouse) carried on a USB stick or a shared drive. The wheelhouse is built once, on a machine with network, and is not part of the repository (`wheelhouse/` is ignored by git).

## Status

Tested on Linux x86_64: the wheelhouse was built (68 wheels, 362 MB, plus the two library wheels), and the installation below was run in a clean copy of the repository inside a network namespace with no network and an empty uv cache. The install check passed, notebook A ran from start to end and the test suite passed. That test was made before notebooks B and C existed. All three notebooks use the one environment of the repository, so the same wheelhouse serves them, but notebooks B and C were not run from an installation without network.

Not tested: the installation on macOS and on Windows. The wheels for these platforms are selected from `uv.lock` by the same code, and `tests/test_environment.py` checks that a wheel exists for every package on each platform, but no installation was run there and their wheelhouse folders were not built. Also not tested: installing Python itself without network.

## Build the wheelhouse (with network)

```bash
uv run python scripts/build_wheelhouse.py --platform all
```

This writes one folder per platform (`linux-x86-64-glibc-2-28`, `macos-arm64-macos-12`, `macos-x86-64-macos-10-15`, `windows-x86-64`) and `wheelhouse/common` with one wheel of SymSeq and one of SeqBench, built from their pinned commits. Every wheel is downloaded from the URL recorded in `uv.lock` and its SHA-256 is verified. Each platform folder has a `requirements.txt` with exact versions. Copy the repository and `wheelhouse/` to the stick.

## Install from the wheelhouse (without network)

Python 3.11 must already be on the laptop: either a system Python 3.11, or one installed beforehand with `uv python install 3.11`. Replace `<platform>` by the folder of the laptop's platform.

```bash
cd symseqbench-tutorials
uv venv --python 3.11
uv pip install --offline --no-index --find-links wheelhouse/common --find-links wheelhouse/<platform> -r wheelhouse/<platform>/requirements.txt
uv pip install --offline --no-index --find-links wheelhouse/common --find-links wheelhouse/<platform> --no-deps -e .
uv run --no-sync --offline marimo edit notebooks/a_generate_and_measure.py
```

The install check at the top of the notebook passes on an installation from the wheelhouse: the wheels of the two libraries record no commit, so the check compares the SHA-256 of their Python sources with the value recorded for the pinned commits.

Notebooks B and C open the same way, with `notebooks/b_behaviour.py` or `notebooks/c_networks.py` in the last command. Both read their saved results from the repository (`data/behaviour_measures/` for notebook B; `data/precomputed/` and `data/complexity_pilot/` or `data/complexity/` for notebook C), and the tables of notebook B are in `data/behaviour/`, so neither needs the network.

## If the installation fails

Every notebook has a folder in `fallback/`: `fallback/a_generate_and_measure/`, `fallback/b_behaviour/` and `fallback/c_networks/`. Each holds a static HTML export of the notebook, which opens in any browser, the final figure (PDF, PNG, SVG) and the numbers of the figure (JSON). The folder of notebook A also holds its three tables and the two exact panels of the worked example. The folder of notebook C holds its two figures, each with its numbers. A participant without a working installation can follow with those, or pair up.
