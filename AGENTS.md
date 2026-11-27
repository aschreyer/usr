# AGENTS.md

Rules for AI coding agents in this repository.

## Setup and commands

uv is the only prerequisite. It installs the Python in `.python-version` and
the locked dependencies. The package supports Python 3.12 and later.

```bash
uv sync --locked             # create or update .venv from uv.lock
uv run ruff check .          # lint
uv run ruff format .         # format
uv run mypy src tests        # strict type check
uv run pytest --cov          # tests and the doctests in src/, with coverage
uv build                     # sdist and wheel
uvx pre-commit run --all-files
```

- **Pass test arguments after `pytest`:** `uv run pytest tests/unit/test_io.py -k gzip`.
- **After changing a dependency or tool, run the checks in a fresh clone:**
  `git clone . <scratch dir>`, then `uv sync --locked` and the commands above.
  A `.venv` keeps packages that `pyproject.toml` no longer lists.

## Commits and releases

- **Conventional Commits:** `type(scope): lower-case imperative subject`.
- **The tag is the version.** hatch-vcs builds `v1.2.0` as `1.2.0`; an untagged
  tree builds a `.devN` version. Pushing a `v*` tag runs `publish.yml`, which
  publishes to PyPI by trusted publishing, then creates the GitHub release with
  generated notes.
- **Release notes group pull requests by label**, as `.github/release.yml`
  lists: label a `feat:` pull request `enhancement`, a `fix:` one `bug` and a
  `docs:` one `documentation`. Unlabelled ones go under "Other changes".
- **Never push, tag or publish unless asked.**
- **A measurement goes in the commit message**, never in a docstring or a
  comment, where nothing keeps it true.

## The science

- **Do not change what USR computes.** It implements Ballester and Richards
  (2007), doi:10.1002/jcc.20681. The moments are the
  mean, the population standard deviation and the cube root of the biased
  skewness of the distances to `ctd`, `cst`, `fct` and `ftf`, in that order.
- **Do not change the moments to match the paper's text.** It calls the second
  and third moments the variance and the skewness, and its Figure 3 matches
  the variance and the raw third central moment. The method's author confirmed
  that the population standard deviation and the cube root of the standardised
  skewness are intended; the code computes those. The README says so.
- **Moments match the reference to within floating-point tolerance.** Tests
  compare them with a tolerance (`pytest.approx` or `assert_allclose`), never
  bit for bit: a change of summation order or of cube-root library moves the
  last bit.

## Code style

- Ruff formats at 88 characters and sorts imports: stdlib, third-party,
  first-party. Absolute imports; no wildcard imports.
- **Import by name when the name says what it is** (`from usr.search import
  screen`). **Import the module when it does not**: `from usr import io`,
  then `io.read_molecules`.
- Full type hints; mypy runs in strict mode. Use PEP 695 `type` aliases. Do not
  write `from __future__ import annotations`; ruff bans it.
- **Public by default.** `usr/__init__.py` exports only `__version__`. A leading
  underscore marks a helper of its module.
- **Specific exceptions with the offending value in the message.** A caller's
  mistake is a `ValueError`; the CLI turns it into one line on stderr and exit
  code 2.
- **Validate before work.** A function that returns an iterator checks its
  arguments when called, not on the first `next`, as `read_molecules` and
  `screen` do.
- Paragraph-oriented formatting: one blank line between logical steps, and
  before the final `return` of a function with several.

## Docstrings and comments

- NumPy-style docstrings on every public module, class and function, in
  concise technical English: one idea to a sentence.
- **A module docstring** says what the module does, its main types and how data
  passes between them, and what calls it.
- **Parameters** say what the value is, its accepted values and the default.
  **Returns** and **Raises** say what, and when, in one sentence each.
- **A docstring states what its object does, not why it exists or its history.**
- **An example runs.** Doctests run with the tests. Seed anything random.
- **A `#` comment is one or two lines** on a non-obvious block or the
  constraint a line satisfies. Delete comments that restate the code.

## Logging

USR is a library. It logs through loguru and calls `logger.disable("usr")` on
import.

- **Never add a sink in `src/`** except in `cli.py`, which replaces
  loguru's handler at `WARNING`, or `DEBUG` with `-v`.
- **Pass values as arguments**: `logger.warning("skipped record {}", number)`.
- **The component that learns a fact logs it**, as `read_molecules` logs a
  skipped record.

## Command line

- Results go to stdout, or to `--out`. Errors, warnings and logs go to stderr.
- Exit codes: 0 for success, 1 for a fault, 2 for a caller's mistake.
- A command parses options, calls the library and writes the result; logic
  belongs in the library. Options both commands take are `Annotated`
  assignments in `cli.py`, since Typer cannot read a `type` alias.

## Tests

- `tests/unit/` mirrors `src/usr/`; CLI tests are in `test_cli.py`. Shared
  fixtures live in `tests/conftest.py`; SD files in `tests/data/`.
- **Name a test as a sentence that states the outcome:**
  `test_an_unreadable_record_is_skipped_with_a_warning`.
- **An exact value needs ground truth.** Derive it by hand or by a separate
  plain computation, and say which in the docstring. Two of our own
  implementations that agree are a regression pin, and the docstring says so.
- **Name the defect a regression test exists for** with a docstring starting
  `Measured:`.
- **Assert the whole message of a guard:** `match=` a prefix, then
  `str(caught.value) ==` the full text.
- **Warnings are errors** (`filterwarnings = ["error"]`). Do not silence one
  without a comment saying why.
- The CLI runs in-process through `typer.testing.CliRunner`. An autouse fixture
  removes its loguru sink after each test.
- `tests/data/chembl26-oral-approved.sdf` is 2D. Use `drugs-3d.sdf` where shape
  matters.

## Project structure

```
src/usr/          # The package: moments, similarity, search, io
src/usr/cli.py    # The `usr` command: generate and search
tests/unit/       # Unit tests, mirroring src/usr/
tests/data/       # SD files the tests read
pyproject.toml    # Package, ruff, mypy, pytest and coverage configuration
.github/          # CI, release and Dependabot
```
