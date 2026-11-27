"""The ``usr`` command line: ``usr generate`` and ``usr search``.

:data:`app` is the Typer application that the ``usr`` console script runs. Each command
parses its options, calls the library and writes CSV; the logic is in the library.

Results go to stdout, or to the file ``--out`` names. Errors and log records go to
stderr, so ``usr search ... > hits.csv`` captures only the results. Log records show
from ``WARNING`` up by default; ``--verbose`` lowers the level to ``DEBUG``. Invoking
:data:`app` configures loguru for the whole process and does not restore it, so call it
in-process only in tests, which reset loguru after each test.

Exit codes are 0 for success, 1 for an internal failure and 2 for a caller's mistake,
such as a file in the wrong format. A mistake prints one line and no traceback.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from usr import __version__, io, moments, search


app = typer.Typer(
    name="usr",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

OutputOption = Annotated[
    Path | None,
    typer.Option(
        "--out",
        "-o",
        dir_okay=False,
        writable=True,
        help="Write the CSV to this file instead of stdout.",
    ),
]
"""``-o/--out``, which both commands take. Absent means stdout.

Typer cannot read a PEP 695 ``type`` alias, so this is a plain assignment.
"""

ExplicitHOption = Annotated[
    bool,
    typer.Option(
        "--explicit-h",
        help=(
            "Count the coordinates of explicit hydrogen atoms in the SD file. "
            "By default they are removed first."
        ),
    ),
]
"""``--explicit-h``. A database and its queries must use the same setting."""


def _report_version(requested: bool) -> None:
    """Print the version and exit, for the eager ``--version`` option."""
    if requested:
        typer.echo(f"usr {__version__}")
        raise typer.Exit(0)


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_report_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Log progress at DEBUG level. Place before the command: usr -v generate ...",
        ),
    ] = False,
) -> None:
    """Calculate USR (Ultrafast Shape Recognition) moments and search by shape.

    Use generate to write a moments database from an SD file, and search to score it
    against query molecules. Results go to stdout unless --out names a file.
    """
    # The CLI opts in to the library's logging. Replace loguru's handler, so each
    # record prints once, at WARNING by default and DEBUG with -v.
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "WARNING",
        format="<level>{level: <8}</level> {message}",
    )
    logger.enable("usr")


@contextmanager
def reporting() -> Iterator[None]:
    """Map the library's exceptions onto exit codes.

    Wraps a command body. Anything not handled here is a fault and propagates with its
    traceback.

    ``BrokenPipeError``
        The reader closed the pipe, as ``| head`` does. Exit 0. Handled first, because
        it is an ``OSError``.
    ``ValueError``, ``OSError``
        A caller's mistake: a file in the wrong format, or one that cannot be read or
        written, such as ``--out`` in a missing directory. One line on stderr, exit 2.

    Yields
    ------
    None
        The command body runs inside the context.
    """
    try:
        yield
    except BrokenPipeError:
        # Point stdout at a sink, so the flush at shutdown cannot raise and override
        # the exit code.
        sys.stdout = StringIO()
        raise typer.Exit(0) from None
    except (ValueError, OSError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error


@app.command(no_args_is_help=True)
def generate(
    input_path: Annotated[
        Path,
        typer.Argument(
            metavar="SDF",
            exists=True,
            dir_okay=False,
            readable=True,
            help="SD file of 3D conformers, *.sdf or *.sdf.gz.",
        ),
    ],
    output_path: OutputOption = None,
    explicit_h: ExplicitHOption = False,
) -> None:
    """Calculate the USR moments of every conformer in an SD file.

    Writes a CSV with one row per SD record: Title, Conformer and the moments M1 to
    M12. The CSV is a database for usr search. A record RDKit cannot read is skipped
    with a warning.
    """
    with reporting():
        # Every moment is calculated before the output opens: no partial file.
        mols = io.read_molecules(input_path, remove_hs=not explicit_h)
        table = moments.get_usr_moment_table(mols, explicit_h=explicit_h)
        io.write_moment_table(table, output_path or sys.stdout)


@app.command("search", no_args_is_help=True)
def search_database(
    query_path: Annotated[
        Path,
        typer.Argument(
            metavar="QUERY",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Query conformers: an SD file (*.sdf, *.sdf.gz) or a moments CSV (*.csv).",
        ),
    ],
    database_path: Annotated[
        Path,
        typer.Argument(
            metavar="DATABASE",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Moments CSV written by usr generate.",
        ),
    ],
    output_path: OutputOption = None,
    top: Annotated[
        int | None,
        typer.Option(
            "--top",
            "-n",
            min=1,
            help="Keep the N best hits per query conformer. By default every row is kept.",
        ),
    ] = None,
    explicit_h: ExplicitHOption = False,
) -> None:
    """Score every database conformer against every query conformer, best first.

    Writes a CSV with one row per hit: Query, QueryConformer, Title, Conformer and
    Score. Score is the USR similarity, from 1 for identical moments towards 0. The
    hits of each query conformer are in descending score order. --explicit-h applies
    to an SD query, and should match the setting the database was generated with.
    """
    with reporting():
        database = io.read_moment_table(database_path)
        if query_path.suffix.lower() == ".csv":
            query = io.read_moment_table(query_path)
        else:
            mols = io.read_molecules(query_path, remove_hs=not explicit_h)
            query = moments.get_usr_moment_table(mols, explicit_h=explicit_h)
        hits = search.screen(query, database, top=top)
        io.write_hits(hits, output_path or sys.stdout)
