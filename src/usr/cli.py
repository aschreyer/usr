"""The ``usr`` and ``usrcat`` command lines, each with ``generate`` and ``search``.

:data:`app` is the Typer application that the ``usr`` console script runs, on the 12
USR moments, and :data:`usrcat_app` the one that ``usrcat`` runs, on the 60 USRCAT
moments. They share their options, logging and error handling. Each command parses its
options, calls the library and writes CSV; the logic is in the library.

Results go to stdout, or to the file ``--out`` names. Errors and log records go to
stderr, so ``usr search ... > hits.csv`` captures only the results. Log records show
from ``WARNING`` up by default; ``--verbose`` lowers the level to ``DEBUG``. Invoking
either application configures loguru for the whole process and does not restore it, so
call it in-process only in tests, which reset loguru after each test.

Exit codes are 0 for success, 1 for an internal failure and 2 for a caller's mistake,
such as a file in the wrong format. A mistake prints one line and no traceback.
"""

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from usr import __version__, io, moments, search, similarity


def _create_application(name: str) -> typer.Typer:
    """Return an empty Typer application named ``name``, with the shared settings."""
    return typer.Typer(
        name=name,
        no_args_is_help=True,
        add_completion=False,
        rich_markup_mode="markdown",
        context_settings={"help_option_names": ["-h", "--help"]},
    )


app = _create_application("usr")
"""The ``usr`` command: USR moments and shape search."""

usrcat_app = _create_application("usrcat")
"""The ``usrcat`` command: USRCAT moments and shape and pharmacophore search."""

SdfArgument = Annotated[
    Path,
    typer.Argument(
        metavar="SDF",
        exists=True,
        dir_okay=False,
        readable=True,
        help="SD file of 3D conformers, `*.sdf` or `*.sdf.gz`.",
    ),
]
"""``SDF``, the input of both ``generate`` commands."""

QueryArgument = Annotated[
    Path,
    typer.Argument(
        metavar="QUERY",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Query conformers: an SD file (`*.sdf`, `*.sdf.gz`) or a moments CSV (`*.csv`).",
    ),
]
"""``QUERY``, the first argument of both ``search`` commands."""

DatabaseArgument = Annotated[
    Path,
    typer.Argument(
        metavar="DATABASE",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Moments CSV written by the generate command.",
    ),
]
"""``DATABASE``, the second argument of both ``search`` commands."""

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
"""``-o/--out``, which every command takes. Absent means stdout.

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

TopOption = Annotated[
    int | None,
    typer.Option(
        "--top",
        "-n",
        min=1,
        help="Keep the N best hits per query conformer. By default every row is kept.",
    ),
]
"""``-n/--top``, which both ``search`` commands take. Absent keeps every row."""

VerboseOption = Annotated[
    bool,
    typer.Option(
        "--verbose",
        "-v",
        help="Log progress at DEBUG level. Place before the command: -v generate ...",
    ),
]
"""``-v/--verbose``, which both applications take before the command."""


def _make_version_reporter(program: str) -> Callable[[bool], None]:
    """Return the callback of an eager ``--version`` option that names ``program``."""

    def report(requested: bool) -> None:
        if requested:
            typer.echo(f"{program} {__version__}")
            raise typer.Exit(0)

    return report


def _configure_logging(verbose: bool) -> None:
    """Replace loguru's handler with one on stderr and enable the library's logging.

    Each record prints once, at WARNING by default and DEBUG with ``verbose``.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "WARNING",
        format="<level>{level: <8}</level> {message}",
    )
    logger.enable("usr")


@app.callback()
def configure_usr(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_make_version_reporter("usr"),
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Calculate USR (Ultrafast Shape Recognition) moments and search by shape.

    Use generate to write a moments database from an SD file, and search to score it
    against query molecules. Results go to stdout unless --out names a file. For
    pharmacophore-aware shape search, use usrcat.
    """
    _configure_logging(verbose)


@usrcat_app.callback()
def configure_usrcat(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_make_version_reporter("usrcat"),
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Calculate USRCAT moments and search by shape and pharmacophore.

    USRCAT (USR with CREDO atom types) adds the moments of hydrophobic, aromatic,
    hydrogen-bond acceptor and donor atoms to the USR moments of all atoms. Use generate
    to write a moments database from an SD file, and search to score it against query
    molecules. Results go to stdout unless --out names a file.
    """
    _configure_logging(verbose)


@contextmanager
def _report_errors() -> Iterator[None]:
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


def _is_csv(path: Path) -> bool:
    """Return whether ``path`` names a moments CSV rather than an SD file."""
    return path.suffix.lower() == ".csv"


def _parse_weights(text: str | None) -> list[float] | None:
    """Return the weights of a comma-separated ``--weights`` value, or None."""
    if text is None:
        return None
    try:
        return [float(value) for value in text.split(",")]
    except ValueError:
        raise ValueError(
            f"--weights must be numbers separated by commas, such as 1,0.5,0.5,1,1, "
            f"not {text!r}"
        ) from None


@app.command("generate", no_args_is_help=True)
def generate_usr(
    input_path: SdfArgument,
    output_path: OutputOption = None,
    explicit_h: ExplicitHOption = False,
) -> None:
    """Calculate the USR moments of every conformer in an SD file.

    Writes a CSV with one row per SD record: Title, Conformer and the moments M1 to
    M12. The CSV is a database for usr search. A record RDKit cannot read is skipped
    with a warning.
    """
    with _report_errors():
        # Every moment is calculated before the output opens: no partial file.
        mols = io.read_molecules(input_path, remove_hs=not explicit_h)
        table = moments.get_usr_moment_table(mols, explicit_h=explicit_h)
        io.write_moment_table(table, output_path or sys.stdout)


@app.command("search", no_args_is_help=True)
def search_usr(
    query_path: QueryArgument,
    database_path: DatabaseArgument,
    output_path: OutputOption = None,
    top: TopOption = None,
    explicit_h: ExplicitHOption = False,
) -> None:
    """Score every database conformer against every query conformer, best first.

    Writes a CSV with one row per hit: Query, QueryConformer, Title, Conformer and
    Score. Score is the USR similarity, from 1 for identical moments towards 0. The
    hits of each query conformer are in descending score order. --explicit-h applies
    to an SD query, and should match the setting the database was generated with.
    """
    with _report_errors():
        database = io.read_moment_table(database_path)
        num_moments = database.moments.shape[1]
        if num_moments != moments.NUM_MOMENTS:
            raise ValueError(
                f"{database_path.name} holds {num_moments} moments per row, not the "
                f"{moments.NUM_MOMENTS} of USR: search it with usrcat search"
            )
        if _is_csv(query_path):
            query = io.read_moment_table(query_path)
        else:
            mols = io.read_molecules(query_path, remove_hs=not explicit_h)
            query = moments.get_usr_moment_table(mols, explicit_h=explicit_h)
        hits = search.screen(query, database, top=top)
        io.write_hits(hits, output_path or sys.stdout)


@usrcat_app.command("generate", no_args_is_help=True)
def generate_usrcat(
    input_path: SdfArgument,
    output_path: OutputOption = None,
    explicit_h: ExplicitHOption = False,
) -> None:
    """Calculate the USRCAT moments of every conformer in an SD file.

    Writes a CSV with one row per SD record: Title, Conformer and the moments M1 to
    M60, twelve each for all atoms and for the hydrophobic, aromatic, acceptor and
    donor atoms. The CSV is a database for usrcat search. A record RDKit cannot read
    is skipped with a warning.
    """
    with _report_errors():
        mols = io.read_molecules(input_path, remove_hs=not explicit_h)
        table = moments.get_usrcat_moment_table(mols, explicit_h=explicit_h)
        io.write_moment_table(table, output_path or sys.stdout)


@usrcat_app.command("search", no_args_is_help=True)
def search_usrcat(
    query_path: QueryArgument,
    database_path: DatabaseArgument,
    output_path: OutputOption = None,
    top: TopOption = None,
    explicit_h: ExplicitHOption = False,
    weights: Annotated[
        str | None,
        typer.Option(
            "--weights",
            "-w",
            metavar="W1,W2,...",
            help=(
                "Weigh each block of 12 moments, all atoms first, then hydrophobic, "
                "aromatic, acceptor and donor atoms: such as 1,0.5,0.5,1,1. Weights "
                "are non-negative and not all zero. By default each weighs 1."
            ),
        ),
    ] = None,
) -> None:
    """Score every database conformer against every query conformer, best first.

    Writes a CSV with one row per hit: Query, QueryConformer, Title, Conformer and
    Score. Score is the weighted USRCAT similarity, from 1 for identical moments
    towards 0. The hits of each query conformer are in descending score order.
    --explicit-h applies to an SD query, and should match the setting the database
    was generated with.
    """
    with _report_errors():
        weight_values = _parse_weights(weights)
        database = io.read_moment_table(database_path)
        num_moments = database.moments.shape[1]
        if num_moments == moments.NUM_MOMENTS:
            raise ValueError(
                f"{database_path.name} holds the {num_moments} moments of USR, not "
                f"USRCAT moments: search it with usr search"
            )
        # Checked before the query is read: a wrong --weights fails fast.
        block_weights = similarity.validate_weights(weight_values, num_moments)
        if _is_csv(query_path):
            query = io.read_moment_table(query_path)
        elif num_moments != moments.NUM_USRCAT_MOMENTS:
            raise ValueError(
                f"{database_path.name} holds {num_moments} moments per row, not the "
                f"{moments.NUM_USRCAT_MOMENTS} of USRCAT: pass a moments CSV with "
                f"{num_moments} moments as the query"
            )
        else:
            mols = io.read_molecules(query_path, remove_hs=not explicit_h)
            query = moments.get_usrcat_moment_table(mols, explicit_h=explicit_h)
        hits = search.screen(query, database, top=top, weights=block_weights)
        io.write_hits(hits, output_path or sys.stdout)
