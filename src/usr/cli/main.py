"""The ``usr`` application object, its root options, and how every command reports.

Results go to stdout, or to the file ``--out`` names. Errors and log records go to
stderr, so ``usr search ... > hits.csv`` captures only the results. Log records show
from ``WARNING`` up by default; ``--verbose`` lowers the level to ``DEBUG``.

Exit codes are 0 for success, 1 for an internal failure and 2 for a caller's mistake,
such as a file in the wrong format. A mistake prints one line and no traceback.

The module is named ``main``, not ``app``, because ``app`` is the Typer instance that
:mod:`usr.cli` re-exports.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from typing import Annotated

import typer
from loguru import logger

from usr import __version__


app = typer.Typer(
    name="usr",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def report_version(requested: bool) -> None:
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
            callback=report_version,
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
    _ = version  # Handled by its eager callback, before this body runs.
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

    Wraps a command body. Three cases are handled; anything else is a fault and
    propagates with its traceback.

    ``ValueError``
        A caller's mistake, such as a file in the wrong format. One line on stderr,
        exit 2.
    ``BrokenPipeError``
        The reader closed the pipe, as ``| head`` does. Exit 0.
    ``OSError``
        A file cannot be read or written, such as ``--out`` in a missing directory.
        One line on stderr, exit 2.

    Yields
    ------
    None
        The command body runs inside the context.
    """
    try:
        yield
    except ValueError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error
    except BrokenPipeError:
        # Point stdout at a sink, so the flush at shutdown cannot raise and override
        # the exit code.
        sys.stdout = StringIO()
        raise typer.Exit(0) from None
    except OSError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error
