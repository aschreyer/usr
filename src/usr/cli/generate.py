"""``usr generate``: write the USR moments of the molecules in an SD file.

Reads the molecules with :func:`usr.io.read_molecules`, calculates their moments with
:func:`usr.moments.get_usr_moment_table` and writes them with
:func:`usr.io.write_moment_table`. Every moment is calculated before the output is
opened, so a failed run leaves no partial file.
"""

import sys
from pathlib import Path
from typing import Annotated

import typer

from usr import io, moments
from usr.cli import main, options


@main.app.command(no_args_is_help=True)
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
    output_path: options.OutputOption = None,
    explicit_h: options.ExplicitHOption = False,
) -> None:
    """Calculate the USR moments of every conformer in an SD file.

    Writes a CSV with one row per SD record: Title, Conformer and the moments M1 to
    M12. The CSV is a database for usr search. A record RDKit cannot read is skipped
    with a warning.
    """
    with main.reporting():
        mols = io.read_molecules(input_path, remove_hs=not explicit_h)
        table = moments.get_usr_moment_table(mols, explicit_h=explicit_h)
        io.write_moment_table(table, output_path or sys.stdout)
