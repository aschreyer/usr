"""``usr search``: score a moments database against query molecules.

The query is an SD file, whose moments are calculated as ``usr generate`` calculates
them, or a moments CSV. The database is a moments CSV. :func:`usr.search.screen`
scores them, and :func:`usr.io.write_hits` writes the hits as they are scored.
"""

import sys
from pathlib import Path
from typing import Annotated

import typer

from usr import io, moments, search
from usr.cli import main, options


@main.app.command("search", no_args_is_help=True)
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
    output_path: options.OutputOption = None,
    top: Annotated[
        int | None,
        typer.Option(
            "--top",
            "-n",
            min=1,
            help="Keep the N best hits per query conformer. By default every row is kept.",
        ),
    ] = None,
    explicit_h: options.ExplicitHOption = False,
) -> None:
    """Score every database conformer against every query conformer, best first.

    Writes a CSV with one row per hit: Query, QueryConformer, Title, Conformer and
    Score. Score is the USR similarity, from 1 for identical moments towards 0. The
    hits of each query conformer are in descending score order. --explicit-h applies
    to an SD query, and should match the setting the database was generated with.
    """
    with main.reporting():
        database = io.read_moment_table(database_path)
        if query_path.suffix.lower() == ".csv":
            query = io.read_moment_table(query_path)
        else:
            mols = io.read_molecules(query_path, remove_hs=not explicit_h)
            query = moments.get_usr_moment_table(mols, explicit_h=explicit_h)
        hits = search.screen(query, database, top=top)
        io.write_hits(hits, output_path or sys.stdout)
