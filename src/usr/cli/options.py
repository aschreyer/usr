"""The options that ``generate`` and ``search`` share.

Each option is an ``Annotated`` alias, so both commands share one help text. Typer
cannot read a PEP 695 ``type`` alias, so each is a plain assignment.
"""

from pathlib import Path
from typing import Annotated

import typer


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
"""``-o/--out``. Absent means stdout."""

ExplicitHOption = Annotated[
    bool,
    typer.Option(
        "--explicit-h/--no-explicit-h",
        help=(
            "Count the coordinates of explicit hydrogen atoms in the SD file. "
            "By default they are removed first."
        ),
    ),
]
"""``--explicit-h``. A database and its queries must use the same setting."""
