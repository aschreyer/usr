"""Reading molecules, and reading and writing USR moments and search results.

:func:`read_molecules` reads an SD file, plain or gzipped, and skips the records RDKit
cannot read. :func:`write_moment_table` and :func:`read_moment_table` store a
:class:`~usr.moments.MomentTable` as CSV with the columns of :data:`MOMENT_HEADER`,
which is the database format of ``usr generate`` and ``usr search``.
:func:`write_hits` writes the scores of :func:`usr.search.screen` as CSV.

A :data:`Stream` is a path, or an open text stream such as ``sys.stdout``, which is
left open.
"""

import contextlib
import csv
import gzip
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
from loguru import logger
from rdkit import Chem

from usr.moments import MOMENT_COLUMNS, NUM_MOMENTS, MomentTable
from usr.search import Hit


type Stream = Path | str | TextIO
"""A path, which the function opens and closes, or an open text stream, left open."""

MOMENT_HEADER = ("Title", "Conformer", *MOMENT_COLUMNS)
"""The header of a moments CSV: the molecule, the conformer id and the moments."""

HIT_HEADER = ("Query", "QueryConformer", "Title", "Conformer", "Score")
"""The header of a search result CSV: the query row, the database row and the score."""


def read_molecules(path: Path | str, *, remove_hs: bool = True) -> Iterator[Chem.Mol]:
    """Return an iterator over the molecules of an SD file.

    The file is opened on the first ``next`` and closed when the iterator is exhausted
    or closed. Molecules are sanitised. A record that RDKit cannot read is skipped, and
    a warning names its position in the file.

    Parameters
    ----------
    path : Path | str
        An SD file. A name ending in ``.gz`` is read as gzip.
    remove_hs : bool, optional
        Whether to remove explicit hydrogen atoms as each record is read. Default is
        True.

    Returns
    -------
    Iterator[Chem.Mol]
        The molecules, in file order.

    Raises
    ------
    ValueError
        If ``path`` is not a file, or its name does not end in ``.sdf`` or
        ``.sdf.gz``. Raised by this call, before any reading.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"cannot read molecules from {path}: no such file")
    if not path.name.lower().endswith((".sdf", ".sdf.gz")):
        raise ValueError(
            f"cannot read molecules from {path.name}: only SD files are supported, "
            f"named *.sdf or *.sdf.gz"
        )

    return _read_records(path, remove_hs=remove_hs)


def _read_records(path: Path, *, remove_hs: bool) -> Iterator[Chem.Mol]:
    """Yield the molecules of the SD file at ``path``, skipping unreadable records."""
    logger.debug("reading molecules from {}", path)
    is_gzipped = path.suffix.lower() == ".gz"
    with gzip.open(path, "rb") if is_gzipped else open(path, "rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, sanitize=True, removeHs=remove_hs)
        # The stubs omit the None that the supplier yields for an unreadable record.
        records: Iterator[Chem.Mol | None] = iter(supplier)
        for number, mol in enumerate(records, 1):
            if mol is None:
                logger.warning(
                    "skipped record {} of {}: RDKit could not read it", number, path
                )
                continue
            yield mol


def _open_text(
    stream: Stream, mode: Literal["r", "w"]
) -> contextlib.AbstractContextManager[TextIO]:
    """Open a path as text in ``mode``, or pass an open stream through unclosed."""
    if isinstance(stream, Path | str):
        return open(stream, mode, newline="", encoding="utf-8")
    return contextlib.nullcontext(stream)


def write_moment_table(table: MomentTable, target: Stream) -> None:
    """Write a moment table as CSV, one row per conformer.

    The columns are :data:`MOMENT_HEADER`. Each moment is written with the digits
    that recover it exactly, and an undefined moment as ``nan``.

    Parameters
    ----------
    table : MomentTable
        The moments to write.
    target : Stream
        Where to write them.
    """
    with _open_text(target, "w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(MOMENT_HEADER)
        for title, conformer, moments in zip(
            table.titles, table.conformers, table.moments.tolist(), strict=True
        ):
            writer.writerow([title, conformer, *moments])
    logger.info("wrote {} rows of USR moments", len(table))


def read_moment_table(source: Stream) -> MomentTable:
    """Read a moment table from CSV, as :func:`write_moment_table` writes it.

    Parameters
    ----------
    source : Stream
        Where to read from.

    Returns
    -------
    MomentTable
        One row per CSV row after the header.

    Raises
    ------
    ValueError
        If the header is not :data:`MOMENT_HEADER`, or a row has the wrong number of
        fields, a conformer id that is not an integer or a moment that is not a number.
        The message names the line.
    """
    with _open_text(source, "r") as stream:
        reader = csv.reader(stream)
        header = tuple(next(reader, ()))
        if header != MOMENT_HEADER:
            raise ValueError(
                f"not a USR moments CSV: the header must be {','.join(MOMENT_HEADER)}, "
                f"not {','.join(header) or 'empty'}"
            )
        titles: list[str] = []
        conformers: list[int] = []
        rows: list[list[float]] = []
        for row in reader:
            if len(row) != len(MOMENT_HEADER):
                raise ValueError(
                    f"line {reader.line_num} of the moments CSV has {len(row)} fields, "
                    f"not {len(MOMENT_HEADER)}"
                )
            try:
                conformers.append(int(row[1]))
                rows.append([float(value) for value in row[2:]])
            except ValueError as error:
                raise ValueError(
                    f"line {reader.line_num} of the moments CSV is not a title, an "
                    f"integer conformer id and {NUM_MOMENTS} numbers: {error}"
                ) from None
            titles.append(row[0])

    moments = np.array(rows, dtype=np.float64).reshape(len(rows), NUM_MOMENTS)

    return MomentTable(tuple(titles), tuple(conformers), moments)


def write_hits(hits: Iterable[Hit], target: Stream) -> None:
    """Write search results as CSV, one row per hit, in the order given.

    The columns are :data:`HIT_HEADER`.

    Parameters
    ----------
    hits : Iterable[Hit]
        The hits. Each is written as it arrives.
    target : Stream
        Where to write them.
    """
    count = 0
    with _open_text(target, "w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HIT_HEADER)
        for hit in hits:
            writer.writerow(
                [hit.query, hit.query_conformer, hit.title, hit.conformer, hit.score]
            )
            count += 1
    logger.info("wrote {} hits", count)
