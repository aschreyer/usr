"""Similarity search: score a database of USR moments against query moments.

:func:`screen` scores each query row against every database row with
:func:`~usr.similarity.calc_usr_sim` and yields the database rows best first, as
:class:`Hit` records. Both sides are :class:`~usr.moments.MomentTable` objects, which
:mod:`usr.moments` calculates and :mod:`usr.io` reads. Every row is a conformer, so a
molecule with several conformers in the database can be hit several times.

Examples
--------
Screen three conformers of aspirin against themselves:

>>> from rdkit import Chem
>>> from rdkit.Chem import rdDistGeom
>>> from usr.moments import get_usr_moment_table
>>> mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
>>> conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42)
>>> table = get_usr_moment_table([mol])
>>> best = next(screen(table, table, top=1))
>>> best.conformer == best.query_conformer, best.score
(True, 1.0)
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from usr.moments import MomentTable
from usr.similarity import calc_usr_sim


@dataclass(frozen=True, slots=True)
class Hit:
    """The score of one database conformer against one query conformer.

    Parameters
    ----------
    query : str
        The title of the query molecule.
    query_conformer : int
        The conformer id of the query row.
    title : str
        The title of the database molecule.
    conformer : int
        The conformer id of the database row.
    score : float
        The USR similarity of the two rows, in the interval ``(0, 1]``, or NaN where
        either has an undefined moment.
    """

    query: str
    query_conformer: int
    title: str
    conformer: int
    score: float


def screen(
    query: MomentTable, database: MomentTable, *, top: int | None = None
) -> Iterator[Hit]:
    """Yield the database rows that score highest against each query row.

    The query rows are taken in order. For each, the database rows come in descending
    score order; rows with equal scores keep their database order, and a NaN score
    comes last.

    Parameters
    ----------
    query : MomentTable
        The query conformers.
    database : MomentTable
        The conformers to score.
    top : int | None, optional
        The number of hits to yield per query row. Default is None, which yields every
        database row.

    Returns
    -------
    Iterator[Hit]
        One hit per query row and database row kept, scored as the iterator advances.

    Raises
    ------
    ValueError
        If ``top`` is less than 1. Raised by this call, before any scoring.
    """
    if top is not None and top < 1:
        raise ValueError(f"top must be at least 1, not {top}")

    return _screen(query, database, top)


def _screen(
    query: MomentTable, database: MomentTable, top: int | None
) -> Iterator[Hit]:
    """Yield the hits that :func:`screen` describes."""
    for title, conformer, moments in zip(
        query.titles, query.conformers, query.moments, strict=True
    ):
        scores = calc_usr_sim(moments, database.moments)
        # Negated, so the stable ascending sort puts the best first and NaN last.
        order = np.argsort(-scores, kind="stable")[:top]
        for row in order.tolist():
            yield Hit(
                title,
                conformer,
                database.titles[row],
                database.conformers[row],
                float(scores[row]),
            )
