"""Tests for screening a table of moments against query moments."""

import math

import numpy as np
import pytest
from rdkit import Chem

from usr.moments import NUM_MOMENTS, MomentTable, get_usr_moment_table
from usr.search import Hit, screen


def table_of(*rows: float) -> MomentTable:
    """Return a table with one row per value, every moment of the row that value."""
    return MomentTable(
        tuple(f"mol-{number}" for number in range(len(rows))),
        tuple(range(len(rows))),
        np.array([[value] * NUM_MOMENTS for value in rows]).reshape(-1, NUM_MOMENTS),
    )


def test_each_query_row_finds_itself_first(aspirin_conformers: Chem.Mol) -> None:
    table = get_usr_moment_table([aspirin_conformers])

    hits = list(screen(table, table, top=1))

    assert [(hit.query_conformer, hit.conformer, hit.score) for hit in hits] == [
        (0, 0, 1.0),
        (1, 1, 1.0),
        (2, 2, 1.0),
    ]


def test_hits_come_best_first_and_ties_keep_database_order() -> None:
    """Derived by hand: against 0, the rows 2, 1, 1 and 0 score 1/3, 1/2, 1/2 and 1."""
    query = table_of(0.0)
    database = table_of(2.0, 1.0, 1.0, 0.0)

    hits = list(screen(query, database))

    assert hits == [
        Hit("mol-0", 0, "mol-3", 3, 1.0),
        Hit("mol-0", 0, "mol-1", 1, 0.5),
        Hit("mol-0", 0, "mol-2", 2, 0.5),
        Hit("mol-0", 0, "mol-0", 0, 1 / 3),
    ]


def test_top_keeps_that_many_hits_per_query_row() -> None:
    hits = list(screen(table_of(0.0, 2.0), table_of(2.0, 1.0, 0.0), top=2))

    assert [(hit.query_conformer, hit.conformer) for hit in hits] == [
        (0, 2),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_a_row_with_an_undefined_moment_comes_last() -> None:
    database = table_of(5.0, 0.0)
    database.moments[1, 2] = math.nan

    hits = list(screen(table_of(0.0), database))

    assert [hit.conformer for hit in hits] == [0, 1]
    assert math.isnan(hits[1].score)


def test_an_empty_database_gives_no_hits() -> None:
    assert list(screen(table_of(0.0), table_of())) == []


def test_a_top_below_one_is_refused_before_scoring() -> None:
    with pytest.raises(ValueError, match="top must be") as caught:
        screen(table_of(0.0), table_of(0.0), top=0)

    assert str(caught.value) == "top must be at least 1, not 0"


def make_usrcat_table(*rows: list[float]) -> MomentTable:
    """Return a table with one row per list of five values, each repeated 12 times."""
    return MomentTable(
        tuple(f"mol-{number}" for number in range(len(rows))),
        tuple(range(len(rows))),
        np.array([np.repeat(row, NUM_MOMENTS) for row in rows]).reshape(-1, 60),
    )


def test_weights_decide_which_usrcat_block_counts() -> None:
    """Derived by hand: against 0, the rows score 1/3 and 1/6, or 1/3 and 1/2.

    Row 0 differs by 2 in the first block only and row 1 by 1 in every block. Equal
    weights score them 1/3 and 1/6; weighing only the
    first block, 1/3 and 1/2.
    """
    query = make_usrcat_table([0.0] * 5)
    database = make_usrcat_table([2.0, 0, 0, 0, 0], [1.0] * 5)

    equal = list(screen(query, database))
    shape_only = list(screen(query, database, weights=[1.0, 0, 0, 0, 0]))

    assert [(hit.conformer, hit.score) for hit in equal] == [(0, 1 / 3), (1, 1 / 6)]
    assert [(hit.conformer, hit.score) for hit in shape_only] == [(1, 0.5), (0, 1 / 3)]


def test_query_and_database_of_different_descriptors_are_refused() -> None:
    with pytest.raises(ValueError, match="the query has") as caught:
        screen(table_of(0.0), make_usrcat_table([0.0] * 5))

    assert str(caught.value) == (
        "the query has 12 moments per row and the database 60: both must hold the "
        "same descriptor"
    )


def test_invalid_weights_are_refused_before_scoring() -> None:
    with pytest.raises(ValueError, match="weights must") as caught:
        screen(make_usrcat_table([0.0] * 5), make_usrcat_table(), weights=[1.0])

    assert str(caught.value) == (
        "weights must hold one weight per block of 12 moments, 5 in all, not 1"
    )
