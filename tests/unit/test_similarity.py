"""Tests for the USR similarity of one set of moments to many."""

import numpy as np
import pytest
from rdkit import Chem

from usr.moments import FloatArray, get_usr_moment_table
from usr.similarity import calc_usr_sim


@pytest.fixture(scope="module")
def drug_scores(drug_mols: list[Chem.Mol]) -> FloatArray:
    """Return the all-against-all scores of the six 3D drugs, hydrogen atoms removed.

    Row ``i`` scores drug ``i`` against all six.
    """
    rows = get_usr_moment_table(drug_mols).moments
    return np.array([calc_usr_sim(row, rows) for row in rows])


def test_moments_differing_by_one_everywhere_score_one_half() -> None:
    """Derived by hand: the mean absolute difference is 1, and 1 / (1 + 1) = 0.5."""
    assert calc_usr_sim(np.zeros(12), np.ones((2, 12))).tolist() == [0.5, 0.5]


def test_the_score_averages_over_the_length_of_the_vectors() -> None:
    """Derived by hand: three differences of 3 average to 3, and 1 / (1 + 3) = 0.25.

    Measured: before the fix, ``calc_usr_sim`` divided by 12 whatever the length, and
    scored this pair 1 / (1 + 9/12).
    """
    assert calc_usr_sim(np.zeros(3), np.full((1, 3), 3.0)).tolist() == [0.25]


def test_a_descriptor_is_identical_to_itself(drug_scores: FloatArray) -> None:
    assert np.diagonal(drug_scores).tolist() == [1.0] * 6


def test_the_score_is_symmetric(drug_scores: FloatArray) -> None:
    np.testing.assert_array_equal(drug_scores, drug_scores.T)


def test_different_drugs_score_below_one(drug_scores: FloatArray) -> None:
    off_diagonal = drug_scores[~np.eye(6, dtype=bool)]

    assert ((off_diagonal > 0) & (off_diagonal < 1)).all()


def test_each_row_scores_as_it_would_alone(drug_mols: list[Chem.Mol]) -> None:
    rows = get_usr_moment_table(drug_mols).moments
    for query in rows:
        alone = [calc_usr_sim(query, row[None])[0] for row in rows]

        assert calc_usr_sim(query, rows).tolist() == alone


@pytest.mark.parametrize(
    ("m1", "m2"),
    [
        (np.zeros(12), np.zeros(12)),
        (np.zeros((1, 12)), np.zeros((1, 12))),
        (np.zeros(12), np.zeros((1, 11))),
    ],
)
def test_mismatched_shapes_are_refused(m1: FloatArray, m2: FloatArray) -> None:
    with pytest.raises(ValueError, match="cannot calculate USR similarities") as caught:
        calc_usr_sim(m1, m2)

    assert str(caught.value) == (
        "cannot calculate USR similarities: m1 must be one set of moments and m2 a "
        f"matrix of sets of the same length, not the shapes {m1.shape} and {m2.shape}"
    )
