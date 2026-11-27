"""Tests for the USR similarity of one pair and of one against many."""

import numpy as np
import pytest
from rdkit import Chem

from usr.moments import MomentTable, get_usr_moment_table
from usr.similarity import calc_usr_sim, calc_usr_sim_nd


@pytest.fixture(scope="module")
def drug_moments(drug_mols: list[Chem.Mol]) -> MomentTable:
    """Return the moments of the six 3D drugs, hydrogen atoms removed."""
    return get_usr_moment_table(drug_mols)


def test_moments_differing_by_one_everywhere_score_one_half() -> None:
    """Derived by hand: the mean absolute difference is 1, and 1 / (1 + 1) = 0.5."""
    assert calc_usr_sim(np.zeros(12), np.ones(12)) == 0.5
    assert calc_usr_sim_nd(np.zeros(12), np.ones((2, 12))).tolist() == [0.5, 0.5]


def test_the_score_averages_over_the_length_of_the_vectors() -> None:
    """Derived by hand: three differences of 3 average to 3, and 1 / (1 + 3) = 0.25.

    Measured: before the fix, ``calc_usr_sim`` divided by 12 whatever the length, and
    scored this pair 1 / (1 + 9/12).
    """
    assert calc_usr_sim(np.zeros(3), np.full(3, 3.0)) == 0.25
    assert calc_usr_sim_nd(np.zeros(3), np.full((1, 3), 3.0)).tolist() == [0.25]


def test_a_descriptor_is_identical_to_itself(drug_moments: MomentTable) -> None:
    for row in drug_moments.moments:
        assert calc_usr_sim(row, row) == 1.0


def test_the_score_is_symmetric(drug_moments: MomentTable) -> None:
    rows = drug_moments.moments
    for first in rows:
        for second in rows:
            assert calc_usr_sim(first, second) == calc_usr_sim(second, first)


def test_different_drugs_score_below_one(drug_moments: MomentTable) -> None:
    scores = calc_usr_sim_nd(drug_moments.moments[0], drug_moments.moments[1:])

    assert ((scores > 0) & (scores < 1)).all()


def test_one_against_many_agrees_with_each_pair(drug_moments: MomentTable) -> None:
    rows = drug_moments.moments
    for query in rows:
        pairwise = [calc_usr_sim(query, row) for row in rows]

        assert calc_usr_sim_nd(query, rows).tolist() == pytest.approx(
            pairwise, rel=1e-15
        )


def test_a_pair_of_different_lengths_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot calculate a USR similarity") as caught:
        calc_usr_sim(np.zeros(12), np.zeros(11))

    assert str(caught.value) == (
        "cannot calculate a USR similarity: m1 and m2 must be one set of moments "
        "each, of the same length, not the shapes (12,) and (11,)"
    )


def test_a_matrix_as_one_of_a_pair_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot calculate a USR similarity"):
        calc_usr_sim(np.zeros(12), np.zeros((1, 12)))


@pytest.mark.parametrize(
    ("m1", "m2"),
    [
        (np.zeros(12), np.zeros(12)),
        (np.zeros((1, 12)), np.zeros((1, 12))),
        (np.zeros(12), np.zeros((1, 11))),
    ],
)
def test_one_against_many_refuses_mismatched_shapes(
    m1: np.ndarray, m2: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="cannot calculate USR similarities") as caught:
        calc_usr_sim_nd(m1, m2)

    assert str(caught.value) == (
        "cannot calculate USR similarities: m1 must be one set of moments and m2 a "
        f"matrix of sets of the same length, not the shapes {m1.shape} and {m2.shape}"
    )
