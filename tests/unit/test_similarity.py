"""Tests for the USR and USRCAT similarity of one set of moments to many."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from usr.moments import FloatArray, get_usr_moment_table, get_usrcat_moment_table
from usr.similarity import calc_usr_sim, calc_usrcat_sim, validate_weights


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


def test_usrcat_blocks_are_averaged_then_weighted() -> None:
    """Derived by hand: blocks that differ by 1 and 3 add up to 4, so 1 / (1 + 4).

    With the weights 2 and 0.5: 1 / (1 + 2 + 1.5) = 1 / 4.5. With the second block
    weighed 0: 1 / (1 + 1).
    """
    m1 = np.zeros(24)
    m2 = np.repeat([1.0, 3.0], 12)[None]

    assert calc_usrcat_sim(m1, m2).tolist() == [0.2]
    assert calc_usrcat_sim(m1, m2, weights=[2.0, 0.5]).tolist() == [1 / 4.5]
    assert calc_usrcat_sim(m1, m2, weights=np.array([1.0, 0.0])).tolist() == [0.5]


def test_usrcat_and_usr_similarity_agree_on_twelve_moments(
    drug_mols: list[Chem.Mol],
) -> None:
    rows = get_usr_moment_table(drug_mols).moments
    for query in rows:
        np.testing.assert_allclose(
            calc_usrcat_sim(query, rows), calc_usr_sim(query, rows), rtol=1e-15
        )


@pytest.mark.parametrize(
    "weights", [None, [1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.25, 0.0, 2.0, 0.5]]
)
def test_usrcat_similarity_agrees_with_rdkit(
    drug_mols: list[Chem.Mol], weights: list[float] | None
) -> None:
    """Checked against a separate implementation: ``rdMolDescriptors.GetUSRScore``."""
    rows = get_usrcat_moment_table(drug_mols).moments
    rdkit_weights = [1.0] * 5 if weights is None else weights
    for query in rows:
        # The stubs type the arguments as Boost.Python objects; RDKit takes lists.
        expected = [
            rdMolDescriptors.GetUSRScore(
                list(query),  # type: ignore[arg-type]
                list(row),  # type: ignore[arg-type]
                weights=rdkit_weights,  # type: ignore[arg-type]
            )
            for row in rows
        ]

        scores = calc_usrcat_sim(query, rows, weights=weights)

        np.testing.assert_allclose(scores, expected, rtol=1e-12)


def test_a_usrcat_descriptor_is_identical_to_itself_and_symmetric(
    drug_mols: list[Chem.Mol],
) -> None:
    rows = get_usrcat_moment_table(drug_mols).moments
    scores = np.array([calc_usrcat_sim(row, rows) for row in rows])

    assert np.diagonal(scores).tolist() == [1.0] * 6
    np.testing.assert_array_equal(scores, scores.T)


@pytest.mark.parametrize(
    ("m1", "m2"),
    [
        (np.zeros(24), np.zeros(24)),
        (np.zeros((1, 24)), np.zeros((1, 24))),
        (np.zeros(24), np.zeros((1, 12))),
    ],
)
def test_mismatched_usrcat_shapes_are_refused(m1: FloatArray, m2: FloatArray) -> None:
    with pytest.raises(ValueError, match="cannot calculate USRCAT") as caught:
        calc_usrcat_sim(m1, m2)

    assert str(caught.value) == (
        "cannot calculate USRCAT similarities: m1 must be one set of moments and m2 "
        f"a matrix of sets of the same length, not the shapes {m1.shape} and "
        f"{m2.shape}"
    )


@pytest.mark.parametrize("size", [0, 3, 13])
def test_usrcat_moments_not_in_blocks_of_twelve_are_refused(size: int) -> None:
    with pytest.raises(ValueError, match="moments come in blocks") as caught:
        calc_usrcat_sim(np.zeros(size), np.zeros((1, size)))

    assert str(caught.value) == (
        f"moments come in blocks of 12, so {size} moments cannot be weighted"
    )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (
            [1.0, 1.0],
            "weights must hold one weight per block of 12 moments, 5 in all, not 2",
        ),
        (
            [[1.0] * 5],
            "weights must be one-dimensional, one weight per block, not of the shape "
            "(1, 5)",
        ),
        (
            1.0,
            "weights must be one-dimensional, one weight per block, not of the shape ()",
        ),
        (
            [1.0, -0.5, 1.0, 1.0, 1.0],
            "weights must be finite and non-negative, not [1.0, -0.5, 1.0, 1.0, 1.0]",
        ),
        (
            [1.0, np.nan, 1.0, 1.0, 1.0],
            "weights must be finite and non-negative, not [1.0, nan, 1.0, 1.0, 1.0]",
        ),
        (
            [np.inf, 1.0, 1.0, 1.0, 1.0],
            "weights must be finite and non-negative, not [inf, 1.0, 1.0, 1.0, 1.0]",
        ),
        ([0.0] * 5, "weights must not all be zero, not [0.0, 0.0, 0.0, 0.0, 0.0]"),
    ],
)
def test_invalid_weights_are_refused(
    weights: list[float] | list[list[float]] | float, message: str
) -> None:
    with pytest.raises(ValueError, match="weights must") as caught:
        calc_usrcat_sim(np.zeros(60), np.zeros((1, 60)), weights=weights)  # type: ignore[arg-type]

    assert str(caught.value) == message


def test_default_weights_are_one_per_block() -> None:
    assert validate_weights(None, 36).tolist() == [1.0, 1.0, 1.0]


def test_an_empty_database_gives_no_usrcat_scores() -> None:
    assert calc_usrcat_sim(np.zeros(60), np.zeros((0, 60))).shape == (0,)


def test_twelve_moments_take_one_weight() -> None:
    """Derived by hand: a difference of 1 weighed 1 scores 1 / 2."""
    scores = calc_usrcat_sim(np.zeros(12), np.ones((1, 12)), weights=(1.0,))

    assert scores.tolist() == [0.5]


def test_huge_finite_weights_are_accepted_without_overflow() -> None:
    """Measured: summing the weights to test them for zero overflowed to inf."""
    weights = [1e308, 1e308]

    assert validate_weights(weights, 24).tolist() == weights
    assert calc_usrcat_sim(np.ones(24), np.ones((1, 24)), weights=weights).tolist() == [
        1.0
    ]
