"""Tests for the USR moments of coordinates, molecules and tables."""

import math
import warnings

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from scipy.stats import skew

from usr.moments import (
    NUM_MOMENTS,
    MomentTable,
    calc_skewness,
    calc_usr_moments_from_coords,
    get_usr_moment_table,
    get_usr_moments,
)


def moments_by_formula(points: list[tuple[float, float, float]]) -> list[float]:
    """Return the USR moments of ``points``, from the definitions, in plain Python.

    The oracle for :func:`calc_usr_moments_from_coords`. It shares no code with it:
    lists, ``math.dist`` and the moment sums written out. A tie between reference atoms
    goes to the first, as ``argmin`` and ``argmax`` break it.
    """
    count = len(points)
    centroid = tuple(sum(point[axis] for point in points) / count for axis in range(3))

    def distances(reference: tuple[float, ...]) -> list[float]:
        return [math.dist(point, reference) for point in points]

    def summarise(values: list[float]) -> list[float]:
        mean = sum(values) / count
        second = sum((value - mean) ** 2 for value in values) / count
        third = sum((value - mean) ** 3 for value in values) / count
        return [mean, math.sqrt(second), math.cbrt(third / second**1.5)]

    to_centroid = distances(centroid)
    closest = points[to_centroid.index(min(to_centroid))]
    farthest = points[to_centroid.index(max(to_centroid))]
    to_farthest = distances(farthest)
    farthest_from_farthest = points[to_farthest.index(max(to_farthest))]

    return [
        *summarise(to_centroid),
        *summarise(distances(closest)),
        *summarise(to_farthest),
        *summarise(distances(farthest_from_farthest)),
    ]


def test_three_collinear_atoms_have_the_hand_derived_moments() -> None:
    """Derived by hand for atoms at x = 0, 0 and 3.

    ctd is x = 1, so its distances are 1, 1, 2: mean 4/3, variance 2/9, third central
    moment 2/27 and skewness 1/sqrt(2). cst is the first atom (distances 0, 0, 3: mean
    1, variance 2, skewness 1/sqrt(2)); fct is the third (3, 3, 0: mean 2, variance 2,
    skewness -1/sqrt(2)); ftf is the first again. The cube root of 1/sqrt(2) is
    2^(-1/6).
    """
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    root = 2 ** (-1 / 6)
    sqrt2 = math.sqrt(2)

    expected = [4 / 3, sqrt2 / 3, root, 1, sqrt2, root, 2, sqrt2, -root, 1, sqrt2, root]

    assert calc_usr_moments_from_coords(coords).tolist() == pytest.approx(
        expected, rel=1e-14
    )


def test_a_regular_tetrahedron_has_an_undefined_centroid_skewness() -> None:
    """Derived by hand: the centroid skewness is 0/0, so the moment is NaN.

    The vertices are alternate corners of the cube of side 2 centred on the origin, so
    every vertex is sqrt(3) from the centroid. Every other reference point is a vertex,
    whose distances are 0 once and 2*sqrt(2) three times: mean 3*sqrt(2)/2, standard
    deviation sqrt(6)/2, and skewness -2/sqrt(3), whose cube root is
    -(2/sqrt(3))^(1/3).
    """
    coords = np.array([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    vertex = [3 * math.sqrt(2) / 2, math.sqrt(6) / 2, -((2 / math.sqrt(3)) ** (1 / 3))]

    moments = calc_usr_moments_from_coords(coords)

    assert moments[:2].tolist() == pytest.approx([math.sqrt(3), 0.0], abs=1e-14)
    assert math.isnan(moments[2])
    assert moments[3:].tolist() == pytest.approx(vertex * 3, rel=1e-14)


def test_moments_of_random_points_agree_with_the_formula() -> None:
    """Checked against a separate computation: :func:`moments_by_formula`.

    That is a plain-Python implementation of the definitions. The points are seeded.
    """
    rng = np.random.default_rng(7)
    for count in (3, 10, 50):
        coords = rng.normal(scale=4.0, size=(count, 3))

        expected = moments_by_formula([tuple(point) for point in coords.tolist()])

        assert calc_usr_moments_from_coords(coords).tolist() == pytest.approx(
            expected, rel=1e-10
        )


def test_moments_are_invariant_under_rotation_and_translation(
    aspirin_conformers: Chem.Mol,
) -> None:
    coords = aspirin_conformers.GetConformer(0).GetPositions()
    rng = np.random.default_rng(11)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1

    moved = coords @ rotation.T + np.array([12.5, -3.0, 40.0])

    np.testing.assert_allclose(
        calc_usr_moments_from_coords(moved),
        calc_usr_moments_from_coords(coords),
        rtol=1e-9,
    )


def test_the_skewness_agrees_bit_for_bit_with_scipy() -> None:
    """Pin the skewness to ``scipy.stats.skew``, which it replaced.

    A regression pin: two implementations of one formula, so agreement shows equal
    rounding, not truth.
    """
    rng = np.random.default_rng(3)
    samples = [rng.gamma(2.0, size=size) * 5 for size in (3, 12, 100, 1000)]
    samples += [np.full(5, 2.5), np.array([0.0])]

    for sample in samples:
        with warnings.catch_warnings():
            # scipy warns of precision loss on equal values; the result is still NaN.
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = float(skew(sample))

        assert calc_skewness(sample) == expected or (
            math.isnan(calc_skewness(sample)) and math.isnan(expected)
        )


def test_seeded_conformer_generation_gives_the_same_moments() -> None:
    def embed(seed: int) -> Chem.Mol:
        mol = Chem.AddHs(Chem.MolFromSmiles("CN1CCC[C@H]1c1cccnc1"))
        rdDistGeom.EmbedMultipleConfs(mol, numConfs=4, randomSeed=seed)
        return mol

    first = get_usr_moments(embed(42))

    assert first.shape == (4, NUM_MOMENTS)
    np.testing.assert_array_equal(get_usr_moments(embed(42)), first)
    assert not np.array_equal(get_usr_moments(embed(43)), first)


def test_hydrogens_are_removed_unless_explicit_h_is_set(
    aspirin_conformers: Chem.Mol,
) -> None:
    """Measured: before the fix, hydrogen atoms counted either way.

    ``Chem.RemoveHs`` returns a new molecule, and the code discarded it.
    """
    heavy = Chem.RemoveHs(aspirin_conformers)
    by_hand = [
        calc_usr_moments_from_coords(conformer.GetPositions())
        for conformer in heavy.GetConformers()
    ]

    without_h = get_usr_moments(aspirin_conformers)
    with_h = get_usr_moments(aspirin_conformers, explicit_h=True)

    np.testing.assert_array_equal(without_h, np.array(by_hand))
    assert with_h.shape == without_h.shape == (3, NUM_MOMENTS)
    assert not np.allclose(with_h, without_h)
    assert aspirin_conformers.GetNumAtoms() == 21


def test_a_molecule_without_conformers_has_no_rows() -> None:
    assert get_usr_moments(Chem.MolFromSmiles("CCO")).shape == (0, NUM_MOMENTS)


def test_explicit_h_on_and_off_both_give_a_row_per_record_and_differ(
    drug_mols: list[Chem.Mol],
) -> None:
    """Measured: before the fix, ``--explicit-h`` wrote a header and no rows.

    The conformer loop sat under ``if not explicit_h``.
    """
    without_h = get_usr_moment_table(drug_mols)
    with_h = get_usr_moment_table(drug_mols, explicit_h=True)

    names = ("aspirin", "caffeine", "ibuprofen", "paracetamol", "nicotine", "diazepam")
    assert without_h.titles == with_h.titles == names
    assert without_h.conformers == with_h.conformers == (0,) * 6
    assert not np.allclose(with_h.moments, without_h.moments)


def test_a_table_names_rows_by_title_and_conformer(
    aspirin_conformers: Chem.Mol,
) -> None:
    untitled = Chem.Mol(aspirin_conformers)
    untitled.SetProp("_Name", "")

    table = get_usr_moment_table([aspirin_conformers, untitled])

    assert table.titles == ("aspirin",) * 3 + ("Mol-2",) * 3
    assert table.conformers == (0, 1, 2) * 2
    np.testing.assert_array_equal(table.moments[:3], table.moments[3:])


def test_a_table_reads_the_title_from_the_named_property(
    aspirin_conformers: Chem.Mol,
) -> None:
    mol = Chem.Mol(aspirin_conformers)
    mol.SetProp("chembl_id", "CHEMBL25")

    assert get_usr_moment_table([mol], title_tag="chembl_id").titles[0] == "CHEMBL25"
    assert get_usr_moment_table([mol], title_tag="missing").titles[0] == "Mol-1"


def test_a_molecule_without_conformers_is_skipped_with_a_warning(
    aspirin_conformers: Chem.Mol, caplog_loguru: list[str]
) -> None:
    flat = Chem.MolFromSmiles("CCO")
    flat.SetProp("_Name", "ethanol")

    table = get_usr_moment_table([flat, aspirin_conformers])

    assert table.titles == ("aspirin",) * 3
    assert caplog_loguru == ["skipped molecule 1 (ethanol): it has no conformers"]


def test_no_molecules_give_an_empty_table() -> None:
    table = get_usr_moment_table([])

    assert len(table) == 0
    assert table.moments.shape == (0, NUM_MOMENTS)


def test_a_table_with_the_wrong_number_of_moments_is_refused() -> None:
    with pytest.raises(ValueError, match="moments must have the shape") as caught:
        MomentTable(("a",), (0,), np.zeros((1, 11)))

    assert str(caught.value) == "moments must have the shape (rows, 12), not (1, 11)"


def test_a_table_whose_columns_differ_in_length_is_refused() -> None:
    with pytest.raises(ValueError, match="one title and one conformer") as caught:
        MomentTable(("a", "b"), (0,), np.zeros((1, NUM_MOMENTS)))

    assert str(caught.value) == (
        "a moment table needs one title and one conformer per row: "
        "2 titles, 1 conformers, 1 rows of moments"
    )
