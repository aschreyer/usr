"""Tests for the USR and USRCAT moments of coordinates, molecules and tables."""

import functools
import math
from collections.abc import Callable, Iterator

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolDescriptors

from usr.chem import get_pharmacophore_subsets
from usr.moments import (
    NUM_MOMENTS,
    NUM_USRCAT_MOMENTS,
    FloatArray,
    MomentTable,
    calc_usr_moments,
    calc_usrcat_moments,
    get_usr_moment_table,
    get_usr_moments,
    get_usrcat_moment_table,
    get_usrcat_moments,
)


def moments_by_formula(points: list[list[float]]) -> list[float]:
    """Return the USR moments of ``points``, from the definitions, in plain Python.

    The oracle for :func:`calc_usr_moments`. It shares no code with it:
    lists, ``math.dist`` and the moment sums written out. A tie between reference atoms
    goes to the first, as ``argmin`` and ``argmax`` break it.
    """
    count = len(points)
    centroid = [sum(point[axis] for point in points) / count for axis in range(3)]

    def distances(reference: list[float]) -> list[float]:
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
    coords = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    root = 2 ** (-1 / 6)
    sqrt2 = math.sqrt(2)

    expected = [4 / 3, sqrt2 / 3, root, 1, sqrt2, root, 2, sqrt2, -root, 1, sqrt2, root]

    assert calc_usr_moments(coords)[0].tolist() == pytest.approx(expected, rel=1e-14)


def test_a_regular_tetrahedron_has_an_undefined_centroid_skewness() -> None:
    """Derived by hand: the centroid skewness is 0/0, so the moment is NaN.

    The vertices are alternate corners of the cube of side 2 centred on the origin, so
    every vertex is sqrt(3) from the centroid. Every other reference point is a vertex,
    whose distances are 0 once and 2*sqrt(2) three times: mean 3*sqrt(2)/2, standard
    deviation sqrt(6)/2, and skewness -2/sqrt(3), whose cube root is
    -(2/sqrt(3))^(1/3).
    """
    coords = np.array([[[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]])
    vertex = [3 * math.sqrt(2) / 2, math.sqrt(6) / 2, -((2 / math.sqrt(3)) ** (1 / 3))]

    moments = calc_usr_moments(coords)[0]

    assert moments[:2].tolist() == pytest.approx([math.sqrt(3), 0.0], abs=1e-14)
    assert math.isnan(moments[2])
    assert moments[3:].tolist() == pytest.approx(vertex * 3, rel=1e-14)


def test_a_single_atom_has_zero_distances_and_undefined_skewness() -> None:
    """Derived by hand: every distance is 0, so each skewness is 0/0."""
    moments = calc_usr_moments(np.array([[[1.5, -2.0, 0.25]]]))[0]

    assert moments[[0, 1, 3, 4, 6, 7, 9, 10]].tolist() == [0.0] * 8
    assert np.isnan(moments[[2, 5, 8, 11]]).all()


def test_moments_of_random_conformers_agree_with_the_formula() -> None:
    """Checked against a separate computation: :func:`moments_by_formula`.

    That is a plain-Python implementation of the definitions, applied to each
    conformer on its own, so the test also shows that conformers do not mix. The
    points are seeded. Two summation orders round differently, so the comparison has
    a tolerance.
    """
    rng = np.random.default_rng(7)
    for count in (3, 10, 50):
        coords = rng.normal(scale=4.0, size=(5, count, 3))

        moments = calc_usr_moments(coords)

        for conformer, row in zip(coords.tolist(), moments.tolist(), strict=True):
            expected = moments_by_formula(conformer)
            assert row == pytest.approx(expected, rel=1e-12)


def test_a_tie_between_reference_atoms_goes_to_the_lower_index() -> None:
    """Checked against :func:`moments_by_formula`, which takes the first of a tie.

    The centroid is the origin, and every squared distance is an integer, so ties are
    exact. Atoms 1 and 5 tie as the closest to the centroid, 0 and 4 as the farthest,
    and 3 and 5 as the farthest from atom 0. Swapping either pair of a tie changes the
    moments, so each of cst, fct and ftf is pinned to the lower index.
    """
    points = [[-3, -1, 2], [-2, 1, -1], [2, -2, 0], [0, -1, -3], [2, 1, 3], [1, 2, -1]]
    coords = [[float(value) for value in point] for point in points]
    moments = calc_usr_moments(np.array([coords]))[0]

    assert moments.tolist() == pytest.approx(moments_by_formula(coords), rel=1e-12)
    for first, second in [(1, 5), (0, 4), (3, 5)]:
        swapped = list(coords)
        swapped[first], swapped[second] = coords[second], coords[first]
        assert not np.allclose(calc_usr_moments(np.array([swapped]))[0], moments)


CALCULATORS: dict[str, Callable[[FloatArray], FloatArray]] = {
    "USR": calc_usr_moments,
    "USRCAT": functools.partial(calc_usrcat_moments, subsets=[[0]]),
}
"""The moments of coordinates, by descriptor, for the checks they share."""


@pytest.mark.parametrize("descriptor", list(CALCULATORS))
@pytest.mark.parametrize("shape", [(4, 3), (1, 4, 2), (2, 1, 4, 3)])
def test_coordinates_in_another_shape_are_refused(
    descriptor: str, shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="coords must have the shape") as caught:
        CALCULATORS[descriptor](np.zeros(shape))

    assert str(caught.value) == (
        f"coords must have the shape (conformers, atoms, 3), not {shape}"
    )


def test_moments_are_invariant_under_rotation_and_translation(
    aspirin_conformers: Chem.Mol,
) -> None:
    coords = np.array([aspirin_conformers.GetConformer(0).GetPositions()])
    rng = np.random.default_rng(11)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1

    moved = coords @ rotation.T + np.array([12.5, -3.0, 40.0])

    np.testing.assert_allclose(
        calc_usr_moments(moved),
        calc_usr_moments(coords),
        rtol=1e-9,
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
    by_hand = calc_usr_moments(
        np.array([conformer.GetPositions() for conformer in heavy.GetConformers()])
    )

    without_h = get_usr_moments(aspirin_conformers)
    with_h = get_usr_moments(aspirin_conformers, explicit_h=True)

    np.testing.assert_array_equal(without_h, by_hand)
    assert with_h.shape == without_h.shape == (3, NUM_MOMENTS)
    assert not np.allclose(with_h, without_h)
    assert aspirin_conformers.GetNumAtoms() == 21


def test_a_molecule_without_conformers_has_no_rows() -> None:
    assert get_usr_moments(Chem.MolFromSmiles("CCO")).shape == (0, NUM_MOMENTS)


def test_explicit_h_on_and_off_both_give_a_row_per_record_and_differ(
    drug_mols: list[Chem.Mol], drug_names: list[str]
) -> None:
    """Measured: before the fix, ``--explicit-h`` wrote a header and no rows.

    The conformer loop sat under ``if not explicit_h``.
    """
    without_h = get_usr_moment_table(drug_mols)
    with_h = get_usr_moment_table(drug_mols, explicit_h=True)

    assert list(without_h.titles) == list(with_h.titles) == drug_names
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


def test_a_molecule_without_atoms_is_skipped_with_a_warning(
    aspirin_conformers: Chem.Mol, caplog_loguru: list[str]
) -> None:
    empty = Chem.RWMol()
    empty.AddConformer(Chem.Conformer(0))

    table = get_usr_moment_table([empty, aspirin_conformers])

    assert table.titles == ("aspirin",) * 3
    assert caplog_loguru == ["skipped molecule 1 (Mol-1): it has no atoms"]


@pytest.mark.parametrize("descriptor", list(CALCULATORS))
def test_coordinates_of_zero_atoms_are_refused(descriptor: str) -> None:
    with pytest.raises(ValueError, match="zero atoms") as caught:
        CALCULATORS[descriptor](np.empty((1, 0, 3)))

    assert str(caught.value) == f"cannot calculate {descriptor} moments of zero atoms"


def test_no_molecules_give_an_empty_table() -> None:
    table = get_usr_moment_table([])

    assert len(table) == 0
    assert table.moments.shape == (0, NUM_MOMENTS)


def test_a_table_whose_columns_differ_in_length_is_refused() -> None:
    with pytest.raises(ValueError, match="one title and one conformer") as caught:
        MomentTable(("a", "b"), (0,), np.zeros((1, NUM_MOMENTS)))

    assert str(caught.value) == (
        "a moment table needs one title and one conformer per row: "
        "2 titles, 1 conformers, 1 rows of moments"
    )


def calc_usrcat_block_by_formula(
    points: list[list[float]], subset: list[int]
) -> list[float]:
    """Return the USRCAT moments of the atoms ``subset`` of ``points``, in plain Python.

    The oracle for the blocks after the first of :func:`calc_usrcat_moments`. The
    reference points come from all atoms, as in :func:`moments_by_formula`; the
    distances, from the subset alone. It shares no code with the implementation.
    """
    count = len(points)
    centroid = [sum(point[axis] for point in points) / count for axis in range(3)]
    to_centroid = [math.dist(point, centroid) for point in points]
    closest = points[to_centroid.index(min(to_centroid))]
    farthest = points[to_centroid.index(max(to_centroid))]
    to_farthest = [math.dist(point, farthest) for point in points]
    farthest_from_farthest = points[to_farthest.index(max(to_farthest))]

    moments: list[float] = []
    for reference in (centroid, closest, farthest, farthest_from_farthest):
        values = [math.dist(points[index], reference) for index in subset]
        mean = sum(values) / len(values)
        second = sum((value - mean) ** 2 for value in values) / len(values)
        third = sum((value - mean) ** 3 for value in values) / len(values)
        moments += [mean, math.sqrt(second), math.cbrt(third / second**1.5)]

    return moments


LINE = np.array([[[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]])
"""Four atoms on the x axis at 0, 1, 2 and 3.

ctd is at 1.5, cst the atom at 1 (the lower index of a tie), fct the atom at 0 and ftf
the atom at 3.
"""


def test_a_one_atom_subset_has_its_distances_as_means_and_zero_spread() -> None:
    """Derived by hand: the atom at 0 is 1.5, 1, 0 and 3 from ctd, cst, fct and ftf."""
    block = calc_usrcat_moments(LINE, [[0]])[0, 12:]

    assert block.tolist() == [1.5, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0]


def test_a_two_atom_subset_has_the_hand_derived_moments() -> None:
    """Derived by hand for the atoms at 0 and 3.

    From ctd both are 1.5 away: no spread, so the standard deviation and skewness are
    0. From cst they are 1 and 2: mean 1.5, deviation 0.5. From fct and ftf, 0 and 3:
    mean 1.5, deviation 1.5. Two values have a third central moment of 0.
    """
    block = calc_usrcat_moments(LINE, [[0, 3]])[0, 12:]

    assert block.tolist() == pytest.approx(
        [1.5, 0, 0, 1.5, 0.5, 0, 1.5, 1.5, 0, 1.5, 1.5, 0], abs=1e-12
    )


def test_an_empty_subset_has_twelve_zero_moments() -> None:
    moments = calc_usrcat_moments(np.repeat(LINE, 2, axis=0), [[], [1]])

    assert moments.shape == (2, 36)
    assert moments[:, 12:24].tolist() == [[0.0] * 12] * 2


def test_the_first_usrcat_block_is_the_usr_moments() -> None:
    rng = np.random.default_rng(3)
    coords = rng.normal(scale=3.0, size=(4, 9, 3))

    moments = calc_usrcat_moments(coords, [[0, 2, 4], [1]])

    np.testing.assert_array_equal(moments[:, :12], calc_usr_moments(coords))


def test_usrcat_moments_of_random_conformers_agree_with_the_formula() -> None:
    """Checked against a separate computation: :func:`calc_usrcat_block_by_formula`.

    The points are seeded and every subset has at least three atoms, so every
    skewness is defined. Two summation orders round differently: a tolerance.
    """
    rng = np.random.default_rng(5)
    coords = rng.normal(scale=4.0, size=(5, 20, 3))
    subsets = [list(range(0, 20, 2)), [3, 7, 19], [1, 4, 5, 6, 18, 0]]

    moments = calc_usrcat_moments(coords, subsets)

    for conformer, row in zip(coords.tolist(), moments.tolist(), strict=True):
        assert row[:12] == pytest.approx(moments_by_formula(conformer), rel=1e-12)
        for number, subset in enumerate(subsets, 1):
            block = row[12 * number : 12 * (number + 1)]
            expected = calc_usrcat_block_by_formula(conformer, subset)
            assert block == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("subsets", "message"),
    [
        ([[0], [4]], "subsets[1] holds the atom index 4, outside 0 to 3"),
        ([[-1]], "subsets[0] holds the atom index -1, outside 0 to 3"),
        ([[0, 2, 0]], "subsets[0] holds the atom index 0 more than once"),
    ],
)
def test_a_subset_with_an_invalid_index_is_refused(
    subsets: list[list[int]], message: str
) -> None:
    with pytest.raises(ValueError, match="subsets") as caught:
        calc_usrcat_moments(LINE, subsets)

    assert str(caught.value) == message


def calc_rdkit_usrcat(mol: Chem.Mol) -> FloatArray:
    """Return RDKit's USRCAT descriptor of every conformer of ``mol`` without its Hs."""
    heavy = Chem.RemoveHs(mol)
    return np.array(
        [
            rdMolDescriptors.GetUSRCAT(heavy, confId=conformer.GetId())
            for conformer in heavy.GetConformers()
        ]
    )


def test_usrcat_moments_agree_with_rdkit(
    drug_mols: list[Chem.Mol], aspirin_conformers: Chem.Mol
) -> None:
    """Checked against a separate implementation: ``rdMolDescriptors.GetUSRCAT``.

    RDKit types the atoms with the same SMARTS and computes the moments in C++. A
    subset of two atoms has a skewness of 0, which rounding leaves near 1e-16 on
    either side; its cube root is then near 5e-6, hence the absolute tolerance.
    """
    for mol in [*drug_mols, aspirin_conformers]:
        np.testing.assert_allclose(
            get_usrcat_moments(mol), calc_rdkit_usrcat(mol), rtol=1e-9, atol=1e-5
        )


def test_usrcat_moments_have_a_row_per_conformer_and_the_usr_moments_first(
    aspirin_conformers: Chem.Mol,
) -> None:
    for explicit_h in (False, True):
        moments = get_usrcat_moments(aspirin_conformers, explicit_h=explicit_h)

        assert moments.shape == (3, NUM_USRCAT_MOMENTS)
        np.testing.assert_array_equal(
            moments[:, :12],
            get_usr_moments(aspirin_conformers, explicit_h=explicit_h),
        )


def test_custom_subsets_and_smarts_give_a_block_each(
    aspirin_conformers: Chem.Mol,
) -> None:
    heavy = Chem.RemoveHs(aspirin_conformers)
    coords = np.array([conformer.GetPositions() for conformer in heavy.GetConformers()])
    oxygens = [atom.GetIdx() for atom in heavy.GetAtoms() if atom.GetSymbol() == "O"]

    by_subsets = get_usrcat_moments(aspirin_conformers, subsets=[oxygens])
    by_smarts = get_usrcat_moments(aspirin_conformers, smarts=["[#8]"])

    assert by_subsets.shape == (3, 24)
    np.testing.assert_array_equal(by_subsets, calc_usrcat_moments(coords, [oxygens]))
    np.testing.assert_array_equal(by_smarts, by_subsets)


def test_explicit_h_moments_use_the_coordinates_with_hydrogens(
    aspirin_conformers: Chem.Mol,
) -> None:
    coords = np.array(
        [conformer.GetPositions() for conformer in aspirin_conformers.GetConformers()]
    )
    subsets = get_pharmacophore_subsets(aspirin_conformers, explicit_h=True)
    expected = calc_usrcat_moments(coords, subsets)

    np.testing.assert_array_equal(
        get_usrcat_moments(aspirin_conformers, explicit_h=True), expected
    )
    np.testing.assert_array_equal(
        get_usrcat_moments(aspirin_conformers, explicit_h=True, subsets=subsets),
        expected,
    )


def test_subsets_and_smarts_together_are_refused(aspirin_conformers: Chem.Mol) -> None:
    with pytest.raises(ValueError, match="pass subsets") as caught:
        get_usrcat_moments(aspirin_conformers, subsets=[[0]], smarts=["[#8]"])

    assert str(caught.value) == "pass subsets or smarts, not both"


def test_a_molecule_without_conformers_has_no_usrcat_rows() -> None:
    mol = Chem.MolFromSmiles("CCO")

    assert get_usrcat_moments(mol).shape == (0, NUM_USRCAT_MOMENTS)
    assert get_usrcat_moments(mol, subsets=[[0]]).shape == (0, 24)


def test_a_usrcat_table_skips_molecules_without_conformers_or_atoms(
    aspirin_conformers: Chem.Mol, caplog_loguru: list[str]
) -> None:
    flat = Chem.MolFromSmiles("CCO")
    flat.SetProp("_Name", "ethanol")
    empty = Chem.RWMol()
    empty.AddConformer(Chem.Conformer(0))

    table = get_usrcat_moment_table([flat, empty, aspirin_conformers])

    assert table.titles == ("aspirin",) * 3
    assert table.conformers == (0, 1, 2)
    np.testing.assert_array_equal(table.moments, get_usrcat_moments(aspirin_conformers))
    assert caplog_loguru == [
        "skipped molecule 1 (ethanol): it has no conformers",
        "skipped molecule 2 (Mol-2): it has no atoms",
    ]


def test_a_usrcat_table_has_a_block_per_custom_smarts(
    drug_mols: list[Chem.Mol], drug_names: list[str]
) -> None:
    table = get_usrcat_moment_table(drug_mols, smarts=["[#7]", "[#8]"], explicit_h=True)

    assert list(table.titles) == drug_names
    assert table.moments.shape == (6, 36)


def test_no_molecules_give_an_empty_usrcat_table() -> None:
    assert get_usrcat_moment_table([]).moments.shape == (0, NUM_USRCAT_MOMENTS)
    assert get_usrcat_moment_table([], smarts=["[#8]"]).moments.shape == (0, 24)


def test_an_invalid_smarts_is_refused_before_any_molecule_is_read() -> None:
    class Unread:
        def __iter__(self) -> Iterator[Chem.Mol]:
            raise AssertionError("a molecule was read")

    with pytest.raises(ValueError, match="smarts") as caught:
        get_usrcat_moment_table(Unread(), smarts=["[C"])

    assert str(caught.value) == "smarts[0] is not a valid SMARTS: '[C'"


@pytest.mark.parametrize("columns", [24, 60])
def test_a_table_accepts_any_positive_multiple_of_twelve_moments(columns: int) -> None:
    assert MomentTable(("a",), (0,), np.zeros((1, columns))).moments.shape[1] == columns


@pytest.mark.parametrize("shape", [(1, 11), (1, 0), (1, 13), (12,)])
def test_a_table_refuses_other_numbers_of_moments(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="moments must have the shape") as caught:
        MomentTable(("a",), (0,), np.zeros(shape))

    assert str(caught.value) == (
        "moments must have the shape (rows, n) for n a positive multiple of 12, "
        f"not {shape}"
    )


@pytest.mark.parametrize("flag", [True, np.True_])
def test_a_boolean_atom_index_is_refused(flag: bool) -> None:
    """Measured: ``operator.index(True)`` is 1, so a mask read as the atom index 1."""
    with pytest.raises(ValueError, match="subsets") as caught:
        calc_usrcat_moments(LINE, [[0], [flag]])

    assert str(caught.value) == "subsets[1] holds the boolean True, not an atom index"


def test_zero_conformers_give_zero_usrcat_rows() -> None:
    assert calc_usrcat_moments(np.empty((0, 4, 3)), [[0], []]).shape == (0, 36)


def test_a_single_atom_has_a_usr_block_and_a_zero_feature_block() -> None:
    """Derived by hand: every distance is 0; USR leaves the skewness NaN, USRCAT 0."""
    moments = calc_usrcat_moments(np.array([[[1.0, 2.0, 3.0]]]), [[0]])[0]

    assert np.isnan(moments[[2, 5, 8, 11]]).all()
    assert moments[[0, 1, 3, 4, 6, 7, 9, 10]].tolist() == [0.0] * 8
    assert moments[12:].tolist() == [0.0] * 12


def test_no_subsets_give_the_usr_moments_alone() -> None:
    np.testing.assert_array_equal(calc_usrcat_moments(LINE, ()), calc_usr_moments(LINE))


@pytest.mark.parametrize(
    ("subsets", "message"),
    [
        ([[-1]], "subsets[0] holds the atom index -1, outside 0 to 2"),
        ([[999]], "subsets[0] holds the atom index 999, outside 0 to 2"),
        ([[0, 0]], "subsets[0] holds the atom index 0 more than once"),
    ],
)
def test_invalid_subsets_are_refused_without_conformers(
    subsets: list[list[int]], message: str
) -> None:
    with pytest.raises(ValueError, match="subsets") as caught:
        get_usrcat_moments(Chem.MolFromSmiles("CCO"), subsets=subsets)

    assert str(caught.value) == message
