"""USR and USRCAT moments: the shape descriptors of Ultrafast Shape Recognition.

USR (Ballester and Richards, 2007) describes the shape of one conformer by the
distributions of its atomic distances to four reference points:

- ``ctd``, the centroid of the atoms;
- ``cst``, the atom closest to ``ctd``;
- ``fct``, the atom farthest from ``ctd``;
- ``ftf``, the atom farthest from ``fct``.

Each distribution contributes its mean, its standard deviation and the cube root of its
skewness, in that order, so a conformer has :data:`NUM_MOMENTS` moments.

USRCAT (Schreyer and Blundell, 2012) adds a block of :data:`NUM_MOMENTS` moments for
each of four pharmacophoric atom types, :data:`usr.chem.USRCAT_SMARTS`: the distances
of the atoms of that type to the same four reference points, which all atoms define. A
conformer has :data:`NUM_USRCAT_MOMENTS` USRCAT moments, the USR moments first.

:func:`calc_usr_moments` and :func:`calc_usrcat_moments` compute the moments of an
array of conformer coordinates. :func:`get_usr_moments` and :func:`get_usrcat_moments`
compute them for every conformer of an RDKit molecule, and
:func:`get_usr_moment_table` and :func:`get_usrcat_moment_table` for many molecules, as
a :class:`MomentTable` whose rows name their molecule and conformer. The USRCAT
functions call :mod:`usr.chem` to type the atoms of a molecule. :mod:`usr.io` reads
and writes the table, and :mod:`usr.similarity` scores it.
"""

import functools
import operator
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from loguru import logger
from rdkit import Chem

from usr import chem


type FloatArray = npt.NDArray[np.float64]
"""A NumPy array of 64-bit floats."""

NUM_MOMENTS = 12
"""The moments per conformer: three for each of the four reference points."""

MOMENT_COLUMNS = tuple(f"M{number}" for number in range(1, NUM_MOMENTS + 1))
"""The names of the moments, ``M1`` to ``M12``, in the order they are computed."""

NUM_USRCAT_MOMENTS = NUM_MOMENTS * (1 + len(chem.USRCAT_SMARTS))
"""The USRCAT moments per conformer: 12 for all atoms and 12 for each atom type."""


@dataclass(frozen=True, slots=True, eq=False)
class MomentTable:
    """USR or USRCAT moments of many conformers, one row each.

    Row ``i`` holds the moments of conformer ``conformers[i]`` of the molecule named
    ``titles[i]``. A molecule with several conformers has several rows.

    Parameters
    ----------
    titles : tuple[str, ...]
        The name of the molecule of each row.
    conformers : tuple[int, ...]
        The RDKit conformer id of each row.
    moments : FloatArray
        The moments, in the shape ``(rows, n)``, where ``n`` is a positive multiple of
        :data:`NUM_MOMENTS`: 12 for USR, :data:`NUM_USRCAT_MOMENTS` for USRCAT.

    Raises
    ------
    ValueError
        If ``moments`` is not two-dimensional with a positive multiple of
        :data:`NUM_MOMENTS` columns, or if ``titles``, ``conformers`` and ``moments``
        do not have the same number of rows.

    Examples
    --------
    >>> table = MomentTable(("benzene",), (0,), np.zeros((1, NUM_MOMENTS)))
    >>> len(table)
    1
    """

    titles: tuple[str, ...]
    conformers: tuple[int, ...]
    moments: FloatArray

    def __post_init__(self) -> None:
        """Check that the three columns describe the same rows."""
        shape = self.moments.shape
        if len(shape) != 2 or not shape[1] or shape[1] % NUM_MOMENTS:
            raise ValueError(
                f"moments must have the shape (rows, n) for n a positive multiple of "
                f"{NUM_MOMENTS}, not {shape}"
            )
        rows = shape[0]
        if not len(self.titles) == len(self.conformers) == rows:
            raise ValueError(
                f"a moment table needs one title and one conformer per row: "
                f"{len(self.titles)} titles, {len(self.conformers)} conformers, "
                f"{rows} rows of moments"
            )

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.titles)


def _find_reference_points(coords: FloatArray) -> FloatArray:
    """Return ``ctd``, ``cst``, ``fct`` and ``ftf`` in the shape ``(conformers, 4, 3)``.

    A tie goes to the atom with the lower index, as ``argmin`` and ``argmax`` break it.
    """
    conformers = np.arange(coords.shape[0])
    ctd = coords.mean(axis=1)
    dist_ctd = np.linalg.norm(coords - ctd[:, None, :], axis=2)
    cst = coords[conformers, dist_ctd.argmin(axis=1)]
    fct = coords[conformers, dist_ctd.argmax(axis=1)]
    dist_fct = np.linalg.norm(coords - fct[:, None, :], axis=2)
    ftf = coords[conformers, dist_fct.argmax(axis=1)]

    return np.stack([ctd, cst, fct, ftf], axis=1)


def _calc_distances(coords: FloatArray, points: FloatArray) -> FloatArray:
    """Return the distance of every atom to every point: ``(conformers, 4, atoms)``."""
    return np.linalg.norm(coords[:, None, :, :] - points[:, :, None, :], axis=3)


def _calc_central_moments(
    dist: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, npt.NDArray[np.bool_]]:
    """Return the mean, second and third central moments of distances over the atoms.

    Each is in the shape ``(conformers, 4)``, as is the last, which is True where the
    variance exceeds the floating-point resolution of the mean. scipy.stats.skew
    applies the same threshold.
    """
    mean = dist.mean(axis=2)
    deviations = dist - mean[:, :, None]
    second = (deviations**2).mean(axis=2)
    third = (deviations**3).mean(axis=2)
    spread = second > (np.finfo(np.float64).eps * mean) ** 2

    return mean, second, third, spread


def _assemble_block(
    mean: FloatArray, std: FloatArray, skewness: FloatArray
) -> FloatArray:
    """Interleave ``(conformers, 4)`` moments point by point: ``(conformers, 12)``."""
    return np.stack([mean, std, np.cbrt(skewness)], axis=2).reshape(-1, NUM_MOMENTS)


def _calc_shape_block(dist: FloatArray) -> FloatArray:
    """Return the USR moments of distances: a skewness without spread is NaN."""
    mean, second, third, spread = _calc_central_moments(dist)
    skewness = np.divide(
        third, second**1.5, out=np.full_like(second, np.nan), where=spread
    )

    return _assemble_block(mean, np.sqrt(second), skewness)


def _calc_feature_block(dist: FloatArray) -> FloatArray:
    """Return the USRCAT moments of distances: std and skewness without spread are 0."""
    mean, second, third, spread = _calc_central_moments(dist)
    std = np.where(spread, np.sqrt(second), 0.0)
    skewness = np.divide(third, second**1.5, out=np.zeros_like(second), where=spread)

    return _assemble_block(mean, std, skewness)


def _check_coords(coords: FloatArray, descriptor: str) -> None:
    """Raise ValueError unless ``coords`` is ``(conformers, atoms, 3)`` with atoms."""
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(
            f"coords must have the shape (conformers, atoms, 3), not {coords.shape}"
        )
    if not coords.shape[1]:
        raise ValueError(f"cannot calculate {descriptor} moments of zero atoms")


def calc_usr_moments(coords: FloatArray) -> FloatArray:
    """Return the USR moments of each conformer in a coordinate array.

    The calculation has three steps. The reference points come first: ``ctd`` from
    the coordinates, ``cst`` and ``fct`` from the distances to ``ctd``, and ``ftf``
    from the distances to ``fct``. Next, the distances of every atom to all four
    points. Last, the mean, standard deviation and cube root of the skewness of each
    distribution of distances. Where two atoms are equally close or far, the reference
    point is the one with the lower index.

    Parameters
    ----------
    coords : FloatArray
        Coordinates in the shape ``(conformers, atoms, 3)``.

    Returns
    -------
    FloatArray
        The moments in the shape ``(conformers, NUM_MOMENTS)``: for ``ctd``, ``cst``,
        ``fct`` and ``ftf`` in turn, the mean, the population standard deviation and
        the cube root of the population skewness of the distances. A skewness is NaN
        where the variance is zero to within the floating-point resolution of the
        mean, as for a single atom.

    Raises
    ------
    ValueError
        If ``coords`` is not in the shape ``(conformers, atoms, 3)``, or holds no atoms.

    Examples
    --------
    One conformer of four atoms on a line, one unit apart:

    >>> line = np.array([[[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]])
    >>> moments = calc_usr_moments(line)
    >>> moments.shape
    (1, 12)
    >>> print(moments[0, :3])
    [1.  0.5 0. ]
    """
    _check_coords(coords, "USR")

    return _calc_shape_block(_calc_distances(coords, _find_reference_points(coords)))


def _convert_subsets(
    subsets: Sequence[Sequence[int]], atoms: int
) -> list[npt.NDArray[np.intp]]:
    """Return each subset as an index array, refusing booleans, strays and repeats."""
    indices: list[npt.NDArray[np.intp]] = []
    for number, subset in enumerate(subsets):
        flags = [index for index in subset if isinstance(index, bool | np.bool_)]
        if flags:
            raise ValueError(
                f"subsets[{number}] holds the boolean {bool(flags[0])}, not an atom index"
            )
        members = np.array([operator.index(index) for index in subset], dtype=np.intp)
        outside = members[(members < 0) | (members >= atoms)]
        if outside.size:
            raise ValueError(
                f"subsets[{number}] holds the atom index {outside[0]}, outside 0 to "
                f"{atoms - 1}"
            )
        unique, counts = np.unique(members, return_counts=True)
        if (counts > 1).any():
            raise ValueError(
                f"subsets[{number}] holds the atom index {unique[counts > 1][0]} "
                f"more than once"
            )
        indices.append(members)

    return indices


def calc_usrcat_moments(
    coords: FloatArray, subsets: Sequence[Sequence[int]]
) -> FloatArray:
    """Return the USRCAT moments of each conformer in a coordinate array.

    The first block is the USR moments of all atoms, as :func:`calc_usr_moments`
    computes them. Each subset adds a block of the same moments of the distances of
    its atoms to the reference points of all atoms. A subset without spread, as one
    atom has, has the mean distance to each point, a standard deviation of 0 and a
    skewness of 0. An empty subset has 12 moments of 0.

    Parameters
    ----------
    coords : FloatArray
        Coordinates in the shape ``(conformers, atoms, 3)``.
    subsets : Sequence[Sequence[int]]
        For each block after the first, the 0-based indices of its atoms, each at most
        once. A subset may be empty.

    Returns
    -------
    FloatArray
        The moments in the shape ``(conformers, NUM_MOMENTS * (1 + len(subsets)))``.

    Raises
    ------
    ValueError
        If ``coords`` is not in the shape ``(conformers, atoms, 3)`` or holds no atoms,
        or a subset holds a boolean, an index outside ``0`` to ``atoms - 1`` or one
        index twice.

    Examples
    --------
    Four atoms on a line, one unit apart: ``cst`` is the second atom, ``fct`` the
    first and ``ftf`` the last. The first atom alone is 1.5, 1, 0 and 3 from them:

    >>> line = np.array([[[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]])
    >>> moments = calc_usrcat_moments(line, [[0], []])
    >>> moments.shape
    (1, 36)
    >>> print(moments[0, 12:24])
    [1.5 0.  0.  1.  0.  0.  0.  0.  0.  3.  0.  0. ]
    """
    _check_coords(coords, "USRCAT")
    indices = _convert_subsets(subsets, coords.shape[1])

    points = _find_reference_points(coords)
    blocks = [_calc_shape_block(_calc_distances(coords, points))]
    for members in indices:
        if members.size:
            blocks.append(
                _calc_feature_block(_calc_distances(coords[:, members], points))
            )
        else:
            blocks.append(np.zeros((coords.shape[0], NUM_MOMENTS)))

    return np.hstack(blocks)


def _get_coordinates(mol: Chem.Mol) -> FloatArray:
    """Return the coordinates of every conformer: ``(conformers, atoms, 3)``."""
    return np.array(
        [conformer.GetPositions() for conformer in mol.GetConformers()],
        dtype=np.float64,
    ).reshape(mol.GetNumConformers(), mol.GetNumAtoms(), 3)


def get_usr_moments(mol: Chem.Mol, *, explicit_h: bool = False) -> FloatArray:
    """Return the USR moments of every conformer of ``mol``.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule. It is not modified.
    explicit_h : bool, optional
        Whether the coordinates of explicit hydrogen atoms count. Default is False,
        which removes them first.

    Returns
    -------
    FloatArray
        One row of :data:`NUM_MOMENTS` moments per conformer, in conformer order, in
        the shape ``(conformers, NUM_MOMENTS)``. A molecule without conformers gives
        zero rows.

    Raises
    ------
    ValueError
        If the molecule has conformers but no atoms.

    Examples
    --------
    >>> from rdkit.Chem import rdDistGeom
    >>> mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    >>> conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42)
    >>> get_usr_moments(mol).shape
    (3, 12)
    """
    if not explicit_h:
        mol = Chem.RemoveHs(mol)
    if not mol.GetNumConformers():
        return np.empty((0, NUM_MOMENTS))

    return calc_usr_moments(_get_coordinates(mol))


def get_usr_moment_table(
    mols: Iterable[Chem.Mol], *, explicit_h: bool = False, title_tag: str = "_Name"
) -> MomentTable:
    """Return the USR moments of every conformer of every molecule.

    Parameters
    ----------
    mols : Iterable[Chem.Mol]
        The molecules. Each is read once and is not modified.
    explicit_h : bool, optional
        Whether the coordinates of explicit hydrogen atoms count. Default is False.
    title_tag : str, optional
        The property that holds the name of a molecule. Default is ``_Name``, the
        title line of an SD record.

    Returns
    -------
    MomentTable
        One row per conformer. A molecule whose title is missing or empty is named
        ``Mol-<n>``, where ``n`` counts the molecules given, from 1. A molecule without
        atoms or without conformers adds no row, and a warning names it.

    Examples
    --------
    >>> from rdkit.Chem import rdDistGeom
    >>> mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    >>> conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42)
    >>> table = get_usr_moment_table([mol])
    >>> table.titles, table.conformers
    (('Mol-1', 'Mol-1', 'Mol-1'), (0, 1, 2))
    """
    return _build_moment_table(
        mols,
        title_tag,
        "USR",
        NUM_MOMENTS,
        functools.partial(get_usr_moments, explicit_h=explicit_h),
    )


def _build_moment_table(
    mols: Iterable[Chem.Mol],
    title_tag: str,
    descriptor: str,
    num_moments: int,
    calc_moments: Callable[[Chem.Mol], FloatArray],
) -> MomentTable:
    """Return a table of ``calc_moments`` of every conformer of every molecule.

    A molecule without atoms or without conformers adds no row, and a warning names it.
    An empty table has ``num_moments`` columns. ``descriptor`` names the moments in the
    log.
    """
    titles: list[str] = []
    conformers: list[int] = []
    rows: list[FloatArray] = []

    for number, mol in enumerate(mols, 1):
        title = _get_title(mol, title_tag, number)
        if not mol.GetNumAtoms():
            logger.warning("skipped molecule {} ({}): it has no atoms", number, title)
            continue
        if not mol.GetNumConformers():
            logger.warning(
                "skipped molecule {} ({}): it has no conformers", number, title
            )
            continue
        moments = calc_moments(mol)
        titles.extend([title] * len(moments))
        conformers.extend(conformer.GetId() for conformer in mol.GetConformers())
        rows.append(moments)

    table = MomentTable(
        tuple(titles),
        tuple(conformers),
        np.vstack(rows) if rows else np.empty((0, num_moments)),
    )
    logger.info("calculated {} moments for {} conformers", descriptor, len(table))

    return table


def _get_title(mol: Chem.Mol, title_tag: str, number: int) -> str:
    """Return the title of ``mol`` from ``title_tag``, or ``Mol-<number>``."""
    title = mol.GetProp(title_tag) if mol.HasProp(title_tag) else ""
    return title or f"Mol-{number}"


def get_usrcat_moments(
    mol: Chem.Mol,
    *,
    explicit_h: bool = False,
    subsets: Sequence[Sequence[int]] | None = None,
    smarts: Sequence[str] | None = None,
) -> FloatArray:
    """Return the USRCAT moments of every conformer of ``mol``.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule. It is not modified.
    explicit_h : bool, optional
        Whether the coordinates of explicit hydrogen atoms count. Default is False,
        which removes them first. Atom types ignore explicit hydrogen atoms either way.
    subsets : Sequence[Sequence[int]] | None, optional
        The atoms of each block after the first, as :func:`calc_usrcat_moments` takes
        them, indexed as :func:`usr.chem.get_pharmacophore_subsets` indexes them.
        Default is None, which types the atoms with ``smarts``.
    smarts : Sequence[str] | None, optional
        One SMARTS per atom type. Default is None, the values of
        :data:`usr.chem.USRCAT_SMARTS`.

    Returns
    -------
    FloatArray
        One row per conformer, in conformer order, in the shape
        ``(conformers, NUM_MOMENTS * (1 + types))``: :data:`NUM_USRCAT_MOMENTS`
        columns by default. A molecule without conformers gives zero rows.

    Raises
    ------
    ValueError
        If both ``subsets`` and ``smarts`` are given, if a SMARTS is empty or invalid,
        if a subset is invalid for the atoms of ``mol``, even without conformers, or if
        the molecule has conformers but no atoms.

    Examples
    --------
    >>> from rdkit.Chem import rdDistGeom
    >>> mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    >>> conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42)
    >>> get_usrcat_moments(mol).shape
    (3, 60)
    """
    if subsets is not None and smarts is not None:
        raise ValueError("pass subsets or smarts, not both")

    target = mol if explicit_h else Chem.RemoveHs(mol)
    if subsets is None:
        subsets = chem.get_pharmacophore_subsets(
            target, explicit_h=explicit_h, smarts=smarts
        )
    else:
        _convert_subsets(subsets, target.GetNumAtoms())
    if not target.GetNumConformers():
        return np.empty((0, NUM_MOMENTS * (1 + len(subsets))))

    return calc_usrcat_moments(_get_coordinates(target), subsets)


def get_usrcat_moment_table(
    mols: Iterable[Chem.Mol],
    *,
    explicit_h: bool = False,
    title_tag: str = "_Name",
    smarts: Sequence[str] | None = None,
) -> MomentTable:
    """Return the USRCAT moments of every conformer of every molecule.

    Parameters
    ----------
    mols : Iterable[Chem.Mol]
        The molecules. Each is read once and is not modified.
    explicit_h : bool, optional
        Whether the coordinates of explicit hydrogen atoms count. Default is False.
    title_tag : str, optional
        The property that holds the name of a molecule. Default is ``_Name``, the
        title line of an SD record.
    smarts : Sequence[str] | None, optional
        One SMARTS per atom type. Default is None, the values of
        :data:`usr.chem.USRCAT_SMARTS`.

    Returns
    -------
    MomentTable
        One row per conformer, with :data:`NUM_USRCAT_MOMENTS` columns by default.
        Molecules are named, and skipped with a warning, as
        :func:`get_usr_moment_table` names and skips them.

    Raises
    ------
    ValueError
        If a SMARTS is empty or invalid. Raised before any molecule is read.

    Examples
    --------
    >>> from rdkit.Chem import rdDistGeom
    >>> mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    >>> conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42)
    >>> table = get_usrcat_moment_table([mol])
    >>> table.conformers, table.moments.shape
    ((0, 1, 2), (3, 60))
    """
    # Checked before any molecule is read; each SMARTS is then compiled once, cached.
    num_types = len(chem.compile_smarts(smarts))

    return _build_moment_table(
        mols,
        title_tag,
        "USRCAT",
        NUM_MOMENTS * (1 + num_types),
        functools.partial(get_usrcat_moments, explicit_h=explicit_h, smarts=smarts),
    )
