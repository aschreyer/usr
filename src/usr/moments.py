"""USR moments: the shape descriptor of Ultrafast Shape Recognition.

USR (Ballester and Richards, 2007) describes the shape of one conformer by the
distributions of its atomic distances to four reference points:

- ``ctd``, the centroid of the atoms;
- ``cst``, the atom closest to ``ctd``;
- ``fct``, the atom farthest from ``ctd``;
- ``ftf``, the atom farthest from ``fct``.

Each distribution contributes its mean, its standard deviation and the cube root of its
skewness, in that order, so a conformer has :data:`NUM_MOMENTS` moments.

:func:`calc_usr_moments_from_coords` computes the moments of one coordinate array.
:func:`get_usr_moments` computes them for every conformer of an RDKit molecule, and
:func:`get_usr_moment_table` for many molecules, as a :class:`MomentTable` whose rows
name their molecule and conformer. :mod:`usr.io` reads and writes the table, and
:mod:`usr.similarity` scores it.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from loguru import logger
from rdkit import Chem
from scipy.special import cbrt


type FloatArray = npt.NDArray[np.float64]
"""A NumPy array of 64-bit floats."""

NUM_MOMENTS = 12
"""The moments per conformer: three for each of the four reference points."""

MOMENT_COLUMNS = tuple(f"M{number}" for number in range(1, NUM_MOMENTS + 1))
"""The names of the moments, ``M1`` to ``M12``, in the order they are computed."""


@dataclass(frozen=True, slots=True, eq=False)
class MomentTable:
    """USR moments of many conformers, one row each.

    Row ``i`` holds the moments of conformer ``conformers[i]`` of the molecule named
    ``titles[i]``. A molecule with several conformers has several rows.

    Parameters
    ----------
    titles : tuple[str, ...]
        The name of the molecule of each row.
    conformers : tuple[int, ...]
        The RDKit conformer id of each row.
    moments : FloatArray
        The moments, in the shape ``(rows, NUM_MOMENTS)``.

    Raises
    ------
    ValueError
        If ``moments`` is not two-dimensional with :data:`NUM_MOMENTS` columns, or if
        ``titles``, ``conformers`` and ``moments`` do not have the same number of rows.

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
        if self.moments.ndim != 2 or self.moments.shape[1] != NUM_MOMENTS:
            raise ValueError(
                f"moments must have the shape (rows, {NUM_MOMENTS}), "
                f"not {self.moments.shape}"
            )
        rows = self.moments.shape[0]
        if not len(self.titles) == len(self.conformers) == rows:
            raise ValueError(
                f"a moment table needs one title and one conformer per row: "
                f"{len(self.titles)} titles, {len(self.conformers)} conformers, "
                f"{rows} rows of moments"
            )

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self.titles)


def calc_dist_to_point(coords: FloatArray, point: FloatArray) -> FloatArray:
    """Return the Euclidean distance of each coordinate to ``point``.

    Parameters
    ----------
    coords : FloatArray
        Coordinates in the shape ``(atoms, dimensions)``.
    point : FloatArray
        One coordinate with the same number of dimensions.

    Returns
    -------
    FloatArray
        The distances, in the shape ``(atoms,)``.

    Examples
    --------
    >>> coords = np.array(
    ...     [
    ...         [8.38667524, 1.13009736, -0.0745602],
    ...         [7.82163887, 1.77554511, -1.323699],
    ...         [6.38320013, 1.76070059, -1.20237391],
    ...     ]
    ... )
    >>> for dist in calc_dist_to_point(coords, np.zeros(3)):
    ...     print(np.round(dist, 3))
    8.463
    8.129
    6.73
    """
    return np.sqrt(((coords - point) ** 2).sum(axis=1))


def calc_skewness(values: FloatArray) -> float:
    """Return the skewness of ``values``.

    The skewness is the third central moment over the 1.5th power of the second. The
    moments are population moments, without a bias correction. This is the
    default of ``scipy.stats.skew``, and the result agrees with it bit for bit.

    Parameters
    ----------
    values : FloatArray
        A one-dimensional sample.

    Returns
    -------
    float
        The skewness, or NaN where the variance is zero to within the floating-point
        resolution of the mean, as for a single value or equal values.

    Examples
    --------
    >>> calc_skewness(np.array([0.0, 0.0, 3.0]))
    0.7071067811865475
    >>> calc_skewness(np.array([2.0, 2.0]))
    nan
    """
    mean = values.mean()
    deviations = values - mean
    second = np.mean(deviations**2)
    third = np.mean(deviations**3)
    # A variance at the rounding error of the mean is zero, and the skewness is
    # undefined. The threshold is the one scipy.stats.skew applies.
    if second <= (np.finfo(np.float64).eps * mean) ** 2:
        return float("nan")

    return float(third / second**1.5)


def calc_usr_moments_from_coords(coords: FloatArray) -> FloatArray:
    """Return the USR moments of one set of atomic coordinates.

    Parameters
    ----------
    coords : FloatArray
        Coordinates in the shape ``(atoms, 3)``.

    Returns
    -------
    FloatArray
        The :data:`NUM_MOMENTS` moments: mean, standard deviation and cube root of
        the skewness of the distances to ``ctd``, ``cst``, ``fct`` and ``ftf``, in
        that order. A skewness that is undefined, as for fewer than three distinct
        distances, gives NaN.

    Examples
    --------
    Four atoms on a line, one unit apart:

    >>> moments = calc_usr_moments_from_coords(
    ...     np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    ... )
    >>> moments.shape
    (12,)
    >>> print(moments[:3])
    [1.  0.5 0. ]
    """
    ctd = coords.mean(axis=0)
    dist_ctd = calc_dist_to_point(coords, ctd)

    cst = coords[dist_ctd.argmin()]
    fct = coords[dist_ctd.argmax()]
    dist_cst = calc_dist_to_point(coords, cst)
    dist_fct = calc_dist_to_point(coords, fct)

    ftf = coords[dist_fct.argmax()]
    dist_ftf = calc_dist_to_point(coords, ftf)

    moments = [
        moment
        for dist in (dist_ctd, dist_cst, dist_fct, dist_ftf)
        for moment in (dist.mean(), dist.std(), cbrt(calc_skewness(dist)))
    ]

    return np.array(moments, dtype=np.float64)


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

    rows = [
        calc_usr_moments_from_coords(conformer.GetPositions())
        for conformer in mol.GetConformers()
    ]

    return np.array(rows, dtype=np.float64).reshape(len(rows), NUM_MOMENTS)


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
        ``Mol-<n>``, where ``n`` counts the molecules from 1. A molecule without
        conformers adds no row, and a warning names it.

    Examples
    --------
    >>> from rdkit.Chem import rdDistGeom
    >>> mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    >>> conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42)
    >>> table = get_usr_moment_table([mol])
    >>> table.titles, table.conformers
    (('Mol-1', 'Mol-1', 'Mol-1'), (0, 1, 2))
    """
    titles: list[str] = []
    conformers: list[int] = []
    rows: list[FloatArray] = []

    for number, mol in enumerate(mols, 1):
        title = mol.GetProp(title_tag) if mol.HasProp(title_tag) else ""
        title = title or f"Mol-{number}"
        moments = get_usr_moments(mol, explicit_h=explicit_h)
        if not len(moments):
            logger.warning(
                "skipped molecule {} ({}): it has no conformers", number, title
            )
            continue
        titles.extend([title] * len(moments))
        conformers.extend(conformer.GetId() for conformer in mol.GetConformers())
        rows.append(moments)

    table = MomentTable(
        tuple(titles),
        tuple(conformers),
        np.vstack(rows) if rows else np.empty((0, NUM_MOMENTS)),
    )
    logger.info("calculated USR moments for {} conformers", len(table))

    return table
