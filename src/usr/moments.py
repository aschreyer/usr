"""USR moments: the shape descriptor of Ultrafast Shape Recognition.

USR (Ballester and Richards, 2007) describes the shape of one conformer by the
distributions of its atomic distances to four reference points:

- ``ctd``, the centroid of the atoms;
- ``cst``, the atom closest to ``ctd``;
- ``fct``, the atom farthest from ``ctd``;
- ``ftf``, the atom farthest from ``fct``.

Each distribution contributes its mean, its standard deviation and the cube root of its
skewness, in that order, so a conformer has :data:`NUM_MOMENTS` moments.

:func:`calc_usr_moments` computes the moments of an array of conformer coordinates.
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
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError(
            f"coords must have the shape (conformers, atoms, 3), not {coords.shape}"
        )
    if not coords.shape[1]:
        raise ValueError("cannot calculate USR moments of zero atoms")
    conformers = np.arange(coords.shape[0])

    # The reference points, each (conformers, 3).
    ctd = coords.mean(axis=1)
    dist_ctd = np.sqrt(((coords - ctd[:, None, :]) ** 2).sum(axis=2))
    cst = coords[conformers, dist_ctd.argmin(axis=1)]
    fct = coords[conformers, dist_ctd.argmax(axis=1)]
    dist_fct = np.sqrt(((coords - fct[:, None, :]) ** 2).sum(axis=2))
    ftf = coords[conformers, dist_fct.argmax(axis=1)]

    # The distances of every atom to every point: (conformers, 4, atoms).
    points = np.stack([ctd, cst, fct, ftf], axis=1)
    dist = np.sqrt(((coords[:, None, :, :] - points[:, :, None, :]) ** 2).sum(axis=3))

    # The population moments over the atoms, each (conformers, 4).
    mean = dist.mean(axis=2)
    deviations = dist - mean[:, :, None]
    second = (deviations**2).mean(axis=2)
    third = (deviations**3).mean(axis=2)
    # A variance at the rounding error of the mean is zero: the skewness is NaN.
    # scipy.stats.skew applies the same threshold.
    defined = second > (np.finfo(np.float64).eps * mean) ** 2
    skewness = np.divide(
        third, second**1.5, out=np.full_like(second, np.nan), where=defined
    )

    # (conformers, 4, 3) -> (conformers, 12), point by point.
    moments = np.stack([mean, np.sqrt(second), np.cbrt(skewness)], axis=2)

    return moments.reshape(-1, NUM_MOMENTS)


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

    coords = np.array(
        [conformer.GetPositions() for conformer in mol.GetConformers()],
        dtype=np.float64,
    )

    return calc_usr_moments(coords)


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
    titles: list[str] = []
    conformers: list[int] = []
    rows: list[FloatArray] = []

    for number, mol in enumerate(mols, 1):
        title = mol.GetProp(title_tag) if mol.HasProp(title_tag) else ""
        title = title or f"Mol-{number}"
        if not mol.GetNumAtoms():
            logger.warning("skipped molecule {} ({}): it has no atoms", number, title)
            continue
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
