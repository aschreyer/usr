"""
This module contains the functions necessary for calculating USR moments.
"""
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from loguru import logger
from scipy.special import cbrt
from scipy.stats.stats import skew


def calc_dist_to_point(coords: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Returns an array containing the distances of each coordinate in the input
    coordinates to the input point.

    Parameters
    ----------
    coords : np.ndarray
        NumPy array containing Euclidean coordinates.
    point : np.ndarray
        Single coordinate having the same dimensions as the coordinates.

    Returns
    -------
    np.ndarray

    Examples
    --------
    >>> coords = np.array([[8.38667524, 1.13009736, -0.0745602],
    ...                    [7.82163887, 1.77554511, -1.323699],
    ...                    [6.38320013, 1.76070059, -1.20237391]])
    >>> for dist in calc_dist_to_point(coords, np.zeros(3)):
    ...     print(np.round(dist, 3))
    8.463
    8.129
    6.73
    """
    return np.sqrt(((coords-point)**2).sum(axis=1))


def calc_usr_moments_from_coords(coords: np.ndarray) -> np.ndarray:
    """
    Returns a NumPy array containing the USR moments calculated from the coordinates
    `coords`.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates to use for calculating the USR moments.

    Returns
    -------
    moments : np.ndarray
        NumPy array containing the 12 USR moments.
    """
    # centroid of the input coordinates
    ctd = coords.mean(axis=0)

    # get the distances to the centroid
    dist_ctd = calc_dist_to_point(coords, ctd)

    # get the closest and furthest coordinate to/from the centroid
    cst, fct = coords[dist_ctd.argmin()], coords[dist_ctd.argmax()]

    # get the distance distributions for the points that are closest/furthest to/from
    # the centroid
    dist_cst = calc_dist_to_point(coords, cst)
    dist_fct = calc_dist_to_point(coords, fct)

    # get the point that is the furthest from the point that is furthest from the
    # centroid
    ftf = coords[dist_fct.argmax()]
    dist_ftf = calc_dist_to_point(coords, ftf)

    moments = [
        dist_ctd.mean(), dist_ctd.std(), cbrt(skew(dist_ctd)),
        dist_cst.mean(), dist_cst.std(), cbrt(skew(dist_cst)),
        dist_fct.mean(), dist_fct.std(), cbrt(skew(dist_fct)),
        dist_ftf.mean(), dist_ftf.std(), cbrt(skew(dist_ftf))
    ]

    return np.array(moments)


def get_usr_moments(mol: Chem.rdchem.Mol, *, explicit_h: bool = False) -> np.ndarray:
    """
    Returns a NumPy array in the shape (number of conformers, 12) containing the
    calculated USR moments of each conformer.

    Parameters
    ----------
    mol: Chem.rdchem.Mol
        Molecule to calculate USR moments of.
    explicit_h: bool = False
        If True, coordinates of hydrogen atoms will be used in the calculation.

    Returns
    -------
    usr_moments : np.ndarray
        NumPy array in the shape (number of conformers, 12).

    Examples
    --------
    >>> from rdkit.Chem import AllChem
    >>> mol = Chem.MolFromSmiles("CCNC(=O)N1CCC(CC1)Nc2ncc(Cl)c(n2)c3c[nH]c4ccccc34")
    >>> conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=10)
    >>> moments = get_usr_moments(mol)
    >>> moments.shape
    (10, 12)
    """
    if not explicit_h:
        Chem.RemoveHs(mol)

    moments = []
    for conformer in mol.GetConformers():
        coords = conformer.GetPositions()
        conf_moments = calc_usr_moments_from_coords(coords)

        moments.append(conf_moments)

    return np.vstack(moments)


def get_usr_moments_df(
        mols: Iterable[Chem.rdchem.Mol], *,
        explicit_h: bool = False, sd_title_tag: str = "_Name"
) -> pd.DataFrame:
    """
    Returns a Pandas data frame containing the generated USR moments for all the
    molecules `mols`.

    Parameters
    ----------
    mols : Iterable[Chem.rdchem.Mol]
        Molecules to generate USR moments for.
    explicit_h : bool = False
        If True, coordinates of hydrogen atoms will be used in the calculation.
    sd_title_tag : str = "_Name"
        Name of the SD tag containing the title of the molecule. Defaults to `_Name`,
        which is the special SD title field.

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> from rdkit.Chem import AllChem
    >>> mol = Chem.MolFromSmiles("CCNC(=O)N1CCC(CC1)Nc2ncc(Cl)c(n2)c3c[nH]c4ccccc34")
    >>> conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=10)
    >>> df = get_usr_moments_df([mol,])
    >>> df.shape
    (10, 12)
    >>> df.columns.tolist()
    ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12']
    """
    data = []

    for i, mol in enumerate(mols, 1):
        title = mol.GetProp(sd_title_tag) if mol.HasProp(sd_title_tag) else f"Mol-{i}"

        if not explicit_h:
            Chem.RemoveHs(mol)

            for conformer in mol.GetConformers():
                coords = conformer.GetPositions()
                moments = calc_usr_moments_from_coords(coords)

                data.append([title, conformer.GetId(), *moments])

    df = pd.DataFrame.from_records(
        data,
        columns=["Title", "Conformer"] + [f"M{i}" for i in range(1, 13)],
        index=["Title", "Conformer"]
    )

    logger.info("created data frame with {} sets of USR moments.", df.shape[0])

    return df
