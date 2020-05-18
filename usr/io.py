"""

"""
import gzip
from enum import Enum
from typing import Iterator, Union
from pathlib import Path

from rdkit import Chem
from loguru import logger


class MomentFormat(str, Enum):
    """
    Supported input/output file formats for USR moments.
    """
    CSV = "CSV"


def is_gzipped(file: Union[str, Path]) -> bool:
    """
    Returns True if the file `file` is gzipped, i.e. the file ends with the `.gz`
    suffix.

    Parameters
    ----------
    file: Path


    Returns
    -------
    bool

    Examples
    --------
    >>> is_gzipped("/foo/bar/mols.sdf")
    False
    >>> is_gzipped("/foo/bar/mols.sdf.gz")
    True
    """
    file = Path(file) if isinstance(file, str) else file

    return file.suffix == ".gz"


def is_supported_format(file: Union[str, Path]) -> bool:
    """
    Returns true if the file `file` is in a supported file format based on its
    extensions.

    Parameters
    ----------
    file : Path

    Returns
    -------
    bool

    Examples
    --------
    >>> is_supported_format("mols.sdf")
    True
    >>> is_supported_format("mols.sdf.gz")
    True
    >>> is_supported_format("mols.gz")
    False
    """
    file = Path(file) if isinstance(file, str) else file

    supported_formats = {".csv", ".mol", ".sdf", ".smi", ".tsv", ".txt"}

    suffixes = file.suffixes[-2:]
    suffix = suffixes[0] if suffixes[-1] == ".gz" else suffixes[-1]

    return suffix in supported_formats


def get_mols_from_sdf_file(
        sdf_file: Union[str, Path], *,
        sanitize: bool = True, suppress_hydrogens: bool = True
) -> Iterator[Chem.rdchem.Mol]:
    """
    Yields the molecules contained in the SDF file `sdf_file`.

    Parameters
    ----------
    sdf_file : Path

    sanitize : bool = True

    suppress_hydrogens : bool = True


    Yields
    ------
    Chem.rdchem.Mol
    """
    path = Path(sdf_file) if isinstance(sdf_file, str) else sdf_file

    if not path.is_file():
        raise ValueError(
            f"cannot read molecules from {path.name}: file is not valid or does not "
            f"exist."
        )

    if not is_supported_format(path):
        raise ValueError(
            f"cannot read molecules from {path.name}: unsupported file format."
        )

    logger.debug("loading chemical structure files from {}.", path.resolve())

    if is_gzipped(sdf_file):
        fh = gzip.open(sdf_file)
        supplier = Chem.ForwardSDMolSupplier(
            fh, sanitize=sanitize, removeHs=suppress_hydrogens
        )

    else:
        supplier = Chem.SDMolSupplier(
            str(sdf_file), sanitize=sanitize, removeHs=suppress_hydrogens
        )

    yield from supplier
