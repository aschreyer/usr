"""Fixtures shared across the test suite."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger
from rdkit import Chem
from rdkit.Chem import rdDistGeom


# Typer and loguru colour their output when one of these is set, and GitHub Actions sets
# the first two. The tests compare that output as plain text. Removed at import, before
# Typer first reads them.
for name in ("CI", "GITHUB_ACTIONS", "FORCE_COLOR", "PY_COLORS"):
    os.environ.pop(name, None)


DATA_DIR = Path(__file__).parent / "data"
"""The SD files the tests read."""


@pytest.fixture(scope="session")
def chembl_sdf() -> Path:
    """Return 983 oral drugs from ChEMBL 26: one record each, with 2D coordinates.

    The z coordinate is zero throughout, so the moments describe a flat drawing, not a
    shape. The file is large enough to exercise a realistic database.
    """
    return DATA_DIR / "chembl26-oral-approved.sdf"


@pytest.fixture(scope="session")
def drugs_sdf() -> Path:
    """Return six small drugs with 3D coordinates and explicit hydrogen atoms.

    Aspirin, caffeine, ibuprofen, paracetamol, nicotine and diazepam, one conformer
    each. Made with RDKit 2026.9: ``Chem.AddHs`` on the SMILES, ``EmbedMolecule`` with
    ``ETKDGv3`` and ``randomSeed = 42``, then ``MMFFOptimizeMolecule``, written with
    ``Chem.SDWriter`` under the drug's name.
    """
    return DATA_DIR / "drugs-3d.sdf"


UNREADABLE_RECORD = """broken
  hand-written

  6  5  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.8000    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
   -1.8000    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.8000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -1.8000    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    1.8000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
  1  6  1  0
M  END
"""
"""An SD record RDKit refuses: carbon with five bonds fails sanitising."""


@pytest.fixture
def sdf_with_a_bad_record(tmp_path: Path, drugs_sdf: Path) -> Path:
    """Return the six 3D drugs with :data:`UNREADABLE_RECORD` second."""
    records = drugs_sdf.read_text().split("$$$$\n")
    path = tmp_path / "with-bad-record.sdf"
    records.insert(1, UNREADABLE_RECORD)
    path.write_text("$$$$\n".join(records))
    return path


@pytest.fixture(scope="session")
def drug_names() -> list[str]:
    """Return the titles of the records in ``drugs_sdf``, in file order."""
    return ["aspirin", "caffeine", "ibuprofen", "paracetamol", "nicotine", "diazepam"]


@pytest.fixture(scope="session")
def drug_mols(drugs_sdf: Path) -> list[Chem.Mol]:
    """Return the molecules of ``drugs_sdf``, hydrogen atoms kept, read by RDKit alone.

    Tests must not modify them: the session shares them.
    """
    return list(Chem.SDMolSupplier(str(drugs_sdf), removeHs=False))


@pytest.fixture(scope="session")
def aspirin_conformers() -> Chem.Mol:
    """Return aspirin with explicit hydrogen atoms and three seeded 3D conformers.

    Tests must not modify it: the session shares it.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    mol.SetProp("_Name", "aspirin")
    rdDistGeom.EmbedMultipleConfs(mol, numConfs=3, randomSeed=42)
    return mol


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Restore the library's silent logging after each test.

    The CLI configures loguru for its process, and ``CliRunner`` runs it in-process, so
    the suite removes the CLI's sink and disables the package again after each test.
    """
    yield
    logger.remove()
    logger.disable("usr")


@pytest.fixture
def caplog_loguru() -> list[str]:
    """Collect this test's ``WARNING`` and worse from loguru, as plain strings.

    Enables the package first, because ``usr`` disables its own logging on import.
    ``caplog`` collects nothing here, because loguru does not route through
    :mod:`logging`. ``_restore_logging`` removes the sink after the test.
    """
    messages: list[str] = []
    logger.add(
        lambda message: messages.append(message.record["message"]), level="WARNING"
    )
    logger.enable("usr")
    return messages
