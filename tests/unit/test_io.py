"""Tests for reading SD files and reading and writing moments and hits as CSV."""

import gzip
import io as stdio
import math
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from usr import io
from usr.moments import NUM_MOMENTS, MomentTable, get_usr_moment_table
from usr.search import Hit


def test_the_molecules_of_an_sd_file_are_read_in_order(drugs_sdf: Path) -> None:
    mols = list(io.read_molecules(drugs_sdf))

    assert [mol.GetProp("_Name") for mol in mols][:2] == ["aspirin", "caffeine"]
    assert len(mols) == 6
    assert mols[0].GetNumAtoms() == 13


def test_hydrogen_atoms_are_kept_on_request(drugs_sdf: Path) -> None:
    assert next(io.read_molecules(drugs_sdf, remove_hs=False)).GetNumAtoms() == 21


def test_an_sd_file_name_is_matched_without_regard_to_case(
    drugs_sdf: Path, tmp_path: Path
) -> None:
    path = tmp_path / "DRUGS.SDF"
    path.write_bytes(drugs_sdf.read_bytes())

    assert len(list(io.read_molecules(path))) == 6


def test_a_gzipped_sd_file_reads_as_the_plain_one(
    drugs_sdf: Path, tmp_path: Path
) -> None:
    gzipped = tmp_path / "drugs.sdf.gz"
    gzipped.write_bytes(gzip.compress(drugs_sdf.read_bytes()))

    plain = [Chem.MolToMolBlock(mol) for mol in io.read_molecules(drugs_sdf)]

    assert [Chem.MolToMolBlock(mol) for mol in io.read_molecules(gzipped)] == plain


def test_an_unreadable_record_is_skipped_with_a_warning(
    sdf_with_a_bad_record: Path, caplog_loguru: list[str]
) -> None:
    names = [mol.GetProp("_Name") for mol in io.read_molecules(sdf_with_a_bad_record)]

    assert names == [
        "aspirin",
        "caffeine",
        "ibuprofen",
        "paracetamol",
        "nicotine",
        "diazepam",
    ]
    assert caplog_loguru == [
        f"skipped record 2 of {sdf_with_a_bad_record}: RDKit could not read it"
    ]


@pytest.mark.parametrize("name", ["mols.mol2", "mols.gz", "mols", "mols.sdf.csv"])
def test_a_file_that_is_not_an_sd_file_is_refused_before_reading(
    name: str, tmp_path: Path, drugs_sdf: Path
) -> None:
    """The file holds SD records, so only the name check can raise."""
    path = tmp_path / name
    path.write_bytes(drugs_sdf.read_bytes())

    with pytest.raises(ValueError, match="cannot read molecules") as caught:
        io.read_molecules(path)

    assert str(caught.value) == (
        f"cannot read molecules from {name}: only SD files are supported, "
        "named *.sdf or *.sdf.gz"
    )


def test_a_missing_file_is_refused_before_reading(tmp_path: Path) -> None:
    path = tmp_path / "missing.sdf"

    with pytest.raises(ValueError, match="cannot read molecules") as caught:
        io.read_molecules(path)

    assert str(caught.value) == f"cannot read molecules from {path}: no such file"


def test_a_moment_table_survives_a_round_trip_exactly(
    aspirin_conformers: Chem.Mol, tmp_path: Path
) -> None:
    table = get_usr_moment_table([aspirin_conformers])
    odd = MomentTable(
        ('Title, with "quotes"',),
        (7,),
        np.array([[math.nan, -0.0, 1e-300, *range(NUM_MOMENTS - 3)]]),
    )
    for original in (table, odd):
        path = tmp_path / "moments.csv"
        io.write_moment_table(original, path)

        copy = io.read_moment_table(path)

        assert copy.titles == original.titles
        assert copy.conformers == original.conformers
        np.testing.assert_array_equal(copy.moments, original.moments)


def test_a_moment_table_is_written_with_its_header(
    aspirin_conformers: Chem.Mol,
) -> None:
    stream = stdio.StringIO()

    io.write_moment_table(get_usr_moment_table([aspirin_conformers]), stream)

    lines = stream.getvalue().splitlines()
    assert lines[0] == "Title,Conformer,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12"
    assert lines[1].startswith("aspirin,0,")
    assert len(lines) == 4
    assert not stream.closed


def test_an_empty_moments_csv_reads_as_an_empty_table() -> None:
    header = ",".join(io.MOMENT_HEADER) + "\n"

    assert len(io.read_moment_table(stdio.StringIO(header))) == 0


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            "",
            "not a USR moments CSV: the header must be "
            "Title,Conformer,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12, not empty",
        ),
        (
            "Title,Conformer,Score\n",
            "not a USR moments CSV: the header must be "
            "Title,Conformer,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12, "
            "not Title,Conformer,Score",
        ),
        (
            "HEADER\nmol,0,1,2\n",
            "line 2 of the moments CSV has 4 fields, not 14",
        ),
        (
            "HEADER\nmol,first,1,2,3,4,5,6,7,8,9,10,11,12\n",
            "line 2 of the moments CSV is not a title, an integer conformer id and 12 "
            "numbers: invalid literal for int() with base 10: 'first'",
        ),
        (
            "HEADER\nmol,0,1,2,3,4,5,6,7,8,9,10,11,twelve\n",
            "line 2 of the moments CSV is not a title, an integer conformer id and 12 "
            "numbers: could not convert string to float: 'twelve'",
        ),
    ],
)
def test_a_malformed_moments_csv_is_refused(text: str, message: str) -> None:
    text = text.replace("HEADER", ",".join(io.MOMENT_HEADER))

    with pytest.raises(ValueError, match="moments CSV") as caught:
        io.read_moment_table(stdio.StringIO(text))

    assert str(caught.value) == message


def test_hits_are_written_in_order_with_their_header(tmp_path: Path) -> None:
    path = tmp_path / "hits.csv"
    hits = [Hit("q", 0, "a", 1, 1.0), Hit("q", 0, "b", 0, 0.25)]

    io.write_hits(iter(hits), path)

    assert path.read_text().splitlines() == [
        "Query,QueryConformer,Title,Conformer,Score",
        "q,0,a,1,1.0",
        "q,0,b,0,0.25",
    ]
