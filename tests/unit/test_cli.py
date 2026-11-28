"""Tests for the ``usr`` and ``usrcat`` command lines: commands, streams, exit codes.

The commands run in-process through ``CliRunner``. stdout carries only results, so each
test parses it as CSV; diagnostics go to stderr.
"""

import csv
import io as stdio
import sys
from pathlib import Path

import pytest
from rdkit import Chem
from typer.testing import CliRunner

from usr import __version__, io, moments
from usr.cli import app, usrcat_app
from usr.moments import MOMENT_COLUMNS


runner = CliRunner()


def rows_of(text: str) -> list[dict[str, str]]:
    """Return the rows of a CSV, keyed by the header."""
    return list(csv.DictReader(stdio.StringIO(text)))


@pytest.fixture
def drugs_csv(tmp_path: Path, drugs_sdf: Path) -> Path:
    """Return the moments of the six 3D drugs, written by ``usr generate``."""
    path = tmp_path / "drugs.csv"
    result = runner.invoke(app, ["generate", str(drugs_sdf), "--out", str(path)])
    assert result.exit_code == 0, result.output
    return path


@pytest.fixture
def usrcat_csv(tmp_path: Path, drugs_sdf: Path) -> Path:
    """Return the USRCAT moments of the six 3D drugs, written by ``usrcat generate``."""
    path = tmp_path / "drugs-usrcat.csv"
    result = runner.invoke(usrcat_app, ["generate", str(drugs_sdf), "-o", str(path)])
    assert result.exit_code == 0, result.output
    return path


def test_bare_usr_shows_help_and_exits_two() -> None:
    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "Usage:" in result.output
    assert "generate" in result.output
    assert "search" in result.output


def test_the_version_is_the_packages_own() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout == f"usr {__version__}\n"


def test_generate_writes_a_row_per_record_of_a_2d_file(
    chembl_sdf: Path, tmp_path: Path
) -> None:
    """The ChEMBL file is 2D: the run shows scale, not shape."""
    path = tmp_path / "chembl.csv"

    result = runner.invoke(app, ["generate", str(chembl_sdf), "-o", str(path)])

    assert result.exit_code == 0
    assert result.stdout == ""
    rows = rows_of(path.read_text())
    assert len(rows) == 983
    assert list(rows[0]) == ["Title", "Conformer", *MOMENT_COLUMNS]
    assert rows[0]["Title"] == "CHEMBL2336071"


def test_generate_writes_to_stdout_without_out(
    drugs_sdf: Path, drug_names: list[str]
) -> None:
    result = runner.invoke(app, ["generate", str(drugs_sdf)])

    assert result.exit_code == 0
    assert [row["Title"] for row in rows_of(result.stdout)] == drug_names
    assert result.stderr == ""


def test_generate_with_explicit_h_writes_different_moments(drugs_sdf: Path) -> None:
    """Measured: before the fix, ``--explicit-h`` wrote a header and no rows."""
    plain = rows_of(runner.invoke(app, ["generate", str(drugs_sdf)]).stdout)
    result = runner.invoke(app, ["generate", str(drugs_sdf), "--explicit-h"])

    with_h = rows_of(result.stdout)
    assert result.exit_code == 0
    assert len(with_h) == len(plain) == 6
    assert with_h[0]["M1"] != plain[0]["M1"]


def test_generate_skips_an_unreadable_record_with_a_warning(
    sdf_with_a_bad_record: Path,
) -> None:
    result = runner.invoke(app, ["generate", str(sdf_with_a_bad_record)])

    assert result.exit_code == 0
    assert len(rows_of(result.stdout)) == 6
    assert (
        f"skipped record 2 of {sdf_with_a_bad_record}: RDKit could not read it"
        in result.stderr
    )


def test_generate_refuses_a_file_that_is_not_an_sd_file(tmp_path: Path) -> None:
    path = tmp_path / "mols.smi"
    path.write_text("CCO ethanol\n")

    result = runner.invoke(app, ["generate", str(path)])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == (
        "error: cannot read molecules from mols.smi: only SD files are supported, "
        "named *.sdf or *.sdf.gz\n"
    )


def test_generate_refuses_a_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["generate", str(tmp_path / "missing.sdf")])

    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_generate_refuses_an_output_in_a_missing_directory(
    drugs_sdf: Path, tmp_path: Path
) -> None:
    path = tmp_path / "missing" / "drugs.csv"

    result = runner.invoke(app, ["generate", str(drugs_sdf), "-o", str(path)])

    assert result.exit_code == 2
    assert result.stderr == f"error: [Errno 2] No such file or directory: '{path}'\n"


def test_verbose_logs_progress_on_stderr(drugs_sdf: Path) -> None:
    result = runner.invoke(app, ["-v", "generate", str(drugs_sdf)])

    assert result.exit_code == 0
    assert "calculated USR moments for 6 conformers" in result.stderr
    assert len(rows_of(result.stdout)) == 6


def test_search_finds_each_query_first_in_its_own_database(
    drugs_sdf: Path, drugs_csv: Path, drug_names: list[str]
) -> None:
    result = runner.invoke(app, ["search", str(drugs_sdf), str(drugs_csv), "-n", "1"])

    rows = rows_of(result.stdout)
    assert result.exit_code == 0
    assert list(rows[0]) == ["Query", "QueryConformer", "Title", "Conformer", "Score"]
    assert [(row["Query"], row["Title"], row["Score"]) for row in rows] == [
        (name, name, "1.0") for name in drug_names
    ]


def test_search_keeps_every_hit_best_first_without_top(
    drugs_sdf: Path, drugs_csv: Path, tmp_path: Path
) -> None:
    path = tmp_path / "hits.csv"

    result = runner.invoke(
        app, ["search", str(drugs_sdf), str(drugs_csv), "--out", str(path)]
    )

    rows = rows_of(path.read_text())
    assert result.exit_code == 0
    assert len(rows) == 36
    for start in range(0, 36, 6):
        scores = [float(row["Score"]) for row in rows[start : start + 6]]
        assert scores == sorted(scores, reverse=True)


def test_search_accepts_a_moments_csv_as_the_query(drugs_csv: Path) -> None:
    result = runner.invoke(app, ["search", str(drugs_csv), str(drugs_csv), "-n", "2"])

    rows = rows_of(result.stdout)
    assert result.exit_code == 0
    assert len(rows) == 12
    assert rows[0]["Query"] == rows[0]["Title"] == "aspirin"


def test_search_with_explicit_h_matches_a_database_generated_with_it(
    drugs_sdf: Path, tmp_path: Path
) -> None:
    database = tmp_path / "drugs-h.csv"
    runner.invoke(
        app, ["generate", str(drugs_sdf), "--explicit-h", "-o", str(database)]
    )

    matched = runner.invoke(
        app, ["search", str(drugs_sdf), str(database), "--explicit-h", "-n", "1"]
    )
    mismatched = runner.invoke(
        app, ["search", str(drugs_sdf), str(database), "-n", "1"]
    )

    assert {row["Score"] for row in rows_of(matched.stdout)} == {"1.0"}
    assert "1.0" not in {row["Score"] for row in rows_of(mismatched.stdout)}


def test_search_refuses_a_database_that_is_not_a_moments_csv(
    drugs_sdf: Path, tmp_path: Path
) -> None:
    path = tmp_path / "hits.csv"
    path.write_text("Query,QueryConformer,Title,Conformer,Score\n")

    result = runner.invoke(app, ["search", str(drugs_sdf), str(path)])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr.startswith("error: not a moments CSV: the header must be")


def test_search_refuses_a_top_below_one(drugs_sdf: Path, drugs_csv: Path) -> None:
    result = runner.invoke(app, ["search", str(drugs_sdf), str(drugs_csv), "-n", "0"])

    assert result.exit_code == 2
    assert "Invalid value for '--top' / '-n'" in result.stderr


def test_a_closed_pipe_exits_zero(
    drugs_sdf: Path, drugs_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``usr search ... | head`` closes the pipe early; that is not a failure."""

    def closed_pipe(*_: object) -> None:
        raise BrokenPipeError

    monkeypatch.setattr(io, "write_hits", closed_pipe)

    before = sys.stdout
    result = runner.invoke(app, ["search", str(drugs_sdf), str(drugs_csv)])

    assert result.exit_code == 0
    assert result.stderr == ""
    # CliRunner restores the stdout that the broken-pipe path replaced.
    assert sys.stdout is before


def test_usr_search_refuses_a_usrcat_database(
    drugs_sdf: Path, usrcat_csv: Path
) -> None:
    result = runner.invoke(app, ["search", str(drugs_sdf), str(usrcat_csv)])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == (
        "error: drugs-usrcat.csv holds 60 moments per row, not the 12 of USR: "
        "search it with usrcat search\n"
    )


def test_bare_usrcat_shows_help_and_exits_two() -> None:
    result = runner.invoke(usrcat_app, [])

    assert result.exit_code == 2
    assert "USRCAT" in result.output
    assert "generate" in result.output
    assert "search" in result.output


def test_the_usrcat_version_is_the_packages_own() -> None:
    result = runner.invoke(usrcat_app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout == f"usrcat {__version__}\n"


def test_usrcat_generate_writes_sixty_moments_per_record(
    drugs_sdf: Path, drug_names: list[str]
) -> None:
    result = runner.invoke(usrcat_app, ["generate", str(drugs_sdf)])

    rows = rows_of(result.stdout)
    assert result.exit_code == 0
    assert list(rows[0]) == list(io.make_moment_header(60))
    assert [row["Title"] for row in rows] == drug_names
    assert result.stderr == ""


def test_usrcat_generate_with_explicit_h_writes_different_moments(
    drugs_sdf: Path,
) -> None:
    plain = rows_of(runner.invoke(usrcat_app, ["generate", str(drugs_sdf)]).stdout)
    result = runner.invoke(usrcat_app, ["generate", str(drugs_sdf), "--explicit-h"])

    with_h = rows_of(result.stdout)
    assert result.exit_code == 0
    assert len(with_h) == len(plain) == 6
    assert with_h[0]["M1"] != plain[0]["M1"]


def test_usrcat_search_finds_each_query_first_in_its_own_database(
    drugs_sdf: Path, usrcat_csv: Path, drug_names: list[str]
) -> None:
    result = runner.invoke(
        usrcat_app, ["search", str(drugs_sdf), str(usrcat_csv), "-n", "1"]
    )

    rows = rows_of(result.stdout)
    assert result.exit_code == 0
    assert [(row["Query"], row["Title"], row["Score"]) for row in rows] == [
        (name, name, "1.0") for name in drug_names
    ]


def test_usrcat_search_with_shape_only_weights_scores_as_usr(
    drugs_csv: Path, usrcat_csv: Path
) -> None:
    """Weighing only the all-atom block gives the USR scores: the block is USR."""
    usr = rows_of(runner.invoke(app, ["search", str(drugs_csv), str(drugs_csv)]).stdout)

    result = runner.invoke(
        usrcat_app,
        ["search", str(usrcat_csv), str(usrcat_csv), "--weights", "1,0,0,0,0"],
    )

    rows = rows_of(result.stdout)
    assert result.exit_code == 0
    assert [float(row["Score"]) for row in rows] == pytest.approx(
        [float(row["Score"]) for row in usr], rel=1e-14
    )


def test_usrcat_search_weights_change_the_scores(usrcat_csv: Path) -> None:
    def scores(*options: str) -> list[str]:
        result = runner.invoke(
            usrcat_app, ["search", str(usrcat_csv), str(usrcat_csv), *options]
        )
        assert result.exit_code == 0, result.output
        return [row["Score"] for row in rows_of(result.stdout)]

    assert scores("-w", "1,1,1,1,1") == scores()
    assert scores("-w", "1,0.5,0.5,1,1") != scores()


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (
            "1,a",
            "error: --weights must be numbers separated by commas, such as "
            "1,0.5,0.5,1,1, not '1,a'\n",
        ),
        (
            "1,1",
            "error: weights must hold one weight per block of 12 moments, 5 in all, "
            "not 2\n",
        ),
        (
            "0,0,0,0,0",
            "error: weights must not all be zero, not [0.0, 0.0, 0.0, 0.0, 0.0]\n",
        ),
    ],
)
def test_usrcat_search_refuses_invalid_weights(
    usrcat_csv: Path, weights: str, message: str
) -> None:
    result = runner.invoke(
        usrcat_app, ["search", str(usrcat_csv), str(usrcat_csv), "-w", weights]
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == message


def test_usrcat_search_refuses_a_usr_database(drugs_sdf: Path, drugs_csv: Path) -> None:
    result = runner.invoke(usrcat_app, ["search", str(drugs_sdf), str(drugs_csv)])

    assert result.exit_code == 2
    assert result.stderr == (
        "error: drugs.csv holds the 12 moments of USR, not USRCAT moments: "
        "search it with usr search\n"
    )


def test_usrcat_search_of_a_custom_database_needs_a_csv_query(
    drug_mols: list[Chem.Mol], drugs_sdf: Path, tmp_path: Path
) -> None:
    path = tmp_path / "custom.csv"
    table = moments.get_usrcat_moment_table(drug_mols, smarts=["[#7]", "[#8]"])
    io.write_moment_table(table, path)

    from_sdf = runner.invoke(usrcat_app, ["search", str(drugs_sdf), str(path)])
    from_csv = runner.invoke(
        usrcat_app, ["search", str(path), str(path), "-n", "1", "-w", "1,2,2"]
    )

    assert from_sdf.exit_code == 2
    assert from_sdf.stderr == (
        "error: custom.csv holds 36 moments per row, not the 60 of USRCAT: pass a "
        "moments CSV with 36 moments as the query\n"
    )
    assert from_csv.exit_code == 0
    assert {row["Score"] for row in rows_of(from_csv.stdout)} == {"1.0"}


def test_usrcat_search_with_explicit_h_matches_a_database_generated_with_it(
    drugs_sdf: Path, tmp_path: Path
) -> None:
    database = tmp_path / "drugs-h.csv"
    runner.invoke(
        usrcat_app, ["generate", str(drugs_sdf), "--explicit-h", "-o", str(database)]
    )

    result = runner.invoke(
        usrcat_app,
        ["search", str(drugs_sdf), str(database), "--explicit-h", "-n", "1"],
    )

    assert {row["Score"] for row in rows_of(result.stdout)} == {"1.0"}


def test_usrcat_verbose_logs_progress_on_stderr(drugs_sdf: Path) -> None:
    result = runner.invoke(usrcat_app, ["-v", "generate", str(drugs_sdf)])

    assert result.exit_code == 0
    assert "calculated USRCAT moments for 6 conformers" in result.stderr


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (
            "",
            "error: --weights must be numbers separated by commas, such as "
            "1,0.5,0.5,1,1, not ''\n",
        ),
        (
            "1,2,",
            "error: --weights must be numbers separated by commas, such as "
            "1,0.5,0.5,1,1, not '1,2,'\n",
        ),
        (
            "nan,1,1,1,1",
            "error: weights must be finite and non-negative, not "
            "[nan, 1.0, 1.0, 1.0, 1.0]\n",
        ),
        (
            "1,inf,1,1,1",
            "error: weights must be finite and non-negative, not "
            "[1.0, inf, 1.0, 1.0, 1.0]\n",
        ),
        (
            "-1,1,1,1,1",
            "error: weights must be finite and non-negative, not "
            "[-1.0, 1.0, 1.0, 1.0, 1.0]\n",
        ),
    ],
)
def test_usrcat_search_refuses_malformed_weights(
    usrcat_csv: Path, weights: str, message: str
) -> None:
    result = runner.invoke(
        usrcat_app, ["search", str(usrcat_csv), str(usrcat_csv), f"--weights={weights}"]
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == message


def test_usrcat_search_weights_may_hold_spaces(usrcat_csv: Path) -> None:
    spaced = runner.invoke(
        usrcat_app,
        ["search", str(usrcat_csv), str(usrcat_csv), "-w", " 1 , 1 , 1 , 1 , 1 "],
    )
    plain = runner.invoke(usrcat_app, ["search", str(usrcat_csv), str(usrcat_csv)])

    assert spaced.exit_code == 0
    assert spaced.stdout == plain.stdout


def test_usrcat_search_checks_weights_before_reading_an_sd_query(
    drugs_sdf: Path, usrcat_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured: the query's moments were calculated before the weights were checked."""

    def unread(*_: object, **__: object) -> None:
        raise AssertionError("the query was read")

    monkeypatch.setattr(io, "read_molecules", unread)

    result = runner.invoke(
        usrcat_app, ["search", str(drugs_sdf), str(usrcat_csv), "-w", "1,1"]
    )

    assert result.exit_code == 2
    assert result.stderr == (
        "error: weights must hold one weight per block of 12 moments, 5 in all, not 2\n"
    )
