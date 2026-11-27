"""Tests for the ``usr`` command line: its commands, streams and exit codes.

The commands run in-process through ``CliRunner``. stdout carries only results, so each
test parses it as CSV; diagnostics go to stderr.
"""

import csv
import io as stdio
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from usr import __version__, io
from usr.cli import app
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
    assert result.stderr.startswith("error: not a USR moments CSV: the header must be")


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
