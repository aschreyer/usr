from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from usr.cli import app


runner = CliRunner()


def test_usr_generate_app(tmp_path: Path, usr_test_dir: Path):
    """
    Tests the USR::Generate command line application.
    """
    input_path = usr_test_dir / "data" / "chembl26-oral-approved.sdf"

    assert input_path.is_file(), \
        f"input file for test {input_path.resolve()} does not exist."

    output_path = tmp_path / "moments.csv"

    result = runner.invoke(
        app,
        [
            "generate",
            "--in", input_path,
            "--out", output_path.resolve()
        ]
    )

    assert result.exit_code == 0
    assert output_path.is_file()

    df = pd.read_csv(output_path.resolve())

    assert df.shape == (983, 14)
    assert df.columns.tolist() == [
        'Title', 'Conformer', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
        'M10', 'M11', 'M12'
    ]
