"""
This module defines the USR::Generate CLI application, which can be used to generate USR
moments for molecules contained in chemical structure files.
"""
from pathlib import Path

from loguru import logger

from usr import io
from usr.moments import get_usr_moments_df


def generate_usr_moments(
        input_path: Path, output_path: Path, *,
        output_format: io.MomentFormat = io.MomentFormat.CSV,
        explicit_h: bool = False, sd_title_tag: str = "_Name",
) -> None:
    """
    Reads molecules from the input file `input_path` and writes the generated USR
    moments to `output_path`.

    This function uses Pandas data frames to simplify output handling, with support for
    different file formats and storage such as S3.

    Parameters
    ----------
    input_path : Path
        Path to a chemical structure file in a supported 3D chemical structure format
        (SDF).
    output_path : Path
        Path to which the calculated USR moments will be written.
    output_format : io.MomentOutputFormat = io.MomentOutputFormat.CSV
        Format for writing USR moments.
    explicit_h: bool = False
        If True, include the coordinates of explicit hydrogens atoms in the USR moment
        generation.
    sd_title_tag: str = "_Name"
        SD tag containing the molecule names.
    """
    mols = io.get_mols_from_sdf_file(
        input_path, sanitize=True, suppress_hydrogens=explicit_h is False
    )

    # create Pandas data frame containing all USR moments
    df = get_usr_moments_df(mols, explicit_h=explicit_h, sd_title_tag=sd_title_tag)

    if output_format == io.MomentFormat.CSV:
        df.to_csv(output_path)
        logger.info("wrote USR moments to {}.", output_path)
