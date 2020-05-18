"""
"""
from pathlib import Path

import pandas as pd
from loguru import logger

from usr import io


def calc_usr_sim_scores(
        input_path: Path, output_path: Path, *,
        input_format: io.MomentFormat = io.MomentFormat.CSV,
        explicit_h: bool = False
) -> None:
    """

    Returns
    -------

    """
    df = pd.read_csv()
