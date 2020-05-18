import sys
from enum import Enum
from pathlib import Path

import typer
from loguru import logger

from usr import __version__
from usr import io
from usr.app import generate_usr_moments, calc_usr_sim_scores


app = typer.Typer(
    name="USR",
    help="USR: Application to generate USR moments and use them for similarity "
         "searching."
)


def version_callback(value: bool) -> None:
    """
    """
    if value:
        typer.echo(f"USR {__version__}")
        raise typer.Exit()


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@app.callback()
def main(
        version: bool = typer.Option(None, "--version", callback=version_callback),
        log_level: LogLevel = typer.Option(
            LogLevel.INFO,
            help="Logging level to use for running the application."
        )
):
    """
    These CLI options are part of root application and need to be listed before any
    subcommands.
    """
    logger.configure(
        handlers=[
            dict(sink=sys.stderr, level=log_level)
        ]
    )


@app.command(no_args_is_help=True)
def generate(
    input_path: Path = typer.Option(
            ..., "--in",
            exists=True, readable=True,
            help="Chemical structure file containing molecules for which USR moments "
                 "should be generated. Only the SDF format is supported (.sdf)."
        ),
        output_path: Path = typer.Option(
            ..., "--out",
            writable=True,
            help="Path to which the USR moments should be written."
        ),
        output_format: io.MomentFormat = typer.Option(
            io.MomentFormat.CSV, "--output-format",
            show_default=True,
            help="Format in which the USR moments should be written."
        ),
        explicit_h: bool = typer.Option(
            False,
            show_default=True,
            help="Include the position of explicit hydrogen atoms in the calculation "
                 "of USR moments. By default, hydrogen atoms will not be included."
        )
) -> None:
    """
    Generate USR moments for chemical structures.

    This application can be used to generate databases of USR moments suitable for
    virtual screening.
    """
    generate_usr_moments(
        input_path, output_path, output_format=output_format, explicit_h=explicit_h
    )


@app.command()
def search(
        input_path: Path = typer.Option(
            ..., "--in",
            exists=True, readable=True,
            help="Chemical structure file containing molecules for which USR moments "
                 "should be generated. Only the SDF format is supported (.sdf)."
        ),
        output_path: Path = typer.Option(
            ..., "--out",
            writable=True,
            help="Path to which the USR moments should be written."
        ),
        input_format: io.MomentFormat = typer.Option(
            io.MomentFormat.CSV, "--output-format",
            show_default=True,
            help="Format in which the USR moments should be written."
        )
) -> None:
    """
    Screen a USR moment database with a reference query.


    """
    calc_usr_sim_scores(input_path, output_path, input_format=input_format)
