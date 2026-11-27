"""The ``usr`` command line.

Re-exports the Typer application, so the console entry point in ``pyproject.toml`` names
the package: ``usr = "usr.cli:app"``. Importing this package registers every command.

Each command parses its options, calls the library and writes the result; the logic is
in the library. Invoking ``app`` configures loguru for the whole process and does not
restore it, so call it in-process only in tests, which reset loguru after each test.

The root command is in :mod:`usr.cli.main`, because a submodule named ``app`` would be
shadowed by the Typer instance of that name.
"""

# Each import runs a module whose decorator registers its command on `app`.
import usr.cli.generate
import usr.cli.search  # noqa: F401
from usr.cli.main import app


__all__ = ["app"]
