"""USR: Ultrafast Shape Recognition moments and similarity search with RDKit.

The modules, from the bottom up:

- :mod:`usr.moments` calculates the USR moments of conformers.
- :mod:`usr.similarity` scores two sets of moments, or one against many.
- :mod:`usr.search` screens a table of moments against query moments.
- :mod:`usr.io` reads SD files and reads and writes moments and results as CSV.

The ``usr`` command line wraps them: ``usr generate`` writes a moments database and
``usr search`` screens it.

USR logs through loguru and is silent until an application calls
``logger.enable("usr")``.
"""

from importlib import metadata

from loguru import logger


try:
    # The distribution name, which differs from the import name.
    __version__ = metadata.version("rd-usr")
except metadata.PackageNotFoundError:
    # Running from a source tree that was never installed.
    __version__ = "0.0.0"

# A library adds no sink. The application opts in with logger.enable("usr") and a
# sink of its own.
logger.disable("usr")

__all__ = ["__version__"]
