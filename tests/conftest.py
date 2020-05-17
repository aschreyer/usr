import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def usr_test_dir() -> Path:
    """
    """
    return Path(os.path.abspath(os.path.dirname(__file__)))
