"""Root conftest — ensure test CWD matches data paths."""

from pathlib import Path

import pytest

# Old tests assumed CWD was repo root and used paths like "tests/data/..."
# Since data now lives at packages/tensortrade/tests/data/, set rootdir
# to packages/tensortrade/ so "tests/data/..." resolves correctly.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _test_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set CWD to core package root so relative data paths resolve."""
    monkeypatch.chdir(_PACKAGE_ROOT)
