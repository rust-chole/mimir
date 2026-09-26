import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "render: end-to-end tests that build real VODs and render real Shorts")


@pytest.fixture(scope="session")
def e2e_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("mimir_e2e")
