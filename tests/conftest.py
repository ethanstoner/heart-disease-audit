import pytest

from heart_audit.data import fetch_raw


@pytest.fixture(scope="session")
def raw_dir():
    return fetch_raw()
