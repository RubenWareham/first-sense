from pathlib import Path

import pytest

from examples.make_tiny_dataset import create_dataset


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    return create_dataset(tmp_path / "dataset")
