"""Publishing a replacement must leave the old artifact intact on failure."""

import pytest

from analysis.io.artifacts import atomic_path


def test_atomic_replacement_and_interrupted_write(tmp_path):
    destination = tmp_path / "summary.csv"
    destination.write_text("previous\n")
    with pytest.raises(RuntimeError, match="interrupted"):
        with atomic_path(destination) as temporary:
            temporary.write_text("partial\n")
            raise RuntimeError("interrupted")
    assert destination.read_text() == "previous\n"
    assert list(tmp_path.iterdir()) == [destination]

    with atomic_path(destination) as temporary:
        temporary.write_text("complete\n")
    assert destination.read_text() == "complete\n"
    assert list(tmp_path.iterdir()) == [destination]
