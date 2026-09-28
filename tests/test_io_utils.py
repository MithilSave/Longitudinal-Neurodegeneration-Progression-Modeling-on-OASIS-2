import numpy as np
import pytest

from oasis2 import io_utils


def test_checkpoint_roundtrip(tmp_path):
    path = io_utils.save_checkpoint({"a": 1}, "mycp", directory=tmp_path)
    assert path.exists()
    assert io_utils.checkpoint_exists("mycp", directory=tmp_path)
    assert io_utils.load_checkpoint("mycp", directory=tmp_path) == {"a": 1}


def test_load_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io_utils.load_checkpoint("nope", directory=tmp_path)


def test_json_roundtrip(tmp_path):
    p = tmp_path / "out.json"
    io_utils.save_json({"x": [1, 2, 3]}, p)
    assert io_utils.load_json(p) == {"x": [1, 2, 3]}


def test_z_score_zero_mean_unit_var():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = io_utils.z_score(arr)
    assert abs(z.mean()) < 1e-8
    assert abs(z.std() - 1.0) < 1e-8


def test_z_score_constant_column_no_div_by_zero():
    arr = np.array([5.0, 5.0, 5.0])
    z = io_utils.z_score(arr)
    assert np.all(z == 0.0)
