import os
from pathlib import Path

from oasis2 import config


def test_get_project_root_defaults_to_repo_root(monkeypatch):
    monkeypatch.delenv("OASIS2_PROJECT_ROOT", raising=False)
    root = config.get_project_root()
    assert isinstance(root, Path)
    assert root.exists()


def test_get_project_root_respects_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    assert config.get_project_root() == tmp_path


def test_get_data_dir_creates_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("OASIS2_PROJECT_ROOT", str(tmp_path))
    d = config.get_data_dir("raw")
    assert d.exists()
    assert d == tmp_path / "data" / "raw"


def test_is_colab_false_outside_colab():
    assert config.is_colab() is False


def test_get_device_returns_torch_device():
    import torch
    assert isinstance(config.get_device(), torch.device)


def test_environment_summary_has_expected_keys():
    info = config.environment_summary()
    for key in ("platform", "python", "colab", "gpu_available", "project_root"):
        assert key in info
