import pyro.model as model
import pytest
import time
import torch
from tempfile import TemporaryDirectory
import pathlib


class BasicModel(model.Model):
    def __init__(self, *args, **kwargs):
        super(BasicModel, self).__init__(*args, **kwargs)
        self.dense = torch.nn.Linear(100, 20)


def test_checkpoint_dir(tmp_path):
    # test that checkpoint directory is lazily created if it does not exist
    model = BasicModel(checkpoint_dir=f"{tmp_path}/checkpoints")
    assert not model.checkpoint_dir.exists()
    model.save("test.pth")
    assert model.checkpoint_dir.exists()


def test_checkpoint_dir_pathlib(tmp_path):
    # test that checkpoint directory is lazily created from pathlib.Path if it does not exist
    path = pathlib.Path(tmp_path)
    model = BasicModel(checkpoint_dir=path / "checkpoints")
    assert not model.checkpoint_dir.exists()
    model.save("test.pth")
    assert model.checkpoint_dir.exists()


def test_checkpoint_clobber():
    # test that checkpoint directory, if already exists, is not clobbered
    with TemporaryDirectory("tempdir") as tempdir:
        tempdir_path = pathlib.Path(tempdir) / "checkpoints"
        tempfile = tempdir_path / "cool_beans.txt"

        model = BasicModel(checkpoint_dir=tempdir_path)
        assert not model.checkpoint_dir.exists()
        tempdir_path.mkdir(parents=True)
        tempfile.touch()
        assert tempfile.exists()
        model.save("test.pth")
        assert tempfile.exists()


def test_checkpoint_dir_create_parents(tmp_path):
    # test that checkpoint directory, if not exists, is created with parents
    model = BasicModel(checkpoint_dir=f"{tmp_path}/cool_beans/checkpoints")
    model.save()
    assert model.checkpoint_dir.exists()
    assert model.checkpoint_dir.parent.exists()


def test_save_custom_name(tmp_path):
    # test that model can be saved with custom name
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    model = BasicModel(checkpoint_dir=tmpdir)
    model.save("test.pth")

    assert (tmpdir / "test.pth").exists()


def test_save_weird_name(tmp_path):
    # test that model can be saved with weird name
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    model = BasicModel(checkpoint_dir=tmpdir)
    model.save("asd1291___203__203__23421__321")
    assert (tmpdir / "asd1291___203__203__23421__321").exists()


def test_load(tmp_path):
    # test that model can be loaded
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    model = BasicModel(checkpoint_dir=tmpdir)
    model.save("test.pth")

    new_model = BasicModel(checkpoint_dir=tmpdir)
    checkpoint = new_model.load(name="test.pth")
    assert checkpoint["name"] == "test.pth"
    assert torch.equal(
        checkpoint["state_dict"]["dense.weight"], model.state_dict()["dense.weight"]
    )
    assert torch.equal(
        checkpoint["state_dict"]["dense.bias"], model.state_dict()["dense.bias"]
    )
    assert torch.equal(
        new_model.state_dict()["dense.weight"], model.state_dict()["dense.weight"]
    )
    assert torch.equal(
        new_model.state_dict()["dense.bias"], model.state_dict()["dense.bias"]
    )


def test_load_nonfile(tmp_path):
    """Tests that loading a non-file checkpoint (directory) fails."""
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    (tmpdir / "test_dir").mkdir(parents=True)

    new_model = BasicModel(checkpoint_dir=tmpdir)
    with pytest.raises(IsADirectoryError):
        new_model.load(name="test_dir")


def test_load_empty_file(tmp_path):
    """Tests that loading an empty file fails."""
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    tmpdir.mkdir(parents=True)
    (tmpdir / "bad_checkpoint").touch()

    new_model = BasicModel(checkpoint_dir=tmpdir)
    with pytest.raises(EOFError):
        new_model.load(name="bad_checkpoint")


def test_load_auto_empty_directory(tmp_path):
    """Tests that auto-loading from an empty directory fails."""
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    tmpdir.mkdir(parents=True)

    new_model = BasicModel(checkpoint_dir=tmpdir)
    with pytest.raises(RuntimeError):
        new_model.load()


def test_load_auto(tmp_path):
    """Tests that auto-loading from a directory with multiple checkpoints succeeds."""
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    model = BasicModel(checkpoint_dir=tmpdir)
    model.save("test.pth")
    time.sleep(1)  # force a different timestamp
    model.save("test2.pth")

    new_model = BasicModel(checkpoint_dir=tmpdir)
    checkpoint = new_model.load()
    assert checkpoint["name"] == "test2.pth"


def test_load_auto_one_choice(tmp_path):
    """Tests that auto-loading from a directory with one checkpoint succeeds."""
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    model = BasicModel(checkpoint_dir=tmpdir)
    model.save("test_only.pth")

    new_model = BasicModel(checkpoint_dir=tmpdir)
    checkpoint = new_model.load()
    assert checkpoint["name"] == "test_only.pth"


def test_model_print():
    """Tests that the model prints correctly."""
    model = BasicModel(checkpoint_dir="checkpoints")
    target = """BasicModel:
==================================================
dense: Linear(in_features=100, out_features=20, bias=True) - 2020 parameters
==================================================
Total: 2020 parameters
Trainable: 2020 parameters (100.00% trainable)
=================================================="""
    assert str(model) == target


def test_model_print_frozen():
    """Tests that the model prints correctly with frozen parameters."""
    model = BasicModel(checkpoint_dir="checkpoints")
    model.dense.requires_grad_ = False
    target = """BasicModel:
==================================================
dense: Linear(in_features=100, out_features=20, bias=True) - 2020 parameters
==================================================
Total: 2020 parameters
Trainable: 0 parameters (0.00% trainable)
=================================================="""
    assert str(model) == target
