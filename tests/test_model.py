import os

import pyro.model as model
import pytest
import time
import torch
import pathlib
from collections import namedtuple


class BasicModel(model.Model):
    def __init__(self, *args, **kwargs):
        super(BasicModel, self).__init__(*args, **kwargs)
        self.dense = torch.nn.Linear(100, 20)


class ConvBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity=activation)
        torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.bn(self.conv(x))


class NestedBasicModel(model.Model):
    def __init__(self, *args, **kwargs):
        super(NestedBasicModel, self).__init__(*args, **kwargs)
        self.features = torch.nn.Sequential(
            ConvBN(3, 32, activation="relu", kernel_size=3, padding=1),
            ConvBN(32, 64, activation="relu", kernel_size=3, padding=1),
            ConvBN(64, 128, activation="relu", kernel_size=3, padding=1),
            ConvBN(128, 8, activation="relu", kernel_size=3, padding=1),
        )
        self.norm = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.norm(x)
        x = self.features(x)
        return x


class EmptyModel(model.Model):
    def __init__(self, *args, **kwargs):
        super(EmptyModel, self).__init__(*args, **kwargs)

    def forward(self, x):
        return x


class EmptyModelWithParams(model.Model):
    def __init__(self, *args, **kwargs):
        super(EmptyModelWithParams, self).__init__(*args, **kwargs)
        self.register_parameter("param", torch.nn.Parameter(torch.rand(10, 2, 3)))

    def forward(self, x):
        return x


@pytest.fixture()
def net(tmp_path):
    return BasicModel(checkpoint_dir=f"{tmp_path}/checkpoints")


def test_checkpoint_dir(tmp_path, net):
    # test that checkpoint directory is lazily created if it does not exist
    assert not net.checkpoint_dir.exists()
    net.save("test.pth")
    assert net.checkpoint_dir.exists()


def test_checkpoint_dir_pathlib(tmp_path, net):
    # test that checkpoint directory is lazily created from pathlib.Path if it does not exist
    assert not net.checkpoint_dir.exists()
    net.save("test.pth")
    assert net.checkpoint_dir.exists()


def test_checkpoint_clobber(tmpdir):
    # test that checkpoint directory, if already exists, is not clobbered
    tempdir_path = pathlib.Path(tmpdir) / "checkpoints"
    file = tempdir_path / "cool_beans.txt"

    net = BasicModel(checkpoint_dir=tempdir_path)
    assert not net.checkpoint_dir.exists()
    tempdir_path.mkdir(parents=True)
    file.touch()
    assert file.exists()
    net.save("test.pth")
    assert file.exists()


def test_checkpoint_dir_create_parents(tmp_path):
    # test that checkpoint directory, if not exists, is created with parents
    net = BasicModel(checkpoint_dir=f"{tmp_path}/cool_beans/checkpoints")
    net.save()
    assert net.checkpoint_dir.exists()
    assert net.checkpoint_dir.parent.exists()


def test_save_custom_name(tmp_path, net):
    # test that model can be saved with custom name
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    net.save("test.pth")
    assert (tmpdir / "test.pth").exists()


def test_save_weird_name(tmp_path, net):
    # test that model can be saved with weird name
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    net.save("asd1291___203__203__23421__321")
    assert (tmpdir / "asd1291___203__203__23421__321").exists()


def test_load(tmp_path, net):
    # test that model can be loaded
    net.save("test.pth")
    new_model = BasicModel(checkpoint_dir=f"{tmp_path}/checkpoints")

    checkpoint = new_model.load(name="test.pth")
    assert checkpoint["name"] == "test.pth"
    assert torch.equal(
        checkpoint["state_dict"]["dense.weight"], net.state_dict()["dense.weight"]
    )
    assert torch.equal(
        checkpoint["state_dict"]["dense.bias"], net.state_dict()["dense.bias"]
    )
    assert torch.equal(
        new_model.state_dict()["dense.weight"], net.state_dict()["dense.weight"]
    )
    assert torch.equal(
        new_model.state_dict()["dense.bias"], net.state_dict()["dense.bias"]
    )


def test_load_non_file(tmp_path):
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
    """Tests that automatic weight loading from an empty directory fails."""
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    tmpdir.mkdir(parents=True)

    new_model = BasicModel(checkpoint_dir=tmpdir)
    with pytest.raises(RuntimeError):
        new_model.load()


def test_load_auto(tmp_path):
    """Tests that automatic weight loading from a directory with multiple checkpoints succeeds."""
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    net = BasicModel(checkpoint_dir=tmpdir)
    net.save("test.pth")
    time.sleep(1)  # force a different timestamp
    net.save("test2.pth")

    new_model = BasicModel(checkpoint_dir=tmpdir)
    checkpoint = new_model.load()
    assert checkpoint["name"] == "test2.pth"


def test_load_auto_one_choice(tmp_path):
    """Tests that automatic weight loading from a directory with one checkpoint succeeds."""
    tmpdir = pathlib.Path(tmp_path) / "checkpoints"
    net = BasicModel(checkpoint_dir=tmpdir)
    net.save("test_only.pth")

    new_model = BasicModel(checkpoint_dir=tmpdir)
    checkpoint = new_model.load()
    assert checkpoint["name"] == "test_only.pth"


def test_model_summary(mocker):
    """Tests that the model is summarized correctly."""
    mocker.patch(
        "os.get_terminal_size",
        return_value=namedtuple("TerminalSize", ["columns", "lines"])(50, 100),
    )

    net = BasicModel(checkpoint_dir="checkpoints")
    target = """ (BasicModel) - 2020 parameters
└── dense (Linear) - 2020 parameters
══════════════════════════════════════════════════
Total: 2020 parameters
Trainable: 2020 parameters (100.00% trainable)"""
    assert net.summary() == target


def test_model_summary_frozen(mocker):
    """Tests that the model is summarized correctly with frozen parameters."""
    mocker.patch(
        "os.get_terminal_size",
        return_value=namedtuple("TerminalSize", ["columns", "lines"])(50, 100),
    )

    net = BasicModel(checkpoint_dir="checkpoints")

    net.dense.weight.requires_grad = False
    net.dense.bias.requires_grad = False

    target = """ (BasicModel) - 2020 parameters
└── dense (Linear) - 2020 parameters
══════════════════════════════════════════════════
Total: 2020 parameters
Trainable: 0 parameters (0.00% trainable)"""
    assert net.summary() == target


def test_nested_model_summary_depth_0(mocker):
    """Tests that the model is summarized correctly with nested layers up to depth 0."""
    mocker.patch(
        "os.get_terminal_size",
        return_value=namedtuple("TerminalSize", ["columns", "lines"])(50, 100),
    )

    net = NestedBasicModel(checkpoint_dir="checkpoints")
    target = """ (NestedBasicModel) - 102942 parameters
══════════════════════════════════════════════════
Total: 102942 parameters
Trainable: 102942 parameters (100.00% trainable)"""
    assert net.summary(max_depth=0) == target


def test_nested_model_summary_depth_1(mocker):
    """Tests that the model is summarized correctly with nested layers up to depth 1."""
    mocker.patch(
        "os.get_terminal_size",
        return_value=namedtuple("TerminalSize", ["columns", "lines"])(50, 100),
    )

    net = NestedBasicModel(checkpoint_dir="checkpoints")
    target = """ (NestedBasicModel) - 102942 parameters
├── features (Sequential) - 102936 parameters
└── norm (BatchNorm2d) - 6 parameters
══════════════════════════════════════════════════
Total: 102942 parameters
Trainable: 102942 parameters (100.00% trainable)"""
    assert net.summary(max_depth=1) == target


def test_nested_model_summary_depth_2(mocker):
    """Tests that the model is summarized correctly with nested layers up to depth 2."""
    mocker.patch(
        "os.get_terminal_size",
        return_value=namedtuple("TerminalSize", ["columns", "lines"])(50, 100),
    )

    net = NestedBasicModel(checkpoint_dir="checkpoints")
    target = """ (NestedBasicModel) - 102942 parameters
├── features (Sequential) - 102936 parameters
│   ├── 0 (ConvBN) - 960 parameters
│   ├── 1 (ConvBN) - 18624 parameters
│   ├── 2 (ConvBN) - 74112 parameters
│   └── 3 (ConvBN) - 9240 parameters
└── norm (BatchNorm2d) - 6 parameters
══════════════════════════════════════════════════
Total: 102942 parameters
Trainable: 102942 parameters (100.00% trainable)"""
    assert net.summary(max_depth=2) == target


def test_nested_model_summary(mocker):
    """Tests that the model is summarized correctly with nested layers."""
    mocker.patch(
        "os.get_terminal_size",
        return_value=namedtuple("TerminalSize", ["columns", "lines"])(50, 100),
    )

    net = NestedBasicModel(checkpoint_dir="checkpoints")
    target = """ (NestedBasicModel) - 102942 parameters
├── features (Sequential) - 102936 parameters
│   ├── 0 (ConvBN) - 960 parameters
│   │   ├── conv (Conv2d) - 896 parameters
│   │   └── bn (BatchNorm2d) - 64 parameters
│   ├── 1 (ConvBN) - 18624 parameters
│   │   ├── conv (Conv2d) - 18496 parameters
│   │   └── bn (BatchNorm2d) - 128 parameters
│   ├── 2 (ConvBN) - 74112 parameters
│   │   ├── conv (Conv2d) - 73856 parameters
│   │   └── bn (BatchNorm2d) - 256 parameters
│   └── 3 (ConvBN) - 9240 parameters
│       ├── conv (Conv2d) - 9224 parameters
│       └── bn (BatchNorm2d) - 16 parameters
└── norm (BatchNorm2d) - 6 parameters
══════════════════════════════════════════════════
Total: 102942 parameters
Trainable: 102942 parameters (100.00% trainable)"""
    assert net.summary() == target


def test_model_summary_invalid_depth(mocker):
    """Tests that an invalid depth raises an error."""
    mocker.patch(
        "os.get_terminal_size",
        return_value=namedtuple("TerminalSize", ["columns", "lines"])(50, 100),
    )
    net = NestedBasicModel(checkpoint_dir="checkpoints")
    with pytest.raises(ValueError):
        net.summary(max_depth=-1)


def test_model_summary_large_depth(mocker):
    """Tests that a large depth is equivalent to no depth."""
    mocker.patch(
        "os.get_terminal_size",
        return_value=namedtuple("TerminalSize", ["columns", "lines"])(50, 100),
    )
    net = NestedBasicModel(checkpoint_dir="checkpoints")
    assert net.summary(max_depth=100) == net.summary()


def test_model_empty(mocker):
    """Tests that summarizer doesn't error on empty model (with no submodules)"""
    mocker.patch(
        "os.get_terminal_size",
        return_value=namedtuple("TerminalSize", ["columns", "lines"])(50, 100),
    )
    net = EmptyModel(checkpoint_dir="checkpoints")
    target = """ (EmptyModel)
══════════════════════════════════════════════════
Total: 0 parameters
Trainable: 0 parameters (0.00% trainable)"""
    assert net.summary() == target


def test_model_no_submodule(mocker):
    """Tests that summarizer doesn't error on model with parameters but with no submodules"""
    mocker.patch(
        "os.get_terminal_size",
        return_value=namedtuple("TerminalSize", ["columns", "lines"])(50, 100),
    )
    net = EmptyModelWithParams(checkpoint_dir="checkpoints")
    target = """ (EmptyModelWithParams) - 60 parameters
══════════════════════════════════════════════════
Total: 60 parameters
Trainable: 60 parameters (100.00% trainable)"""
    assert net.summary() == target
