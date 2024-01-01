import torch
import os
import pathlib
from typing import Union, Iterable
from functools import reduce
from datetime import datetime

LENGTH = 50


class Model(torch.nn.Module):
    """
    Extensible layer on top of `torch.nn.Module` to add amenities such as
    - Object-oriented model checkpointing
    - Verbose model summary (TensorFlow style)
    Note: Only top level modules should extend this class - all other submodules should defer to `torch.nn.Module`.
    """

    def __init__(self, checkpoint_dir: Union[str, os.PathLike], *args, **kwargs):
        """
        Initialize the Model object.

        Args:
            checkpoint_dir (Union[str, os.PathLike]): The directory path where checkpoints will be saved.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super(Model, self).__init__(*args, **kwargs)
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)

    def save(self, name: str = None, **metadata):
        """
        Save model parameters and metadata to file. Checkpoint is saved as "{name}.pth"

        Args:
            name (str, optional): Name of model checkpoint. Defaults to current timestamp.
        """
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)

        if name is None:
            name = datetime.now().strftime("%m-%d-%y_%H-%M-%S") + ".pth"

        # create the state dictionary
        state = {
            "name": name,
            "state_dict": self.state_dict(),
        }
        for key in metadata:
            state[key] = metadata[key]
        filename = self.checkpoint_dir / name
        torch.save(state, filename)

    def load(self, name: str = None, device: str = "cpu"):
        """
        Load model parameters from file.

        Model parameters are stored internally to model and metadata stored at runtime is returned to user.

        Args:
            device (str, optional): Device to load parameters to. Defaults to "cpu".
            name (str, optional): Name of the model checkpoint to load from checkpoint
                                directory. If unspecified, loads the most recent model (by timestamp).

        Returns:
            dict: Metadata associated with checkpoint
        """
        if name is None:
            try:
                file = find_file_by_timestamp(self.checkpoint_dir.iterdir())
            except TypeError:  # occurs when iterable to reduce is empty
                raise RuntimeError(
                    "Could not load checkpoint: checkpoint directory empty"
                )
        else:
            file = self.checkpoint_dir / name

        with file.open("rb") as fstream:
            checkpoint = torch.load(fstream, map_location=device)

        try:
            self.load_state_dict(checkpoint["state_dict"])
        except KeyError:
            raise RuntimeError(
                'Invalid checkpoint file: could not find key "state_dict"'
            )

        return checkpoint

    def summary(self) -> str:
        """
        Return a summary of the model, including the number of parameters.

        Returns:
            str: A string representation of the model summary.
        """
        output = []
        total_param, trainable_param = 0, 0

        separator = "\n" + "─" * LENGTH + "\n"
        thick_separator = "=" * LENGTH

        def extract(module_info):
            nonlocal total_param, trainable_param
            name, module = module_info
            count = sum(map(lambda x: x.numel(), module.parameters()))
            total_param += count
            if module.requires_grad_:
                trainable_param += count
            return f"{name}: {module} - {count} parameters"

        output.append(f"{self.__class__.__name__}:")
        output.append(thick_separator)
        output.append(separator.join(map(extract, self.named_children())))
        output.append(thick_separator)
        output.append(
            f"Total: {total_param} parameters\n"
            f"Trainable: {trainable_param} parameters ({(trainable_param / total_param * 100):.2f}% trainable)"
        )
        output.append(thick_separator)

        return "\n".join(output)
# --- utils ---


def find_file_by_timestamp(files: Iterable[pathlib.Path], latest=True) -> pathlib.Path:
    """
    Find the file with the latest or earliest timestamp among the given files.

    Args:
        files (Iterable[pathlib.Path]): A collection of pathlib.Path objects representing the files to search.
        latest (bool, optional): If True, find the file with the latest timestamp. If False,
        find the file with the earliest timestamp. Defaults to True.

    Returns:
        pathlib.Path: The path of the file with the latest or earliest timestamp.
    """

    def cmp(t1: int, t2: int) -> bool:
        return t1 > t2 if latest else t2 < t1

    reducer = (
        lambda file1, file2: file1
        if cmp(file1.stat().st_mtime_ns, file2.stat().st_mtime_ns)
        else file2
    )
    return reduce(reducer, files)
