import os
import pathlib
from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from typing import Callable, Iterable, Union

import torch


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
        self._summary_stack = []
        self._summarizer = build_summarizer(self._summary_stack)

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

    def summary(self, max_depth: int = None) -> str:
        """
        Return a summary of the model in the form of a submodule tree.

        Args:
            max_depth (int, optional): The maximum depth of submodules to display. Defaults to None,
                                     in which case no maximum depth is applied.

        Returns:
            str: A summary of the model.
        """
        total_count = self._summarizer(self, "module", max_depth)
        percent_trainable = total_count.trainable / total_count.total * 100
        with_parameter_counts = [
            "\n".join(self._summary_stack[::-1]),
            "═" * os.get_terminal_size().columns,
            f"Total: {total_count.total} parameters",
            f"Trainable: {total_count.trainable} parameters ({percent_trainable:.2f}% trainable)"
        ]
        return "\n".join(with_parameter_counts)


# --- utils ---


@dataclass
class _ParameterCount:
    trainable: int = 0
    total: int = 0


def build_summarizer(
        stack: list[str],
) -> Callable[[torch.nn.Module, str, int | None], _ParameterCount]:
    """
    Build a summarizer function that can be used to generate a model summary.

    Args:
        stack (list[str]): The stack to which the summarizer should append its results.

    Returns:
        Callable[[torch.nn.Module, str, int | None], _ParameterCount]: A summarizer function that can be used
        to generate a model summary.
    """
    terminal_states = []

    def build_prefix() -> str:
        """
        Build the prefix for a line in the model summary.
        """
        if len(terminal_states) == 0:
            return ""

        components = ["    " if terminal else "│   " for terminal in terminal_states[:-1]]
        components.append("└── " if terminal_states[-1] else "├── ")
        return "".join(components)

    def build_line(
            module: torch.nn.Module,
            name: str,
            count: _ParameterCount,
    ) -> str:
        """
        Build a line in the model summary.
        """
        module_type = module.__class__.__name__  # get class name for module
        prefix = build_prefix()

        name_portion = f"{prefix}{name} ({module_type})"
        if count.total > 0:
            param_portion = f" - {count.total} parameters"
        else:
            param_portion = ""
        return name_portion + param_portion

    def summarizer(module: torch.nn.Module, name: str, max_depth: int = None) -> _ParameterCount:
        """
        Summarize a module and its submodules.
        """
        children = list(module.named_children())
        count = _ParameterCount()

        if len(children) != 0:
            terminal_states.append(True)
            for index, (child_name, child) in enumerate(children[::-1]):
                result = summarizer(child, child_name, max_depth)
                count.trainable += result.trainable
                count.total += result.total
                if index == 0:
                    terminal_states[-1] = False
            terminal_states.pop()
        else:
            for parameter in module.parameters():
                parameter_count = parameter.numel()
                count.total += parameter_count
                if parameter.requires_grad:
                    count.trainable += parameter_count

        if max_depth is None or len(terminal_states) <= max_depth:
            stack.append(build_line(module, name, count))

        return count

    return summarizer


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
