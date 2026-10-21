"""Compatibility for pre-existing trusted local Ultralytics checkpoints on modern torch."""
from pathlib import Path
from functools import partial
from contextlib import contextmanager


@contextmanager
def trusted_checkpoint_loading():
    import torch
    original=torch.load
    torch.load=partial(original,weights_only=False)
    try:yield
    finally:torch.load=original


def load_yolo(path,task=None):
    from ultralytics import YOLO
    path=Path(path)
    if not path.is_file():raise ValueError("model must be an existing trusted local checkpoint or YAML")
    with trusted_checkpoint_loading():
        return YOLO(str(path),task=task)


def train_yolo(model,**kwargs):
    # Older Ultralytics also reloads its own saved best checkpoint during final validation.
    with trusted_checkpoint_loading():
        return model.train(**kwargs)
