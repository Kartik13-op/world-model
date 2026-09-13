from pathlib import Path

import torch

from .config import ProjectPaths
from .upscaler import (
    ReconstructionNet,
    VisualReconstructionNet,
    fast_sr_loss,
)
from .upscaler import _load_model_state
from .upscaler import train_reconstructor as _train_reconstructor

__all__ = [
    "ReconstructionNet",
    "VisualReconstructionNet",
    "fast_sr_loss",
    "train_reconstructor",
    "load_reconstructor",
]


def train_reconstructor(
    project: str | Path,
    epochs: int = 6,
    batch_size: int = 8,
    lr: float = 3e-4,
    base_ch: int = 32,
    n_res: int = 4,
    patch_size: int | None = None,
    device: str | None = None,
    log_fn=None,
    compile_model: bool = False,
    edge_loss_every: int = 8,
    resume: bool = False,
    progress_fn=None,
    train_size: int | None = 256,
    max_samples: int | None = 512,
) -> Path:
    """Train the idle-time whole-scene reconstructor only."""
    return _train_reconstructor(
        project=project,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        base_ch=base_ch,
        n_res=n_res,
        patch_size=patch_size,
        device=device,
        log_fn=log_fn,
        compile_model=compile_model,
        edge_loss_every=edge_loss_every,
        resume=resume,
        progress_fn=progress_fn,
        train_size=train_size,
        max_samples=max_samples,
    )


def load_reconstructor(project: str | Path, device: str) -> VisualReconstructionNet | None:
    """Load the only display model used by the playback loop."""
    paths = ProjectPaths(Path(project))
    if not paths.reconstructor_file.exists():
        return None

    checkpoint = torch.load(paths.reconstructor_file, map_location=device, weights_only=True)
    model = VisualReconstructionNet(
        base_ch=int(checkpoint.get("base_ch", 32)),
        n_res=int(checkpoint.get("n_res", 4)),
    ).to(device)
    _load_model_state(model, checkpoint["model"])
    model.eval()
    return model
