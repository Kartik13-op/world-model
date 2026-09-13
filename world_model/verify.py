from pathlib import Path

import numpy as np
import torch

from .config import ProjectPaths
from .model import WorldModel
from .reconstructor import load_reconstructor


def verify_pipeline(project: str | Path, device: str = "cpu") -> list[str]:
    """Run read-only integrity checks for processed data and checkpoints."""
    paths = ProjectPaths(Path(project))
    messages: list[str] = []

    if not paths.frames_file.exists() or not paths.actions_file.exists():
        raise RuntimeError("Missing processed data: run preprocess first.")

    frames = np.load(paths.frames_file, mmap_mode="r")
    actions = np.load(paths.actions_file, mmap_mode="r")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"frames.npy has unexpected shape {frames.shape}.")
    if actions.ndim != 2 or actions.shape[0] != frames.shape[0] or actions.shape[1] not in (4, 5):
        raise RuntimeError(f"actions.npy has unexpected shape {actions.shape}; expected ({frames.shape[0]}, 4 or 5).")
    if frames.shape[0] < 3:
        raise RuntimeError("At least 3 processed frames are required.")
    messages.append(f"data OK: {frames.shape[0]} frames, {frames.shape[1]}x{frames.shape[2]}, actions={actions.shape[1]}D")

    if paths.model_file.exists():
        checkpoint = torch.load(paths.model_file, map_location=device, weights_only=True)
        model = WorldModel(latent_channels=int(checkpoint["latent_channels"]))
        model.load_state_dict(checkpoint["model"])
        model.eval()
        sample = torch.from_numpy(np.asarray(frames[:1]).copy()).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            output = model.decode(model.encode(sample))
        if output.shape != sample.shape:
            raise RuntimeError(f"world model round-trip shape mismatch: {tuple(output.shape)} vs {tuple(sample.shape)}")
        messages.append(f"world checkpoint OK: epoch={checkpoint.get('epoch', '?')}")
    else:
        messages.append("world checkpoint missing: run `train` before play")

    if paths.reconstructor_file.exists():
        reconstructor = load_reconstructor(project, device)
        if reconstructor is None:
            raise RuntimeError("Reconstructor checkpoint exists but could not be loaded.")
        sample = torch.from_numpy(np.asarray(frames[:1]).copy()).permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            output = reconstructor(sample)
        if output.shape != sample.shape:
            raise RuntimeError(f"reconstructor shape mismatch: {tuple(output.shape)} vs {tuple(sample.shape)}")
        messages.append("reconstructor checkpoint OK: display-only path is loadable")
    else:
        messages.append("reconstructor checkpoint missing: optional; run `reconstructor` to train it")

    return messages
