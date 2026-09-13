"""Display-side visual reconstruction network for the world model.

This module replaces the conventional super-resolution objective with a
video-specific reconstruction objective: the model learns a per-scene visual
prior from the original training frames and reconstructs degraded world-model
outputs back toward the most likely meaningful frame.

The world-model latent/state is never modified by this network; it is strictly
for display-side restoration.
"""

import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ProjectPaths

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class _ResidualBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class VisualReconstructionNet(nn.Module):
    """Lightweight U-Net style reconstructor for degraded world-model frames."""

    def __init__(self, base_ch: int = 32, n_res: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.res1 = nn.Sequential(*[_ResidualBlock(base_ch) for _ in range(n_res)])

        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.res2 = nn.Sequential(*[_ResidualBlock(base_ch * 2) for _ in range(n_res)])

        self.mid = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(base_ch, 3, kernel_size=3, padding=1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x1 = self.res1(x1)
        x2 = self.down2(x1)
        x2 = self.res2(x2)
        x2 = self.mid(x2)
        x2 = self.up2(x2)
        x2 = x2 + x1
        x = self.up1(x2)
        x = x + x0
        residual = self.out(x)
        return (residual_input + residual).clamp(0.0, 1.0)


ReconstructionNet = VisualReconstructionNet


def _load_model_state(model: nn.Module, state: dict) -> None:
    """Load both normal and older torch.compile-wrapped state dictionaries."""
    if state and all(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)


class _ReconstructionDataset(Dataset):
    """Training pairs: degraded input -> original training frame target."""

    def __init__(self, frames_file: Path, train_size: int | None = 256) -> None:
        frames = np.load(frames_file, mmap_mode="r")
        # np.load(..., mmap_mode="r") returns a read-only array. Copy it
        # before converting so PyTorch never receives a non-writable tensor.
        self.frames = torch.from_numpy(np.asarray(frames).copy()).permute(0, 3, 1, 2).float() / 255.0
        self.train_size = train_size

    def __len__(self) -> int:
        return self.frames.shape[0]

    def _degrade(self, frame: torch.Tensor) -> torch.Tensor:
        frame = frame.clamp(0.0, 1.0)
        h, w = frame.shape[-2:]

        blur = F.avg_pool2d(frame.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
        low = F.interpolate(frame.unsqueeze(0), scale_factor=0.72, mode="bilinear", align_corners=False)
        low = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)

        mixed = 0.55 * blur + 0.45 * low
        mixed = mixed.clamp(0.0, 1.0)

        channel_bias = torch.stack(
            [
                mixed[0] * 0.95 + mixed[1] * 0.05,
                mixed[1] * 0.9 + mixed[0] * 0.1 + mixed[2] * 0.05,
                mixed[2] * 0.9 + mixed[1] * 0.1,
            ],
            dim=0,
        )

        noise = torch.randn_like(channel_bias) * 0.04
        degraded = channel_bias + noise
        degraded = degraded.clamp(0.0, 1.0)

        if torch.rand(()) > 0.5:
            tx = (torch.rand(()) - 0.5) * 0.08
            ty = (torch.rand(()) - 0.5) * 0.08
            theta = torch.tensor([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=degraded.dtype, device=degraded.device)
            theta = theta.unsqueeze(0)
            grid = F.affine_grid(theta, degraded.unsqueeze(0).shape, align_corners=False)
            degraded = F.grid_sample(degraded.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(0)

        return degraded.clamp(0.0, 1.0)

    def __getitem__(self, idx: int):
        target = self.frames[idx]
        if self.train_size is not None and max(target.shape[-2:]) > self.train_size:
            scale = self.train_size / max(target.shape[-2:])
            size = (max(32, int(target.shape[-2] * scale)), max(32, int(target.shape[-1] * scale)))
            target = F.interpolate(target.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
        return self._degrade(target), target


class _PatchDataset(Dataset):
    def __init__(self, base: Dataset, patch_size: int = 64) -> None:
        self.base = base
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        degraded, target = self.base[idx]
        _, h, w = degraded.shape
        ps = min(self.patch_size, h, w)
        if h == ps:
            y = 0
        else:
            y = torch.randint(0, h - ps + 1, ()).item()
        if w == ps:
            x = 0
        else:
            x = torch.randint(0, w - ps + 1, ()).item()
        return degraded[:, y : y + ps, x : x + ps], target[:, y : y + ps, x : x + ps]


def fast_sr_loss(pred: torch.Tensor, target: torch.Tensor, edge_loss: bool = False) -> torch.Tensor:
    loss = F.l1_loss(pred, target)
    if edge_loss:
        pred_x = pred[..., :, 1:] - pred[..., :, :-1]
        target_x = target[..., :, 1:] - target[..., :, :-1]
        pred_y = pred[..., 1:, :] - pred[..., :-1, :]
        target_y = target[..., 1:, :] - target[..., :-1, :]
        loss += 0.25 * (F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y))
    return loss


def train_reconstructor(
    project: str | Path,
    epochs: int = 12,
    batch_size: int = 16,
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
    paths = ProjectPaths(Path(project))
    if not paths.frames_file.exists():
        raise RuntimeError("Run preprocess first — frames.npy not found.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.startswith("cuda")
    if not is_cuda:
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch only permits this before parallel work has started.
            pass

    def _log(msg: str) -> None:
        print(msg, flush=True)
        if log_fn:
            log_fn(msg)

    dataset = _ReconstructionDataset(paths.frames_file, train_size=train_size)
    if max_samples is not None and max_samples > 0 and max_samples < len(dataset):
        original_count = len(dataset)
        generator = torch.Generator().manual_seed(7)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
        _log(f"Fast mode: using {len(dataset)} of {original_count} selected training frames")
    if patch_size is not None:
        dataset = _PatchDataset(dataset, patch_size=patch_size)
        _log(f"Using random {patch_size}×{patch_size} reconstruction patches")
    else:
        _log("Using full-frame reconstruction training")
    if train_size is not None:
        _log(f"Training resolution capped at {train_size}px on the long side")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0, pin_memory=is_cuda)

    checkpoint_path = paths.reconstructor_file
    checkpoint = None
    start_epoch = 1
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        saved_base_ch = int(checkpoint.get("base_ch", base_ch))
        saved_n_res = int(checkpoint.get("n_res", n_res))
        if (saved_base_ch, saved_n_res) != (base_ch, n_res):
            raise ValueError(
                "Cannot resume reconstructor with a different architecture: "
                f"checkpoint=({saved_base_ch}, {saved_n_res}), "
                f"requested=({base_ch}, {n_res})."
            )

    model = VisualReconstructionNet(base_ch=base_ch, n_res=n_res).to(device)
    if checkpoint is not None:
        _load_model_state(model, checkpoint["model"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        _log(f"Resuming reconstructor from epoch {start_epoch} ({checkpoint_path})")
    if is_cuda:
        model = model.to(memory_format=torch.channels_last)

    if compile_model and hasattr(torch, "compile"):
        _log("Compiling reconstructor...")
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)
    if checkpoint is not None:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint and is_cuda:
            scaler.load_state_dict(checkpoint["scaler"])

    n_params = sum(p.numel() for p in model.parameters())
    _log(f"VisualReconstructionNet: {n_params / 1e3:.1f} K params | device={device}")
    _log(f"Dataset: {len(dataset)} samples")

    global_step = 0
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        start_time = time.perf_counter()

        for degraded_batch, target_batch in loader:
            global_step += 1
            degraded_batch = degraded_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)
            if is_cuda:
                degraded_batch = degraded_batch.contiguous(memory_format=torch.channels_last)
                target_batch = target_batch.contiguous(memory_format=torch.channels_last)

            use_edge_loss = edge_loss_every > 0 and global_step % edge_loss_every == 0
            with torch.autocast(device_type="cuda", enabled=is_cuda):
                pred = model(degraded_batch)
                loss = fast_sr_loss(pred, target_batch, edge_loss=use_edge_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if not is_cuda:
                time.sleep(0.005)

            total_loss += float(loss.detach().item())
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, n_batches)
        epoch_time = time.perf_counter() - start_time
        _log(f"Reconstructor epoch {epoch}/{epochs}: loss={avg_loss:.4f} time={epoch_time:.1f}s")
        if progress_fn:
            progress_fn(epoch - start_epoch + 1, epochs)
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()

        # Save every epoch so an interrupted fine-tune can continue without
        # restarting the reconstructor. The world-model checkpoint is never
        # read or modified here.
        paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        ckpt_data = {
            "model": save_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if is_cuda else None,
            "epoch": epoch,
            "global_step": global_step,
            "base_ch": base_ch,
            "n_res": n_res,
            "objective": "video_reconstruction",
        }
        torch.save(ckpt_data, paths.reconstructor_file)

    _log(f"Reconstructor saved → {paths.reconstructor_file}")
    return paths.reconstructor_file


def load_reconstructor(project: str | Path, device: str) -> VisualReconstructionNet | None:
    paths = ProjectPaths(Path(project))
    checkpoint_path = paths.reconstructor_file
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = VisualReconstructionNet(base_ch=int(checkpoint.get("base_ch", 32)), n_res=int(checkpoint.get("n_res", 4))).to(device)
    _load_model_state(model, checkpoint["model"])
    model.eval()
    return model
