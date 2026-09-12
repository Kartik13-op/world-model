"""
Lightweight 2× super-resolution upscaler.

Architecture: Residual encoder → pixel-shuffle 2× head.
Tiny parameter count → trains in minutes on CPU, seconds on GPU.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ProjectPaths


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class UpscalerNet(nn.Module):
    """
    Takes [B, 3, H, W] in [0,1] → outputs [B, 3, 2H, 2W] in [0,1].
    Very small: ~300 K params. Trains fast.
    """

    def __init__(self, base_ch: int = 32, n_res: int = 4) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, base_ch, 3, padding=1)
        self.body = nn.Sequential(*[_ResBlock(base_ch) for _ in range(n_res)])
        # Pixel-shuffle: output base_ch*4 channels → shuffle → base_ch channels
        self.upsample = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),          # → base_ch channels at 2× resolution
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.tail = nn.Conv2d(base_ch, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.leaky_relu(self.head(x), 0.1, inplace=False)
        feat = self.body(feat)
        feat = self.upsample(feat)
        out = torch.sigmoid(self.tail(feat))
        return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _HRLRDataset(Dataset):
    """
    Loads full-res frames from frames.npy.
    Returns (lr_frame, hr_frame) pairs; LR is generated on-the-fly by
    downsampling 2× with bilinear, giving the model a clean target.
    """

    def __init__(self, frames_file: Path) -> None:
        frames = np.load(frames_file)              # [N, H, W, 3]  uint8
        self.hr = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0

    def __len__(self) -> int:
        return len(self.hr)

    def __getitem__(self, idx: int):
        hr = self.hr[idx]                          # [3, H, W]
        lr = F.interpolate(
            hr.unsqueeze(0), scale_factor=0.5,
            mode="bilinear", align_corners=False,
        ).squeeze(0)                               # [3, H/2, W/2]
        return lr, hr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_upscaler(
    project: str | Path,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 2e-4,
    base_ch: int = 32,
    n_res: int = 4,
    device: str | None = None,
    log_fn=None,
) -> Path:
    """
    Train UpscalerNet on the preprocessed frames and save checkpoint.

    Returns the path to the saved checkpoint.
    """
    paths = ProjectPaths(Path(project))
    if not paths.frames_file.exists():
        raise RuntimeError("Run preprocess first — frames.npy not found.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if "cpu" in device.lower():
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    dataset = _HRLRDataset(paths.frames_file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )

    model = UpscalerNet(base_ch=base_ch, n_res=n_res).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=lr * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=("cuda" in device))

    n_params = sum(p.numel() for p in model.parameters())
    _log(f"UpscalerNet: {n_params/1e3:.1f} K params | device={device}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for lr_batch, hr_batch in loader:
            lr_batch = lr_batch.to(device, non_blocking=True)
            hr_batch = hr_batch.to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.split(":")[0],
                enabled=("cuda" in device),
            ):
                pred = model(lr_batch)
                # L1 pixel loss + gradient loss (keeps edges sharp)
                loss_pixel = F.l1_loss(pred, hr_batch)
                gx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
                gx_hr   = hr_batch[:, :, :, 1:] - hr_batch[:, :, :, :-1]
                gy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
                gy_hr   = hr_batch[:, :, 1:, :] - hr_batch[:, :, :-1, :]
                loss_grad = F.l1_loss(gx_pred, gx_hr) + F.l1_loss(gy_pred, gy_hr)
                loss = loss_pixel + 0.5 * loss_grad

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg = total_loss / max(1, n_batches)
        _log(f"Upscaler epoch {epoch}/{epochs}: loss={avg:.4f}")

        gc.collect()
        if "cuda" in device:
            torch.cuda.empty_cache()

    # Save
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ckpt_data = {
        "model": model.state_dict(),
        "base_ch": base_ch,
        "n_res": n_res,
    }
    torch.save(ckpt_data, paths.upscaler_file)
    _log(f"Upscaler saved → {paths.upscaler_file}")
    return paths.upscaler_file


def load_upscaler(project: str | Path, device: str) -> UpscalerNet | None:
    """Load upscaler checkpoint if it exists, else return None."""
    paths = ProjectPaths(Path(project))
    if not paths.upscaler_file.exists():
        return None
    ckpt = torch.load(paths.upscaler_file, map_location=device, weights_only=True)
    model = UpscalerNet(
        base_ch=int(ckpt.get("base_ch", 32)),
        n_res=int(ckpt.get("n_res", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
