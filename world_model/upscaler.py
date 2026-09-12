"""
Ultra-lightweight 2× super-resolution upscaler for AI World Model.

Architecture:
    LR
      ↓
    Bilinear 2× baseline
      ↓
    Tiny residual CNN
      ↓
    learned detail correction
      ↓
    HR

Designed for:
    - extremely fast training
    - low VRAM usage
    - CPU/GPU friendly operation
    - concurrent training with a world model
    - preserving meaningful structures when the world model
      starts producing blurry / colour-glob outputs

The network learns the HIGH-FREQUENCY CORRECTION rather than
reconstructing the entire image from scratch.
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


# ============================================================================
# Performance configuration
# ============================================================================

if torch.cuda.is_available():
    # Enable TensorFloat-32 on supported NVIDIA GPUs.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Let cuDNN select the fastest convolution algorithms.
    torch.backends.cudnn.benchmark = True


# ============================================================================
# Tiny residual block
# ============================================================================

class _ResBlock(nn.Module):
    """
    Extremely small residual block.

    Compared with the original version:
        4 blocks → 2 blocks by default

    This is intentional. The upscaler is a helper network, not
    the primary world model.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            ch,
            ch,
            kernel_size=3,
            padding=1,
        )

        self.act = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            ch,
            ch,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


# ============================================================================
# Upscaler
# ============================================================================

class UpscalerNet(nn.Module):
    """
    Lightweight 2× residual super-resolution network.

    Input:
        [B, 3, H, W]

    Output:
        [B, 3, 2H, 2W]

    The network does NOT reconstruct the entire HR image.

    Instead:

        bilinear(LR) + learned_detail

    This makes learning considerably easier and faster.
    """

    def __init__(
        self,
        base_ch: int = 24,
        n_res: int = 2,
    ) -> None:

        super().__init__()

        # Feature extraction
        self.head = nn.Conv2d(
            3,
            base_ch,
            kernel_size=3,
            padding=1,
        )

        # Tiny residual body
        self.body = nn.Sequential(
            *[
                _ResBlock(base_ch)
                for _ in range(n_res)
            ]
        )

        # 2× pixel shuffle
        self.up = nn.Sequential(
            nn.Conv2d(
                base_ch,
                base_ch * 4,
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        # Predict RGB residual/detail
        self.tail = nn.Conv2d(
            base_ch,
            3,
            kernel_size=3,
            padding=1,
        )

        # Start with almost-zero correction.
        #
        # This means the freshly initialized network behaves
        # approximately like a normal bilinear upscaler.
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Cheap baseline.
        baseline = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        # Learn only the missing information.
        feat = F.relu(
            self.head(x),
            inplace=True,
        )

        feat = self.body(feat)
        feat = self.up(feat)

        residual = self.tail(feat)

        # Residual correction.
        out = baseline + residual

        # Keep output valid.
        return out.clamp_(0.0, 1.0)


# ============================================================================
# Dataset
# ============================================================================

class _HRLRDataset(Dataset):
    """
    Dataset for super-resolution training.

    The important optimization here is:

        LR images are generated ONCE.

    The original implementation generated LR images inside
    __getitem__, which means interpolation was repeatedly performed
    throughout training.
    """

    def __init__(
        self,
        frames_file: Path,
    ) -> None:

        frames = np.load(
            frames_file,
            mmap_mode="r",
        )

        # Convert the complete dataset once.
        #
        # If frames.npy is reasonably sized, this is considerably
        # faster than repeatedly converting individual frames.
        hr = torch.from_numpy(
            np.asarray(frames)
        ).permute(
            0,
            3,
            1,
            2,
        ).float()

        hr.div_(255.0)

        self.hr = hr.contiguous()

        # Precompute LR dataset ONCE.
        #
        # This removes interpolation from the training loop.
        with torch.no_grad():

            self.lr = F.interpolate(
                self.hr,
                scale_factor=0.5,
                mode="bilinear",
                align_corners=False,
            )

        self.lr = self.lr.contiguous()

    def __len__(self) -> int:
        return self.hr.shape[0]

    def __getitem__(self, idx: int):

        return (
            self.lr[idx],
            self.hr[idx],
        )


# ============================================================================
# Optional patch dataset
# ============================================================================

class _PatchDataset(Dataset):
    """
    Random-crop wrapper around the HR/LR dataset.

    This is MUCH faster when your frames are large.

    Example:

        HR: 256×256
        LR: 128×128

    with patch_size=64:

        LR patch: 64×64
        HR patch: 128×128

    The model processes only 1/4 of the LR pixels instead of
    the entire frame.
    """

    def __init__(
        self,
        base: _HRLRDataset,
        patch_size: int = 64,
    ) -> None:

        self.base = base
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):

        lr, hr = self.base[idx]

        _, h, w = lr.shape

        ps = min(
            self.patch_size,
            h,
            w,
        )

        if h == ps:
            y = 0
        else:
            y = torch.randint(
                0,
                h - ps + 1,
                (),
            ).item()

        if w == ps:
            x = 0
        else:
            x = torch.randint(
                0,
                w - ps + 1,
                (),
            ).item()

        lr_patch = lr[
            :,
            y:y + ps,
            x:x + ps,
        ]

        # HR coordinates are exactly 2×.
        hy = y * 2
        hx = x * 2
        hps = ps * 2

        hr_patch = hr[
            :,
            hy:hy + hps,
            hx:hx + hps,
        ]

        return lr_patch, hr_patch


# ============================================================================
# Loss
# ============================================================================

def fast_sr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    edge_loss: bool = False,
) -> torch.Tensor:
    """
    Fast reconstruction loss.

    Default:
        L1 only.

    Optional:
        lightweight gradient/edge loss.

    Edge loss should NOT necessarily run on every batch.
    """

    loss = F.l1_loss(
        pred,
        target,
    )

    if edge_loss:

        pred_x = pred[..., :, 1:] - pred[..., :, :-1]
        target_x = target[..., :, 1:] - target[..., :, :-1]

        pred_y = pred[..., 1:, :] - pred[..., :-1, :]
        target_y = target[..., 1:, :] - target[..., :-1, :]

        loss += 0.25 * (
            F.l1_loss(pred_x, target_x)
            + F.l1_loss(pred_y, target_y)
        )

    return loss


# ============================================================================
# Training
# ============================================================================

def train_upscaler(
    project: str | Path,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 3e-4,
    base_ch: int = 24,
    n_res: int = 2,
    patch_size: int | None = 64,
    device: str | None = None,
    log_fn=None,
    compile_model: bool = False,
    edge_loss_every: int = 8,
) -> Path:

    paths = ProjectPaths(Path(project))

    if not paths.frames_file.exists():
        raise RuntimeError(
            "Run preprocess first — frames.npy not found."
        )

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    is_cuda = device.startswith("cuda")
    is_cpu = not is_cuda

    if is_cpu:

        # Keep the world-model machine responsive.
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)

    # ------------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------------

    def _log(msg: str) -> None:

        print(
            msg,
            flush=True,
        )

        if log_fn:
            log_fn(msg)

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    _log("Loading super-resolution dataset...")

    base_dataset = _HRLRDataset(
        paths.frames_file
    )

    if patch_size is not None:

        dataset = _PatchDataset(
            base_dataset,
            patch_size=patch_size,
        )

        _log(
            f"Using random {patch_size}×{patch_size} LR patches"
        )

    else:

        dataset = base_dataset

        _log("Using full-frame training")

    # ------------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=is_cuda,
    )

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    model = UpscalerNet(
        base_ch=base_ch,
        n_res=n_res,
    ).to(device)

    # Channels-last can improve NVIDIA convolution performance.
    if is_cuda:

        model = model.to(
            memory_format=torch.channels_last
        )

    # ------------------------------------------------------------------------
    # Optional torch.compile
    # ------------------------------------------------------------------------

    if compile_model and hasattr(
        torch,
        "compile",
    ):

        _log("Compiling upscaler...")

        model = torch.compile(
            model,
            mode="reduce-overhead",
        )

    # ------------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.05,
    )

    # ------------------------------------------------------------------------
    # AMP
    # ------------------------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=is_cuda,
    )

    # ------------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------------

    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    _log(
        f"UpscalerNet: {n_params / 1e3:.1f} K params "
        f"| device={device}"
    )

    _log(
        f"Dataset: {len(dataset)} samples"
    )

    total_batches = len(loader)

    # ------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------

    global_step = 0

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        total_loss = 0.0
        n_batches = 0

        start_time = time.perf_counter()

        for lr_batch, hr_batch in loader:

            global_step += 1

            # --------------------------------------------------------------
            # Transfer
            # --------------------------------------------------------------

            if is_cuda:

                lr_batch = lr_batch.to(
                    device,
                    non_blocking=True,
                )

                hr_batch = hr_batch.to(
                    device,
                    non_blocking=True,
                )

                lr_batch = lr_batch.contiguous(
                    memory_format=torch.channels_last
                )

                hr_batch = hr_batch.contiguous(
                    memory_format=torch.channels_last
                )

            else:

                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)

            # --------------------------------------------------------------
            # Edge loss only occasionally
            # --------------------------------------------------------------

            use_edge_loss = (
                edge_loss_every > 0
                and global_step % edge_loss_every == 0
            )

            # --------------------------------------------------------------
            # Forward
            # --------------------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                enabled=is_cuda,
            ):

                pred = model(
                    lr_batch
                )

                loss = fast_sr_loss(
                    pred,
                    hr_batch,
                    edge_loss=use_edge_loss,
                )

            # --------------------------------------------------------------
            # Backward
            # --------------------------------------------------------------

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

            # --------------------------------------------------------------
            # CPU breathing room
            # --------------------------------------------------------------

            if is_cpu:

                time.sleep(
                    0.005
                )

            # --------------------------------------------------------------
            # Logging
            # --------------------------------------------------------------

            loss_value = float(
                loss.detach().item()
            )

            total_loss += loss_value
            n_batches += 1

            if (
                n_batches % 20 == 0
                or n_batches == total_batches
            ):

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                batches_per_sec = (
                    n_batches
                    / max(elapsed, 1e-6)
                )

                _log(
                    f"  [epoch {epoch}/{epochs}] "
                    f"batch {n_batches}/{total_batches} "
                    f"loss={loss_value:.4f} "
                    f"{batches_per_sec:.2f} batch/s"
                )

        # --------------------------------------------------------------------
        # Epoch end
        # --------------------------------------------------------------------

        scheduler.step()

        avg_loss = (
            total_loss
            / max(
                1,
                n_batches,
            )
        )

        epoch_time = (
            time.perf_counter()
            - start_time
        )

        _log(
            f"Upscaler epoch "
            f"{epoch}/{epochs}: "
            f"loss={avg_loss:.4f} "
            f"time={epoch_time:.1f}s"
        )

        gc.collect()

        if is_cuda:

            torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    paths.checkpoints_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ckpt_data = {
        "model": model.state_dict(),
        "base_ch": base_ch,
        "n_res": n_res,
    }

    torch.save(
        ckpt_data,
        paths.upscaler_file,
    )

    _log(
        f"Upscaler saved → "
        f"{paths.upscaler_file}"
    )

    return paths.upscaler_file


# ============================================================================
# Loading
# ============================================================================

def load_upscaler(
    project: str | Path,
    device: str,
) -> UpscalerNet | None:

    paths = ProjectPaths(
        Path(project)
    )

    if not paths.upscaler_file.exists():
        return None

    checkpoint = torch.load(
        paths.upscaler_file,
        map_location=device,
        weights_only=True,
    )

    model = UpscalerNet(
        base_ch=int(
            checkpoint.get(
                "base_ch",
                24,
            )
        ),
        n_res=int(
            checkpoint.get(
                "n_res",
                2,
            )
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    return model