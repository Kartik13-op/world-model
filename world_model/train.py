import gc
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import ProjectPaths
from .model import WorldModel


class VideoPairs(Dataset):
    def __init__(self, frames_file: Path, actions_file: Path):
        frames = np.load(frames_file)
        actions = np.load(actions_file)
        self.frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        self.actions = torch.from_numpy(actions).float()

    def __len__(self) -> int:
        return len(self.frames) - 1

    def __getitem__(self, idx: int):
        return self.frames[idx], self.frames[idx + 1], self.actions[idx + 1]


def make_synthetic_camera_batch(images: torch.Tensor, strength: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, h, w = images.shape
    device = images.device

    strafe = torch.empty(batch, device=device).uniform_(-1.0, 1.0)
    forward = torch.empty(batch, device=device).uniform_(-1.0, 1.0)
    yaw = torch.empty(batch, device=device).uniform_(-1.0, 1.0)
    zoom = torch.empty(batch, device=device).uniform_(0.5, 1.5) * forward * 0.3

    scale = 1.0 - zoom * strength
    tx = -(strafe * strength + yaw * strength * 0.65)
    ty = forward * strength * 0.45
    shear = yaw * strength * 0.45

    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = scale
    theta[:, 0, 1] = shear
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = -shear * 0.25
    theta[:, 1, 1] = scale
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, images.shape, align_corners=False)
    warped = F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)
    action = torch.stack([strafe, forward, yaw, zoom], dim=1)
    return warped, action


def _free_memory(device: str) -> None:
    gc.collect()
    if "cuda" in device:
        torch.cuda.empty_cache()


def train_world_model(
    project: str | Path,
    epochs: int = 50,
    batch_size: int = 2,
    accum_steps: int = 4,
    save_every: int = 10,
    lr: float = 1e-3,
    latent_channels: int = 64,
    device: str | None = None,
    synthetic_controls: bool = True,
    synthetic_strength: float = 0.12,
) -> Path:
    paths = ProjectPaths(Path(project))
    if not paths.frames_file.exists() or not paths.actions_file.exists():
        raise RuntimeError("Run preprocess before training.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoPairs(paths.frames_file, paths.actions_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = WorldModel(latent_channels=latent_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr * accum_steps, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * accum_steps * 0.01)
    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0
        opt.zero_grad(set_to_none=True)
        pbar = tqdm(loader, desc=f"epoch {epoch}/{epochs}")

        for step_idx, (current, nxt, action) in enumerate(pbar):
            current = current.to(device, non_blocking=True)
            nxt = nxt.to(device, non_blocking=True)
            action = action.to(device, non_blocking=True)

            z = model.encode(current)
            z_next = model.encode(nxt).detach()
            recon = model.decode(z)
            predicted_z = model.step(z, action)
            predicted_frame = model.decode(predicted_z)

            loss_recon = l1(recon, current)
            loss_latent = mse(predicted_z, z_next)
            loss_frame = l1(predicted_frame, nxt)
            loss = loss_recon + loss_frame + 0.25 * loss_latent

            if synthetic_controls:
                synth_n = max(1, batch_size // 2)
                synth_next, synth_action = make_synthetic_camera_batch(
                    current[:synth_n], synthetic_strength
                )
                z_synth = model.encode(current[:synth_n])
                synth_z_next = model.encode(synth_next).detach()
                synth_predicted_z = model.step(z_synth, synth_action)
                synth_predicted_frame = model.decode(synth_predicted_z)
                loss = loss + l1(synth_predicted_frame, synth_next) + 0.25 * mse(synth_predicted_z, synth_z_next)

            (loss / accum_steps).backward()

            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

            total += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if epoch >= max(2, epochs // 4):
            model.train()
            rollout_loss = 0.0
            n_rollout = 0
            rpbar = tqdm(loader, desc=f"  rollout epoch {epoch}/{epochs}")
            opt.zero_grad(set_to_none=True)
            for ri, (current, nxt, action) in enumerate(rpbar):
                current = current.to(device, non_blocking=True)
                nxt = nxt.to(device, non_blocking=True)
                action = action.to(device, non_blocking=True)

                z = model.encode(current)
                z = model.step(z, action)
                z_target = model.encode(nxt).detach()
                frame_pred = model.decode(z)
                rloss = mse(z, z_target) + l1(frame_pred, nxt)

                z = z.detach()
                z2 = model.step(z, action)
                z2_target = model.encode(current).detach()
                rloss = rloss + mse(z2, z2_target) * 0.5

                if synthetic_controls:
                    synth_n = max(1, current.shape[0] // 2)
                    synth_next, synth_action = make_synthetic_camera_batch(
                        current[:synth_n], synthetic_strength
                    )
                    z_synth = model.encode(current[:synth_n])
                    synth_z_pred = model.step(z_synth, synth_action)
                    synth_z_target = model.encode(synth_next).detach()
                    rloss = rloss + mse(synth_z_pred, synth_z_target) * 0.5

                (rloss / accum_steps).backward()

                if (ri + 1) % accum_steps == 0 or (ri + 1) == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

                rollout_loss += float(rloss.item())
                n_rollout += 1
                rpbar.set_postfix(roloss=f"{rloss.item():.4f}")

            print(f"  rollout loss={rollout_loss / max(1, n_rollout):.4f}")

        scheduler.step()
        print(f"epoch {epoch}: loss={total / max(1, n_batches):.4f}")
        _free_memory(device)

        if epoch % save_every == 0 or epoch == epochs:
            paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "latent_channels": latent_channels,
                    "frame_shape": tuple(dataset.frames.shape[-2:]),
                    "synthetic_controls": synthetic_controls,
                    "epoch": epoch,
                },
                paths.model_file,
            )
            print(f"  checkpoint saved to {paths.model_file}")

    return paths.model_file
