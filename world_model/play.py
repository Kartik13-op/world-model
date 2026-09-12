from pathlib import Path
import random

import numpy as np
import pygame
import torch
import torch.nn.functional as F

from .config import ProjectPaths
from .model import WorldModel


def _keys_to_action(keys) -> tuple[torch.Tensor, np.ndarray]:
    strafe = float(keys[pygame.K_a]) - float(keys[pygame.K_d])
    forward = float(keys[pygame.K_s]) - float(keys[pygame.K_w])
    yaw = float(keys[pygame.K_LEFT]) - float(keys[pygame.K_RIGHT])
    pitch = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])
    zoom = forward * 0.25
    raw = np.array([strafe, forward, yaw, pitch, zoom], dtype=np.float32)
    return torch.from_numpy(raw).unsqueeze(0), raw





def _draw_hud(screen, font, action: np.ndarray, tick: int, fps_display: float) -> None:
    labels = []
    if action[1] > 0.05:
        labels.append("W forward")
    elif action[1] < -0.05:
        labels.append("S back")
    if action[0] > 0.05:
        labels.append("D right")
    elif action[0] < -0.05:
        labels.append("A left")
    if action[2] > 0.05:
        labels.append("→ yaw right")
    elif action[2] < -0.05:
        labels.append("← yaw left")
    if action[3] > 0.05:
        labels.append("↓ pitch down")
    elif action[3] < -0.05:
        labels.append("↑ pitch up")
    text = " | ".join(labels) if labels else "W/S=forward/back, A/D=strafe, ←/→=yaw, ↑/↓=pitch | Esc quit"
    text += f"  |  tick {tick}  |  {fps_display:.0f} fps"
    surface = font.render(text, True, (245, 245, 245), (15, 15, 15))
    screen.blit(surface, (8, 8))

    if not labels:
        hint = "W/S forward/back, A/D strafe, ←/→ yaw, ↑/↓ pitch, R reset, Esc quit"
        surface_hint = font.render(hint, True, (200, 200, 100), (15, 15, 15))
        screen.blit(surface_hint, (8, screen.get_height() - 28))


def _warp_latent(z: torch.Tensor, raw_action: np.ndarray) -> torch.Tensor:
    strafe, forward, yaw, pitch, zoom = raw_action
    scale = 1.0 - (forward * 0.03 + zoom * 0.03)
    tx = -(strafe * 0.03 + yaw * 0.02)
    ty = forward * 0.02 + pitch * 0.02
    shear = yaw * 0.02
    batch = z.shape[0]
    device = z.device
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = float(scale)
    theta[:, 0, 1] = float(shear)
    theta[:, 0, 2] = float(tx)
    theta[:, 1, 0] = float(-shear * 0.25)
    theta[:, 1, 1] = float(scale)
    theta[:, 1, 2] = float(ty)
    grid = F.affine_grid(theta, z.shape, align_corners=False)
    warped = F.grid_sample(z, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return warped


def _match_latent_distribution(z: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    mean = z.mean(dim=(2, 3), keepdim=True)
    std = z.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    normalized = (z - mean) / std
    return normalized * target_std + target_mean


def _init_from_frame(frames, start_frame, model, device):
    if start_frame < 0:
        start = random.randrange(len(frames))
    else:
        start = min(start_frame, len(frames) - 1)
    current_frame = torch.from_numpy(frames[start]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    with torch.no_grad():
        z = model.encode(current_frame)
        running_mean = z.mean(dim=(2, 3), keepdim=True)
        running_std = z.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    return current_frame, z, running_mean, running_std


def play_world_model(
    project: str | Path,
    fps: int = 30,
    device: str | None = None,
    action_strength: float = 1.0,
    latent_damping: float = 1.0,
    start_frame: int = -1,
    physics_blend: float = 0.92,
    normalize_latent: bool = True,
) -> None:
    paths = ProjectPaths(Path(project))
    if not paths.model_file.exists():
        raise RuntimeError("Train the model before playing.")
    if not paths.frames_file.exists():
        raise RuntimeError("Missing processed frames. Run preprocess first.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(paths.model_file, map_location=device, weights_only=True)
    model = WorldModel(latent_channels=int(ckpt["latent_channels"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    frames = np.load(paths.frames_file)
    current_frame, z, running_mean, running_std = _init_from_frame(frames, start_frame, model, device)

    pygame.init()
    pygame.key.set_repeat(1, 1)
    h, w = frames.shape[1], frames.shape[2]
    scale = max(1, min(6, 768 // max(h, w)))
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("AI World Model")
    font = pygame.font.SysFont("Segoe UI", max(14, 14 * scale // 2))
    clock = pygame.time.Clock()
    ema_decay = 0.95

    running = True
    tick = 0
    with torch.no_grad():
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        current_frame, z, running_mean, running_std = _init_from_frame(
                            frames, -1, model, device
                        )
                        tick = 0

            keys = pygame.key.get_pressed()
            action_tensor, raw_action = _keys_to_action(keys)
            action = (action_tensor * action_strength).to(device)

            is_idle = bool(np.abs(raw_action).sum() < 0.01)

            if not is_idle:
                # Direct latent warping prevents compound pixel-reencoding blur!
                z_physics = _warp_latent(z, raw_action)
                
                # Neural transition + delta clamping to suppress early-epoch hallucinated globs
                raw_z_pred = model.step(z, action)
                delta = torch.clamp(raw_z_pred - z, -0.08, 0.08)
                z_pred = z + delta

                z = physics_blend * z_physics + (1.0 - physics_blend) * z_pred

                if latent_damping < 1.0:
                    z = z * latent_damping

                if normalize_latent:
                    z = _match_latent_distribution(z, running_mean, running_std)
                    running_mean = ema_decay * running_mean + (1.0 - ema_decay) * z.mean(dim=(2, 3), keepdim=True)
                    running_std = ema_decay * running_std + (1.0 - ema_decay) * z.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)

                current_frame = model.decode(z).clamp(0, 1)

            tick += 1

            # Continuous Real-time Bicubic Upscaling (2x Spatial Upsampling)
            upscaled_frame = F.interpolate(current_frame, scale_factor=2.0, mode="bicubic", align_corners=False).clamp(0, 1)

            frame_np = upscaled_frame[0].permute(1, 2, 0).cpu().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)

            surface = pygame.surfarray.make_surface(np.swapaxes(frame_np, 0, 1))
            display_w = w * scale
            display_h = h * scale
            surface = pygame.transform.smoothscale(surface, (display_w, display_h))
            screen.blit(surface, (0, 0))
            _draw_hud(screen, font, raw_action, tick, clock.get_fps())
            pygame.display.flip()
            clock.tick(fps)

    pygame.quit()
