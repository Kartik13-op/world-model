from pathlib import Path
import random

import numpy as np
import pygame
import torch
import torch.nn.functional as F

from .config import ProjectPaths
from .model import WorldModel


def _keys_to_action(keys) -> tuple[torch.Tensor, np.ndarray]:
    strafe = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
    forward = float(keys[pygame.K_w] or keys[pygame.K_UP]) - float(keys[pygame.K_s] or keys[pygame.K_DOWN])
    yaw = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
    zoom = forward * 0.25
    raw = np.array([strafe, forward, yaw, zoom], dtype=np.float32)
    return torch.from_numpy(raw).unsqueeze(0), raw


def _draw_hud(screen, font, action: np.ndarray, tick: int, fps_display: float) -> None:
    labels = []
    if action[1] > 0.05:
        labels.append("W/UP forward")
    elif action[1] < -0.05:
        labels.append("S/DOWN back")
    if action[0] > 0.05:
        labels.append("D right")
    elif action[0] < -0.05:
        labels.append("A left")
    if action[2] > 0.05:
        labels.append("RIGHT turn")
    elif action[2] < -0.05:
        labels.append("LEFT turn")
    text = " | ".join(labels) if labels else "WASD + arrows to move  |  Esc to quit"
    text += f"  |  tick {tick}  |  {fps_display:.0f} fps"
    surface = font.render(text, True, (245, 245, 245), (15, 15, 15))
    screen.blit(surface, (8, 8))

    if not labels:
        hint = "WASD / arrows to drive the physical world model transition!"
        surface_hint = font.render(hint, True, (200, 200, 100), (15, 15, 15))
        screen.blit(surface_hint, (8, screen.get_height() - 28))


def _warp_image_by_camera_motion(img: torch.Tensor, raw_action: np.ndarray) -> torch.Tensor:
    strafe, forward, yaw, zoom = raw_action
    scale = 1.0 - (forward * 0.04 + zoom * 0.04)
    tx = -(strafe * 0.04 + yaw * 0.03)
    ty = forward * 0.03
    shear = yaw * 0.03
    batch = 1
    device = img.device
    theta = torch.zeros(batch, 2, 3, device=device)
    theta[:, 0, 0] = float(scale)
    theta[:, 0, 1] = float(shear)
    theta[:, 0, 2] = float(tx)
    theta[:, 1, 0] = float(-shear * 0.25)
    theta[:, 1, 1] = float(scale)
    theta[:, 1, 2] = float(ty)
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    warped = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return warped


def _match_latent_distribution(z: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    mean = z.mean(dim=(2, 3), keepdim=True)
    std = z.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    normalized = (z - mean) / std
    return normalized * target_std + target_mean


def play_world_model(
    project: str | Path,
    fps: int = 30,
    device: str | None = None,
    action_strength: float = 1.0,
    latent_damping: float = 1.0,
) -> None:
    paths = ProjectPaths(Path(project))
    if not paths.model_file.exists():
        raise RuntimeError("Train the model before playing.")
    if not paths.frames_file.exists():
        raise RuntimeError("Missing processed frames. Run preprocess first.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(paths.model_file, map_location=device)
    model = WorldModel(latent_channels=int(ckpt["latent_channels"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    frames = np.load(paths.frames_file)
    
    start = random.randrange(len(frames))
    current_frame = torch.from_numpy(frames[start]).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        z = model.encode(current_frame)
        starting_mean = z.mean(dim=(2, 3), keepdim=True)
        starting_std = z.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)

    pygame.init()
    pygame.key.set_repeat(1, 1)
    h, w = frames.shape[1], frames.shape[2]
    scale = max(1, min(6, 768 // max(h, w)))
    screen = pygame.display.set_mode((w * scale, h * scale))
    pygame.display.set_caption("AI World Model")
    font = pygame.font.SysFont("Segoe UI", max(14, 14 * scale // 2))
    clock = pygame.time.Clock()

    running = True
    tick = 0
    with torch.no_grad():
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            action_tensor, raw_action = _keys_to_action(keys)
            action = (action_tensor * action_strength).to(device)

            # 1. Warp pixel image (sharp, grounded in real pixels)
            warped_frame = _warp_image_by_camera_motion(current_frame, raw_action)

            # 2. Encode the warped pixels → physics latent
            z_physics = model.encode(warped_frame)

            # 3. Neural transition prediction
            z_pred = model.step(z, action)

            # 4. Blend: physics keeps it stable, prediction adds learned dynamics
            z = 0.85 * z_physics + 0.15 * z_pred

            if latent_damping < 1.0:
                z = z * latent_damping

            z = _match_latent_distribution(z, starting_mean, starting_std)
            tick += 1

            # 5. Decode → next frame
            current_frame = model.decode(z).clamp(0, 1)

            frame_np = current_frame[0].permute(1, 2, 0).cpu().numpy()
            frame_np = (frame_np * 255).astype(np.uint8)

            surface = pygame.surfarray.make_surface(np.swapaxes(frame_np, 0, 1))
            if scale != 1:
                surface = pygame.transform.scale(surface, (w * scale, h * scale))
            screen.blit(surface, (0, 0))
            _draw_hud(screen, font, raw_action, tick, clock.get_fps())
            pygame.display.flip()
            clock.tick(fps)

    pygame.quit()
