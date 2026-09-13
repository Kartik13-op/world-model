"""
AI World Model Player
=====================

Runtime pipeline:

    PLAYER INPUT
        |
        v
    latent physics + WorldModel.step()
        |
        v
    WorldModel.decode()
        |
        v
    degraded generated frame
        |
        +--------------------+
        |                    |
     MOVING                IDLE
        |                    |
        v                    v
    Display scaling   Visual reconstructor
        |                    |
        +---------+----------+
                  |
                  v
               Pygame

IMPORTANT:
    The visual reconstructor is DISPLAY-ONLY.

    Its output is NEVER fed back into the world-model latent.
    While moving, the reconstructor is completely bypassed.
"""

from pathlib import Path
import random
import time

import numpy as np
import pygame
import torch
import torch.nn.functional as F

from .config import ProjectPaths
from .model import WorldModel
from .reconstructor import load_reconstructor


# ============================================================================
# Input
# ============================================================================

def _keys_to_action(keys) -> tuple[torch.Tensor, np.ndarray]:
    """
    Convert keyboard input into the 5-dimensional world-model action.

    Action layout:

        [strafe, forward, yaw, pitch, zoom]
    """

    strafe = (
        float(keys[pygame.K_a])
        - float(keys[pygame.K_d])
    )

    # W/Up are forward (+1); S/Down are backward (-1). Use the same
    # forward axis for both keyboard layouts so the controls cannot invert.
    forward = (
        max(float(keys[pygame.K_w]), float(keys[pygame.K_UP]))
        - max(float(keys[pygame.K_s]), float(keys[pygame.K_DOWN]))
    )

    yaw = (
        float(keys[pygame.K_LEFT])
        - float(keys[pygame.K_RIGHT])
    )

    # Up/Down are forward/back controls, not pitch controls.
    pitch = 0.0

    # Zoom follows forward/back movement.
    zoom = forward * 0.25

    raw = np.array(
        [
            strafe,
            forward,
            yaw,
            pitch,
            zoom,
        ],
        dtype=np.float32,
    )

    action_tensor = (
        torch.from_numpy(raw)
        .unsqueeze(0)
    )

    return action_tensor, raw


# ============================================================================
# HUD
# ============================================================================

def _draw_hud(
    screen,
    font,
    action: np.ndarray,
    tick: int,
    fps_display: float,
    sr_time_ms: float,
    sr_difference: float,
    using_sr: bool,
) -> None:
    """
    Draw runtime information.

    The reconstruction diagnostics are deliberately shown so you can immediately
    determine whether the trained reconstructor is actually regenerating the image.
    """

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

    if labels:
        movement_text = " | ".join(labels)
    else:
        movement_text = (
        "W/S or ↑/↓=forward/back, "
            "A/D=strafe, "
            "←/→=yaw, "
            "↑/↓=pitch"
        )

    mode = "RECON" if using_sr else "BICUBIC"

    text = (
        f"{movement_text}"
        f" | tick {tick}"
        f" | {fps_display:.0f} FPS"
        f" | {mode}"
    )

    if using_sr:
        text += (
            f" | recon {sr_time_ms:.1f} ms"
            f" | Δ {sr_difference:.6f}"
        )

    surface = font.render(
        text,
        True,
        (245, 245, 245),
        (15, 15, 15),
    )

    screen.blit(
        surface,
        (8, 8),
    )

    if not labels:

        hint = (
            "W/S forward/back, "
            "A/D strafe, "
            "←/→ yaw, "
            "↑/↓ pitch, "
            "R reset, "
            "Esc quit"
        )

        surface_hint = font.render(
            hint,
            True,
            (200, 200, 100),
            (15, 15, 15),
        )

        screen.blit(
            surface_hint,
            (
                8,
                screen.get_height() - 28,
            ),
        )


# ============================================================================
# Latent physics
# ============================================================================

def _warp_latent(
    z: torch.Tensor,
    raw_action: np.ndarray,
) -> torch.Tensor:
    """
    Apply cheap spatial motion directly to the latent.

    This helps preserve spatial continuity without repeatedly decoding
    and re-encoding images.
    """

    strafe, forward, yaw, pitch, zoom = raw_action

    scale = (
        1.0
        - (
            forward * 0.03
            + zoom * 0.03
        )
    )

    tx = -(
        strafe * 0.03
        + yaw * 0.02
    )

    ty = (
        forward * 0.02
        + pitch * 0.02
    )

    shear = yaw * 0.02

    batch = z.shape[0]
    device = z.device

    theta = torch.zeros(
        batch,
        2,
        3,
        device=device,
        dtype=z.dtype,
    )

    theta[:, 0, 0] = float(scale)
    theta[:, 0, 1] = float(shear)
    theta[:, 0, 2] = float(tx)

    theta[:, 1, 0] = float(-shear * 0.25)
    theta[:, 1, 1] = float(scale)
    theta[:, 1, 2] = float(ty)

    grid = F.affine_grid(
        theta,
        z.shape,
        align_corners=False,
    )

    warped = F.grid_sample(
        z,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    return warped


# ============================================================================
# Latent distribution stabilization
# ============================================================================

def _match_latent_distribution(
    z: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """
    Match the current latent distribution to the initial latent statistics.

    This helps reduce runaway latent drift during long playback sessions.
    """

    mean = z.mean(
        dim=(2, 3),
        keepdim=True,
    )

    std = z.std(
        dim=(2, 3),
        keepdim=True,
    ).clamp_min(1e-4)

    normalized = (
        z - mean
    ) / std

    return (
        normalized
        * target_std
        + target_mean
    )


# ============================================================================
# Initial frame
# ============================================================================

def _init_from_frame(
    frames,
    start_frame,
    model,
    device,
):
    """
    Select a starting frame and encode it into the world-model latent.
    """

    if start_frame < 0:

        start = random.randrange(
            len(frames)
        )

    else:

        start = min(
            start_frame,
            len(frames) - 1,
        )

    current_frame = (
        torch.from_numpy(
            frames[start]
        )
        .permute(
            2,
            0,
            1,
        )
        .unsqueeze(0)
        .float()
        .to(device)
        / 255.0
    )

    with torch.no_grad():

        z = model.encode(
            current_frame
        )

        running_mean = z.mean(
            dim=(2, 3),
            keepdim=True,
        )

        running_std = z.std(
            dim=(2, 3),
            keepdim=True,
        ).clamp_min(1e-4)

    return (
        current_frame,
        z,
        running_mean,
        running_std,
    )


# ============================================================================
# Main player
# ============================================================================

def play_world_model(
    project: str | Path,
    fps: int = 30,
    device: str | None = None,
    action_strength: float = 1.0,
    latent_damping: float = 1.0,
    start_frame: int = -1,
    physics_blend: float = 0.92,
    normalize_latent: bool = True,

    # ------------------------------------------------------------------------
    # Idle-time whole-scene reconstructor
    # ------------------------------------------------------------------------

    use_reconstructor: bool = True,

) -> None:
    """
    Play the trained AI world model.

    Reconstructor behavior:

        MOVING:
        WorldModel → display scaling → display

        IDLE:
            WorldModel → visual reconstructor → display

    The reconstructor is intentionally NOT fed back into the world model.
    """

    paths = ProjectPaths(
        Path(project)
    )

    # ========================================================================
    # Validate project
    # ========================================================================

    if not paths.model_file.exists():
        raise RuntimeError(
            "Train the model before playing."
        )

    if not paths.frames_file.exists():
        raise RuntimeError(
            "Missing processed frames. "
            "Run preprocess first."
        )

    # ========================================================================
    # Device
    # ========================================================================

    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[world] device = {device}",
        flush=True,
    )

    # ========================================================================
    # Load WorldModel
    # ========================================================================

    checkpoint = torch.load(
        paths.model_file,
        map_location=device,
        weights_only=True,
    )

    model = WorldModel(
        latent_channels=int(
            checkpoint["latent_channels"]
        )
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()
    model.requires_grad_(False)

    print(
        "[world] WorldModel loaded.",
        flush=True,
    )

    # ========================================================================
    # Load display reconstructor
    # ========================================================================

    reconstructor = None

    if use_reconstructor:

        reconstructor = load_reconstructor(
            project,
            device,
        )

        if reconstructor is not None:

            reconstructor.eval()
            reconstructor.requires_grad_(False)

            print(
                "[reconstructor] visual checkpoint loaded.",
                flush=True,
            )

            print(
                "[reconstructor] display-only reconstruction runs while idle.",
                flush=True,
            )

        else:

            print(
                "[reconstructor] no checkpoint found.",
                flush=True,
            )

    # ========================================================================
    # Load frames
    # ========================================================================

    frames = np.load(
        paths.frames_file
    )

    h = int(frames.shape[1])
    w = int(frames.shape[2])

    print(
        f"[world] training frame resolution = {w}x{h}",
        flush=True,
    )

    # ========================================================================
    # Initialize world
    # ========================================================================

    (
        current_frame,
        z,
        running_mean,
        running_std,
    ) = _init_from_frame(
        frames,
        start_frame,
        model,
        device,
    )

    print(
        f"[world] current frame tensor = "
        f"{tuple(current_frame.shape)}",
        flush=True,
    )

    # ========================================================================
    # Pygame
    # ========================================================================

    pygame.init()

    pygame.key.set_repeat(
        1,
        1,
    )

    # Keep approximately the same display sizing as the original.
    display_scale = max(
        1,
        min(
            6,
            768 // max(h, w),
        ),
    )

    display_w = w * display_scale
    display_h = h * display_scale

    screen = pygame.display.set_mode(
        (
            display_w,
            display_h,
        )
    )

    pygame.display.set_caption(
        "AI World Model"
    )

    font = pygame.font.SysFont(
        "Segoe UI",
        max(
            14,
            14 * display_scale // 2,
        ),
    )

    clock = pygame.time.Clock()

    # ========================================================================
    # Latent stabilization
    # ========================================================================

    ema_decay = 0.95

    # ========================================================================
    # Runtime state
    # ========================================================================

    running = True
    tick = 0

    # Last reconstructor result.
    #
    # This is not fed into the world model.
    last_recon_frame = None

    # Diagnostics.
    recon_time_ms = 0.0
    recon_difference = 0.0

    # Print reconstructor diagnostics periodically rather than spamming the terminal.
    diagnostic_counter = 0

    # ========================================================================
    # Main loop
    # ========================================================================

    with torch.no_grad():

        while running:

            # =================================================================
            # Events
            # =================================================================

            for event in pygame.event.get():

                if event.type == pygame.QUIT:

                    running = False

                elif event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_ESCAPE:

                        running = False

                    elif event.key == pygame.K_r:

                        print(
                            "[world] Resetting world...",
                            flush=True,
                        )

                        (
                            current_frame,
                            z,
                            running_mean,
                            running_std,
                        ) = _init_from_frame(
                            frames,
                            -1,
                            model,
                            device,
                        )

                        # Do not keep a reconstruction result from the old world.
                        last_sr_frame = None

                        sr_time_ms = 0.0
                        sr_difference = 0.0

                        tick = 0

            # =================================================================
            # Read input
            # =================================================================

            keys = pygame.key.get_pressed()

            action_tensor, raw_action = (
                _keys_to_action(keys)
            )

            action = (
                action_tensor
                * action_strength
            ).to(device)

            # True only when absolutely no meaningful movement is happening.
            is_idle = bool(
                np.abs(raw_action).sum()
                < 0.01
            )

            # =================================================================
            # World-model transition
            # =================================================================

            if not is_idle:

                # -------------------------------------------------------------
                # Direct latent physics
                # -------------------------------------------------------------

                z_physics = _warp_latent(
                    z,
                    raw_action,
                )

                # -------------------------------------------------------------
                # Neural world-model prediction
                # -------------------------------------------------------------

                raw_z_pred = model.step(
                    z,
                    action,
                )

                # -------------------------------------------------------------
                # Clamp neural latent movement.
                #
                # Prevents unstable early-model predictions from causing
                # massive latent jumps.
                # -------------------------------------------------------------

                delta = torch.clamp(
                    raw_z_pred - z,
                    -0.08,
                    0.08,
                )

                z_pred = z + delta

                # -------------------------------------------------------------
                # Physics + learned transition
                # -------------------------------------------------------------

                z = (
                    physics_blend
                    * z_physics
                    + (
                        1.0
                        - physics_blend
                    )
                    * z_pred
                )

                # -------------------------------------------------------------
                # Optional damping
                # -------------------------------------------------------------

                if latent_damping < 1.0:

                    z = (
                        z
                        * latent_damping
                    )

                # -------------------------------------------------------------
                # Latent distribution stabilization
                # -------------------------------------------------------------

                if normalize_latent:

                    z = _match_latent_distribution(
                        z,
                        running_mean,
                        running_std,
                    )

                    running_mean = (
                        ema_decay
                        * running_mean
                        + (
                            1.0
                            - ema_decay
                        )
                        * z.mean(
                            dim=(2, 3),
                            keepdim=True,
                        )
                    )

                    running_std = (
                        ema_decay
                        * running_std
                        + (
                            1.0
                            - ema_decay
                        )
                        * z.std(
                            dim=(2, 3),
                            keepdim=True,
                        ).clamp_min(1e-4)
                    )

                # -------------------------------------------------------------
                # Decode generated world frame
                # -------------------------------------------------------------

                current_frame = (
                    model.decode(z)
                    .clamp(
                        0,
                        1,
                    )
                )

            # =================================================================
            # Tick
            # =================================================================

            tick += 1

            # =================================================================
            # DISPLAY PIPELINE
            # =================================================================
            #
            # IMPORTANT:
            #
            #     current_frame remains untouched.
            #
            #     The reconstructor only receives current_frame and its result
            #     is used for rendering.
            #
            # =================================================================

            if is_idle and reconstructor is not None:

                # =============================================================
                # IDLE → DISPLAY RECONSTRUCTION
                # =============================================================

                recon_start = time.perf_counter()

                recon_frame = (
                    reconstructor(
                        current_frame
                    )
                    .clamp(
                        0,
                        1,
                    )
                )

                recon_time_ms = (
                    time.perf_counter()
                    - recon_start
                ) * 1000.0

                recon_difference = (
                    recon_frame - current_frame
                ).abs().mean().item()

                # -------------------------------------------------------------
                # Use reconstructed result for display only.
                #
                # DO NOT write this back into current_frame or z.
                # -------------------------------------------------------------

                display_frame = recon_frame

                last_recon_frame = recon_frame

                diagnostic_counter += 1

                if diagnostic_counter % 30 == 0:

                    print(
                        "[reconstructor] "
                        f"time={recon_time_ms:.2f} ms | "
                        f"mean_difference="
                        f"{recon_difference:.6f} | "
                        f"input={tuple(current_frame.shape)} | "
                        f"output={tuple(recon_frame.shape)}",
                        flush=True,
                    )

            else:

                # =============================================================
                # MOVING → BICUBIC ONLY
                # =============================================================

                display_frame = F.interpolate(
                    current_frame,
                    scale_factor=2.0,
                    mode="bicubic",
                    align_corners=False,
                ).clamp(
                    0,
                    1,
                )

                # Reset visual reconstruction diagnostic state.
                #
                # The reconstructor isn't being run here.
                recon_time_ms = 0.0
                recon_difference = 0.0

            # =================================================================
            # Convert PyTorch tensor → NumPy
            # =================================================================

            frame_np = (
                display_frame[0]
                .permute(
                    1,
                    2,
                    0,
                )
                .cpu()
                .numpy()
            )

            frame_np = (
                frame_np * 255.0
            ).clip(
                0,
                255,
            ).astype(
                np.uint8
            )

            # =================================================================
            # NumPy → Pygame
            # =================================================================

            surface = (
                pygame.surfarray.make_surface(
                    np.swapaxes(
                        frame_np,
                        0,
                        1,
                    )
                )
            )

            # =================================================================
            # Display scaling
            # =================================================================

            surface = (
                pygame.transform.smoothscale(
                    surface,
                    (
                        display_w,
                        display_h,
                    ),
                )
            )

            screen.blit(
                surface,
                (0, 0),
            )

            # =================================================================
            # HUD
            # =================================================================

            _draw_hud(
                screen,
                font,
                raw_action,
                tick,
                clock.get_fps(),
                recon_time_ms,
                recon_difference,
                is_idle and reconstructor is not None,
            )

            # =================================================================
            # Present
            # =================================================================

            pygame.display.flip()

            # =================================================================
            # FPS limiter
            # =================================================================

            clock.tick(
                fps
            )

    pygame.quit()
