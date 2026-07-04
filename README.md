# AI World Model

A PyTorch-based world model that learns to simulate a video from raw footage. It estimates camera motion via optical flow, trains an autoencoder and a latent-space transition U-Net, and lets you interactively steer through the learned world using keyboard controls.

## How It Works

1. **Preprocessing** — Extracts frames from a video, estimates camera motion between consecutive frames using Farneback optical flow, and saves frame-action pairs.
2. **Training** — Trains an image autoencoder (encoder/decoder) to compress frames into latent feature maps, and a Latent U-Net that predicts the next latent state given the current latent and an action vector. Synthetic camera-warps are used as data augmentation to teach the model how keyboard-driven movement should affect the scene.
3. **Play** — Starts from a random real frame, then runs the latent transition model in a loop. At each step, the current pixel frame is warped by a physical camera transform based on your keyboard input, encoded into the latent space, blended with the learned transition prediction, and decoded into the next frame. This hybrid approach keeps the output stable while incorporating learned scene dynamics.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requirements: `numpy`, `opencv-python`, `pygame`, `torch`, `tqdm`

## Quick Start

```powershell
python main.py init --project my_world
python main.py preprocess --project my_world --video path\to\video.mp4 --size 128 --max-frames 500
python main.py train --project my_world --epochs 50
python main.py play --project my_world
```

Or use the desktop GUI:

```powershell
python main.py gui
```

## Controls (Play Mode)

| Key | Action |
|---|---|
| W / Up Arrow | Move forward |
| S / Down Arrow | Move backward |
| A | Strafe left |
| D | Strafe right |
| Left Arrow | Rotate camera left |
| Right Arrow | Rotate camera right |
| Esc | Quit |

## Project Structure

```
world_model/
├── config.py          — ProjectPaths dataclass, folder creation
├── model.py           — Encoder, Decoder, LatentUNet, WorldModel
├── preprocess.py      — Video frame extraction + optical-flow action estimation
├── train.py           — Autoencoder + transition training with gradient accumulation
│                        and optional synthetic camera augmentation
├── play.py            — Interactive pygame-based playback with physics-bottleneck
│                        latent blending
main.py                — CLI entry point
gui.py                 — Tkinter GUI wrapper
```

## CLI Reference

```
python main.py init --project <name>
  Create project folder structure.

python main.py preprocess --project <name> --video <path> [--size 128] [--max-frames N]
  Convert video to frames and action vectors.

python main.py train --project <name> [--epochs 50] [--batch-size 2] [--accum-steps 4]
                      [--lr 0.001] [--latent-channels 64] [--save-every 10]
                      [--device cpu] [--no-synthetic-controls] [--synthetic-strength 0.12]
  Train the autoencoder and latent transition model.

python main.py play --project <name> [--fps 30] [--action-strength 1.0] [--device cpu]
  Launch the interactive world model.

python main.py gui
  Open the Tkinter desktop interface.
```

## Training Tips

- **Memory:** Default `--batch-size 2 --accum-steps 4` gives an effective batch of 8 while keeping peak memory low. Increase `--batch-size` if you have a powerful GPU.
- **Quality:** Higher `--latent-channels` (64 or 128) captures more visual detail. The synthetic camera augmentations (`--synthetic-strength 0.12`) help the model generalize to novel movements.
- **Checkpoints:** The model is saved every `--save-every` epochs (default 10) to `checkpoints/world_model.pt`. You can interrupt training at any checkpoint and run `play`.

## How the Physics-Bottleneck Hybrid Works

The play loop uses a hybrid transition that prevents the "blurry collapse" common in autoregressive video models:

1. **Physical warp** — The previous pixel frame is warped with a 3D affine camera transform based on your keyboard input (translation, scale, shear).
2. **Encode** — The warped pixels are encoded into the latent space, grounding the representation in real image structure.
3. **Neural prediction** — The learned transition U-Net also predicts the next latent.
4. **Blend** — `z = 0.85 * z_physics + 0.15 * z_pred` — the physics anchor keeps it stable, the neural net adds learned dynamics.
5. **Decode** — The blended latent is decoded into the next frame.

This ensures the output never diverges into noise while staying purely autoregressive (no video-frame scrubbing).

## License

MIT
