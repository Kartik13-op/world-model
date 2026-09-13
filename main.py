import argparse
from pathlib import Path

from world_model.config import create_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI world model from raw video.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create project folders.")
    init.add_argument("--project", required=True, help="Project folder.")

    prep = sub.add_parser("preprocess", help="Convert video to frames and action vectors.")
    prep.add_argument("--project", required=True)
    prep.add_argument("--video", required=True)
    prep.add_argument("--size", type=int, default=128)
    prep.add_argument("--max-frames", type=int, default=None)

    train = sub.add_parser("train", help="Train the autoencoder and latent U-Net.")
    train.add_argument("--project", required=True)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--accum-steps", type=int, default=4)
    train.add_argument("--save-every", type=int, default=10)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--latent-channels", type=int, default=64)
    train.add_argument("--device", default=None)
    train.add_argument("--no-synthetic-controls", action="store_true")
    train.add_argument("--synthetic-strength", type=float, default=0.12)

    recon = sub.add_parser("reconstructor", help="Train or resume the display-side reconstructor only.")
    recon.add_argument("--project", required=True)
    recon.add_argument("--epochs", type=int, default=6, help="Additional epochs to run.")
    recon.add_argument("--batch-size", type=int, default=8, help="Batch size for fast complete-frame training.")
    recon.add_argument("--lr", type=float, default=3e-4)
    recon.add_argument("--base-ch", type=int, default=32)
    recon.add_argument("--n-res", type=int, default=4)
    recon.add_argument("--patch-size", type=int, default=0, help="Random patch size; 0 trains on complete frames.")
    recon.add_argument("--train-size", type=int, default=256, help="Training long-side cap; 0 keeps original resolution.")
    recon.add_argument("--max-samples", type=int, default=512, help="Frames sampled per epoch; 0 uses every frame.")
    recon.add_argument("--device", default=None)
    recon.add_argument("--resume", action="store_true", help="Continue from reconstructor.pt.")
    recon.add_argument("--compile", action="store_true", help="Use torch.compile when available.")

    verify = sub.add_parser("verify", help="Read-only verification of data and model pipeline.")
    verify.add_argument("--project", required=True)
    verify.add_argument("--device", default="cpu")

    play = sub.add_parser("play", help="Play the trained world model.")
    play.add_argument("--project", required=True)
    play.add_argument("--fps", type=int, default=30)
    play.add_argument("--device", default=None)
    play.add_argument("--action-strength", type=float, default=1.0)
    play.add_argument("--latent-damping", type=float, default=1.0)
    play.add_argument("--start-frame", type=int, default=-1, help="Frame index to start from (-1 = random)")
    play.add_argument("--physics-blend", type=float, default=0.85, help="Weight of physics warp vs neural prediction (0-1)")
    play.add_argument("--no-normalize-latent", action="store_true", help="Disable latent distribution normalization")

    sub.add_parser("gui", help="Open the Tkinter desktop interface.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "init":
        paths = create_project(args.project)
        print(f"Created world model project at {paths.root.resolve()}")
    elif args.command == "preprocess":
        from world_model.preprocess import preprocess_video

        paths = preprocess_video(
            project=args.project,
            video=args.video,
            size=args.size,
            max_frames=args.max_frames,
        )
        print(f"Saved processed data in {paths.processed_dir.resolve()}")
    elif args.command == "train":
        from world_model.train import train_world_model

        ckpt = train_world_model(
            project=args.project,
            epochs=args.epochs,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
            save_every=args.save_every,
            lr=args.lr,
            latent_channels=args.latent_channels,
            device=args.device,
            synthetic_controls=not args.no_synthetic_controls,
            synthetic_strength=args.synthetic_strength,
        )
        print(f"Saved checkpoint to {Path(ckpt).resolve()}")
    elif args.command == "reconstructor":
        from world_model.reconstructor import train_reconstructor

        ckpt = train_reconstructor(
            project=args.project,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            base_ch=args.base_ch,
            n_res=args.n_res,
            patch_size=args.patch_size or None,
            train_size=args.train_size or None,
            max_samples=args.max_samples or None,
            device=args.device,
            compile_model=args.compile,
            resume=args.resume,
        )
        print(f"Saved reconstructor checkpoint to {Path(ckpt).resolve()}")
    elif args.command == "verify":
        from world_model.verify import verify_pipeline

        for message in verify_pipeline(args.project, device=args.device):
            print(f"[verify] {message}")
    elif args.command == "play":
        from world_model.play import play_world_model

        play_world_model(
            project=args.project,
            fps=args.fps,
            device=args.device,
            action_strength=args.action_strength,
            latent_damping=args.latent_damping,
            start_frame=args.start_frame,
            physics_blend=args.physics_blend,
            normalize_latent=not args.no_normalize_latent,
        )
    elif args.command == "gui":
        from gui import main as gui_main

        gui_main()


if __name__ == "__main__":
    main()
