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

    play = sub.add_parser("play", help="Play the trained world model.")
    play.add_argument("--project", required=True)
    play.add_argument("--fps", type=int, default=30)
    play.add_argument("--device", default=None)
    play.add_argument("--action-strength", type=float, default=1.0)
    play.add_argument("--latent-damping", type=float, default=1.0)

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
    elif args.command == "play":
        from world_model.play import play_world_model

        play_world_model(
            project=args.project,
            fps=args.fps,
            device=args.device,
            action_strength=args.action_strength,
            latent_damping=args.latent_damping,
        )
    elif args.command == "gui":
        from gui import main as gui_main

        gui_main()


if __name__ == "__main__":
    main()
