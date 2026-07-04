from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def frames_file(self) -> Path:
        return self.processed_dir / "frames.npy"

    @property
    def actions_file(self) -> Path:
        return self.processed_dir / "actions.npy"

    @property
    def latents_file(self) -> Path:
        return self.processed_dir / "latents.npy"

    @property
    def model_file(self) -> Path:
        return self.checkpoints_dir / "world_model.pt"

    @property
    def meta_file(self) -> Path:
        return self.processed_dir / "meta.json"


def create_project(root: str | Path) -> ProjectPaths:
    paths = ProjectPaths(Path(root))
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return paths
