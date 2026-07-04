import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 4, 2, 1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentUNet(nn.Module):
    def __init__(self, latent_channels: int = 64, action_dim: int = 4):
        super().__init__()
        self.action = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.SiLU(),
            nn.Linear(64, latent_channels),
        )
        self.norm_in = nn.GroupNorm(num_groups=min(8, latent_channels * 2), num_channels=latent_channels * 2)
        self.down1 = nn.Conv2d(latent_channels * 2, 128, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, 128)
        self.down2 = nn.Conv2d(128, 128, 4, 2, 1)
        self.mid = nn.Sequential(
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.up = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.norm_up = nn.GroupNorm(8, 256)
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(256, latent_channels, 3, padding=1),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_map = self.action(action).view(action.shape[0], -1, 1, 1).expand_as(z)
        x = torch.cat([z, action_map], dim=1)
        x = self.norm_in(x)
        skip = F.silu(self.down1(x))
        skip = self.norm1(skip)
        low = self.mid(self.down2(skip))
        up = self.up(low)
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        combined = self.norm_up(torch.cat([up, skip], dim=1))
        delta = self.out(combined)
        return z + delta


class WorldModel(nn.Module):
    def __init__(self, latent_channels: int = 64):
        super().__init__()
        self.encoder = Encoder(latent_channels)
        self.decoder = Decoder(latent_channels)
        self.transition = LatentUNet(latent_channels)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def step(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.transition(latents, actions)
