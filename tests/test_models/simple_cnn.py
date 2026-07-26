"""Simple CNN policy for local testing without model downloads."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SimpleCNNPolicy"]


class SimpleCNNPolicy(nn.Module):
    """A minimal CNN policy for testing quantization and export pipelines.

    This model accepts image observations and state vectors,
    processes them through a small CNN, and outputs actions.
    """

    def __init__(
        self,
        image_channels: int = 3,
        image_height: int = 224,
        image_width: int = 224,
        state_dim: int = 2,
        action_dim: int = 2,
    ) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.image_height = image_height
        self.image_width = image_width
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )

        cnn_output_size = 64 * 4 * 4
        self.fusion = nn.Sequential(
            nn.Linear(cnn_output_size + state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        features = self.cnn(image)
        combined = torch.cat([features, state], dim=-1)
        return self.fusion(combined)

    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        image = batch.get("observation.images.front", batch.get("observation.images"))
        state = batch.get("observation.state")
        if image is None or state is None:
            raise ValueError(
                f"Expected 'observation.images' and 'observation.state' in batch. "
                f"Got keys: {list(batch.keys())}"
            )
        return self.forward(image, state)

    def reset(self) -> None:
        pass
