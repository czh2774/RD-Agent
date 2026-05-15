from __future__ import annotations

import torch
from torch import Tensor, nn


class QlibAshareTemporalScoreModel(nn.Module):
    """Reference model for Qlib A-share temporal prediction-score benchmark tasks."""

    def __init__(self, num_features: int, num_timesteps: int = 1, hidden_size: int = 32) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")

        self.num_features = num_features
        self.num_timesteps = num_timesteps
        self.temporal_encoder = nn.GRU(input_size=num_features, hidden_size=hidden_size, batch_first=True)
        self.score_head = nn.Linear(hidden_size, 1)

    def forward(self, feature_window: Tensor) -> Tensor:
        if feature_window.ndim != 3:
            raise ValueError("Qlib A-share reference model expects [batch, datetime_window, feature] input")
        if feature_window.shape[1] != self.num_timesteps:
            raise ValueError("feature_window datetime dimension does not match num_timesteps")
        if feature_window.shape[2] != self.num_features:
            raise ValueError("feature_window feature dimension does not match num_features")

        _, hidden = self.temporal_encoder(feature_window)
        return self.score_head(hidden[-1])


model_cls = QlibAshareTemporalScoreModel


if __name__ == "__main__":
    feature_window = torch.load("feature_window.pt")
    model = QlibAshareTemporalScoreModel(
        num_features=feature_window.size(-1),
        num_timesteps=feature_window.size(1),
    )
    torch.save(model(feature_window), "gt_output.pt")
