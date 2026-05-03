import torch
import torch.nn as nn


class ChessNet(nn.Module):
    """
    Input:  (batch, 13, 8, 8) board tensor from engine.features.board_to_tensor
    Output: (batch, 1) scalar in [-1, 1]; +1 = white wins, -1 = black wins
    """

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


def load(path: str, device: str = "cpu") -> ChessNet:
    model = ChessNet()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def make_eval_fn(path_or_model, device: str = "cpu"):
    """
    Return an eval_fn(board) -> float for use in alphabeta.
    Accepts a checkpoint path or an already-loaded ChessNet.
    Inference always runs on CPU to avoid per-node GPU transfer overhead.
    """
    import numpy as np
    from engine.features import board_to_tensor

    model = load(path_or_model, device) if isinstance(path_or_model, str) else path_or_model.to(device).eval()

    def eval_fn(board):
        x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(x).item()

    return eval_fn
