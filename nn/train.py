import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from nn.model import ChessNet


class ChessDataset(Dataset):
    def __init__(self, path: str):
        data = torch.load(path, weights_only=True)
        self.tensors = data["tensors"]
        self.outcomes = data["outcomes"].unsqueeze(1)

    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        return self.tensors[idx], self.outcomes[idx]


def load_dataset(data_path: str) -> Dataset:
    """Accept a single .pt file or a glob pattern (e.g. 'nn/data/*.pt')."""
    paths = sorted(glob.glob(data_path)) if "*" in data_path else [data_path]
    if not paths:
        raise FileNotFoundError(f"No data files matched: {data_path}")
    datasets = [ChessDataset(p) for p in paths]
    print(f"Loaded {sum(len(d) for d in datasets):,} positions from {len(paths)} file(s)")
    return ConcatDataset(datasets)


def train(
    data_path: str,
    checkpoint_dir: str = "nn/checkpoints",
    resume: str = None,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str = None,
) -> ChessNet:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = load_dataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device == "cuda"))

    model = ChessNet().to(device)
    if resume:
        model.load_state_dict(torch.load(resume, map_location=device, weights_only=True))
        print(f"Resumed from {resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        avg_loss = total_loss / len(dataset)
        ckpt = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.6f}  saved {ckpt}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Path to .pt dataset or glob (e.g. 'nn/data/*.pt')")
    parser.add_argument("--checkpoint-dir", default="nn/checkpoints")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        args.data,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
