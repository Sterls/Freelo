"""
Supervised pretraining from human Lichess games.

How to get data:
    1. Download a monthly PGN from https://database.lichess.org
       e.g. lichess_db_standard_rated_2026-04.pgn.zst
    2. Run this script:
       python -m nn.pretrain --pgn lichess_db_standard_rated_2026-04.pgn.zst

After pretraining, run nn.rl_loop normally — it will resume from the pretrained best.pt.
"""
import argparse
import glob as glob_module
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nn import storage
from nn.lichess_games import generate_from_pgn
from nn.model import ChessNet
from nn.train import ChessDataset

CHECKPOINT_DIR = "nn/checkpoints"
PRETRAIN_DIR = "nn/data/pretrain"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pt")


def _train_streaming(data_glob, checkpoint_dir, resume, epochs, batch_size=512, lr=1e-3):
    """
    Train on chunked data files, loading one at a time to keep RAM bounded.
    Each epoch iterates through all chunk files sequentially.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    chunk_files = sorted(glob_module.glob(data_glob))
    print(f"Found {len(chunk_files)} chunk files")

    model = ChessNet().to(device)
    if resume and os.path.exists(resume):
        model.load_state_dict(torch.load(resume, map_location=device, weights_only=True))
        print(f"Resumed from {resume}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        total_v = total_p = total_n = 0

        for chunk_path in chunk_files:
            dataset = ChessDataset(chunk_path)
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
                pin_memory=(device == "cuda"),
            )
            model.train()
            for x, policy_target, outcome in loader:
                x = x.to(device)
                policy_target = policy_target.to(device)
                outcome = outcome.to(device)

                optimizer.zero_grad()
                value, policy_logits = model(x)
                v_loss = F.mse_loss(value, outcome)
                p_loss = -(F.log_softmax(policy_logits, dim=-1) * policy_target).sum(-1).mean()
                (v_loss + p_loss).backward()
                optimizer.step()

                total_v += v_loss.item() * len(outcome)
                total_p += p_loss.item() * len(outcome)
                total_n += len(outcome)

        ckpt = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save(model.state_dict(), ckpt)
        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"value={total_v/total_n:.4f}  policy={total_p/total_n:.4f}  "
            f"saved {ckpt}"
        )

    return model


def run(
    pgn_path: str = None,
    n_games: int = 50_000,
    min_elo: int = 1500,
    max_moves: int = 60,
    epochs: int = 20,
    chunk_size: int = 5_000,
    skip_parse: bool = False,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PRETRAIN_DIR, exist_ok=True)

    storage.pull(CHECKPOINT_DIR, "checkpoints")

    data_glob = os.path.join(PRETRAIN_DIR, "pretrain_*.pt")

    if not skip_parse:
        print(f"Parsing up to {n_games:,} games from {pgn_path}  (min_elo={min_elo}) ...")
        data_glob = generate_from_pgn(
            pgn_source=pgn_path,
            output_dir=PRETRAIN_DIR,
            n_games=n_games,
            min_elo=min_elo,
            max_moves=max_moves,
            chunk_size=chunk_size,
        )

    resume = BEST_CHECKPOINT if os.path.exists(BEST_CHECKPOINT) else None
    print(f"\n{'Resuming from ' + resume if resume else 'Training from scratch'}.")
    print(f"Training for {epochs} epochs (streaming one chunk at a time) ...")
    model = _train_streaming(
        data_glob,
        checkpoint_dir=CHECKPOINT_DIR,
        resume=resume,
        epochs=epochs,
    )

    torch.save(model.state_dict(), BEST_CHECKPOINT)
    print(f"\nPretrained model saved → {BEST_CHECKPOINT}")

    storage.push(CHECKPOINT_DIR, "checkpoints")
    print("Done. Run nn.rl_loop to continue with self-play RL.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain ChessNet on human Lichess games.")
    parser.add_argument("--pgn",         help="Path to .pgn / .pgn.bz2 / .pgn.zst file")
    parser.add_argument("--n-games",     type=int,   default=50_000)
    parser.add_argument("--min-elo",     type=int,   default=1500)
    parser.add_argument("--max-moves",   type=int,   default=60)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--chunk-size",  type=int,   default=5_000)
    parser.add_argument("--skip-parse",  action="store_true",
                        help="Skip PGN parsing and train on existing chunks in PRETRAIN_DIR")
    args = parser.parse_args()

    run(
        pgn_path=args.pgn,
        n_games=args.n_games,
        min_elo=args.min_elo,
        max_moves=args.max_moves,
        epochs=args.epochs,
        chunk_size=args.chunk_size,
        skip_parse=args.skip_parse,
    )
