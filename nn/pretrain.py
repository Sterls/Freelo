"""
Supervised pretraining from human Lichess games.

How to get data:
    1. Download a monthly PGN from https://database.lichess.org
       e.g. lichess_db_standard_rated_2024-01.pgn.bz2  (~10 GB compressed)
    2. Run this script (it reads .bz2 directly, no manual decompression needed):
       python -m nn.pretrain --pgn lichess_db_standard_rated_2024-01.pgn.bz2

After pretraining, run nn.rl_loop normally — it will resume from the pretrained best.pt.
"""
import argparse
import os
import torch

from nn import storage
from nn.lichess_games import generate_from_pgn
from nn.train import train

CHECKPOINT_DIR = "nn/checkpoints"
DATA_DIR = "nn/data"
PRETRAIN_DATA = os.path.join(DATA_DIR, "pretrain.pt")
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pt")


def run(
    pgn_path: str,
    n_games: int = 50_000,
    min_elo: int = 1500,
    max_moves: int = 60,
    epochs: int = 20,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    storage.pull(CHECKPOINT_DIR, "checkpoints")

    print(f"Parsing up to {n_games:,} games from {pgn_path}  (min_elo={min_elo}) ...")
    generate_from_pgn(
        pgn_source=pgn_path,
        output_path=PRETRAIN_DATA,
        n_games=n_games,
        min_elo=min_elo,
        max_moves=max_moves,
    )

    resume = BEST_CHECKPOINT if os.path.exists(BEST_CHECKPOINT) else None
    print(f"\n{'Resuming from ' + resume if resume else 'Training from scratch'}.")
    print(f"Training for {epochs} epochs on pretrain data ...")
    model = train(
        PRETRAIN_DATA,
        checkpoint_dir=CHECKPOINT_DIR,
        resume=resume,
        epochs=epochs,
    )

    torch.save(model.state_dict(), BEST_CHECKPOINT)
    print(f"\nPretrained model saved → {BEST_CHECKPOINT}")

    storage.push(CHECKPOINT_DIR, "checkpoints")
    storage.push(DATA_DIR, "data")
    print("Done. Run nn.rl_loop to continue with self-play RL.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain ChessNet on human Lichess games.")
    parser.add_argument("--pgn",       required=True, help="Path to .pgn or .pgn.bz2 file")
    parser.add_argument("--n-games",   type=int, default=50_000)
    parser.add_argument("--min-elo",   type=int, default=1500)
    parser.add_argument("--max-moves", type=int, default=60)
    parser.add_argument("--epochs",    type=int, default=20)
    args = parser.parse_args()

    run(
        pgn_path=args.pgn,
        n_games=args.n_games,
        min_elo=args.min_elo,
        max_moves=args.max_moves,
        epochs=args.epochs,
    )
