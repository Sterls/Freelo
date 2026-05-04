"""
Generate supervised training data from human Lichess PGN games.

Policy target: one-hot on the move actually played.
Value target:  game outcome from white's perspective (+1 / -1 / 0).

Data is saved in chunks to keep memory usage bounded.
"""
import bz2
import gzip
import os

import chess
import chess.pgn
import numpy as np
import torch

from engine.features import board_to_tensor
from engine.moves import policy_to_tensor


def _outcome(game):
    result = game.headers.get("Result", "*")
    if result == "1-0":       return  1.0
    if result == "0-1":       return -1.0
    if result == "1/2-1/2":  return  0.0
    return None


def _elo_ok(game, min_elo: int) -> bool:
    for key in ("WhiteElo", "BlackElo"):
        try:
            if int(game.headers.get(key, "0")) < min_elo:
                return False
        except ValueError:
            return False
    return True


def _open_pgn(source):
    """Open a PGN file, transparently decompressing .bz2 / .gz / .zst."""
    if isinstance(source, str):
        if source.endswith(".bz2"):
            return bz2.open(source, "rt", encoding="utf-8", errors="ignore"), True
        if source.endswith(".gz"):
            return gzip.open(source, "rt", encoding="utf-8", errors="ignore"), True
        if source.endswith(".zst"):
            import io
            import zstandard as zstd
            ctx = zstd.ZstdDecompressor()
            raw = open(source, "rb")
            return io.TextIOWrapper(ctx.stream_reader(raw), encoding="utf-8", errors="ignore"), True
        return open(source, encoding="utf-8", errors="ignore"), True
    return source, False


def _save_chunk(tensors, policies, outcomes, path):
    dataset = {
        "tensors":  torch.tensor(np.array(tensors), dtype=torch.uint8),
        "policies": torch.stack(policies),
        "outcomes": torch.tensor(outcomes, dtype=torch.float32),
    }
    torch.save(dataset, path)
    return len(outcomes)


def generate_from_pgn(
    pgn_source,
    output_dir: str,
    n_games: int = 10_000,
    min_elo: int = 1500,
    min_moves: int = 10,
    max_moves: int = 60,
    chunk_size: int = 5_000,
) -> str:
    """
    Parse a PGN file and save training data as chunked .pt files.

    Saves chunk_size games per file to keep memory bounded (~4 GB/chunk).
    Returns a glob pattern matching all saved files.

    pgn_source  — path to .pgn / .pgn.bz2 / .pgn.gz / .pgn.zst, or file object.
    output_dir  — directory to write pretrain_NNN.pt chunks into.
    n_games     — total games to accept.
    chunk_size  — games per chunk file (controls peak RAM usage).
    """
    os.makedirs(output_dir, exist_ok=True)

    pgn_file, should_close = _open_pgn(pgn_source)
    chunk_tensors, chunk_policies, chunk_outcomes = [], [], []
    used = skipped = chunk_idx = total_positions = 0

    try:
        while used < n_games:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            outcome = _outcome(game)
            if outcome is None or not _elo_ok(game, min_elo):
                skipped += 1
                continue

            board = game.board()
            game_tensors, game_policies = [], []

            for i, move in enumerate(game.mainline_moves()):
                if i >= max_moves or board.is_game_over():
                    break
                game_tensors.append(board_to_tensor(board))
                game_policies.append(policy_to_tensor({move: 1.0}))
                board.push(move)

            if len(game_tensors) < min_moves:
                skipped += 1
                continue

            chunk_tensors.extend(game_tensors)
            chunk_policies.extend(game_policies)
            chunk_outcomes.extend([outcome] * len(game_tensors))
            used += 1

            if used % 500 == 0:
                print(f"  {used}/{n_games} games  ({total_positions + len(chunk_outcomes):,} positions)")

            # Flush chunk to disk
            if used % chunk_size == 0:
                path = os.path.join(output_dir, f"pretrain_{chunk_idx:03d}.pt")
                n = _save_chunk(chunk_tensors, chunk_policies, chunk_outcomes, path)
                total_positions += n
                print(f"  Saved chunk {chunk_idx} → {path}  ({n:,} positions)")
                chunk_tensors, chunk_policies, chunk_outcomes = [], [], []
                chunk_idx += 1

    finally:
        if should_close:
            pgn_file.close()

    # Save any remaining positions
    if chunk_tensors:
        path = os.path.join(output_dir, f"pretrain_{chunk_idx:03d}.pt")
        n = _save_chunk(chunk_tensors, chunk_policies, chunk_outcomes, path)
        total_positions += n
        print(f"  Saved chunk {chunk_idx} → {path}  ({n:,} positions)")

    if total_positions == 0:
        raise RuntimeError(
            "No valid games found — check your PGN file and filters.\n"
            "Get data from: https://database.lichess.org"
        )

    glob_pattern = os.path.join(output_dir, "pretrain_*.pt")
    print(f"\nTotal: {total_positions:,} positions from {used} games  (skipped {skipped})")
    print(f"Data glob: {glob_pattern}")
    return glob_pattern
