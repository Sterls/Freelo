"""
Generate supervised training data from human Lichess PGN games.

Policy target: one-hot on the move actually played.
Value target:  game outcome from white's perspective (+1 / -1 / 0).
"""
import bz2
import gzip
import math
import os

import chess
import chess.pgn
import numpy as np
import torch

from engine.features import board_to_tensor
from engine.moves import policy_to_tensor
from engine.search import evaluate as static_eval

STATIC_EVAL_SCALE = 600


def _outcome(game):
    result = game.headers.get("Result", "*")
    if result == "1-0":   return  1.0
    if result == "0-1":   return -1.0
    if result == "1/2-1/2": return 0.0
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
    return source, False  # already a file object, caller manages it


def generate_from_pgn(
    pgn_source,
    output_path: str,
    n_games: int = 10_000,
    min_elo: int = 1500,
    min_moves: int = 10,
    max_moves: int = 60,
) -> dict:
    """
    Parse a PGN file and produce a training dataset.

    pgn_source  — path to a .pgn, .pgn.bz2, or .pgn.gz file, or a file object.
    n_games     — stop after this many accepted games.
    min_elo     — skip games where either player is below this rating.
    min_moves   — skip very short games.
    max_moves   — truncate games longer than this (half-moves).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    pgn_file, should_close = _open_pgn(pgn_source)
    all_tensors, all_policies, all_outcomes = [], [], []
    used = skipped = 0

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

            all_tensors.extend(game_tensors)
            all_policies.extend(game_policies)
            all_outcomes.extend([outcome] * len(game_tensors))
            used += 1

            if used % 500 == 0:
                print(f"  {used}/{n_games} games  ({len(all_outcomes):,} positions)")

    finally:
        if should_close:
            pgn_file.close()

    if not all_tensors:
        raise RuntimeError(
            "No valid games found — check your PGN file and filters.\n"
            "Get data from: https://database.lichess.org"
        )

    dataset = {
        "tensors":  torch.tensor(np.array(all_tensors), dtype=torch.uint8),
        "policies": torch.stack(all_policies),
        "outcomes": torch.tensor(all_outcomes, dtype=torch.float32),
    }
    torch.save(dataset, output_path)
    print(f"Saved {len(all_outcomes):,} positions from {used} games → {output_path}")
    print(f"(Skipped {skipped} games due to filters)")
    return dataset
