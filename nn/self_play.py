import os
import random
import numpy as np
import torch
import chess

from engine.features import board_to_tensor
from engine.search import best_move, evaluate
from nn.model import make_eval_fn  # noqa: F401 — re-exported for rl_loop convenience

MAX_MOVES = 200


def _pick_move(board, eval_fn, depth, epsilon):
    """Pick a move epsilon-greedily: random with prob epsilon, best otherwise."""
    if epsilon > 0 and random.random() < epsilon:
        return random.choice(list(board.legal_moves))
    return best_move(board, depth=depth, eval_fn=eval_fn)


def play_game(eval_fn=evaluate, depth=1, epsilon=0.15):
    """
    Play one self-play game. Returns (tensors, outcome).
    outcome: +1 white wins, -1 black wins, 0 draw.
    epsilon: probability of playing a random move (breaks repetition cycles).
    """
    board = chess.Board()
    tensors = []

    for _ in range(MAX_MOVES):
        if board.is_game_over():
            break
        tensors.append(board_to_tensor(board))
        move = _pick_move(board, eval_fn, depth, epsilon)
        if move is None:
            break
        board.push(move)

    result = board.result()
    outcome = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0

    return tensors, outcome


def play_game_vs(eval_fn_white, eval_fn_black, depth=1):
    """
    Play a game with separate evaluators for each side (no epsilon — used for pit).
    Returns outcome from white's perspective: +1, -1, or 0.
    """
    board = chess.Board()

    for _ in range(MAX_MOVES):
        if board.is_game_over():
            break
        eval_fn = eval_fn_white if board.turn == chess.WHITE else eval_fn_black
        move = best_move(board, depth=depth, eval_fn=eval_fn)
        if move is None:
            break
        board.push(move)

    result = board.result()
    return 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0


def generate(n_games, output_path, eval_fn=evaluate, depth=1, epsilon=0.15):
    """
    Play n_games self-play games and save the dataset to output_path.
    Dataset: dict with 'tensors' (N, 13, 8, 8) and 'outcomes' (N,).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    all_tensors = []
    all_outcomes = []

    for i in range(n_games):
        tensors, outcome = play_game(eval_fn=eval_fn, depth=depth, epsilon=epsilon)
        all_tensors.extend(tensors)
        all_outcomes.extend([outcome] * len(tensors))
        print(f"  game {i + 1}/{n_games}  moves={len(tensors)}  outcome={outcome:+.0f}")

    dataset = {
        "tensors": torch.tensor(np.array(all_tensors), dtype=torch.uint8),
        "outcomes": torch.tensor(all_outcomes, dtype=torch.float32),
    }
    torch.save(dataset, output_path)
    print(f"Saved {len(all_outcomes)} positions → {output_path}")
    return dataset


if __name__ == "__main__":
    print("Generating 20 games with static evaluator...")
    generate(20, "nn/data/gen0.pt", depth=1)
