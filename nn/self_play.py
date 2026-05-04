import os
import numpy as np
import torch
import chess

from engine.features import board_to_tensor
from engine.moves import policy_to_tensor
from engine import mcts

MAX_MOVES = 200
TEMP_THRESHOLD = 30  # sample proportionally for first N moves, then argmax


def play_game(model, n_simulations: int = 50) -> tuple:
    """
    Play one self-play game using MCTS.
    Returns (tensors, policy_targets, outcome).
    outcome: +1 white wins, -1 black wins, 0 draw.
    """
    board = chess.Board()
    tensors = []
    policy_targets = []

    for move_num in range(MAX_MOVES):
        if board.is_game_over():
            break

        policy = mcts.search(board, model, n_simulations, add_noise=True)
        if not policy:
            break

        tensors.append(board_to_tensor(board))
        policy_targets.append(policy_to_tensor(policy))

        moves = list(policy.keys())
        probs = np.array([policy[m] for m in moves], dtype=np.float64)
        probs /= probs.sum()

        if move_num < TEMP_THRESHOLD:
            move = moves[np.random.choice(len(moves), p=probs)]
        else:
            move = moves[probs.argmax()]

        board.push(move)

    result = board.result()
    outcome = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0
    return tensors, policy_targets, outcome


def play_game_vs(model_white, model_black, n_simulations: int = 25) -> float:
    """Pit two models. Returns outcome from white's perspective (+1/-1/0)."""
    board = chess.Board()

    for _ in range(MAX_MOVES):
        if board.is_game_over():
            break
        model = model_white if board.turn == chess.WHITE else model_black
        move = mcts.best_move(board, model, n_simulations)
        if move is None:
            break
        board.push(move)

    result = board.result()
    return 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0


def generate(n_games: int, output_path: str, model, n_simulations: int = 50):
    """Play n_games self-play games and save dataset."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    all_tensors, all_policies, all_outcomes = [], [], []

    for i in range(n_games):
        tensors, policies, outcome = play_game(model, n_simulations)
        all_tensors.extend(tensors)
        all_policies.extend(policies)
        all_outcomes.extend([outcome] * len(tensors))
        print(f"  game {i + 1}/{n_games}  moves={len(tensors)}  outcome={outcome:+.0f}")

    dataset = {
        "tensors": torch.tensor(np.array(all_tensors), dtype=torch.uint8),
        "policies": torch.stack(all_policies),
        "outcomes": torch.tensor(all_outcomes, dtype=torch.float32),
    }
    torch.save(dataset, output_path)
    print(f"Saved {len(all_outcomes)} positions → {output_path}")
    return dataset
