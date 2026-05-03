import os
import numpy as np
import torch
import chess

from engine.features import board_to_tensor
from engine.search import best_move, evaluate

MAX_MOVES = 200


def make_nn_eval(model, device="cpu"):
    """Wrap a ChessNet into an eval_fn compatible with alphabeta."""
    def eval_fn(board: chess.Board) -> float:
        x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(x).item()
    return eval_fn


def play_game(eval_fn=evaluate, depth=1):
    """
    Play one self-play game. Returns (tensors, outcome).
    outcome: +1 white wins, -1 black wins, 0 draw.
    """
    board = chess.Board()
    tensors = []

    for _ in range(MAX_MOVES):
        if board.is_game_over():
            break
        tensors.append(board_to_tensor(board))
        move = best_move(board, depth=depth, eval_fn=eval_fn)
        if move is None:
            break
        board.push(move)

    result = board.result()
    outcome = 1.0 if result == "1-0" else -1.0 if result == "0-1" else 0.0

    return tensors, outcome


def generate(n_games, output_path, eval_fn=evaluate, depth=1):
    """
    Play n_games self-play games and save the dataset to output_path.
    Dataset: dict with 'tensors' (N, 13, 8, 8) and 'outcomes' (N,).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    all_tensors = []
    all_outcomes = []

    for i in range(n_games):
        tensors, outcome = play_game(eval_fn=eval_fn, depth=depth)
        all_tensors.extend(tensors)
        all_outcomes.extend([outcome] * len(tensors))
        print(f"  game {i + 1}/{n_games}  moves={len(tensors)}  outcome={outcome:+.0f}")

    dataset = {
        "tensors": torch.tensor(np.array(all_tensors), dtype=torch.uint8),  # binary planes: 4x smaller than float32
        "outcomes": torch.tensor(all_outcomes, dtype=torch.float32),
    }
    torch.save(dataset, output_path)
    print(f"Saved {len(all_outcomes)} positions → {output_path}")
    return dataset


if __name__ == "__main__":
    print("Generating 20 games with static evaluator...")
    generate(20, "nn/data/gen0.pt", depth=1)
