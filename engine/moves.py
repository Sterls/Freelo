import chess
import torch

POLICY_SIZE = 64 * 64  # from_square * 64 + to_square


def move_to_idx(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def legal_policy(board: chess.Board, logits: torch.Tensor) -> dict:
    """
    Softmax over legal moves only. Groups moves sharing the same (from, to)
    index (promotions) — queen promotion wins ties.
    Returns {chess.Move: float}.
    """
    idx_to_moves: dict[int, list] = {}
    for move in board.legal_moves:
        idx_to_moves.setdefault(move_to_idx(move), []).append(move)

    if not idx_to_moves:
        return {}

    idxs = list(idx_to_moves.keys())
    probs = torch.softmax(logits[torch.tensor(idxs)], dim=0).tolist()

    result = {}
    for idx, prob in zip(idxs, probs):
        moves = idx_to_moves[idx]
        queen = [m for m in moves if m.promotion == chess.QUEEN]
        result[queen[0] if queen else moves[0]] = prob
    return result


def policy_to_tensor(policy: dict) -> torch.Tensor:
    """Convert {move: prob} to a (POLICY_SIZE,) float tensor."""
    t = torch.zeros(POLICY_SIZE)
    for move, prob in policy.items():
        t[move_to_idx(move)] += prob
    return t
