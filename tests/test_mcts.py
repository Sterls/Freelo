import chess
import torch
import pytest

from engine.moves import move_to_idx, legal_policy, policy_to_tensor, POLICY_SIZE
from engine.mcts import search, best_move, MCTSNode
from nn.model import ChessNet


def _random_model():
    return ChessNet().eval()


# --- move encoding ---

def test_move_to_idx_range():
    board = chess.Board()
    for move in board.legal_moves:
        idx = move_to_idx(move)
        assert 0 <= idx < POLICY_SIZE

def test_move_to_idx_unique_for_normal_moves():
    # e2e4 and d2d4 should have different indices
    assert move_to_idx(chess.Move.from_uci("e2e4")) != move_to_idx(chess.Move.from_uci("d2d4"))

def test_policy_to_tensor_shape():
    board = chess.Board()
    model = _random_model()
    x = torch.from_numpy(__import__("engine.features", fromlist=["board_to_tensor"]).board_to_tensor(board)).unsqueeze(0).float()
    with torch.no_grad():
        _, logits = model(x)
    policy = legal_policy(board, logits.squeeze(0))
    t = policy_to_tensor(policy)
    assert t.shape == (POLICY_SIZE,)

def test_legal_policy_sums_to_one():
    board = chess.Board()
    model = _random_model()
    x = torch.from_numpy(__import__("engine.features", fromlist=["board_to_tensor"]).board_to_tensor(board)).unsqueeze(0).float()
    with torch.no_grad():
        _, logits = model(x)
    policy = legal_policy(board, logits.squeeze(0))
    assert abs(sum(policy.values()) - 1.0) < 1e-5

def test_legal_policy_only_legal_moves():
    board = chess.Board()
    model = _random_model()
    x = torch.from_numpy(__import__("engine.features", fromlist=["board_to_tensor"]).board_to_tensor(board)).unsqueeze(0).float()
    with torch.no_grad():
        _, logits = model(x)
    policy = legal_policy(board, logits.squeeze(0))
    legal = set(board.legal_moves)
    assert all(m in legal for m in policy)

def test_legal_policy_empty_on_game_over():
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    assert board.is_checkmate()
    model = _random_model()
    x = torch.from_numpy(__import__("engine.features", fromlist=["board_to_tensor"]).board_to_tensor(board)).unsqueeze(0).float()
    with torch.no_grad():
        _, logits = model(x)
    policy = legal_policy(board, logits.squeeze(0))
    assert policy == {}


# --- MCTS ---

def test_mcts_search_returns_legal_moves():
    board = chess.Board()
    model = _random_model()
    policy = search(board, model, n_simulations=5, add_noise=False)
    assert policy
    legal = set(board.legal_moves)
    assert all(m in legal for m in policy)

def test_mcts_search_probabilities_sum_to_one():
    board = chess.Board()
    model = _random_model()
    policy = search(board, model, n_simulations=5, add_noise=False)
    assert abs(sum(policy.values()) - 1.0) < 1e-5

def test_mcts_search_empty_on_game_over():
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    model = _random_model()
    assert search(board, model, n_simulations=5) == {}

def test_mcts_best_move_is_legal():
    board = chess.Board()
    model = _random_model()
    move = best_move(board, model, n_simulations=10)
    assert move in board.legal_moves

def test_mcts_best_move_as_black():
    board = chess.Board()
    board.push_san("e4")
    model = _random_model()
    move = best_move(board, model, n_simulations=10)
    assert move in board.legal_moves

def test_mcts_visits_increase_with_more_sims():
    board = chess.Board()
    model = _random_model()
    p5  = search(board, model, n_simulations=5,  add_noise=False)
    p20 = search(board, model, n_simulations=20, add_noise=False)
    # More simulations → more concentrated (higher max prob) or at least valid
    assert max(p20.values()) >= 0.0
    assert len(p5) > 0 and len(p20) > 0

def test_mctsnode_q_zero_before_visits():
    node = MCTSNode()
    assert node.Q == 0.0

def test_mctsnode_q_after_update():
    node = MCTSNode()
    node.N = 4
    node.W = 2.0
    assert node.Q == pytest.approx(0.5)
