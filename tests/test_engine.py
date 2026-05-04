import chess
import pytest

from engine.search import evaluate, best_move, INF


# --- evaluate ---

def test_evaluate_starting_position_is_zero():
    assert evaluate(chess.Board()) == 0

def test_evaluate_white_up_a_queen():
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert evaluate(board) > 0

def test_evaluate_black_up_a_queen():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    assert evaluate(board) < 0

def test_evaluate_checkmate_black_to_move():
    # Fool's mate — black is checkmated, white wins
    board = chess.Board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    board.turn = chess.BLACK
    board.push(chess.Move.null())
    # Use a known checkmate position instead
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    assert board.is_checkmate()
    assert evaluate(board) == INF   # white wins → +INF

def test_evaluate_stalemate_is_zero():
    # Classic stalemate position
    board = chess.Board("k7/8/1Q6/8/8/8/8/7K b - - 0 1")
    if board.is_stalemate():
        assert evaluate(board) == 0


# --- best_move ---

def test_best_move_returns_legal_move():
    board = chess.Board()
    move = best_move(board, depth=1)
    assert move in board.legal_moves

def test_best_move_captures_free_piece():
    # White queen can take an undefended black rook — depth 1 should find it
    # Valid position: kings on a1/a8, white queen on d1, black rook on d8
    board = chess.Board("k7/8/8/8/8/8/8/K2Q2r1 w - - 0 1")
    if not board.is_game_over():
        move = best_move(board, depth=1)
        assert move is not None
        assert board.is_capture(move)

def test_best_move_avoids_none_on_game_over():
    # Checkmate position — no legal moves
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3")
    if board.is_game_over():
        assert best_move(board, depth=1) is None

def test_best_move_as_black():
    board = chess.Board()
    board.push_san("e4")
    move = best_move(board, depth=1)
    assert move in board.legal_moves
    assert board.turn == chess.BLACK
