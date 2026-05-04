import chess
import numpy as np
import pytest

from engine.features import board_to_tensor


def test_output_shape():
    assert board_to_tensor(chess.Board()).shape == (13, 8, 8)

def test_output_dtype():
    assert board_to_tensor(chess.Board()).dtype == np.float32

def test_values_binary():
    t = board_to_tensor(chess.Board())
    assert set(t.flatten().tolist()).issubset({0.0, 1.0})

def test_side_to_move_plane_white():
    t = board_to_tensor(chess.Board())
    assert t[12].all()   # all 1s when white to move

def test_side_to_move_plane_black():
    board = chess.Board()
    board.push_san("e4")
    t = board_to_tensor(board)
    assert not t[12].any()   # all 0s when black to move

def test_white_king_present():
    t = board_to_tensor(chess.Board())
    # Plane 5 = white king; starting square e1 = rank 0, file 4
    assert t[5, 0, 4] == 1.0

def test_black_king_present():
    t = board_to_tensor(chess.Board())
    # Plane 11 = black king; starting square e8 = rank 7, file 4
    assert t[11, 7, 4] == 1.0

def test_empty_board_has_no_pieces():
    board = chess.Board(fen=None)
    t = board_to_tensor(board)
    assert t[:12].sum() == 0

def test_piece_count_starting_position():
    t = board_to_tensor(chess.Board())
    # 16 white pieces + 16 black pieces = 32 occupied squares across planes 0-11
    assert t[:12].sum() == 32
