import numpy as np
import chess

_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Encode a board as a (13, 8, 8) float32 tensor.

    Planes 0-5:  white pieces  (P, N, B, R, Q, K)
    Planes 6-11: black pieces  (P, N, B, R, Q, K)
    Plane 12:    side to move  (all 1s = white, all 0s = black)
    """
    tensor = np.zeros((13, 8, 8), dtype=np.float32)

    for i, piece_type in enumerate(_PIECE_TYPES):
        for sq in board.pieces(piece_type, chess.WHITE):
            tensor[i, sq >> 3, sq & 7] = 1.0
        for sq in board.pieces(piece_type, chess.BLACK):
            tensor[i + 6, sq >> 3, sq & 7] = 1.0

    if board.turn == chess.WHITE:
        tensor[12] = 1.0

    return tensor
