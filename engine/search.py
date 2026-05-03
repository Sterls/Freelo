import chess

INF = 10**9


# --- basic evaluation ---
def evaluate(board: chess.Board):
    if board.is_checkmate():
        return -INF if board.turn else INF

    if board.is_stalemate():
        return 0

    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
    }

    score = 0

    for piece, value in values.items():
        score += len(board.pieces(piece, chess.WHITE)) * value
        score -= len(board.pieces(piece, chess.BLACK)) * value

    return score


# --- minimax with alpha-beta ---
def alphabeta(board, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    if maximizing:
        best = -INF
        for move in board.legal_moves:
            board.push(move)
            val = alphabeta(board, depth - 1, alpha, beta, False)
            board.pop()

            best = max(best, val)
            alpha = max(alpha, val)

            if beta <= alpha:
                break

        return best

    else:
        best = INF
        for move in board.legal_moves:
            board.push(move)
            val = alphabeta(board, depth - 1, alpha, beta, True)
            board.pop()

            best = min(best, val)
            beta = min(beta, val)

            if beta <= alpha:
                break

        return best


def best_move(board, depth=2):
    maximizing = board.turn == chess.WHITE
    best = None
    best_value = -INF if maximizing else INF

    for move in board.legal_moves:
        board.push(move)
        val = alphabeta(board, depth - 1, -INF, INF, not maximizing)
        board.pop()

        if maximizing and val > best_value or not maximizing and val < best_value:
            best_value = val
            best = move

    return best