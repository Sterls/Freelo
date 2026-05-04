import chess

INF = 10**9

_PIECE_VAL = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20_000,
}


def _order_moves(board: chess.Board, moves):
    """Recaptures → captures (MVV-LVA) → escapes → quiet moves."""
    last = board.peek() if board.move_stack else None
    recap_sq = last.to_square if last else None
    opp = not board.turn

    def key(move):
        victim = board.piece_at(move.to_square)
        aggressor = board.piece_at(move.from_square)
        agg_val = _PIECE_VAL.get(aggressor.piece_type, 0) if aggressor else 0

        if victim:
            vic_val = _PIECE_VAL.get(victim.piece_type, 0)
            tier = 0 if move.to_square == recap_sq else 1
            return (tier, -vic_val, agg_val)

        if aggressor and board.is_attacked_by(opp, move.from_square):
            atk_min = min(
                (_PIECE_VAL.get(board.piece_at(sq).piece_type, 0)
                 for sq in board.attackers(opp, move.from_square)
                 if board.piece_at(sq)),
                default=0,
            )
            if atk_min < agg_val:
                return (2, 0, 0)

        return (3, 0, 0)

    return sorted(moves, key=key)


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


def alphabeta(board, depth, alpha, beta, maximizing, eval_fn=evaluate):
    if board.is_game_over():
        return evaluate(board)  # static eval for terminals — checkmate/stalemate values are outside NN range

    if depth == 0:
        return eval_fn(board)

    if maximizing:
        best = -INF
        for move in _order_moves(board, board.legal_moves):
            board.push(move)
            val = alphabeta(board, depth - 1, alpha, beta, False, eval_fn)
            board.pop()

            best = max(best, val)
            alpha = max(alpha, val)

            if beta <= alpha:
                break

        return best

    else:
        best = INF
        for move in _order_moves(board, board.legal_moves):
            board.push(move)
            val = alphabeta(board, depth - 1, alpha, beta, True, eval_fn)
            board.pop()

            best = min(best, val)
            beta = min(beta, val)

            if beta <= alpha:
                break

        return best


def best_move(board, depth=3, eval_fn=evaluate):
    maximizing = board.turn == chess.WHITE
    best = None
    best_value = -(INF + 1) if maximizing else (INF + 1)

    for move in _order_moves(board, board.legal_moves):
        board.push(move)
        val = alphabeta(board, depth - 1, -INF, INF, not maximizing, eval_fn)
        board.pop()

        if maximizing and val > best_value or not maximizing and val < best_value:
            best_value = val
            best = move

    return best
