import chess
from engine.bobby_eval import evaluate_fen

INF = 10**9
cache = {}

def evaluate_fen(fen):
    if fen in cache:
        return cache[fen]

    score = run_julia(fen)
    cache[fen] = score
    return score

def evaluate(board: chess.Board):
    # delegate to Julia
    return evaluate_fen(board.fen())


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


def best_move(board, depth=3):
    best_move = None
    best_value = -INF

    for move in board.legal_moves:
        board.push(move)
        val = alphabeta(board, depth - 1, -INF, INF, False)
        board.pop()

        if val > best_value:
            best_value = val
            best_move = move

    return best_move
