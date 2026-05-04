import chess
import berserk
from engine.search import best_move as alphabeta_best_move, evaluate

GAME_OVER_STATUSES = {"mate", "resign", "stalemate", "timeout", "draw",
                      "outoftime", "cheat", "noStart", "unknownFinish", "aborted"}


def _to_ms(t) -> int:
    """Normalize berserk clock value to milliseconds (int or timedelta)."""
    return int(t.total_seconds() * 1000) if hasattr(t, "total_seconds") else int(t)


def _pick_simulations(my_time_ms: int) -> int:
    """Scale MCTS simulations with remaining clock time."""
    if my_time_ms < 10_000:
        return 25
    if my_time_ms < 30_000:
        return 35
    return 50


def _pick_depth(my_time_ms: int) -> int:
    """Fallback depth for alpha-beta when no model is loaded."""
    if my_time_ms < 10_000:
        return 1
    return 2


class LichessBot:
    def __init__(self, token: str, model=None):
        session = berserk.TokenSession(token)
        self.client = berserk.Client(session=session)
        self.my_id = self.client.account.get()["id"]
        self.model = model  # ChessNet or None

    def run(self):
        print("Bot running...")

        for event in self.client.bots.stream_incoming_events():

            if event["type"] == "challenge":
                challenge_id = event["challenge"]["id"]
                try:
                    self.client.bots.accept_challenge(challenge_id)
                    print("Accepted challenge", challenge_id)
                except berserk.exceptions.ResponseError as e:
                    print("Could not accept challenge:", e)
                    self.client.bots.decline_challenge(challenge_id)

            elif event["type"] == "gameStart":
                game_id = event["game"]["id"]
                print("Game started:", game_id)
                self.play_game(game_id)

    def challenge_bots(self, n: int):
        sent = 0
        for bot in self.client.bots.get_online_bots():
            if sent >= n:
                break
            if bot["id"] == self.my_id:
                continue
            try:
                self.client.challenges.create(
                    bot["id"],
                    rated=True,
                    clock_limit=180,
                    clock_increment=0,
                )
                print(f"Challenged {bot['id']}")
                sent += 1
            except berserk.exceptions.ResponseError as e:
                print(f"Could not challenge {bot['id']}:", e)

    def _pick_move(self, board: chess.Board, my_time_ms: int):
        if self.model is not None:
            from engine.mcts import best_move as mcts_best_move
            n_sims = _pick_simulations(my_time_ms)
            return mcts_best_move(board, self.model, n_sims)
        depth = _pick_depth(my_time_ms)
        return alphabeta_best_move(board, depth=depth, eval_fn=evaluate)

    def play_game(self, game_id: str):
        is_white = None
        my_time_ms = 180_000

        for state in self.client.bots.stream_game_state(game_id):

            if state["type"] == "gameFull":
                is_white = (state["white"]["id"] == self.my_id)
                moves = state["state"]["moves"]
                status = state["state"].get("status", "started")
                my_time_ms = _to_ms(state["state"]["wtime" if is_white else "btime"])

            elif state["type"] == "gameState":
                moves = state["moves"]
                status = state.get("status", "started")
                if is_white is not None:
                    my_time_ms = _to_ms(state["wtime" if is_white else "btime"])

            else:
                continue

            if status in GAME_OVER_STATUSES:
                print(f"Game over: {status}")
                break

            board = chess.Board()
            if moves:
                for m in moves.split():
                    board.push_uci(m)

            if is_white is None:
                continue

            if board.turn != is_white:
                continue

            move = self._pick_move(board, my_time_ms)

            if move:
                print(f"Playing: {move.uci()}  (time={my_time_ms//1000}s)")
                try:
                    self.client.bots.make_move(game_id, move.uci())
                except Exception as e:
                    print(f"make_move failed: {e}")
                    break
