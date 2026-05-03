import chess
import berserk
from engine.search import best_move, evaluate


class LichessBot:
    def __init__(self, token: str, eval_fn=None):
        session = berserk.TokenSession(token)
        self.client = berserk.Client(session=session)
        self.my_id = self.client.account.get()["id"]
        self.eval_fn = eval_fn or evaluate

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

    def play_game(self, game_id: str):
        is_white = None

        for state in self.client.bots.stream_game_state(game_id):

            if state["type"] == "gameFull":
                is_white = (state["white"]["id"] == self.my_id)
                moves = state["state"]["moves"]

            elif state["type"] == "gameState":
                moves = state["moves"]

            else:
                continue

            board = chess.Board()

            if moves:
                for m in moves.split():
                    board.push_uci(m)

            if is_white is None:
                continue

            if board.turn != is_white:
                continue

            move = best_move(board, depth=3, eval_fn=self.eval_fn)

            if move:
                print("Playing:", move.uci())
                self.client.bots.make_move(game_id, move.uci())
