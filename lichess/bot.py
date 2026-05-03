import chess
import berserk
from engine.search import best_move


class LichessBot:
    def __init__(self, token: str):
        session = berserk.TokenSession(token)
        self.client = berserk.Client(session=session)
        self.my_id = self.client.account.get()["id"]

    def run(self):
        print("Bot running...")

        for event in self.client.bots.stream_incoming_events():

            if event["type"] == "challenge":
                print("Accepting challenge")
                self.client.bots.accept_challenge(event["challenge"]["id"])

            elif event["type"] == "gameStart":
                game_id = event["game"]["id"]
                print("Game started:", game_id)
                self.play_game(game_id)

    def play_game(self, game_id: str):
        is_white = None

        for state in self.client.bots.stream_game_state(game_id):

            # initial state
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

            # only move on your turn
            if board.turn != is_white:
                continue

            move = best_move(board, depth=2)

            if move:
                print("Playing:", move.uci())
                self.client.bots.make_move(game_id, move.uci())