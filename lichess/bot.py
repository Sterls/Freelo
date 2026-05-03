import chess
import berserk
from engine.search import best_move


class LichessBot:
    def __init__(self, token: str):
        session = berserk.TokenSession(token)
        self.client = berserk.Client(session=session)

    def run(self):
        for event in self.client.bots.stream_incoming_events():

            if event["type"] == "challenge":
                self.client.bots.accept_challenge(event["challenge"]["id"])

            elif event["type"] == "gameStart":
                game_id = event["game"]["id"]
                self.play_game(game_id)

    def play_game(self, game_id: str):
        board = chess.Board()

        for state in self.client.bots.stream_game_state(game_id):

            if state["type"] != "gameState":
                continue

            fen = state.get("fen")
            if fen is None:
                print("Fen is None")
                continue

            board.set_fen(fen)

            move = best_move(board, depth=3)

            if move is None:
                continue

            self.client.bots.make_move(game_id, move.uci())