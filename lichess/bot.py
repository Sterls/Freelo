import berserk
import chess
from engine.search import best_move
from engine.board import GameBoard

class LichessBot:
    def __init__(self, token):
        session = berserk.TokenSession(token)
        self.client = berserk.Client(session=session)

    def run(self):
        for event in self.client.bots.stream_incoming_events():

            if event["type"] == "challenge":
                self.client.bots.accept_challenge(event["challenge"]["id"])

            if event["type"] == "gameStart":
                game_id = event["game"]["id"]
                self.play_game(game_id)

    def play_game(self, game_id):
        board = GameBoard()

        for state in self.client.bots.stream_game_state(game_id):

            if state["type"] == "gameState":
                board.set_fen(state["fen"])

                if board.board.turn == chess.WHITE:
                    move = best_move(board.board, depth=3)
                    if move:
                        self.client.bots.make_move(game_id, move.uci())
