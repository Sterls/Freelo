from datetime import timedelta
from unittest.mock import MagicMock, patch, call
import pytest

from lichess.bot import _to_ms, _pick_depth, GAME_OVER_STATUSES, LichessBot


# --- _to_ms ---

def test_to_ms_int():
    assert _to_ms(180_000) == 180_000

def test_to_ms_timedelta():
    assert _to_ms(timedelta(seconds=180)) == 180_000

def test_to_ms_timedelta_partial():
    assert _to_ms(timedelta(seconds=9, milliseconds=500)) == 9_500


# --- _pick_depth ---

def test_pick_depth_low_time():
    assert _pick_depth(5_000) == 1

def test_pick_depth_medium_time():
    assert _pick_depth(15_000) == 2

def test_pick_depth_full_time():
    assert _pick_depth(180_000) == 2

def test_pick_depth_boundary_10s():
    assert _pick_depth(10_000) == 2   # exactly 10s is NOT under 10s

def test_pick_depth_boundary_9999ms():
    assert _pick_depth(9_999) == 1


# --- play_game: game-over handling ---

def _make_bot():
    with patch("lichess.bot.berserk.Client") as mock_client_cls:
        client = mock_client_cls.return_value
        client.account.get.return_value = {"id": "testbot"}
        bot = LichessBot.__new__(LichessBot)
        bot.client = client
        bot.my_id = "testbot"
        bot.eval_fn = None
        return bot, client


def test_play_game_stops_on_outoftime():
    """Bot must not call make_move after receiving a game-over status."""
    bot, client = _make_bot()

    # Bot is black; first state is white to move (bot skips), second is outoftime
    states = [
        {
            "type": "gameFull",
            "white": {"id": "opponent"},
            "black": {"id": "testbot"},
            "state": {"moves": "", "status": "started", "wtime": 180_000, "btime": 180_000},
        },
        {
            "type": "gameState",
            "moves": "",
            "status": "outoftime",
            "wtime": 180_000,
            "btime": 0,
        },
    ]
    client.bots.stream_game_state.return_value = iter(states)

    bot.play_game("abc123")

    client.bots.make_move.assert_not_called()


def test_play_game_stops_on_mate():
    bot, client = _make_bot()

    states = [
        {
            "type": "gameFull",
            "white": {"id": "opponent"},
            "black": {"id": "testbot"},
            "state": {"moves": "", "status": "mate", "wtime": 100_000, "btime": 100_000},
        },
    ]
    client.bots.stream_game_state.return_value = iter(states)

    bot.play_game("abc123")

    client.bots.make_move.assert_not_called()


def test_play_game_skips_opponents_turn():
    """Bot should not move when it is the opponent's turn."""
    bot, client = _make_bot()

    # Bot is black; starting position is white to move → bot should skip
    states = [
        {
            "type": "gameFull",
            "white": {"id": "opponent"},
            "black": {"id": "testbot"},
            "state": {"moves": "", "status": "started", "wtime": 180_000, "btime": 180_000},
        },
    ]
    client.bots.stream_game_state.return_value = iter(states)

    bot.play_game("abc123")

    client.bots.make_move.assert_not_called()


def test_play_game_makes_move_on_our_turn():
    """Bot must call make_move when it is the bot's turn."""
    bot, client = _make_bot()
    from engine.search import evaluate
    bot.eval_fn = evaluate

    # Bot is white; starting position is white to move
    states = [
        {
            "type": "gameFull",
            "white": {"id": "testbot"},
            "black": {"id": "opponent"},
            "state": {"moves": "", "status": "started", "wtime": 180_000, "btime": 180_000},
        },
    ]
    client.bots.stream_game_state.return_value = iter(states)

    bot.play_game("abc123")

    client.bots.make_move.assert_called_once()
    move_uci = client.bots.make_move.call_args[0][1]
    # Must be a legal UCI move from the starting position
    import chess
    board = chess.Board()
    assert chess.Move.from_uci(move_uci) in board.legal_moves


def test_play_game_make_move_error_does_not_crash():
    """A ResponseError from make_move must not propagate."""
    import berserk.exceptions
    bot, client = _make_bot()
    from engine.search import evaluate
    bot.eval_fn = evaluate

    states = [
        {
            "type": "gameFull",
            "white": {"id": "testbot"},
            "black": {"id": "opponent"},
            "state": {"moves": "", "status": "started", "wtime": 180_000, "btime": 180_000},
        },
    ]
    client.bots.stream_game_state.return_value = iter(states)
    client.bots.make_move.side_effect = Exception("Game already finished")

    bot.play_game("abc123")   # must not raise
