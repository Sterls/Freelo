# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the bot

```bash
python main.py
```

Requires a `config.py` in the repo root (gitignored) with:
```python
TOKEN = "your_lichess_api_token"
```

## Dependencies

```bash
pip install python-chess berserk
```

## Architecture

The bot connects to Lichess via the [berserk](https://github.com/lichess-org/berserk) client and plays games using a custom chess engine.

**`main.py`** — entry point. Calls `challenge_bots(10)` then `run()`.

**`lichess/bot.py` — `LichessBot`**
- `run()`: streams incoming Lichess events, accepts challenges, dispatches to `play_game()`
- `challenge_bots(n)`: challenges n online bots to rated 3+0 games before entering the event loop
- `play_game(game_id)`: reconstructs the full board from the UCI move history on every state update, then calls the engine on the bot's turn

**`engine/search.py`** — the chess engine
- `evaluate(board)`: static material evaluation; positive = white advantage
- `alphabeta(board, depth, alpha, beta, maximizing)`: minimax with alpha-beta pruning
- `best_move(board, depth=3)`: top-level call; reads `board.turn` to determine whether to maximize (white) or minimize (black); `best_value` is initialized to `±(INF+1)` — one unit outside the range of `evaluate()` — so a move is always returned even when all lines lose

## Key invariants

- `evaluate()` returns values strictly within `[-INF, INF]` where `INF = 10**9`. Keep `best_value` initialization outside this range so the first legal move is always selected.
- `board.turn` is `chess.WHITE` (`True`) / `chess.BLACK` (`False`). The engine and bot both rely on this for color detection.
- `play_game()` rebuilds the board from scratch on every event rather than maintaining incremental state.
