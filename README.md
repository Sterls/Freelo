# Tally-Bot

A Lichess chess bot that teaches itself to play through reinforcement learning. Built for fun.

## What it does

Tally connects to [Lichess](https://lichess.org) via the bot API, accepts challenges, and plays games autonomously. Under the hood it runs a minimax search with alpha-beta pruning. The twist: instead of a hand-coded evaluation function, it trains a small neural network by playing thousands of games against itself and learning which positions tend to win.

## How it works

### Playing games

`main.py` connects to Lichess using a personal API token, challenges online bots to rated games, then sits in an event loop accepting incoming challenges. For each game it reconstructs the full board from the move history on every turn, calls the engine to pick a move, and sends it back to Lichess.

### The engine

The search is a standard minimax with alpha-beta pruning (`engine/search.py`). It explores moves to a fixed depth and scores leaf positions with an evaluation function. Before any training, that function is a simple material count (pawn=100, knight=320, bishop=330, rook=500, queen=900). After training, it's replaced by the neural network.

### The neural network

A small CNN (`nn/model.py`) takes the board as a `(13, 8, 8)` tensor — 12 binary planes for piece positions (one per piece type × color) plus a side-to-move plane — and outputs a scalar in `[-1, 1]` where `+1` means white is winning and `-1` means black is winning.

### Self-play and training

The RL loop (`nn/rl_loop.py`) runs continuously:

1. **Self-play** — the current best model plays against itself for N games, recording every position and the final outcome
2. **Train** — the network is trained on a rolling window of the last ~50k positions to predict game outcomes from board positions
3. **Pit** — the new model plays 20 games against the previous best; it's only promoted to `best.pt` if it wins more than 55% of them
4. **Sync** — checkpoints and datasets are pushed to Google Drive after each generation

Run the training loop:
```bash
python -m nn.rl_loop --generations 10 --games-per-gen 200 --epochs-per-gen 10
```

### Storage

Training data (`.pt` files) and model checkpoints are synced to Google Drive via rclone after every generation. Locally, old data files are pruned to stay within a position budget (~50k positions). On Drive, everything is kept permanently so no work is ever lost.

## Setup

```bash
pip install -r requirements.txt
```

Add a `config.py` with your Lichess bot token:
```python
TOKEN = "your_lichess_api_token"
```

For training sync, configure rclone with a Google Drive remote named `gdrive`:
```bash
rclone config
```

## Running

Play games only (uses `nn/checkpoints/best.pt` if it exists, otherwise static eval):
```bash
python main.py
```

Train a new generation of models:
```bash
python -m nn.rl_loop
```
