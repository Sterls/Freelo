# NN Evaluator & RL Training Plan

Replace the static material `evaluate()` in `engine/search.py` with a neural network
trained via self-play reinforcement learning.

## Hardware

RTX 3060 Laptop GPU (6 GB VRAM). All training runs on CUDA; inference during search
runs on CPU to avoid GPU→CPU transfer overhead per node.

## Dependencies to add

```
torch          # GPU training
numpy          # board tensor construction
```

---

## Phase 1 — Board encoding (`engine/features.py`)

Convert a `chess.Board` to a `(13, 8, 8)` float32 tensor:
- Planes 0–11: one binary plane per piece type × color (P, N, B, R, Q, K for each side)
- Plane 12: side-to-move (all 1s if white to move, all 0s if black)

This is the sole input format for the network.

## Phase 2 — Network architecture (`nn/model.py`)

Small CNN; input `(13, 8, 8)` → scalar output ∈ `[-1, 1]`.

```
Conv2d(13, 64, 3, padding=1) → BatchNorm → ReLU
Conv2d(64, 128, 3, padding=1) → BatchNorm → ReLU
Conv2d(128, 128, 3, padding=1) → BatchNorm → ReLU
Flatten → Linear(8192, 256) → ReLU → Linear(256, 1) → Tanh
```

Output convention: `+1` = white wins, `-1` = black wins (same sign as `evaluate()`).

## Phase 3 — Self-play data collection (`nn/self_play.py`)

- Run bot vs itself locally (no Lichess), using the current best model for both sides
- Record every position as `(fen, outcome)` where outcome ∈ `{+1, -1, 0}`
- Outcome is assigned at game end and propagated back to all positions in that game
- Store to disk as a `.pt` file (list of tensors + labels)
- Use search depth 1 during self-play for speed; depth can be increased later

## Phase 4 — Training loop (`nn/train.py`)

- Load dataset from disk
- Loss: MSE between network output and game outcome
- Optimizer: Adam, lr=1e-3
- Train on GPU (`model.to("cuda")`), batch size 512
- Save checkpoint after each epoch to `nn/checkpoints/`
- Log train loss per epoch to stdout

## Phase 5 — Integration (`engine/search.py`)

- Add `load_nn_evaluator(path)` to load a checkpoint
- Replace `evaluate(board)` call in `alphabeta` with NN inference when a model is loaded
- Fall back to material evaluation when no model is present
- Inference runs on CPU (avoids per-node GPU overhead at search depth)

## Phase 6 — RL iteration loop (`nn/rl_loop.py`)

Repeat:
1. Run self-play with current model → collect N games
2. Append to training dataset (keep a rolling window of last ~50k positions)
3. Train for E epochs
4. Save new checkpoint as next generation
5. Optionally pit new model vs previous; promote only if win rate > 55%

Start with N=200 games, E=10 epochs per iteration. Tune as needed based on GPU
utilization and game quality.

---

## File layout when complete

```
engine/
  features.py       # board → tensor
  search.py         # alpha-beta, calls evaluate() or NN
nn/
  model.py          # CNN definition
  self_play.py      # local self-play, writes dataset
  train.py          # training loop
  rl_loop.py        # ties self-play + train together
  checkpoints/      # saved model weights (gitignored)
  data/             # self-play datasets (gitignored)
```
