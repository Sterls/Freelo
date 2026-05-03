import os
import glob
import tempfile
import torch

from engine.search import evaluate as static_eval
from nn.model import ChessNet, make_eval_fn
from nn.self_play import generate, play_game_vs
from nn.train import train

CHECKPOINT_DIR = "nn/checkpoints"
DATA_DIR = "nn/data"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pt")


def pit(eval_fn_new, eval_fn_old, n_games=20, depth=1):
    """
    Play n_games between new and old model, alternating colors.
    Returns win rate of new model (wins + 0.5*draws) / n_games.
    """
    wins = draws = losses = 0

    for i in range(n_games):
        if i % 2 == 0:
            outcome = play_game_vs(eval_fn_new, eval_fn_old, depth=depth)
            if outcome > 0:   wins += 1
            elif outcome == 0: draws += 1
            else:             losses += 1
        else:
            outcome = play_game_vs(eval_fn_old, eval_fn_new, depth=depth)
            if outcome < 0:   wins += 1
            elif outcome == 0: draws += 1
            else:             losses += 1

    win_rate = (wins + 0.5 * draws) / n_games
    print(f"  Pit result: {wins}W {draws}D {losses}L  win_rate={win_rate:.2f}")
    return win_rate


def _prune_old_data(max_positions):
    """Remove oldest gen files so the remaining total stays within max_positions."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "gen*.pt")))
    if not files:
        return

    sizes = [len(torch.load(f, weights_only=True)["outcomes"]) for f in files]

    total = 0
    keep_from = 0
    for i in range(len(files) - 1, -1, -1):
        if total + sizes[i] > max_positions:
            keep_from = i + 1
            break
        total += sizes[i]

    for f in files[:keep_from]:
        os.remove(f)
        print(f"  Pruned {f}")


def run(
    generations: int = 10,
    games_per_gen: int = 200,
    epochs_per_gen: int = 10,
    depth: int = 1,
    pit_games: int = 20,
    promote_threshold: float = 0.55,
    max_positions: int = 50_000,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    existing_gens = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "gen*.pt")))
    start_gen = len(existing_gens)

    if os.path.exists(BEST_CHECKPOINT):
        current_eval = make_eval_fn(BEST_CHECKPOINT)
        print(f"Resuming from {BEST_CHECKPOINT}")
    else:
        current_eval = None
        print("No checkpoint — bootstrapping with static evaluator")

    for gen in range(start_gen, start_gen + generations):
        print(f"\n{'='*50}")
        print(f"Generation {gen}")
        print(f"{'='*50}")

        # 1. self-play
        data_path = os.path.join(DATA_DIR, f"gen{gen:03d}.pt")
        eval_fn = current_eval if current_eval is not None else static_eval
        print(f"Self-play: {games_per_gen} games, depth={depth}")
        generate(games_per_gen, data_path, eval_fn=eval_fn, depth=depth)

        # 2. rolling window
        _prune_old_data(max_positions)

        # 3. train — use a temp dir for per-epoch checkpoints to keep checkpoints/ clean
        data_glob = os.path.join(DATA_DIR, "gen*.pt")
        resume = BEST_CHECKPOINT if os.path.exists(BEST_CHECKPOINT) else None
        print(f"Training: {epochs_per_gen} epochs")
        with tempfile.TemporaryDirectory() as tmpdir:
            new_model = train(
                data_glob,
                checkpoint_dir=tmpdir,
                resume=resume,
                epochs=epochs_per_gen,
            )

        gen_ckpt = os.path.join(CHECKPOINT_DIR, f"gen{gen:03d}.pt")
        torch.save(new_model.state_dict(), gen_ckpt)
        print(f"Saved {gen_ckpt}")

        # 4. pit new vs current best
        new_eval = make_eval_fn(new_model)

        if current_eval is None or pit_games == 0:
            promote = True
            print("  No baseline to pit against — auto-promoting")
        else:
            print(f"Pitting new vs current best ({pit_games} games)...")
            win_rate = pit(new_eval, current_eval, n_games=pit_games, depth=depth)
            promote = win_rate >= promote_threshold

        # 5. promote if better
        if promote:
            torch.save(new_model.state_dict(), BEST_CHECKPOINT)
            current_eval = new_eval
            print(f"  Promoted gen{gen:03d} → best.pt")
        else:
            print(f"  Not promoted — keeping previous best")

    print("\nRL loop complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--generations",        type=int,   default=10)
    parser.add_argument("--games-per-gen",      type=int,   default=200)
    parser.add_argument("--epochs-per-gen",     type=int,   default=10)
    parser.add_argument("--depth",              type=int,   default=1)
    parser.add_argument("--pit-games",          type=int,   default=20)
    parser.add_argument("--promote-threshold",  type=float, default=0.55)
    parser.add_argument("--max-positions",      type=int,   default=50_000)
    args = parser.parse_args()

    run(
        generations=args.generations,
        games_per_gen=args.games_per_gen,
        epochs_per_gen=args.epochs_per_gen,
        depth=args.depth,
        pit_games=args.pit_games,
        promote_threshold=args.promote_threshold,
        max_positions=args.max_positions,
    )
