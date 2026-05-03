import os
from lichess.bot import LichessBot
from config import TOKEN

BEST_CHECKPOINT = "nn/checkpoints/best.pt"

if __name__ == "__main__":
    eval_fn = None
    if os.path.exists(BEST_CHECKPOINT):
        from nn.model import make_eval_fn
        eval_fn = make_eval_fn(BEST_CHECKPOINT, device="cpu")
        print(f"Loaded NN evaluator from {BEST_CHECKPOINT}")
    else:
        print("No checkpoint found, using static evaluator")

    bot = LichessBot(TOKEN, eval_fn=eval_fn)
    bot.challenge_bots(10)
    bot.run()
