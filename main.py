import os
from lichess.bot import LichessBot
from config import TOKEN

BEST_CHECKPOINT = "nn/checkpoints/best.pt"

if __name__ == "__main__":
    model = None
    if os.path.exists(BEST_CHECKPOINT):
        from nn.model import load as load_model
        model = load_model(BEST_CHECKPOINT, device="cpu")
        print(f"Loaded NN model from {BEST_CHECKPOINT}")
    else:
        print("No checkpoint found, using static evaluator")

    bot = LichessBot(TOKEN, model=model)
    bot.challenge_bots(1)
    bot.run()
