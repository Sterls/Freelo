from lichess.bot import LichessBot
from config import TOKEN

if __name__ == "__main__":
    bot = LichessBot(TOKEN)
    bot.run()
