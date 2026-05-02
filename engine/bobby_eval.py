import subprocess
import chess

JULIA_CMD = ["julia", "--project=./Bobby.jl"]  # adjust path if needed


def evaluate_fen(fen: str) -> float:
    """
    Calls Bobby.jl to evaluate a position.
    Assumes Julia script prints a numeric score.
    """

    process = subprocess.run(
        JULIA_CMD + ["--eval", fen],
        capture_output=True,
        text=True,
        check=True
    )

    try:
        return float(process.stdout.strip())
    except Exception:
        return 0.0
