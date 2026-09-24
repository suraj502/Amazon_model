"""Orchestrate EDA and the remaining pipeline stages in order."""

from scripts.run_blocking import main as run_blocking
from scripts.run_evaluate import main as run_evaluate
from scripts.run_features import main as run_features
from scripts.run_eda import main as run_eda
from scripts.run_normalize import main as run_normalize
from scripts.run_train import main as run_train


def main() -> None:
    """Call EDA first, then each pipeline stage in order as implemented."""
    run_eda()
    run_normalize()
    run_blocking()
    run_features()
    run_train()
    run_evaluate()


if __name__ == "__main__":
    main()
