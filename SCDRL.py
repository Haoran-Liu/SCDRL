import argparse
import importlib
from typing import Optional, Sequence

DATASET_MODULES = {
    "simulation": "SCDRL_simulation",
    "mouse_human": "SCDRL_mouse_human",
    "haniffa": "SCDRL_haniffa",
}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch to a dataset-specific SCDRL experiment runner."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_MODULES),
        help="Dataset-specific runner to execute.",
    )
    parser.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected dataset runner.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    runner_args = list(args.runner_args)
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    module = importlib.import_module(DATASET_MODULES[args.dataset])
    module.main(runner_args)


if __name__ == "__main__":
    main()
