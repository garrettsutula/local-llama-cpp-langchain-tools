import argparse


def get_args() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Batch inference tool for locally-hosted llama.cpp models.",
        formatter_class=argparse.HelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="Job file name to load and run from './jobs/'",
    )
    return parser.parse_args()
