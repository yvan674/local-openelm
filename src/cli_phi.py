"""Command Line Interface for Phi-3.

Provides a command line interface to simply return a single answer using Phi-3.
"""
from argparse import ArgumentParser
from pathlib import Path

from mlx_phi import MlxPhi


def parse_args():
    p = ArgumentParser()

    p.add_argument("--model-fp", type=Path,
                   default=None)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--sys-prompt", type=str, default=None)

    return p.parse_args()


def main(prompt: str, model_fp: Path = None, sys_prompt: str = None):
    if model_fp is None:
        model_fp = (Path.home() / "Models" /
                    "Phi-3-mini-4k-instruct-4bit-no-q-embed")

    phi = MlxPhi(model_fp, sys_prompt)
    print(phi.generate(prompt))


if __name__ == '__main__':
    args = parse_args()
    main(args.prompt, model_fp=args.model_fp, sys_prompt=args.sys_prompt)
