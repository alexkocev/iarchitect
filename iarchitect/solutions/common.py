import json
from argparse import Namespace
from pathlib import Path


def read_args(folder):
    p = Path(folder) / "args.json"
    d = dict()
    print(p)
    if p.exists():
        with open(p,"r") as f:
            d = json.load(f)
    return Namespace(**d)


def write_args(folder, args):
    with open(folder / "args.json", "w") as f:
        json.dump(dict(args._get_kwargs()),f)


class ValidEnv:
    aligned_rows_big = "aligned_rows_big"
    windows_env_big_network = "windows_env_big_network"



