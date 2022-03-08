import json
from argparse import Namespace
from pathlib import Path
from .aligned_rows_big import make_environment as aligned_rows_big_make_environment
from .windows_env_big_network import make_environment as windows_env_big_network_make_environment


def read_args(folder):
    p = Path(folder) / "args.json"
    d = dict()
    print(p)
    if p.exists():
        with open(p,"r") as f:
            d = json.load(f)
    return Namespace(**d)

VALID_NAME_ENV = ["aligned_rows_big","windows_env_big_network"]

def make_environment(args):
    if "name_env" not in args:
        setattr(args,"name_env",VALID_NAME_ENV[0])
    name = args.name_env
    assert name in VALID_NAME_ENV
    if name == VALID_NAME_ENV[0]:
        return aligned_rows_big_make_environment(args)
    elif name == VALID_NAME_ENV[1]:
        return windows_env_big_network_make_environment(args)
