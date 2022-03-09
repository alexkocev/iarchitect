from .common import ValidEnv,read_args,write_args
from .aligned_rows_big import make_environment as aligned_rows_big_make_environment
from .windows_env_big_network import make_environment as windows_env_big_network_make_environment



def make_environment(args):
    if "name_env" not in args:
        setattr(args,"name_env",ValidEnv.aligned_rows_big)
    name = args.name_env
    if name == ValidEnv.aligned_rows_big:
        return aligned_rows_big_make_environment(args)
    elif name == ValidEnv.windows_env_big_network:
        return windows_env_big_network_make_environment(args)
