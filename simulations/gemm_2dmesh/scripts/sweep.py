import os
import sys
import subprocess
from itertools import product
from copy import deepcopy
from pathlib import Path
import yaml
from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT'))
DOCKER_PROJECT_ROOT = Path(os.getenv('DOCKER_PROJECT_ROOT'))
PYUTILS_RDIR = Path(os.getenv('PYUTILS_RDIR'))
sys.path.insert(0, str(PROJECT_ROOT / PYUTILS_RDIR))
from xlogging import logger
from update_hyperparameters import HyperparameterUpdater, RUN_SCRIPT_PATH

SIMULATION_RDIR = Path('simulations/gemm_2dmesh')
SIMULATION_ROOT = PROJECT_ROOT / SIMULATION_RDIR
TEMPLATES_DIR = SIMULATION_ROOT / 'scripts/resource'
DEFAULT_HPS_PATH = SIMULATION_ROOT / 'hyperparameters.yml'
SWEEP_HPS_PATH = SIMULATION_ROOT / 'sweep_hyperparameters.yml'

def gen_sweep_hps(sweep_params):
    """ Generator of hyperparameter sets for the sweep
    
    Note
    --------
    Written with LLM, not checked
    """
    def flatten(d, path=()):
        for key, value in d.items():
            if isinstance(value, dict):
                yield from flatten(value, path + (key,))
            else:
                yield path + (key,), value
    params = list(flatten(sweep_params))
    for combination in product(*(values for _, values in params)):
        result = {}
        for (path, _), value in zip(params, combination):
            d = result
            for key in path[:-1]:
                d = d.setdefault(key, {})
            d[path[-1]] = value.copy()
        yield result

def merge_hps(default, override):
    """ Merges the default and sweep hypreparameter sets
    
    Note
    --------
    Written with LLM, not checked
    """
    result = deepcopy(default)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(result.get(key), dict)
            and "value" not in value
        ):
            result[key] = merge_hps(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

if __name__ == '__main__':
    with open(DEFAULT_HPS_PATH, 'r', encoding='utf-8') as f:
        default_hps:dict = yaml.safe_load(f)
    with open(SWEEP_HPS_PATH, 'r', encoding='utf-8') as f:
        sweep_params:dict = yaml.safe_load(f)

    for sweep_hps in gen_sweep_hps(sweep_params):
        hps = merge_hps(default_hps, sweep_hps)
        hu = HyperparameterUpdater(hps, TEMPLATES_DIR)
        hu.update()
        subprocess.run(['bash', str(RUN_SCRIPT_PATH)], check=True)
