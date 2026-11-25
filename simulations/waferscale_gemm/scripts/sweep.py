import os
import sys
import subprocess
import re
from tqdm import tqdm
from datetime import datetime as dt
from tqdm.contrib.logging import logging_redirect_tqdm
from itertools import product
from copy import deepcopy
from pathlib import Path
from typing import Iterator
import yaml
from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT'))
DOCKER_PROJECT_ROOT = Path(os.getenv('DOCKER_PROJECT_ROOT'))
PYUTILS_RDIR = Path(os.getenv('PYUTILS_RDIR'))
sys.path.insert(0, str(PROJECT_ROOT / PYUTILS_RDIR))
from xlogging import logger
from update_hyperparameters import HyperparameterUpdater, RUN_SCRIPT_PATH

SIMULATION_RDIR = Path('simulations/waferscale_gemm')
SIMULATION_ROOT = PROJECT_ROOT / SIMULATION_RDIR
TEMPLATES_DIR = SIMULATION_ROOT / 'scripts/resource'
DEFAULT_HPS_PATH = SIMULATION_ROOT / 'default-hyperparams.yml'
SWEEP_HPS_PATH = SIMULATION_ROOT / 'sweep-hyperparams.yml'

def gen_sweep_hps(sweep_params) -> Iterator[dict]:
    """ Generator of hyperparameter sets for the sweep
    
    Note
    --------
    Written with LLM, not checked
    """
    if not sweep_params:
        yield {}
        return
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
    if not override:
        return default
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
        sweep_params = {} if sweep_params is None else sweep_params

    # Progress bar
    start = dt.now()
    logger.info(f'Sweep started')
    total_runs = sum(1 for _ in gen_sweep_hps(sweep_params))
    with logging_redirect_tqdm(), tqdm(
            gen_sweep_hps(sweep_params), total=total_runs,
            unit='run', dynamic_ncols=True,
            bar_format='{n_fmt}/{total_fmt} |{bar}| {desc}'
        ) as pbar:
            for sweep_hps in pbar:
                pbar.set_description_str('Updating hyperparameters')
                hps = merge_hps(default_hps, sweep_hps)
                hu = HyperparameterUpdater(hps, TEMPLATES_DIR)
                hu.update()

                pbar.set_description_str('Setting up environment')
                run_dir = None
                round_x_phase_x = re.compile(r'\*+\s*(Round \d+ Phase \d+)\s*\*+')
                with subprocess.Popen(['bash', str(RUN_SCRIPT_PATH)],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT,
                                      text=True) as p:
                    # This for loop prints simulation status and
                    # reads the output dir's path
                    for line in p.stdout:
                        line:str
                        round_match = round_x_phase_x.search(line)
                        if ('setup_environment succeeded' in line
                            or 'make: Entering directory' in line):
                            pbar.set_description_str('Building simulation')
                        elif line.startswith('Output directory: '):
                            docker_run_dir = Path(line.removeprefix('Output directory: ').strip())
                            run_dir = PROJECT_ROOT / docker_run_dir.relative_to(DOCKER_PROJECT_ROOT)
                        elif '==== LegoSim Chiplet Simulator ====' in line:
                            pbar.set_description_str('Initializing LEGOSim')
                        elif round_match:
                            pbar.set_description_str(round_match.group(1))
                        elif 'End of Simulation' in line:
                            pbar.set_description_str('Ending Simulation')
                    retcode = p.wait()

                if retcode != 0:
                    logger.warning(
                        f'Simulation to {run_dir} failed '
                        f'for sweep parameter set {sweep_hps!r}. '
                    )
                else:
                    pbar.set_description_str('Simulation completed')
    stop = dt.now()
    elapsed = stop - start
    d = elapsed.days
    h, remainder = divmod(elapsed.seconds, 3600)
    m, s = divmod(remainder, 60)
    logger.info(f'Sweep ended. Elapsed time: {d}d {h}h {m}m {s}s')
