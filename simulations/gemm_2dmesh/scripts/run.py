import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT'))
PYUTILS_RDIR = Path(os.getenv('PYUTILS_RDIR'))
sys.path.insert(0, str(PROJECT_ROOT / PYUTILS_RDIR))
from xlogging import logger
from update_hyperparameters import HyperparameterUpdater, RUN_SCRIPT_PATH

SIMULATION_ROOT = PROJECT_ROOT / 'simulations/gemm_2dmesh'
HYPERPARAMS_PATH = SIMULATION_ROOT / 'hyperparameters.yml'
TEMPLATES_DIR = SIMULATION_ROOT / 'scripts/resource'

if __name__ == '__main__':
    hu = HyperparameterUpdater(HYPERPARAMS_PATH, TEMPLATES_DIR)
    hu.update()
    subprocess.run(['bash', str(RUN_SCRIPT_PATH)], check=True)
