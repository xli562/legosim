import os
import sys
import copy
from pathlib import Path
import yaml
from pint import UnitRegistry
from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT'))
PYUTILS_RDIR = Path(os.getenv('PYUTILS_RDIR'))
sys.path.insert(0, str(PROJECT_ROOT / PYUTILS_RDIR))
from xlogging import logger

from models import ASIC

HYPERPARAMS_PATH = PROJECT_ROOT / 'benchmark/gemm_2dmesh/hyperparameters.yml'
TEMPLATES_DIR = PROJECT_ROOT / 'benchmark/gemm_2dmesh/scripts/resource'
PINT_UNITS_PATH = PROJECT_ROOT / 'dependencies/pyutils/resource/pint_units.txt'

ureg = UnitRegistry(PINT_UNITS_PATH)

class HyperparameterUpdater:
    OUTPUT_DIR = PROJECT_ROOT / 'benchmark/gemm_2dmesh/cfg'
    SIM_CONFIG_TEMPLATE_FNAME = 'sim_config_template.yml'

    def __init__(self, hyperparams_path:Path, templates_dir:Path):
        # Input processing
        required_templates = {self.SIM_CONFIG_TEMPLATE_FNAME}
        templates = {f.name for f in templates_dir.iterdir() if f.is_file()}
        missing_templates = required_templates - templates
        if missing_templates:
            raise FileNotFoundError('Required templates not found '
                                    f'in {templates_dir}: '
                                    f'{', '.join(sorted(missing_templates))}')

        with open(hyperparams_path, 'r', encoding='utf-8') as f:
            self.hyperparams:dict = yaml.safe_load(f)
        self._init_asic()
        self._init_sim_config(templates_dir)

    @property
    def _asic_cnt(self) -> int:
        mesh_dim = self.hyperparams['topology']['mesh-dimension']['value']
        return mesh_dim[0] * mesh_dim[1] - 1

    @property
    def _mkn(self) -> list:
        r""" List of M, K, and N.

        Note
        -------- 
        Matrix sizes: A (M\*K), B (K\*N), C(M\*N)
        """
        return self.hyperparams['workload']['matrices-dimensions']['value']

    def _init_sim_config(self, templates_dir:Path):
        """ Initializes `self.sim_config` from the sim_config template """
        sim_config_path = templates_dir / self.SIM_CONFIG_TEMPLATE_FNAME
        with open(sim_config_path, 'r', encoding='utf-8') as f:
            self.sim_config:dict = yaml.safe_load(f)
        asic_template = self.sim_config['phase1'].pop()
        for asic_id in range(self._asic_cnt):
            asic_config = copy.deepcopy(asic_template)
            asic_config['args'] = [str(asic_id), self.asic.latency]
            asic_config['log'] = f'asic_{asic_id}.log'
            self.sim_config['phase1'].append(asic_config)

    def _init_asic(self):
        """ Initializes the ASICs' analytical model """
        self.asic = ASIC(self._mkn)

    def _clock(self):
        """ Update the clock hyperparameters
         
        Note
        --------
        `clock_rate` in the sim_config is (simlet_clk_freq)/(e-NoC_clk_freq)
        """
        # The e-NoC clock is used as reference frequency for all other clocks.
        enoc_clk_config = self.hyperparams['clock']['electrical-network']
        enoc_freq = ureg.Quantity(enoc_clk_config['value'],
                                  enoc_clk_config['unit'])

        # CPU clock
        cpu_clk_config = self.hyperparams['clock']['cpu']
        cpu_freq = ureg.Quantity(cpu_clk_config['value'],
                                 cpu_clk_config['unit'])
        # Update clock rate
        cpu_clk_rate = (cpu_freq / enoc_freq).m_as('rate')
        self.sim_config['phase1'][0]['clock_rate'] = cpu_clk_rate
        # Update SniperSim arguments
        cpu_ghz = str(cpu_freq.m_as('GHz'))
        cpu_freq_args = ['-g', f'perf_model/core/frequency={cpu_ghz}']
        self.sim_config['phase1'][0]['args'][:0] = cpu_freq_args

        # ASIC clock
        asic_clk_config = self.hyperparams['clock']['asic']
        asic_freq = ureg.Quantity(asic_clk_config['value'],
                                  asic_clk_config['unit'])
        asic_clk_rate = (asic_freq / enoc_freq).m_as('rate')
        for asic_id in range(self._asic_cnt):
            self.sim_config['phase1'][asic_id+1]['clock_rate'] = asic_clk_rate

    def update(self):
        self._clock()
        output_path = self.OUTPUT_DIR / 'sim_config.yml'
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.sim_config, f, indent=2)


if __name__ == '__main__':
    hu = HyperparameterUpdater(HYPERPARAMS_PATH, TEMPLATES_DIR)
    print(hu.sim_config['phase1'])
    hu.update()