import os
import sys
import copy
from pathlib import Path
import yaml
from pint import UnitRegistry
from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT'))
DOCKER_PROJECT_ROOT = Path(os.getenv('DOCKER_PROJECT_ROOT'))
PYUTILS_RDIR = Path(os.getenv('PYUTILS_RDIR'))
sys.path.insert(0, str(PROJECT_ROOT / PYUTILS_RDIR))
from xlogging import logger

from models import ASIC
from math_utils import factor_pairs
from file_utils import replace_cpp_vars, Mesh2DGraph, RunScriptBuilder

SIMULATION_RDIR = Path('simulations/gemm_2dmesh')
SIMULATION_ROOT = PROJECT_ROOT / SIMULATION_RDIR
DOCKER_SIMULATION_ROOT = DOCKER_PROJECT_ROOT / SIMULATION_RDIR
CONFIG_DIR = SIMULATION_ROOT / 'cfg'
SIM_CONFIG_PATH = CONFIG_DIR / 'sim_config.yml'
MESH_GRAPHVIZ_PATH = CONFIG_DIR / '2dmesh.gv'
RUN_SCRIPT_PATH = CONFIG_DIR / 'run.sh'
DOCKER_CONFIG_DIR = DOCKER_SIMULATION_ROOT / 'cfg'
DOCKER_SIM_CONFIG_PATH = DOCKER_CONFIG_DIR / 'sim_config.yml'
DOCKER_MESH_GRAPHVIZ_PATH = DOCKER_CONFIG_DIR / '2dmesh.gv'
DOCKER_RUN_SCRIPT_PATH = DOCKER_CONFIG_DIR / 'run.sh'  
COMMS_PROTOCOL_H_PATH = SIMULATION_ROOT / 'src/comms_protocol.h'

PINT_UNITS_PATH = PROJECT_ROOT / 'dependencies/pyutils/resource/pint_units.txt'
ureg = UnitRegistry(PINT_UNITS_PATH)

def _chiplet_layout(mesh_x_dim, mesh_y_dim, cpu_pos) -> tuple[str,str]:
        """ Returns chiplet layout based on mesh dimension and cpu position

        :return asic_layout: 
        :return cpu_layout:
        
        Note
        --------
        Written with LLM, not checked
        """
        cpu_pos = tuple(cpu_pos)
        rows = []
        for x in range(mesh_x_dim):
            coords = [
                f"{{{x},{y}}}"
                for y in range(mesh_y_dim)
                if (x, y) != cpu_pos
            ]
            if coords:
                rows.append("    " + ", ".join(coords))
        asic_layout = "{\n" + ",\n".join(rows) + "\n}"
        cpu_layout = f"{{\n    {{{cpu_pos[0]},{cpu_pos[1]}}}\n}}"
        return asic_layout, cpu_layout

class HyperparameterUpdater:
    SIM_CONFIG_TEMPLATE_FNAME = 'sim_config_template.yml'
    RUN_SCRIPT_TEMPLATE_FNAME = 'run_template.sh'

    def __init__(self, hyperparams:Path|dict, templates_dir:Path):
        # Input processing
        required_templates = {self.SIM_CONFIG_TEMPLATE_FNAME,
                              self.RUN_SCRIPT_TEMPLATE_FNAME}
        templates = {f.name for f in templates_dir.iterdir() if f.is_file()}
        missing_templates = required_templates - templates
        if missing_templates:
            raise FileNotFoundError('Required templates not found '
                                    f'in {templates_dir}: '
                                    f'{', '.join(sorted(missing_templates))}')

        if isinstance(hyperparams, Path):
            with open(hyperparams, 'r', encoding='utf-8') as f:
                self.hp:dict = yaml.safe_load(f)
        elif isinstance(hyperparams, dict):
            self.hp = hyperparams
        else:
            raise ValueError('hyperparams must be either a dictionary'
                             'or a path to a YAML file. '
                             f'Got {hyperparams}')
        self._init_asic()
        self._init_sim_config(templates_dir)
        self.graph = Mesh2DGraph()
        self._init_run_script_builder(templates_dir)
        self.comms_prtc_replacements = []

    @property
    def _asic_cnt(self) -> int:
        mesh_dim = self.hp['topology']['mesh-dimension']['value']
        return mesh_dim[0] * mesh_dim[1] - 1

    @property
    def _mesh_xy_dim(self) -> tuple:
        return self.hp['topology']['mesh-dimension']['value']

    @property
    def _mkn(self) -> list:
        r""" List of M, K, and N.

        Note
        -------- 
        Matrix sizes: A (M\*K), B (K\*N), C(M\*N)
        """
        return self.hp['workload']['matrices-dimensions']['value']

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
        self.asic = ASIC(*self._mkn)

    def _init_run_script_builder(self, templates_dir:Path):
        """ Initializes `self.run_script_builder` """
        template_script_path = templates_dir / self.RUN_SCRIPT_TEMPLATE_FNAME
        self.run_script_builder = RunScriptBuilder(template_script_path)
        self.run_script_builder.fill('SIMULATION_ROOT', DOCKER_SIMULATION_ROOT)
        self.run_script_builder.fill('SIM_CONFIG_YAML_PATH', DOCKER_SIM_CONFIG_PATH)
        self.run_script_builder.fill('GRAPHVIZ_FILE_PATH', DOCKER_MESH_GRAPHVIZ_PATH)

    def _clock(self):
        """ Update the clock hyperparameters
         
        Note
        --------
        `clock_rate` in the sim_config is (simlet_clk_freq)/(e-NoC_clk_freq)
        """
        # The e-NoC clock is used as reference frequency for all other clocks.
        enoc_clk_config = self.hp['clock']['electrical-network']
        enoc_freq = ureg.Quantity(enoc_clk_config['value'],
                                  enoc_clk_config['unit'])

        # CPU clock
        cpu_clk_config = self.hp['clock']['cpu']
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
        asic_clk_config = self.hp['clock']['asic']
        asic_freq = ureg.Quantity(asic_clk_config['value'],
                                  asic_clk_config['unit'])
        asic_clk_rate = (asic_freq / enoc_freq).m_as('rate')
        for asic_id in range(self._asic_cnt):
            self.sim_config['phase1'][asic_id+1]['clock_rate'] = asic_clk_rate

    def _topology(self):
        # Input check
        M, K, N = self._mkn
        valid_tile_layouts = factor_pairs(self._asic_cnt)
        for valid_tile_layout in valid_tile_layouts:
            row_block_count, col_block_count = valid_tile_layout
            if (M % row_block_count == 0) and (N % col_block_count == 0):
                break
        else:
            raise ValueError('Infeasible matrix dimensions '
                             f'M = {M}, K = {K}, N = {N}. '
                             '(M, N) needs to be divisible by '
                             f'at least one of {valid_tile_layouts}')

        # Mesh dimension
        # Update dimension definitions in the C++ header file
        mesh_x_dim, mesh_y_dim = self._mesh_xy_dim
        self.comms_prtc_replacements.append(['static constexpr int',
                                             'MESH_X_DIM', mesh_x_dim])
        self.comms_prtc_replacements.append(['static constexpr int',
                                             'MESH_Y_DIM', mesh_y_dim])
        # Update SniperSim core count and Popnet node count
        cpu_core_count_args = ['-n', str(self._asic_cnt)]
        self.sim_config['phase1'][0]['args'][:0] = cpu_core_count_args
        popnet_node_count_args = ['-A', str(self._asic_cnt)]
        self.sim_config['phase2'][0]['args'][:0] = popnet_node_count_args
        # Update the graphviz topology description
        self.graph.build_mesh(mesh_x_dim, mesh_y_dim)
        # Update the run script
        self.run_script_builder.fill('MESH_WIDTH', mesh_x_dim)
        self.run_script_builder.fill('MESH_HEIGHT', mesh_y_dim)

        # CPU position
        # Update chiplet layout definitions in the C++ header file
        asic_layout, cpu_layout = _chiplet_layout(
            *self._mesh_xy_dim,
            self.hp['topology']['cpu-position']['value']
        )
        self.comms_prtc_replacements.append(['static constexpr Mesh2DCoord',
                                             'ASIC_SLOTS[]', asic_layout])
        self.comms_prtc_replacements.append(['static constexpr Mesh2DCoord',
                                             'CPU_SLOTS[]', cpu_layout])

        # Flit size
        flit_size = int(ureg.Quantity(
            self.hp['topology']['flit-size']['value'],
            self.hp['topology']['flit-size']['unit']).m_as('byte'))
        # Suggested by LLM: 'FLITS_PER_PACKAGE' = flit_size / 64
        if flit_size % 8 != 0:
            raise ValueError(f'Cannot support flit size {flit_size}. '
                             'Flit size must be multiples of 8 bytes')
        # Update the run script
        self.run_script_builder.fill('FLITS_PER_PACKAGE', flit_size // 8)
        # Update Popnet args in sim_config
        popnet_flit_args = ['-F', str(flit_size // 8)]
        self.sim_config['phase2'][0]['args'][:0] = popnet_flit_args
        # comms_protocol.h
        self.comms_prtc_replacements.append(['static constexpr int',
                                             'FLIT_SIZE', flit_size])

    def _workload(self):
        # Compatilibity with the specified topology is
        # checked in self._topology()
        M, K, N = self._mkn
        self.comms_prtc_replacements.append(['static constexpr int', 'M', M])
        self.comms_prtc_replacements.append(['static constexpr int', 'K', K])
        self.comms_prtc_replacements.append(['static constexpr int', 'N', N])

    def _router(self):
        # Number of virtual channels and their depths
        vc_count = self.hp['router']['virtual-channel-count']['value']
        vc_depth = self.hp['router']['virtual-channel-depth']['value']
        popnet_vc_args = ['-V', str(vc_count), '-B', str(vc_depth)]
        self.sim_config['phase2'][0]['args'][:0] = popnet_vc_args

    def update(self):
        self._clock()
        self._topology()
        # Write the graphviz topology file
        self.graph.save(MESH_GRAPHVIZ_PATH)
        # Write the simulation's run script
        self.run_script_builder.write_file(RUN_SCRIPT_PATH)
        # Write the comms protocol C++ header file
        replace_cpp_vars(COMMS_PROTOCOL_H_PATH,
                         self.comms_prtc_replacements)
        # Write the sim_config YAML file
        with open(SIM_CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.sim_config, f, indent=2)
