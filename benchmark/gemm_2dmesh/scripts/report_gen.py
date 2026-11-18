import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
import numpy as np

load_dotenv()
PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT'))
PYUTILS_RDIR = Path(os.getenv('PYUTILS_RDIR'))
sys.path.insert(0, str(PROJECT_ROOT / PYUTILS_RDIR))
from xlogging import logger

# Mirrors of comms_protocol.h
M = 150
K = 60
N = 200
RNG_P1 = 7
RNG_P2 = 13
SEED_A = 321
SEED_B = 37
element_t = np.int64
MESH_X_DIM = 3
MESH_Y_DIM = 3
CPU_SLOT = (1, 1)
FLIT_BITS = 16 * 8

CPU_NODE = CPU_SLOT[0] + MESH_X_DIM * CPU_SLOT[1]
CPU_PROC_IDX = 0    # Position of the CPU in the YAML's phase1 list
TOPOLOGY_FILENAME = '2dmesh.gv'
REPORT_FILENAME = 'report.json'
LINK_JSON_PATH = Path(__file__).parent.parent / 'link_characteristics.json'


def matPrng(pos:int, colCnt:int, p1:int, p2:int, seed:int) -> int:
    """ Python version of the `matPrng` function in `cpu.cpp` """
    i = pos // colCnt
    j = pos % colCnt
    return (i*p1 + j*p2 + seed) % 51

def gen_input_matrix(mat_name:Literal['A','B']) -> np.ndarray:
    """ Generates matA and matB exactly as in `cpu.cpp` """
    row_cnt = M if mat_name == 'A' else K
    col_cnt = K if mat_name == 'A' else N
    seed = SEED_A if mat_name == 'A' else SEED_B
    retmat = np.ndarray((row_cnt, col_cnt), dtype=element_t)
    for i in range(row_cnt * col_cnt):
        retmat.flat[i] = matPrng(i, col_cnt, RNG_P1, RNG_P2, seed)
    return retmat

def verify_output_matrix(matA, matB, matC_path) -> bool:
    """ Calculates matA@matB and checks matC against the result """
    matC = np.fromfile(matC_path, dtype='<i8')
    if matC.size != M * N:
        logger.error(f'Output matrix at {matC_path} has size {matC.size}, '
                     f'{M * N} expected')
        return False
    matC = matC.reshape(M, N)
    return True if np.all(matA @ matB == matC) else False

def find_final_round(run_dir:Path) -> int:
    """ Reads the number of the last round from `coordinator.log`. Refuses
    runs whose round 1 loaded delay records left over by an earlier run. """
    log = (run_dir / 'coordinator.log').read_text()
    loaded = re.search(r'Load (\d+) delay records', log)
    if loaded is None or int(loaded.group(1)) != 0:
        logger.critical(f'Stale run directory {run_dir}: round 1 did not '
                        'start from 0 delay records, no metrics reported')
    rounds = re.findall(r'\*{4} Round (\d+) Phase 1 \*{4}', log)
    if not rounds:
        logger.critical(f'No round found in {run_dir / "coordinator.log"}')
    return int(rounds[-1])

def read_transfers(run_dir:Path) -> list[dict]:
    """ Pairs every `bench.txt` row with its `delayInfo.txt` row.

    Rows of the same (src, dst) pair are matched in time order, as the
    coordinator does.

    :param run_dir: Run output directory
    :return: One dict per transfer, sorted by send cycle. Keys:
            `src`, `dst`, `flits`, `send` (cycle the send is issued),
            `send_finish` (send + sender-side delay),
            `arrival` (send + receiver-side delay),
            `recv_finish` (max of arrival and the cycle the receive is
            issued; the value the coordinator applies).
    """
    bench = np.loadtxt(run_dir / 'bench.txt', dtype=np.int64, ndmin=2)
    delay = np.loadtxt(run_dir / 'delayInfo.txt', dtype=np.int64, ndmin=2)
    if len(bench) != len(delay):
        logger.critical(f'bench.txt has {len(bench)} rows but delayInfo.txt '
                        f'has {len(delay)}')
    delays = defaultdict(list)
    for _, src, dst, _, _, d_src, d_dst in delay[np.argsort(delay[:, 0])]:
        delays[(src, dst)].append((d_src, d_dst))

    transfers = []
    for send, recv_call, src, dst, flits, _ in bench[np.argsort(bench[:, 0])]:
        if not delays[(src, dst)]:
            logger.critical(f'No delay record left for transfer {src}->{dst}')
        d_src, d_dst = delays[(src, dst)].pop(0)
        transfers.append({
            'src': int(src), 'dst': int(dst), 'flits': int(flits),
            'send': int(send),
            'send_finish': int(send + d_src),
            'arrival': int(send + d_dst),
            'recv_finish': int(max(send + d_dst, recv_call)),
        })
    return transfers

def read_edges(gv_path:Path) -> list[tuple[int,int]]:
    """ Reads the undirected edges `a--b` from a Graphviz topology file """
    text = re.sub(r'//.*', '', gv_path.read_text())
    return [(int(a), int(b)) for a, b in re.findall(r'(\d+)\s*--\s*(\d+)', text)]

def mesh_cuts() -> dict[str,set]:
    """ Lists the straight cuts of the mesh.

    :return: Cut name -> set of nodes on the low-index side of the cut
    """
    nodes = [(x, y) for y in range(MESH_Y_DIM) for x in range(MESH_X_DIM)]
    cuts = {}
    for c in range(MESH_X_DIM - 1):
        cuts[f'col {c}|{c+1}'] = {x + MESH_X_DIM*y for x, y in nodes if x <= c}
    for r in range(MESH_Y_DIM - 1):
        cuts[f'row {r}|{r+1}'] = {x + MESH_X_DIM*y for x, y in nodes if y <= r}
    return cuts

def calc_latency(transfers:list[dict]) -> dict:
    """ Latency from the CPU's last send-finish to its last receive-finish.

    :return: Cycles of both events and the latency, with receive-finish
            taken as the coordinator's value (`coordinator`) and as the
            network arrival (`arrival`).
    """
    sent = [t for t in transfers if t['src'] == CPU_NODE]
    received = [t for t in transfers if t['dst'] == CPU_NODE]
    last_send_finish = max(t['send_finish'] for t in sent)
    last_recv_finish = max(t['recv_finish'] for t in received)
    last_arrival = max(t['arrival'] for t in received)
    return {
        'last_send_finish_cycle': last_send_finish,
        'last_recv_finish_cycle': {'coordinator': last_recv_finish,
                                   'arrival': last_arrival},
        'latency_cycles': {'coordinator': last_recv_finish - last_send_finish,
                           'arrival': last_arrival - last_send_finish},
    }

def calc_comm_window(transfers:list[dict]) -> dict:
    """ Window from the CPU's first send to its last receive-finish
    (coordinator's value). """
    start = min(t['send'] for t in transfers if t['src'] == CPU_NODE)
    end = max(t['recv_finish'] for t in transfers if t['dst'] == CPU_NODE)
    return {'start_cycle': start, 'end_cycle': end, 'cycles': end - start}

def calc_cut_throughput(transfers:list[dict], edges:list[tuple[int,int]],
                        window_cycles:int) -> dict:
    """ Measured traffic and capacity of every straight cut of the mesh.

    Popnet routes on shortest paths, so a transfer crosses a straight cut
    once if its endpoints are on opposite sides, and never otherwise.

    :param transfers: Output of `read_transfers`
    :param edges: Undirected edges of the topology
    :param window_cycles: Length of the communication window
    :return: Cut name -> `edges` crossing the cut, `capacity_bits_per_cycle`
            (both directions), `flits` that crossed (all flits, both
            directions), `measured_bits_per_cycle`, `utilisation`.
    """
    retdict = {}
    for name, side in mesh_cuts().items():
        edge_cnt = sum((a in side) != (b in side) for a, b in edges)
        flits = sum(t['flits'] for t in transfers
                    if (t['src'] in side) != (t['dst'] in side))
        capacity = edge_cnt * 2 * FLIT_BITS
        measured = flits * FLIT_BITS / window_cycles
        retdict[name] = {
            'edges': edge_cnt,
            'capacity_bits_per_cycle': capacity,
            'flits': flits,
            'measured_bits_per_cycle': measured,
            'utilisation': measured / capacity,
        }
    return retdict

def hop_count(src:int, dst:int) -> int:
    """ Hops on a shortest path between two mesh nodes """
    dx = abs(src % MESH_X_DIM - dst % MESH_X_DIM)
    dy = abs(src // MESH_X_DIM - dst // MESH_X_DIM)
    return dx + dy

def calc_area(edges:list[tuple[int,int]], link:dict, router:dict) -> dict:
    """ NoC area: every link direction has its own waveguide and interface.

    :param edges: Undirected edges of the topology
    :param link: `optical` block of the link characteristics
    :param router: `router` block of the link characteristics
    :return: `links_mm2`, `routers_mm2` and `total_mm2`
    """
    direction_area = (link['length_mm'] * link['routing_pitch_um'] * 1e-3
                      + link['interface_area_mm2_per_direction'])
    links = 2 * len(edges) * direction_area
    routers = len({n for e in edges for n in e}) * router['area_mm2_per_router']
    return {'links_mm2': links, 'routers_mm2': routers,
            'total_mm2': links + routers}

def calc_energy(transfers:list[dict], edges:list[tuple[int,int]],
                window_cycles:int, costs:dict) -> dict:
    """ NoC power and energy over the communication window.

    A flit on a path of h hops crosses h links and visits h + 1 routers.
    Static power covers every link direction and every router.

    :param transfers: Output of `read_transfers`
    :param edges: Undirected edges of the topology
    :param window_cycles: Length of the communication window
    :param costs: Contents of the link characteristics JSON
    :return: Window duration, static and dynamic power (W), static,
            dynamic and total energy (J), bits delivered (all flits), and
            energy per bit (J/bit) as `total` and `dynamic_only`.
    """
    link, router = costs['optical'], costs['router']
    e_link = link['dynamic_energy_j_per_transmitted_bit_hop']
    e_router = router['dynamic_energy_j_per_transmitted_bit_per_visit']
    window_s = window_cycles / costs['simulated_clock_hz']

    dynamic_link_j = dynamic_router_j = 0.0
    for t in transfers:
        bits = t['flits'] * FLIT_BITS
        hops = hop_count(t['src'], t['dst'])
        dynamic_link_j += bits * hops * e_link
        dynamic_router_j += bits * (hops + 1) * e_router
    dynamic_j = dynamic_link_j + dynamic_router_j

    static_link_w = 2 * len(edges) * link['static_power_w_per_direction']
    static_router_w = (len({n for e in edges for n in e})
                       * router['static_power_w_per_router'])
    static_w = static_link_w + static_router_w
    static_j = static_w * window_s

    bits_delivered = sum(t['flits'] for t in transfers) * FLIT_BITS
    return {
        'window_s': window_s,
        'static_power_w': {'links': static_link_w, 'routers': static_router_w,
                           'total': static_w},
        'dynamic_power_w': {'links': dynamic_link_j / window_s,
                            'routers': dynamic_router_j / window_s,
                            'total': dynamic_j / window_s},
        'energy_j': {'static': static_j, 'dynamic': dynamic_j,
                     'total': static_j + dynamic_j},
        'bits_delivered': bits_delivered,
        'energy_per_bit_j': {'total': (static_j + dynamic_j) / bits_delivered,
                             'dynamic_only': dynamic_j / bits_delivered},
    }

def calc_asic_breakdown(transfers:list[dict]) -> dict:
    """ Per-ASIC traffic and how long its C tile waited at the CPU.

    :param transfers: Output of `read_transfers`
    :return: ASIC node -> `hops` to the CPU, `flits_received` (A and B
            blocks), `flits_sent` (C tile), `tile_arrival_cycle`,
            `tile_recv_finish_cycle` and `tile_wait_cycles` (receive-finish
            minus arrival; 0 if the CPU was already waiting).
    """
    retdict = {}
    for t in transfers:
        if t['src'] == CPU_NODE:
            asic = retdict.setdefault(t['dst'], defaultdict(int))
            asic['hops'] = hop_count(CPU_NODE, t['dst'])
            asic['flits_received'] += t['flits']
        else:
            asic = retdict.setdefault(t['src'], defaultdict(int))
            asic['flits_sent'] += t['flits']
            asic['tile_arrival_cycle'] = t['arrival']
            asic['tile_recv_finish_cycle'] = t['recv_finish']
            asic['tile_wait_cycles'] = t['recv_finish'] - t['arrival']
    return {node: dict(retdict[node]) for node in sorted(retdict)}

def print_report(report:dict):
    latency = report['latency']
    window = report['comm_window']
    print(f'Run directory : {report["run_dir"]}')
    print(f'Final round   : {report["final_round"]}')
    print(f'Comm window   : {window["cycles"]} cycles '
          f'({window["start_cycle"]} -> {window["end_cycle"]})')
    print('Latency (CPU last send-finish -> last receive-finish)')
    for key in ('coordinator', 'arrival'):
        print(f'  {key:<12}: {latency["latency_cycles"][key]} cycles')
    print('Cut throughput (all flits, both directions)')
    print(f'  {"cut":<8} {"edges":>5} {"capacity b/cyc":>15} '
          f'{"measured b/cyc":>15} {"utilisation":>12}')
    for name, cut in report['cuts'].items():
        print(f'  {name:<8} {cut["edges"]:>5} '
              f'{cut["capacity_bits_per_cycle"]:>15} '
              f'{cut["measured_bits_per_cycle"]:>15.4f} '
              f'{cut["utilisation"]:>12.3e}')
    print('Per ASIC')
    print(f'  {"node":>4} {"hops":>4} {"flits in":>9} {"flits out":>9} '
          f'{"tile wait (cycles)":>19}')
    for node, asic in report['asics'].items():
        print(f'  {node:>4} {asic["hops"]:>4} {asic["flits_received"]:>9} '
              f'{asic["flits_sent"]:>9} {asic["tile_wait_cycles"]:>19}')
    area = report['area']
    print(f'Area          : {area["total_mm2"]:.4f} mm2 '
          f'(links {area["links_mm2"]:.4f}, '
          f'routers {area["routers_mm2"]:.4f})')
    energy = report['energy']
    print(f'Static power  : {energy["static_power_w"]["total"]:.4e} W')
    print(f'Dynamic power : {energy["dynamic_power_w"]["total"]:.4e} W')
    print(f'Energy        : {energy["energy_j"]["total"]:.4e} J '
          f'(static {energy["energy_j"]["static"]:.4e}, '
          f'dynamic {energy["energy_j"]["dynamic"]:.4e}) '
          f'over {energy["window_s"]:.4e} s')
    print(f'Energy per bit: {energy["energy_per_bit_j"]["total"]:.4e} J/bit '
          f'(dynamic only '
          f'{energy["energy_per_bit_j"]["dynamic_only"]:.4e})')

def main():
    parser = argparse.ArgumentParser(
        description='Report metrics of one gemm_2dmesh run')
    parser.add_argument('run_dir', type=Path, help='Run output directory')
    run_dir = parser.parse_args().run_dir.resolve()

    final_round = find_final_round(run_dir)
    matC_path = (run_dir / f'proc_r{final_round}_p1_t{CPU_PROC_IDX}'
                 / 'matC.bin')
    if not matC_path.is_file():
        logger.critical(f'Output matrix not found at {matC_path}')
    if not verify_output_matrix(gen_input_matrix('A'), gen_input_matrix('B'),
                                matC_path):
        logger.critical(f'Incorrect output matrix at {matC_path}, '
                        'no metrics reported')
    logger.debug('Checked AB == C')

    costs = json.loads(LINK_JSON_PATH.read_text())
    if costs['flit_bits'] != FLIT_BITS:
        logger.critical(f'{LINK_JSON_PATH} has flit_bits={costs["flit_bits"]}, '
                        f'{FLIT_BITS} expected')
    transfers = read_transfers(run_dir)
    edges = read_edges(run_dir / TOPOLOGY_FILENAME)
    window = calc_comm_window(transfers)
    report = {
        'run_dir': str(run_dir),
        'final_round': final_round,
        'flit_bits': FLIT_BITS,
        'comm_window': window,
        'latency': calc_latency(transfers),
        'cuts': calc_cut_throughput(transfers, edges, window['cycles']),
        'asics': calc_asic_breakdown(transfers),
        'area': calc_area(edges, costs['optical'], costs['router']),
        'energy': calc_energy(transfers, edges, window['cycles'], costs),
    }
    print_report(report)
    report_path = run_dir / REPORT_FILENAME
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    logger.debug(f'Wrote {report_path}')

if __name__ == '__main__':
    main()
