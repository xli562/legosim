import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import report_gen as rg

FIGURES_DIRNAME = 'figures'
SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3df'
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'font.size': 15,
    'font.family': 'DejaVu Sans', 'text.color': TEXT,
    'axes.labelcolor': TEXT_SECONDARY, 'axes.edgecolor': GRID,
    'xtick.color': TEXT_SECONDARY, 'ytick.color': TEXT_SECONDARY,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.axisbelow': True,
    'grid.color': GRID, 'grid.linewidth': 1.0,
    'legend.frameon': False, 'ytick.left': False,
})


def new_figure(title:str, subtitle:str, height:float=6.5):
    """ Creates a 16:9-style figure with a left-aligned title block.

    :return: (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(12.8, height), dpi=200)
    fig.subplots_adjust(left=0.19, right=0.95, top=1 - 1.75/height,
                        bottom=1.0/height)
    fig.text(0.03, 1 - 0.42/height, title, fontsize=21, fontweight='bold',
             va='center')
    fig.text(0.03, 1 - 0.85/height, subtitle, fontsize=14,
             color=TEXT_SECONDARY, va='center')
    return fig, ax

def asic_nodes(transfers:list[dict]) -> list[int]:
    """ ASIC nodes in the order the CPU first sends to them """
    nodes = []
    for t in transfers:
        if t['src'] == rg.CPU_NODE and t['dst'] not in nodes:
            nodes.append(t['dst'])
    return nodes

def plot_timeline(transfers:list[dict], report:dict, out_path:Path):
    """ Message events of every ASIC across the communication window """
    nodes = asic_nodes(transfers)
    start = report['comm_window']['start_cycle']
    fig, ax = new_figure(
        'The CPU, not the network, sets the pace',
        'Preparing one ASIC\'s inputs takes the CPU about 290k cycles; '
        'a packet is on the wire for about 2k.')
    ax.set_ylim(-0.7, len(nodes) - 0.5)
    styles = {'A': (BLUE, 'o', 'A block sent'),
              'B': (ORANGE, 's', 'B block sent'),
              'C': (AQUA, '^', 'C tile received')}
    for row, node in enumerate(nodes):
        sends = [t for t in transfers if t['dst'] == node]
        tile = next(t for t in transfers if t['src'] == node)
        events = {'A': sends[0]['send'], 'B': sends[1]['send'],
                  'C': tile['recv_finish']}
        ax.hlines(row, 0, report['comm_window']['cycles'] / 1e6,
                  color=GRID, linewidth=1)
        for key, cycle in events.items():
            color, marker, _ = styles[key]
            # The tile returns ~0.3% of the axis after the B block, so
            # lift its mark off the line to keep both visible
            y = row - 0.3 if key == 'C' else row
            ax.plot((cycle - start) / 1e6, y, marker=marker, color=color,
                    markersize=11, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, linestyle='none')
    ax.set_yticks(range(len(nodes)), [f'ASIC node {n}' for n in nodes])
    ax.invert_yaxis()
    ax.set_xlabel('Cycles since the CPU\'s first send (millions)')
    ax.set_xlim(-0.03, report['comm_window']['cycles'] / 1e6 + 0.03)
    ax.legend(handles=[Line2D([], [], marker=m, color=c, linestyle='none',
                              markersize=11, label=label)
                       for c, m, label in styles.values()],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3,
              handletextpad=0.2, columnspacing=1.5)
    fig.savefig(out_path)
    plt.close(fig)

def plot_round_trip(transfers:list[dict], out_path:Path):
    """ Where each ASIC's round trip goes, from B block send to tile
    receive-finish """
    nodes = asic_nodes(transfers)
    parts = [('B block on the wire', BLUE), ('ASIC compute', ORANGE),
             ('C tile on the wire', AQUA), ('Tile waits for the CPU', YELLOW)]
    fig, ax = new_figure(
        'Round trip per ASIC: only the CPU wake-up varies',
        'Cycles from the CPU sending the B block to the CPU finishing its '
        'receive of the C tile, per ASIC.')
    for row, node in enumerate(nodes):
        b_block = [t for t in transfers if t['dst'] == node][1]
        tile = next(t for t in transfers if t['src'] == node)
        edges = [b_block['send'], b_block['arrival'], tile['send'],
                 tile['arrival'], tile['recv_finish']]
        for (_, color), left, right in zip(parts, edges[:-1], edges[1:]):
            ax.barh(row, right - left, left=left - edges[0], height=0.6,
                    color=color, edgecolor=SURFACE, linewidth=2)
        hops = rg.hop_count(rg.CPU_NODE, node)
        ax.text(edges[-1] - edges[0] + 250, row,
                f'{edges[-1] - edges[0]:,}', va='center', fontsize=13,
                color=TEXT_SECONDARY)
        nodes[row] = f'Node {node} ({hops} hop{"s" if hops > 1 else ""})'
    ax.set_yticks(range(len(nodes)), nodes)
    ax.invert_yaxis()
    ax.set_xlabel('Cycles since the B block was sent')
    ax.xaxis.set_major_formatter(lambda x, _: f'{x:,.0f}')
    ax.grid(axis='x')
    ax.margins(x=0.08)
    ax.legend(handles=[Patch(color=c, label=label) for label, c in parts],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=4,
              handlelength=1.0, handletextpad=0.5, columnspacing=1.5)
    fig.savefig(out_path)
    plt.close(fig)

def plot_power(report:dict, out_path:Path):
    """ Static and dynamic NoC power by component, on a log axis """
    energy = report['energy']
    static_share = energy['energy_j']['static'] / energy['energy_j']['total']
    bars = [('Links, static', energy['static_power_w']['links'], BLUE),
            ('Routers, static', energy['static_power_w']['routers'], BLUE),
            ('Routers, dynamic', energy['dynamic_power_w']['routers'], ORANGE),
            ('Links, dynamic', energy['dynamic_power_w']['links'], ORANGE)]
    fig, ax = new_figure(
        f'Idle optical links draw {static_share:.0%} of the NoC energy',
        'Average NoC power over the communication window (log scale). '
        f'Energy per bit: {energy["energy_per_bit_j"]["total"]*1e12:.0f} pJ '
        f'in total, {energy["energy_per_bit_j"]["dynamic_only"]*1e12:.2f} pJ '
        'dynamic only.', height=5.4)
    for row, (_, watts, color) in enumerate(bars):
        ax.barh(row, watts * 1e3, height=0.55, color=color)
        ax.text(watts * 1e3 * 1.12, row, f'{watts*1e3:.3g} mW', va='center',
                fontsize=14)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars])
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(lambda x, _: f'{x:g}')
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.set_xlim(0.1, 2000)
    ax.set_xlabel('Power (mW)')
    ax.grid(axis='x')
    ax.legend(handles=[Patch(color=BLUE, label='Static (always on)'),
                       Patch(color=ORANGE, label='Dynamic (traffic)')],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2,
              handlelength=1.0, handletextpad=0.5)
    fig.savefig(out_path)
    plt.close(fig)

def plot_area(report:dict, costs:dict, out_path:Path):
    """ NoC area by component """
    link = costs['optical']
    directions = round(report['area']['links_mm2']
                       / (link['length_mm'] * link['routing_pitch_um'] * 1e-3
                          + link['interface_area_mm2_per_direction']))
    bars = [('Waveguides', directions * link['length_mm']
                           * link['routing_pitch_um'] * 1e-3),
            ('Routers', report['area']['routers_mm2']),
            ('Link interfaces',
             directions * link['interface_area_mm2_per_direction'])]
    fig, ax = new_figure(
        f'NoC area is {report["area"]["total_mm2"]:.2f} mm², '
        'mostly waveguide routing',
        f'{directions} link directions of {link["length_mm"]} mm at '
        f'{link["routing_pitch_um"]:.0f} µm pitch, plus 9 routers.',
        height=4.6)
    for row, (_, area) in enumerate(bars):
        ax.barh(row, area, height=0.55, color=BLUE)
        ax.text(area + 0.015, row, f'{area:.3f} mm²', va='center',
                fontsize=14)
    ax.set_yticks(range(len(bars)), [b[0] for b in bars])
    ax.invert_yaxis()
    ax.set_xlabel('Area (mm²)')
    ax.set_xlim(0, max(b[1] for b in bars) * 1.2)
    ax.grid(axis='x')
    fig.savefig(out_path)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description='Plot the report of one gemm_2dmesh run')
    parser.add_argument('run_dir', type=Path, help='Run output directory '
                        'that already holds report.json')
    run_dir = parser.parse_args().run_dir.resolve()

    report = json.loads((run_dir / rg.REPORT_FILENAME).read_text())
    costs = json.loads(rg.LINK_JSON_PATH.read_text())
    transfers = rg.read_transfers(run_dir)
    out_dir = run_dir / FIGURES_DIRNAME
    out_dir.mkdir(exist_ok=True)

    plot_timeline(transfers, report, out_dir / 'timeline.png')
    plot_round_trip(transfers, out_dir / 'round_trip.png')
    plot_power(report, out_dir / 'power.png')
    plot_area(report, costs, out_dir / 'area.png')
    print(f'Wrote figures to {out_dir}')

if __name__ == '__main__':
    main()
