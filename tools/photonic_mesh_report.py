#!/usr/bin/env python3
"""Estimate illustrative photonic link costs from a completed LEGOSim mesh run.

This accounting model reads the actual Popnet input trace. It is a placeholder
for device characterization, not an optical circuit or wavelength simulator.
"""

import argparse
import csv
import json
from collections import deque
from pathlib import Path
import re

from analyze_run import parse_coordinator, read_network_reports, read_workload_validation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOPOLOGY = ROOT / "benchmark/matmul_test/topology/photonic_mesh_2_2.gv"
DEFAULT_ASSUMPTIONS = ROOT / "benchmark/matmul_test/photonic_mesh_assumptions.json"


def read_edges(path):
    """Read simple numeric undirected DOT edges; reject unsupported edge syntax."""
    text = path.read_text()
    matches = re.findall(r"^\s*(\d+)\s*--\s*(\d+)\s*;\s*$", text, re.MULTILINE)
    if not matches:
        raise ValueError("No simple numeric undirected edges in %s" % path)
    edges = set()
    adjacency = {}
    for left, right in matches:
        a, b = int(left), int(right)
        if a == b:
            raise ValueError("Self-link in topology")
        edge = tuple(sorted((a, b)))
        if edge in edges:
            raise ValueError("Duplicate link %s" % (edge,))
        edges.add(edge)
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return edges, adjacency


def hop_count(adjacency, source, destination):
    if source not in adjacency or destination not in adjacency:
        raise ValueError("Trace endpoint missing from topology")
    queue = deque([(source, 0)])
    seen = {source}
    while queue:
        node, hops = queue.popleft()
        if node == destination:
            return hops
        for next_node in adjacency[node] - seen:
            seen.add(next_node)
            queue.append((next_node, hops + 1))
    raise ValueError("Disconnected trace endpoints: %s -> %s" % (source, destination))


def read_trace(path, adjacency, flit_bits):
    transfers = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError("Expected six fields in %s line %s" % (path, number))
        source_cycle, destination_cycle, source, destination, flits, descriptor = map(int, fields)
        if source_cycle < 0 or destination_cycle < 0 or flits < 1:
            raise ValueError("Invalid cycle or flit count in trace line %s" % number)
        if source == destination:
            raise ValueError("Local transfer in trace line %s needs separate accounting" % number)
        hops = hop_count(adjacency, source, destination)
        transfers.append({
            "source_cycle": source_cycle,
            "destination_cycle": destination_cycle,
            "source": source,
            "destination": destination,
            "flits_including_header": flits,
            "encoded_payload_bytes_upper_bound": (flits - 1) * flit_bits // 8,
            "hops": hops,
            "transmitted_bit_hops_including_header": flits * flit_bits * hops,
            "descriptor": descriptor,
        })
    if not transfers:
        raise ValueError("Empty communication trace")
    return transfers


def estimate(summary, transfers, edges, params):
    """Compute link subsystem metrics using explicit placeholder parameters."""
    frequency = params["simulated_clock_hz"]
    flit_bits = params["flit_bits"]
    directions = params["assumed_directions_per_edge"]
    if frequency <= 0 or flit_bits <= 0 or flit_bits % 8 or directions != 2:
        raise ValueError("Expected positive clock, whole-byte flits, and two directions per edge")
    for name in (
        "dynamic_energy_j_per_transmitted_bit_per_hop", "laser_power_w_per_direction",
        "ring_tuning_power_w_per_direction", "link_length_mm", "waveguide_pitch_um",
        "transceiver_area_mm2_per_direction", "router_dynamic_energy_j_per_transmitted_bit_per_router",
        "router_static_power_w_per_router", "router_area_mm2_per_router",
    ):
        if params[name] < 0:
            raise ValueError("Negative physical cost assumption: %s" % name)
    if params["wavelengths_per_direction"] * params["bit_rate_per_wavelength_bps"] != flit_bits * frequency:
        raise ValueError("Assumed WDM line rate must equal one flit per network cycle")
    directed_links = len(edges) * directions
    router_count = params["coordinate_width"] ** 2
    duration = summary["final_cycles"] / frequency
    bit_hops = sum(item["transmitted_bit_hops_including_header"] for item in transfers)
    bit_router_visits = sum(
        item["flits_including_header"] * flit_bits * (item["hops"] + 1)
        for item in transfers
    )
    payload_bound = sum(item["encoded_payload_bytes_upper_bound"] for item in transfers)
    dynamic = bit_hops * params["dynamic_energy_j_per_transmitted_bit_per_hop"]
    laser = directed_links * params["laser_power_w_per_direction"]
    tuning = directed_links * params["ring_tuning_power_w_per_direction"]
    static = (laser + tuning) * duration
    footprint_per_direction = params["link_length_mm"] * params["waveguide_pitch_um"] / 1000
    footprint = directed_links * footprint_per_direction
    transceiver_area = directed_links * params["transceiver_area_mm2_per_direction"]
    router_dynamic = bit_router_visits * params["router_dynamic_energy_j_per_transmitted_bit_per_router"]
    router_static = router_count * params["router_static_power_w_per_router"] * duration
    router_area = router_count * params["router_area_mm2_per_router"]
    return {
        "kind": "illustrative_photonic_mesh_noc",
        "status": summary["status"],
        "workload_completion_cycles": summary["final_cycles"],
        "assumed_workload_duration_s": duration,
        "number_of_undirected_edges": len(edges),
        "number_of_directed_optical_links": directed_links,
        "number_of_electrical_routers": router_count,
        "transfer_count": len(transfers),
        "transmitted_bit_hops_including_headers": bit_hops,
        "transmitted_bit_router_visits_including_headers": bit_router_visits,
        "encoded_payload_bytes_upper_bound": payload_bound,
        "encoded_payload_throughput_Bps_upper_bound": payload_bound / duration,
        "assumed_peak_bit_rate_bps_per_direction": params["wavelengths_per_direction"] * params["bit_rate_per_wavelength_bps"],
        "dynamic_optical_link_energy_j": dynamic,
        "laser_static_power_w": laser,
        "ring_tuning_static_power_w": tuning,
        "static_optical_link_energy_j": static,
        "total_optical_link_energy_j": dynamic + static,
        "waveguide_routing_footprint_mm2": footprint,
        "transceiver_area_mm2": transceiver_area,
        "total_optical_link_area_mm2": footprint + transceiver_area,
        "router_dynamic_energy_j": router_dynamic,
        "router_static_energy_j": router_static,
        "total_router_energy_j": router_dynamic + router_static,
        "total_noc_energy_j": dynamic + static + router_dynamic + router_static,
        "total_router_area_mm2": router_area,
        "total_noc_area_mm2": footprint + transceiver_area + router_area,
        "exclusions": [
            "Photonic device behavior beyond the fixed edge delay and stated cost assumptions",
            "Wavelength contention, laser gating, fabrication variation, and thermal coupling",
            "Workload numerical correctness",
        ],
    }


def plot_metrics(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(["Optical\ndynamic", "Optical\nstatic", "Router\ndynamic", "Router\nstatic"],
                [result["dynamic_optical_link_energy_j"] * 1e6,
                 result["static_optical_link_energy_j"] * 1e6,
                 result["router_dynamic_energy_j"] * 1e6,
                 result["router_static_energy_j"] * 1e6],
                color=["#246b91", "#e29c42", "#6b8797", "#ac8a67"])
    axes[0].set(ylabel="NoC energy (µJ)", title="Illustrative NoC energy")
    axes[1].bar(["Waveguide", "Transceivers", "Routers"],
                [result["waveguide_routing_footprint_mm2"],
                 result["transceiver_area_mm2"], result["total_router_area_mm2"]],
                color=["#54a88a", "#8f73ae", "#7794b7"])
    axes[1].set(ylabel="Estimated area (mm²)", title="Illustrative NoC area")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output / "photonic_link_metrics.png", dpi=180)
    fig.savefig(output / "photonic_link_metrics.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--topology", type=Path, help="Default: snapshot in run directory")
    parser.add_argument("--assumptions", type=Path, help="Default: snapshot in run directory")
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/analysis")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        summary = parse_coordinator((args.run_dir / "coordinator.log").read_text())
        topology = args.topology or args.run_dir / DEFAULT_TOPOLOGY.name
        assumptions = args.assumptions or args.run_dir / DEFAULT_ASSUMPTIONS.name
        params = json.loads(assumptions.read_text())
        edges, adjacency = read_edges(topology)
        if len(adjacency) != params["coordinate_width"] ** 2 or set(adjacency) != set(range(len(adjacency))):
            raise ValueError("Topology nodes must match the square coordinate width and IDs")
        if not re.search(r"edge\s*\[weight\s*=\s*%s\s*\]" % params["graph_edge_delay_cycles"], topology.read_text()):
            raise ValueError("Graph edge delay differs from the assumptions file")
        transfers = read_trace(args.run_dir / "bench.txt", adjacency, params["flit_bits"])
        result = estimate(summary, transfers, edges, params)
        reports = read_network_reports(args.run_dir)
        expected_reports = len(summary["rounds"]) - int(summary["status"] == "converged")
        if len(reports) != expected_reports:
            raise ValueError("Expected %s complete network reports, found %s" % (expected_reports, len(reports)))
        for report in reports:
            log_text = (args.run_dir / report["log"]).read_text(errors="replace")
            expected_config = {
                "ary size": len(adjacency),
                "cube dimension": 1,
                "flit size": params["flit_bits"] // 64,
                "Routing algorithm": 4,
            }
            for name, value in expected_config.items():
                if not re.search(r"^%s:\s*%s\s*$" % (re.escape(name), value), log_text, re.MULTILINE):
                    raise ValueError("Popnet %s does not match %s=%s" % (report["log"], name, value))
            if report["finished_packets"] != len(transfers):
                raise ValueError("Network packet count differs from final trace")
        result["last_completed_network_report"] = reports[-1] if reports else None
        result["workload_validation"] = read_workload_validation(args.run_dir, len(summary["rounds"]))
        if result["workload_validation"]["status"] == "passed":
            result["exclusions"].remove("Workload numerical correctness")
        result["assumptions"] = params
        result["topology_file"] = str(topology)
        result["assumptions_file"] = str(assumptions)
        result["trace_basis"] = "Last phase-1 bench.txt; may be from a later round than the last phase-2 network report."
        output = args.output or args.run_dir / "analysis"
        output.mkdir(parents=True, exist_ok=True)
        (output / "photonic_link_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        with (output / "photonic_trace.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(transfers[0]))
            writer.writeheader()
            writer.writerows(transfers)
        with (output / "metrics.csv").open("w", newline="") as handle:
            row = {
                "scenario": args.run_dir.name,
                "status": result["status"],
                "workload_completion_cycles": result["workload_completion_cycles"],
                "network_report_round": reports[-1]["round"] if reports else "",
                "network_average_packet_latency_cycles": (
                    reports[-1]["average_packet_latency_network_cycles"] if reports else ""
                ),
                "encoded_payload_throughput_MBps_upper_bound": (
                    result["encoded_payload_throughput_Bps_upper_bound"] / 1e6
                ),
                "optical_link_energy_uJ": result["total_optical_link_energy_j"] * 1e6,
                "optical_link_area_mm2": result["total_optical_link_area_mm2"],
                "total_noc_energy_uJ": result["total_noc_energy_j"] * 1e6,
                "total_noc_area_mm2": result["total_noc_area_mm2"],
            }
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        if args.plot:
            plot_metrics(result, output)
    except (OSError, ValueError, KeyError, ImportError, json.JSONDecodeError) as error:
        parser.exit(1, "Photonic report failed: %s\n" % error)
    print("%s: %s encoded payload bytes (upper bound), %.3f µJ illustrative NoC energy; %s" % (
        result["status"], result["encoded_payload_bytes_upper_bound"],
        result["total_noc_energy_j"] * 1e6, output))


if __name__ == "__main__":
    main()
