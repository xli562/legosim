#!/usr/bin/env python3
"""Run a finite-window Popnet offered-load study on generated mesh graphs.

This measures delivered flits and completed-packet latency under synthetic
traffic. It is separate from the checked matrix workload and does not model
optical wavelength contention or device energy.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import subprocess

from mesh_study import (ROOT, check_assumptions, graph_text, load_spec,
                        verify_edge_profile_sources)


POPNET = ROOT / "popnet_chiplet/build/popnet"
CONTAINER_ROOT = "/workspace/sim"


def container_path(path):
    return CONTAINER_ROOT + "/" + path.resolve().relative_to(ROOT).as_posix()


def generate_trace(nodes, rate, cycles, packet_flits, seed, pattern="uniform",
                   hotspot=0):
    """Use the same random draw at every source/cycle across rate settings."""
    if nodes < 2 or not 0 < rate <= 1 or cycles <= 0 or packet_flits <= 0:
        raise ValueError("Invalid traffic size, injection probability, or packet length")
    if pattern not in ("uniform", "hotspot") or not 0 <= hotspot < nodes:
        raise ValueError("Invalid traffic pattern or hotspot node")
    rng = random.Random(seed)
    packets = []
    for cycle in range(cycles):
        for source in range(nodes):
            inject = rng.random()
            choice = rng.randrange(nodes - 1)
            random_destination = choice + int(choice >= source)
            destination = (hotspot if pattern == "hotspot" and source != hotspot
                           else random_destination)
            if inject < rate:
                packets.append((cycle, source, destination, packet_flits))
    return packets


def parse_completions(path, packets, expected_total):
    offered = Counter((cycle, source, destination)
                      for cycle, source, destination, _ in packets)
    completed = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError("Invalid Popnet completion record on line %d" % number)
        start, source, destination, delay = map(int, fields)
        key = (start, source, destination)
        if delay < 0 or offered[key] < 1:
            raise ValueError("Completion does not match offered packet on line %d" % number)
        offered[key] -= 1
        completed.append((start, source, destination, delay))
    if len(completed) != expected_total:
        raise ValueError("Popnet finished count differs from completion records")
    return completed


def summarize_window(packets, completions, warmup, measure, nodes, packet_flits):
    end = warmup + measure
    offered = sum(warmup <= packet[0] < end for packet in packets)
    received = [item for item in completions if warmup <= item[0] + item[3] < end]
    delays = sorted(item[3] for item in received)
    return {
        "injected_packets_total": len(packets),
        "completed_packets_total": len(completions),
        "undelivered_packets_at_horizon": len(packets) - len(completions),
        "offered_packets_in_window": offered,
        "delivered_packets_in_window": len(received),
        "offered_flits_per_node_cycle": offered * packet_flits / (nodes * measure),
        "delivered_flits_per_node_cycle": len(received) * packet_flits / (nodes * measure),
        "completed_packet_mean_latency_cycles": statistics.mean(delays) if delays else None,
        "completed_packet_p95_latency_cycles": delays[math.ceil(0.95 * len(delays)) - 1]
        if delays else None,
    }


def run_case(scenario_path, spec, params, rate, rate_index, seed, trace_path, packets,
             output, warmup, measure, packet_flits, pattern, container):
    case = output / spec["name"] / ("seed_%d" % seed) / ("rate_%02d" % rate_index)
    work = case / "work"
    work.mkdir(parents=True, exist_ok=True)
    graph = case / "topology.gv"
    graph.write_text(graph_text(spec, params))
    (case / "scenario.json").write_text(json.dumps(spec, indent=2) + "\n")
    (case / "link_assumptions.json").write_text(json.dumps(params, indent=2) + "\n")
    log_path = case / "popnet.log"
    command = ["docker", "exec", "-w", container_path(work), container,
               container_path(POPNET), "-A", str(params["coordinate_width"] ** 2),
               "-c", "1", "-V", "3", "-B", "12", "-O", "12", "-F", "2",
               "-L", "1000", "-T", str(warmup + measure), "-r", "1",
               "-I", container_path(trace_path), "-G", container_path(graph),
               "-R", "4", "-D", "delayInfo.txt"]
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError("Popnet failed for %s at rate %g; see %s" %
                           (spec["name"], rate, log_path))
    text = log_path.read_text(errors="replace")
    finished = re.findall(r"^total finished:\s*(\d+)\s*$", text, re.MULTILINE)
    if len(finished) != 1:
        raise ValueError("Popnet finished summary missing or duplicated in %s" % log_path)
    completions = parse_completions(work / "delayInfo.txt", packets, int(finished[0]))
    row = {"scenario": spec["name"], "seed": seed,
           "requested_packets_per_node_cycle": rate,
           "pattern": "uniform_random" if pattern == "uniform" else "hotspot",
           "packet_flits": packet_flits, "warmup_cycles": warmup,
           "measurement_cycles": measure,
           **summarize_window(packets, completions, warmup, measure,
                              params["coordinate_width"] ** 2, packet_flits)}
    row["delivered_aggregate_Gbps_including_headers"] = (
        row["delivered_flits_per_node_cycle"] * params["coordinate_width"] ** 2 *
        params["flit_bits"] * params["simulated_clock_hz"] / 1e9)
    (case / "result.json").write_text(json.dumps({
        "metrics": row,
        "command": command,
        "scenario_source": str(scenario_path.resolve()),
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return row


def summarize_seeds(rows):
    """Report seed spread without treating it as physical-model uncertainty."""
    groups = {}
    for row in rows:
        key = (row["scenario"], row["requested_packets_per_node_cycle"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (scenario, rate), group in sorted(groups.items()):
        offered = [row["offered_flits_per_node_cycle"] for row in group]
        delivered = [row["delivered_flits_per_node_cycle"] for row in group]
        latency = [row["completed_packet_mean_latency_cycles"] for row in group]
        if len({row["seed"] for row in group}) != len(group):
            raise ValueError("Repeated seed in offered-load summary")
        valid_latency = [value for value in latency if value is not None]
        summary.append({
            "scenario": scenario,
            "requested_packets_per_node_cycle": rate,
            "seed_count": len(group),
            "offered_flits_per_node_cycle_mean": statistics.mean(offered),
            "delivered_flits_per_node_cycle_mean": statistics.mean(delivered),
            "delivered_flits_per_node_cycle_min": min(delivered),
            "delivered_flits_per_node_cycle_max": max(delivered),
            "completed_packet_mean_latency_cycles_seed_mean": (
                statistics.mean(valid_latency) if valid_latency else None),
            "completed_packet_mean_latency_cycles_seed_min": (
                min(valid_latency) if valid_latency else None),
            "completed_packet_mean_latency_cycles_seed_max": (
                max(valid_latency) if valid_latency else None),
            "undelivered_packets_at_horizon_mean": statistics.mean(
                row["undelivered_packets_at_horizon"] for row in group),
        })
    return summary


def plot_results(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = sorted({row["scenario"] for row in summary})
    for name in names:
        selected = sorted((row for row in summary if row["scenario"] == name),
                          key=lambda row: row["offered_flits_per_node_cycle_mean"])
        x = [row["offered_flits_per_node_cycle_mean"] for row in selected]
        label = name.replace("_", " ")
        line = axes[0].plot(x, [row["delivered_flits_per_node_cycle_mean"] for row in selected],
                            "o-", label=label)[0]
        axes[0].fill_between(x,
                             [row["delivered_flits_per_node_cycle_min"] for row in selected],
                             [row["delivered_flits_per_node_cycle_max"] for row in selected],
                             color=line.get_color(), alpha=0.15)
        if all(row["completed_packet_mean_latency_cycles_seed_mean"] is not None
               for row in selected):
            axes[1].plot(x, [row["completed_packet_mean_latency_cycles_seed_mean"]
                             for row in selected], "o-", label=label, color=line.get_color())
            axes[1].fill_between(
                x, [row["completed_packet_mean_latency_cycles_seed_min"] for row in selected],
                [row["completed_packet_mean_latency_cycles_seed_max"] for row in selected],
                color=line.get_color(), alpha=0.15)
    axes[0].set_ylabel("Delivered flits / node / cycle")
    axes[1].set_ylabel("Completed-packet mean latency (cycles)")
    for ax in axes:
        ax.set_xlabel("Offered flits / node / cycle")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Finite-window synthetic traffic; lines are seed means, shading is seed range")
    fig.tight_layout()
    fig.savefig(output / "offered_load.png", dpi=180)
    fig.savefig(output / "offered_load.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", nargs="+", required=True, type=Path)
    parser.add_argument("--assumptions", required=True, type=Path)
    parser.add_argument("--rates", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2, 0.4])
    parser.add_argument("--warmup-cycles", type=int, default=500)
    parser.add_argument("--measure-cycles", type=int, default=1500)
    parser.add_argument("--packet-flits", type=int, default=4)
    seeds_group = parser.add_mutually_exclusive_group()
    seeds_group.add_argument("--seed", type=int, default=17)
    seeds_group.add_argument("--seeds", nargs="+", type=int,
                             help="Independent seeds for a matched multi-run summary")
    parser.add_argument("--pattern", choices=("uniform", "hotspot"), default="uniform")
    parser.add_argument("--hotspot-node", type=int, default=0)
    parser.add_argument("--container", default="chiplet")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.relative_to(ROOT)
        if args.warmup_cycles < 0 or args.measure_cycles <= 0 or args.packet_flits <= 0:
            raise ValueError("Warmup, measurement window, and packet length must be valid")
        if not args.rates or any(not math.isfinite(rate) or not 0 < rate <= 1
                                 for rate in args.rates):
            raise ValueError("Each injection probability must be in (0, 1]")
        if len(set(args.rates)) != len(args.rates):
            raise ValueError("Injection probabilities must be unique")
        seeds = args.seeds if args.seeds is not None else [args.seed]
        if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            raise ValueError("Traffic seeds must be distinct nonnegative integers")
        specs = [load_spec(path) for path in args.scenario]
        if len({spec["name"] for spec in specs}) != len(specs):
            raise ValueError("Scenario names must be unique")
        widths = {spec["coordinate_width"] for spec in specs}
        if len(widths) != 1:
            raise ValueError("Offered-load comparison needs one mesh width")
        width = widths.pop()
        params = json.loads(args.assumptions.read_text())
        params["coordinate_width"] = width
        check_assumptions(params)
        if not 0 <= args.hotspot_node < width * width:
            raise ValueError("Hotspot node is outside the mesh")
        for spec in specs:
            verify_edge_profile_sources(spec, params=params)
        output.mkdir(parents=True, exist_ok=True)
        (output / "sweep_inputs.json").write_text(json.dumps({
            "scenarios": [str(path.resolve()) for path in args.scenario],
            "scenario_sha256": {spec["name"]: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path, spec in zip(args.scenario, specs)},
            "assumptions_sha256": hashlib.sha256(args.assumptions.read_bytes()).hexdigest(),
            "popnet_sha256": hashlib.sha256(POPNET.read_bytes()).hexdigest(),
            "rates": args.rates, "warmup_cycles": args.warmup_cycles,
            "measurement_cycles": args.measure_cycles,
            "packet_flits": args.packet_flits, "seeds": seeds,
            "pattern": args.pattern, "hotspot_node": args.hotspot_node,
            "clock_hz": params["simulated_clock_hz"],
            "interpretation": "Finite-window synthetic offered/delivered flits; completed-packet latency is conditional on delivery in the window",
        }, indent=2) + "\n")
        rows = []
        for index, rate in enumerate(args.rates):
            for seed in seeds:
                packets = generate_trace(width * width, rate,
                                         args.warmup_cycles + args.measure_cycles,
                                         args.packet_flits, seed, args.pattern,
                                         args.hotspot_node)
                trace = output / ("traffic_seed_%d_rate_%02d.txt" % (seed, index))
                trace.write_text("".join("%d %d %d %d\n" % packet for packet in packets))
                for scenario_path, spec in zip(args.scenario, specs):
                    row = run_case(scenario_path, spec, params, rate, index, seed,
                                   trace, packets, output, args.warmup_cycles,
                                   args.measure_cycles, args.packet_flits,
                                   args.pattern, args.container)
                    rows.append(row)
                    print("%s seed=%d rate=%g: %.3f delivered flits/node/cycle" % (
                        spec["name"], seed, rate,
                        row["delivered_flits_per_node_cycle"]), flush=True)
        with (output / "offered_load.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / "offered_load.json").write_text(json.dumps(rows, indent=2) + "\n")
        summary = summarize_seeds(rows)
        with (output / "offered_load_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        (output / "offered_load_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if args.plot:
            plot_results(summary, output)
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.SubprocessError,
            ImportError) as error:
        parser.exit(1, "Popnet offered-load study failed: %s\n" % error)


if __name__ == "__main__":
    main()
