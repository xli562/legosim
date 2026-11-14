#!/usr/bin/env python3
"""Recompute mesh energy and area from validated traces without rerunning Popnet.

Only cost coefficients may change. Topology, traffic, rates, lengths, and
all timing inputs must match the original simulations.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from analyze_run import parse_coordinator, read_network_reports
from mesh_study import (calculate, check_assumptions, compare, load_spec,
                        read_scenario_trace, row)


def timing_signature(params):
    """Inputs that determine route/timing behavior or the modeled link geometry."""
    return {
        "coordinate_width": params["coordinate_width"],
        "flit_bits": params["flit_bits"],
        "simulated_clock_hz": params["simulated_clock_hz"],
        "router_pipeline_stages": params["router_pipeline_stages"],
        "router_pipeline_stage_delay_cycles": params["router_pipeline_stage_delay_cycles"],
        "optical": {
            "edge_delay_cycles": params["optical"]["edge_delay_cycles"],
            "length_mm": params["optical"]["length_mm"],
            "wavelengths_per_direction": params["optical"]["wavelengths_per_direction"],
            "bit_rate_per_wavelength_bps": params["optical"]["bit_rate_per_wavelength_bps"],
        },
        "electrical": {
            "edge_delay_cycles": params["electrical"]["edge_delay_cycles"],
            "length_mm": params["electrical"]["length_mm"],
            "assumed_bit_rate_bps_per_direction": params["electrical"]["assumed_bit_rate_bps_per_direction"],
        },
    }


def reprice(run_dirs, assumptions_path, output, make_plot=False):
    run_dirs = [path.resolve() for path in run_dirs]
    output = output.resolve()
    # compare() fully validates every saved run and their shared workload,
    # binaries, original assumptions, and fixed simulator input.
    originals = compare(run_dirs, output / "source_validation", False)
    target = json.loads(assumptions_path.read_text())
    check_assumptions(target)
    if timing_signature(target) != timing_signature(originals[0]["assumptions"]):
        raise ValueError("Timing, rate, or link geometry changed; run a new mesh simulation")

    results = []
    for run_dir in run_dirs:
        # compare() sorts its results by name; match each run by the saved name.
        spec = load_spec(run_dir / "scenario.json")
        source = next(item for item in originals if item["scenario"] == spec["name"])
        summary = parse_coordinator((run_dir / "coordinator.log").read_text())
        summary["run_dir"] = run_dir
        transfers = read_scenario_trace(run_dir / "bench.txt", spec, target)
        reports = read_network_reports(run_dir)
        result = calculate(summary, transfers, spec, target, reports)
        if "npu_validation" in source:
            result["npu_validation"] = source["npu_validation"]
        result["useful_payload_bytes"] = source["useful_payload_bytes"]
        result["useful_payload_throughput_MBps"] = source["useful_payload_throughput_MBps"]
        for key in ("traffic_signature", "workload_completion_cycles",
                    "network_average_packet_latency_cycles",
                    "encoded_payload_throughput_MBps_upper_bound"):
            if result[key] != source[key]:
                raise ValueError("Saved traffic or timing changed during repricing: %s" % key)
        result["source_run"] = str(run_dir)
        result["cost_only_repricing"] = True
        results.append(result)
    results.sort(key=lambda result: result["scenario"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "repriced.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row(results[0])))
        writer.writeheader()
        writer.writerows(row(result) for result in results)
    (output / "repriced.json").write_text(json.dumps({
        "source_runs": [str(path) for path in run_dirs],
        "target_assumptions_path": str(assumptions_path.resolve()),
        "target_assumptions_sha256": hashlib.sha256(assumptions_path.read_bytes()).hexdigest(),
        "method": "Recompute energy and area from validated saved traces; no network resimulation",
        "unchanged": "Topology, traffic, route delays, link lengths/rates, clock, latency, and throughput",
        "results": results,
    }, indent=2, allow_nan=False, default=str) + "\n")
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [result["scenario"].replace("_", "\n") for result in results]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, (key, title) in zip(axes, (
                ("total_noc_energy_uJ", "Repriced NoC energy (µJ)"),
                ("total_noc_area_mm2", "Repriced NoC area (mm²)"))):
            axis.bar(labels, [result[key] for result in results], color="#287b91")
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        fig.suptitle("Cost-only analysis of validated saved traffic")
        fig.tight_layout()
        fig.savefig(output / "repriced.png", dpi=180)
        fig.savefig(output / "repriced.svg")
        plt.close(fig)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--assumptions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        results = reprice(args.run_dirs, args.assumptions, args.output, args.plot)
    except (OSError, KeyError, ValueError, ImportError) as error:
        parser.exit(1, "Mesh repricing failed: %s\n" % error)
    for result in results:
        print("%s: %.3f µJ, %.3f mm²" % (
            result["scenario"], result["total_noc_energy_uJ"],
            result["total_noc_area_mm2"]))


if __name__ == "__main__":
    main()
