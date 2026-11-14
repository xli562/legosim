#!/usr/bin/env python3
"""Run vendored SCALE-Sim for the matmul NPU tile and save timing provenance.

Use a Python environment with NumPy and tqdm. The SCALE-Sim core does not
need pandas; this wrapper reads its COMPUTE_REPORT.csv with the standard library.
"""

import argparse
import configparser
import contextlib
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCALE_ROOT = ROOT / "scale-sim-v2"
DEFAULT_CONFIG = ROOT / "benchmark/matmul_test/scalesim_npu_16x16.cfg"
DEFAULT_TOPOLOGY = ROOT / "benchmark/matmul_test/scalesim_npu_tile.csv"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_tree_sha256():
    digest = hashlib.sha256()
    for path in sorted((SCALE_ROOT / "scalesim").rglob("*.py")):
        digest.update(path.relative_to(SCALE_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_tile(topology):
    with topology.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) != 2 or [field.strip() for field in rows[1][:4]][1:] != ["10", "10", "100"]:
        raise ValueError("NPU wrapper requires a one-layer GEMM topology with M=10, N=10, K=100")


def read_compute_cycles(report):
    with report.open(newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError("Empty SCALE-Sim compute report")
        column = next((name for name in reader.fieldnames if name.strip() == "Total Cycles"), None)
        if column is None:
            raise ValueError("SCALE-Sim compute report lacks Total Cycles")
        values = []
        for row in reader:
            try:
                value = Decimal(row[column].strip())
            except (InvalidOperation, AttributeError) as error:
                raise ValueError("Invalid Total Cycles in SCALE-Sim report") from error
            if not value.is_finite() or value <= 0 or value != value.to_integral_value():
                raise ValueError("SCALE-Sim Total Cycles must be a positive integer")
            values.append(int(value))
    if len(values) != 1:
        raise ValueError("Expected one SCALE-Sim layer for the NPU tile")
    return values[0]


def run(config, topology, output):
    config = config.resolve()
    topology = topology.resolve()
    output = output.resolve()
    check_tile(topology)
    parsed = configparser.ConfigParser()
    if not parsed.read(config):
        raise ValueError("Cannot read SCALE-Sim config")
    run_name = parsed.get("general", "run_name")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SCALE_ROOT))
    from scalesim.scale_sim import scalesim  # imported after adding vendored source

    log = output / "scalesim.stdout.txt"
    with log.open("w") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
        model = scalesim(save_disk_space=True, verbose=False, config=str(config),
                         topology=str(topology), input_type_gemm=True)
        model.run_scale(top_path=str(output))
    report = output / run_name / "COMPUTE_REPORT.csv"
    cycles = read_compute_cycles(report)
    manifest = {
        "model": "vendored SCALE-Sim v2 GEMM cycle model",
        "compute_cycles": cycles,
        "gemm": {"m": 10, "n": 10, "k": 100},
        "array_rows": parsed.getint("architecture_presets", "ArrayHeight"),
        "array_cols": parsed.getint("architecture_presets", "ArrayWidth"),
        "dataflow": parsed.get("architecture_presets", "Dataflow"),
        "interface_bandwidth_mode": parsed.get("run_presets", "InterfaceBandwidth"),
        "config_path": str(config),
        "config_sha256": sha256(config),
        "topology_path": str(topology),
        "topology_sha256": sha256(topology),
        "compute_report_path": str(report),
        "compute_report_sha256": sha256(report),
        "scale_sim_source_sha256": source_tree_sha256(),
    }
    path = output / "npu_timing.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path, cycles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--output", type=Path, default=ROOT / "run_outputs/scalesim_npu_tile")
    args = parser.parse_args()
    try:
        path, cycles = run(args.config, args.topology, args.output)
    except (OSError, ImportError, ValueError, KeyError) as error:
        parser.exit(1, "NPU SCALE-Sim run failed: %s\n" % error)
    print("%s: %d cycles" % (path, cycles))


if __name__ == "__main__":
    main()
