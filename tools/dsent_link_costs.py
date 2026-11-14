#!/usr/bin/env python3
"""Run vendored DSENT's electrical and optical link models for mesh assumptions.

The resulting costs are model estimates. They are saved separately from the
mesh-study placeholders, so a technology choice is never changed silently.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "gem5/ext/dsent"
ASSUMPTIONS = ROOT / "benchmark/matmul_test/mesh_link_assumptions.json"
CONTAINER_ROOT = "/workspace/sim"
CONTAINER_BINARY = "/tmp/legosim_dsent_cli"
RESULT_RE = re.compile(r"^DSENT_RESULT ([a-z0-9_]+) ([^ ]+)$", re.MULTILINE)


def container_path(path):
    return CONTAINER_ROOT + "/" + str(path.relative_to(ROOT))


def run_checked(command):
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError("Command failed: %s\n%s\n%s" %
                           (" ".join(command), completed.stdout, completed.stderr))
    return completed.stdout


def build_cli(container):
    sources = sorted(SOURCE.rglob("*.cc"))
    sources = [path for path in sources if path.name != "interface.cc"]
    command = (["docker", "exec", "-w", CONTAINER_ROOT, container, "g++",
                "-std=c++11", "-O2", "-I", "gem5/ext/dsent", "tools/dsent_cli.cc"] +
               [str(path.relative_to(ROOT)) for path in sources] +
               ["-o", CONTAINER_BINARY])
    run_checked(command)


def config_text(kind, assumptions):
    bits = assumptions["flit_bits"]
    clock = assumptions["simulated_clock_hz"]
    electrical_tech = CONTAINER_ROOT + "/gem5/ext/dsent/tech/tech_models/Bulk45LVT.model"
    common = ["ElectricalTechModelFilename = " + electrical_tech,
              "IsPerformTimingOptimization = false"]
    if kind == "electrical":
        link = assumptions[kind]
        queries = {
            "send_energy_j_per_word": "Energy>>RepeatedLink:Send",
            "leakage_w": "NddPower>>RepeatedLink:Leakage",
            "active_area_m2": "Area>>RepeatedLink:Active",
            "wire_area_m2": "Area>>RepeatedLink:GlobalWire",
        }
        fields = ["ModelName = RepeatedLink", "NumberBits = %d" % bits,
                  "WireLayer = Global", "WireWidthMultiplier = 1.0",
                  "WireSpacingMultiplier = 1.0",
                  "WireLength = %.17g" % (link["length_mm"] / 1000),
                  "Delay = %.17g" % (link["edge_delay_cycles"] / clock)]
    else:
        link = assumptions[kind]
        wavelengths = link["wavelengths_per_direction"]
        line_rate = link["bit_rate_per_wavelength_bps"]
        if wavelengths * line_rate != bits * clock:
            raise ValueError("Optical line rate must match one flit per cycle")
        queries = {
            "send_energy_j_per_word": "Energy>>SWSRLink:Send",
            "leakage_w": "NddPower>>SWSRLink:Leakage",
            "laser_w": "NddPower>>SWSRLink:Laser",
            "ring_tuning_w": "NddPower>>SWSRLink:RingTuning",
            "active_area_m2": "Area>>SWSRLink:Active",
            "photonic_area_m2": "Area>>SWSRLink:Photonic",
            "waveguide_area_m2": "Area>>SWSRLink:Waveguide",
        }
        fields = ["ModelName = SWSRLink", "NumberBits = %d" % bits,
                  "CoreDataRate = %.17g" % clock,
                  "LinkDataRate = %.17g" % line_rate,
                  "OptimizeLoss = true", "OptUtil = 0.5",
                  "LaserType = Standard",
                  "RingTuningMethod = ThermalWithBitReshuffle",
                  "Length = %.17g" % (link["length_mm"] / 1000),
                  "PhotonicTechModelFilename = " + CONTAINER_ROOT +
                  "/gem5/ext/dsent/tech/tech_models/Photonics.model"]
    expressions = ["print \"%s\" $(%s);" % pair for pair in sorted(queries.items())]
    return "\n".join(common + fields +
                     ["EvaluateString = " + " ".join(expressions), ""])


def read_values(output, expected_keys):
    values = {name: float(value) for name, value in RESULT_RE.findall(output)}
    if set(values) != set(expected_keys):
        raise ValueError("Unexpected DSENT results: %s" % sorted(values))
    if not all(math.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError("DSENT returned invalid cost")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "run_outputs/dsent_link_costs")
    parser.add_argument("--assumptions", type=Path, default=ASSUMPTIONS)
    parser.add_argument("--container", default="chiplet")
    parser.add_argument("--rebuild", action="store_true")
    for kind in ("electrical", "optical"):
        parser.add_argument("--%s-length-mm" % kind, type=float)
        parser.add_argument("--%s-edge-delay-cycles" % kind, type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    try:
        args.output.relative_to(ROOT)
    except ValueError:
        parser.error("--output must be inside %s so the container can read it" % ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    assumptions = json.loads(args.assumptions.read_text())
    for kind in ("electrical", "optical"):
        length = getattr(args, kind + "_length_mm")
        delay = getattr(args, kind + "_edge_delay_cycles")
        if length is not None:
            if not math.isfinite(length) or length <= 0:
                parser.error("--%s-length-mm must be positive and finite" % kind)
            assumptions[kind]["length_mm"] = length
        if delay is not None:
            if delay <= 0:
                parser.error("--%s-edge-delay-cycles must be positive" % kind)
            assumptions[kind]["edge_delay_cycles"] = delay
    effective = args.output / "input_assumptions.json"
    effective.write_text(json.dumps(assumptions, indent=2, sort_keys=True) + "\n")
    existing = subprocess.run(["docker", "exec", args.container,
                               "test", "-x", CONTAINER_BINARY], capture_output=True)
    if args.rebuild or existing.returncode:
        build_cli(args.container)

    report = {"source": "vendored DSENT analytical model",
              "technology": {"electrical": "Bulk45LVT.model", "photonic": "Photonics.model"},
              "assumptions_sha256": hashlib.sha256(args.assumptions.read_bytes()).hexdigest(),
              "effective_assumptions_sha256": hashlib.sha256(effective.read_bytes()).hexdigest(),
              "flit_bits": assumptions["flit_bits"], "links": {}}
    source_files = (
        "gem5/ext/dsent/DSENT.cc",
        "gem5/ext/dsent/model/electrical/RepeatedLink.cc",
        "gem5/ext/dsent/model/optical/SWSRLink.cc",
        "gem5/ext/dsent/tech/tech_models/Bulk45LVT.model",
        "gem5/ext/dsent/tech/tech_models/Photonics.model",
        "tools/dsent_cli.cc",
    )
    report["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in source_files
    }
    for kind in ("electrical", "optical"):
        cfg = args.output / (kind + ".cfg")
        cfg.write_text(config_text(kind, assumptions))
        output = run_checked(["docker", "exec", "-w", CONTAINER_ROOT,
                              args.container, CONTAINER_BINARY, container_path(cfg)])
        (args.output / (kind + ".stdout.txt")).write_text(output)
        keys = ("send_energy_j_per_word", "leakage_w", "active_area_m2", "wire_area_m2")
        if kind == "optical":
            keys = ("send_energy_j_per_word", "leakage_w", "laser_w",
                    "ring_tuning_w", "active_area_m2", "photonic_area_m2",
                    "waveguide_area_m2")
        values = read_values(output, keys)
        values["send_energy_j_per_bit"] = values["send_energy_j_per_word"] / assumptions["flit_bits"]
        values["non_data_dependent_power_w"] = sum(
            values[name] for name in ("leakage_w", "laser_w", "ring_tuning_w") if name in values)
        report["links"][kind] = values
    destination = args.output / "link_costs.json"
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    calibrated = copy.deepcopy(assumptions)
    calibrated["purpose"] = (
        "DSENT-derived link energy, idle power, and area; router cost and edge delay remain illustrative")
    calibrated["dsent_source"] = {
        "link_costs_path": str(destination.resolve()),
        "link_costs_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    for kind, values in report["links"].items():
        link = calibrated[kind]
        link["dynamic_energy_j_per_transmitted_bit_hop"] = values["send_energy_j_per_bit"]
        link["static_power_w_per_direction"] = values["non_data_dependent_power_w"]
        route_area = (values["wire_area_m2"] if kind == "electrical"
                      else values["waveguide_area_m2"])
        total_area = values["active_area_m2"] + route_area
        if kind == "optical":
            if values["photonic_area_m2"] < route_area:
                raise ValueError("DSENT photonic area is below its waveguide area")
            total_area += values["photonic_area_m2"] - route_area
        link["routing_pitch_um"] = route_area / (link["length_mm"] / 1000) * 1e6
        link["interface_area_mm2_per_direction"] = (total_area - route_area) * 1e6
    calibrated_path = args.output / "mesh_link_assumptions_dsent.json"
    calibrated_path.write_text(json.dumps(calibrated, indent=2, sort_keys=True) + "\n")
    print(destination)
    print(calibrated_path)


if __name__ == "__main__":
    main()
