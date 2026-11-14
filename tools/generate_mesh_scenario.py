#!/usr/bin/env python3
"""Generate an editable electrical/photonic square-mesh scenario JSON file."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from mesh_study import (ROOT, default_placements, mesh_edges, profile_from_source,
                        resolve_placements)


LAYOUTS = ("all-optical", "all-electrical", "vertical-optical", "horizontal-optical")


def make_scenario(width, layout, name, overrides=(), worker3_kind="gpu",
                  npu_compute_cycles=None, placements=None):
    if not isinstance(width, int) or width < 2:
        raise ValueError("--width must be an integer of at least 2")
    if layout not in LAYOUTS:
        raise ValueError("Unknown layout: %s" % layout)
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Scenario name must use lowercase letters, digits, and underscores")
    edges = {}
    for a, b in mesh_edges(width):
        vertical = b - a == width
        if layout == "all-optical":
            kind = "optical"
        elif layout == "all-electrical":
            kind = "electrical"
        elif layout == "vertical-optical":
            kind = "optical" if vertical else "electrical"
        else:
            kind = "electrical" if vertical else "optical"
        edges["%d-%d" % (a, b)] = kind
    for override in overrides:
        if "=" not in override:
            raise ValueError("Edge override must look like 0-1=optical")
        edge, kind = override.split("=", 1)
        if edge not in edges or kind not in ("optical", "electrical"):
            raise ValueError("Invalid edge override: %s" % override)
        edges[edge] = kind
    if worker3_kind not in ("gpu", "npu"):
        raise ValueError("worker3_kind must be gpu or npu")
    scenario = {"name": name, "coordinate_width": width, "edges": edges}
    if placements is not None:
        scenario["placements"] = resolve_placements(
            {"coordinate_width": width, "placements": placements})
    if worker3_kind == "npu":
        if not isinstance(npu_compute_cycles, int) or npu_compute_cycles <= 0:
            raise ValueError("NPU worker needs positive integer compute cycles")
        scenario["worker3_kind"] = "npu"
        scenario["npu_compute_cycles"] = npu_compute_cycles
    elif npu_compute_cycles is not None:
        raise ValueError("--npu-compute-cycles requires --worker3-kind npu")
    return scenario


def parse_position(text):
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("position must be X,Y")
    try:
        return [int(part) for part in parts]
    except ValueError as error:
        raise argparse.ArgumentTypeError("position must contain integer X,Y") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--layout", required=True, choices=LAYOUTS)
    parser.add_argument("--name", help="Defaults to layout_widthxwidth")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        metavar="EDGE=KIND", help="Override one link, e.g. 0-1=optical")
    parser.add_argument("--edge-profile", action="append", default=[],
                        metavar="EDGE=JSON", help="Use complete per-edge costs from a JSON profile")
    parser.add_argument("--worker3-kind", choices=("gpu", "npu"), default="gpu")
    for role in ("cpu", "worker1", "worker2", "worker3"):
        parser.add_argument("--%s" % role, type=parse_position, metavar="X,Y",
                            help="Place %s at mesh coordinate X,Y" % role)
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument("--npu-compute-cycles", type=int)
    timing.add_argument("--npu-timing", type=Path,
                        help="npu_timing.json written by tools/npu_scalesim.py")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    name = args.name or "%s_%dx%d%s" % (
        args.layout.replace("-", "_"), args.width, args.width,
        "_npu" if args.worker3_kind == "npu" else "")
    try:
        timing_source = None
        cycles = args.npu_compute_cycles
        if args.npu_timing is not None:
            timing_path = args.npu_timing.resolve()
            relative = timing_path.relative_to(ROOT)
            timing_data = json.loads(timing_path.read_text())
            if timing_data.get("gemm") != {"m": 10, "n": 10, "k": 100}:
                raise ValueError("NPU timing manifest is for a different GEMM tile")
            cycles = timing_data["compute_cycles"]
            timing_source = {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(timing_path.read_bytes()).hexdigest(),
            }
        placements = default_placements(args.width)
        moved = False
        for role in placements:
            position = getattr(args, role)
            if position is not None:
                placements[role] = position
                moved = True
        scenario = make_scenario(args.width, args.layout, name, args.overrides,
                                 args.worker3_kind, cycles, placements if moved else None)
        if args.edge_profile:
            scenario["edge_parameters"] = {}
            scenario["edge_profile_sources"] = {}
            for assignment in args.edge_profile:
                if "=" not in assignment:
                    raise ValueError("--edge-profile must look like 0-1=path.json")
                edge, filename = assignment.split("=", 1)
                if edge not in scenario["edges"] or edge in scenario["edge_parameters"]:
                    raise ValueError("Unknown or repeated edge profile: %s" % edge)
                source = Path(filename).resolve()
                relative = source.relative_to(ROOT)
                data = json.loads(source.read_text())
                scenario["edge_parameters"][edge] = profile_from_source(
                    data, scenario["edges"][edge], edge)
                scenario["edge_profile_sources"][edge] = {
                    "path": relative.as_posix(),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
        if timing_source is not None:
            scenario["npu_timing"] = timing_source
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, "Mesh scenario generation failed: %s\n" % error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scenario, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
