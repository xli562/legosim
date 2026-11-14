#!/usr/bin/env python3
"""Prepare, run, report, and compare square electrical/optical meshes.

The graph controls Popnet's per-edge timing. Energy and area are analytical
estimates from the trace and values in the saved assumptions file.
"""

import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

from analyze_run import parse_coordinator, read_network_reports, read_workload_validation
from npu_scalesim import check_tile, read_compute_cycles
from photonic_mesh_report import read_trace


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmark/matmul_test"
ASSUMPTIONS = BENCHMARK / "mesh_link_assumptions.json"
TEMPLATE = BENCHMARK / "matmul_photonic_mesh.yml"
KINDS = ("optical", "electrical")
EDGE_COST_FIELDS = ("edge_delay_cycles", "dynamic_energy_j_per_transmitted_bit_hop",
                    "static_power_w_per_direction", "length_mm", "routing_pitch_um",
                    "interface_area_mm2_per_direction")
MATRIX_INPUT_BYTES = 100 * 100 * 8
MATRIX_TILE_BYTES = 10 * 10 * 8
NPU_MODEL_SNAPSHOTS = (
    ("config_path", "config_sha256", "npu_scalesim_config.cfg"),
    ("topology_path", "topology_sha256", "npu_scalesim_topology.csv"),
    ("compute_report_path", "compute_report_sha256", "npu_scalesim_compute_report.csv"),
)


def mesh_edges(width):
    edges = []
    for y in range(width):
        for x in range(width):
            node = x + y * width
            if x + 1 < width:
                edges.append((node, node + 1))
            if y + 1 < width:
                edges.append((node, node + width))
    return tuple(edges)


def default_placements(width):
    return {"cpu": [0, 0], "worker1": [0, width - 1],
            "worker2": [width - 1, 0], "worker3": [width - 1, width - 1]}


def resolve_placements(spec):
    width = spec.get("coordinate_width", 2)
    positions = spec.get("placements", default_placements(width))
    if not isinstance(positions, dict) or set(positions) != set(default_placements(width)):
        raise ValueError("placements must give cpu and worker1/worker2/worker3")
    for role, position in positions.items():
        if (not isinstance(position, list) or len(position) != 2 or
                any(type(value) is not int or value < 0 or value >= width for value in position)):
            raise ValueError("Invalid %s placement for %dx%d mesh" % (role, width, width))
    if len({tuple(position) for position in positions.values()}) != 4:
        raise ValueError("CPU and workers must occupy distinct mesh nodes")
    return positions


def placement_nodes(spec):
    width = spec.get("coordinate_width", 2)
    return {role: x + y * width for role, (x, y) in resolve_placements(spec).items()}


def edge_profile(spec, params, edge):
    return {**params[spec["edges"][edge]],
            **spec.get("edge_parameters", {}).get(edge, {})}


def check_edge_profile(edge, profile):
    if not isinstance(profile, dict) or set(profile) != set(EDGE_COST_FIELDS):
        raise ValueError("Edge %s needs all six delay, energy, power, and area fields" % edge)
    for field, value in profile.items():
        if field == "edge_delay_cycles":
            valid = type(value) is int and value >= 0
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            if field == "length_mm":
                valid = valid and value > 0
        if not valid:
            raise ValueError("Invalid %s value for edge %s" % (field, edge))


def profile_from_source(data, kind, edge):
    source = data[kind] if kind in data else data
    profile = {field: source[field] for field in EDGE_COST_FIELDS}
    check_edge_profile(edge, profile)
    return profile


def edge_profile_snapshot_name(edge):
    return "edge_profile_%s.json" % edge.replace("-", "_")


def verify_edge_profile_sources(spec, saved_dir=None, params=None):
    paths = {}
    for edge, source in spec.get("edge_profile_sources", {}).items():
        path = (saved_dir / edge_profile_snapshot_name(edge) if saved_dir is not None
                else (ROOT / source["path"]).resolve())
        if saved_dir is None:
            path.relative_to(ROOT)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Edge profile source hash differs for %s" % edge)
        data = json.loads(path.read_text())
        if profile_from_source(data, spec["edges"][edge], edge) != spec["edge_parameters"][edge]:
            raise ValueError("Edge profile values differ from source for %s" % edge)
        if params is not None and "flit_bits" in data:
            if (data["flit_bits"] != params["flit_bits"] or
                    data["simulated_clock_hz"] != params["simulated_clock_hz"]):
                raise ValueError("Edge profile clock/flit width differs from mesh assumptions")
            kind = spec["edges"][edge]
            rate_key = ("wavelengths_per_direction", "bit_rate_per_wavelength_bps") if kind == "optical" else (
                "assumed_bit_rate_bps_per_direction",)
            if any(data[kind][key] != params[kind][key] for key in rate_key):
                raise ValueError("Edge profile line rate differs from mesh assumptions")
        paths[edge_profile_snapshot_name(edge)] = path
    return paths


def load_spec(path):
    spec = json.loads(path.read_text())
    if not re.fullmatch(r"[a-z][a-z0-9_]*", spec["name"]):
        raise ValueError("Scenario name must use lowercase letters, digits, and underscores")
    width = spec.get("coordinate_width", 2)
    if not isinstance(width, int) or width < 2:
        raise ValueError("Mesh coordinate_width must be an integer of at least 2")
    expected = {"%d-%d" % edge for edge in mesh_edges(width)}
    if set(spec["edges"]) != expected:
        raise ValueError("A %dx%d mesh requires exactly edges %s" %
                         (width, width, sorted(expected)))
    if any(kind not in KINDS for kind in spec["edges"].values()):
        raise ValueError("Every edge must be optical or electrical")
    overrides = spec.get("edge_parameters", {})
    if not isinstance(overrides, dict) or not set(overrides) <= expected:
        raise ValueError("edge_parameters must name existing mesh edges")
    for edge, profile in overrides.items():
        check_edge_profile(edge, profile)
    sources = spec.get("edge_profile_sources", {})
    if not isinstance(sources, dict) or not set(sources) <= set(overrides):
        raise ValueError("edge_profile_sources must match overridden edges")
    for edge, source in sources.items():
        if (not isinstance(source, dict) or set(source) != {"path", "sha256"} or
                not isinstance(source["path"], str) or
                not isinstance(source["sha256"], str) or
                not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])):
            raise ValueError("Invalid edge profile source for %s" % edge)
    worker = spec.get("worker3_kind", "gpu")
    if worker not in ("gpu", "npu"):
        raise ValueError("worker3_kind must be gpu or npu")
    if worker == "npu" and (not isinstance(spec.get("npu_compute_cycles"), int) or
                            spec["npu_compute_cycles"] <= 0):
        raise ValueError("NPU worker needs positive integer npu_compute_cycles")
    if "npu_timing" in spec and (worker != "npu" or
                                  set(spec["npu_timing"]) != {"path", "sha256"}):
        raise ValueError("npu_timing requires an NPU worker and path/SHA-256")
    resolve_placements(spec)
    return spec


def verify_npu_timing(spec, path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["npu_timing"]["sha256"]:
        raise ValueError("NPU timing manifest hash differs from scenario")
    manifest = json.loads(path.read_text())
    if (manifest.get("compute_cycles") != spec["npu_compute_cycles"] or
            manifest.get("gemm") != {"m": 10, "n": 10, "k": 100}):
        raise ValueError("NPU timing manifest does not match scenario/workload")
    return manifest


def verify_npu_model_files(manifest, saved_dir=None):
    paths = {}
    for source_key, hash_key, snapshot_name in NPU_MODEL_SNAPSHOTS:
        path = (saved_dir / snapshot_name if saved_dir is not None
                else Path(manifest[source_key]))
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[hash_key]:
            raise ValueError("SCALE-Sim input/report hash mismatch: %s" % snapshot_name)
        paths[snapshot_name] = path
    check_tile(paths["npu_scalesim_topology.csv"])
    if read_compute_cycles(paths["npu_scalesim_compute_report.csv"]) != manifest["compute_cycles"]:
        raise ValueError("SCALE-Sim compute report cycles differ from timing manifest")
    return paths


def check_assumptions(params):
    def nonnegative(value):
        return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0

    if not isinstance(params["coordinate_width"], int) or params["coordinate_width"] < 2:
        raise ValueError("Mesh coordinate_width must be an integer of at least 2")
    if params["flit_bits"] != 128:
        raise ValueError("This benchmark uses 128-bit Popnet flits")
    if params["simulated_clock_hz"] <= 0 or params["router_pipeline_stage_delay_cycles"] <= 0:
        raise ValueError("Clock and router stage delay must be positive")
    if params["router_pipeline_stages"] != 5:
        raise ValueError("Popnet shortest-path code currently models five pipeline stages")
    expected_rate = params["flit_bits"] * params["simulated_clock_hz"]
    if (params["optical"]["wavelengths_per_direction"] *
            params["optical"]["bit_rate_per_wavelength_bps"] != expected_rate):
        raise ValueError("Optical line rate must equal one flit per cycle")
    if params["electrical"]["assumed_bit_rate_bps_per_direction"] != expected_rate:
        raise ValueError("Electrical line rate must equal one flit per cycle")
    for kind in KINDS:
        for name in ("edge_delay_cycles", "dynamic_energy_j_per_transmitted_bit_hop",
                     "static_power_w_per_direction", "length_mm", "routing_pitch_um",
                     "interface_area_mm2_per_direction"):
            if not nonnegative(params[kind][name]):
                raise ValueError("Invalid %s cost: %s" % (kind, name))
    for name in ("dynamic_energy_j_per_transmitted_bit_per_visit",
                 "static_power_w_per_router", "area_mm2_per_router"):
        if not nonnegative(params["router"][name]):
            raise ValueError("Invalid router cost: %s" % name)


def graph_text(spec, params):
    width = params["coordinate_width"]
    lines = ["graph mesh_%d_%d {" % (width, width),
             "    node [pipeline_stage_delay=%s]" % params["router_pipeline_stage_delay_cycles"]]
    for a, b in mesh_edges(width):
        kind = spec["edges"]["%d-%d" % (a, b)]
        delay = edge_profile(spec, params, "%d-%d" % (a, b))["edge_delay_cycles"]
        lines.append('    %d--%d [weight=%s, id="%s"];' % (a, b, delay, kind))
    return "\n".join(lines + ["}", ""])


def render_yaml(template, run_dir, width, worker3_kind="gpu", npu_compute_cycles=None,
                placements=None):
    old_graph = "$SIMULATOR_ROOT/benchmark/matmul_test/topology/photonic_mesh_2_2.gv"
    if template.count(old_graph) != 1 or "$BENCHMARK_ROOT/bin/" not in template:
        raise ValueError("The source YAML template changed; inspect its graph and binary paths")
    positions = resolve_placements({"coordinate_width": width, "placements":
                                    placements if placements is not None else default_placements(width)})
    relative = run_dir.relative_to(ROOT).as_posix()
    template = template.replace(old_graph, "$SIMULATOR_ROOT/%s/topology.gv" % relative)
    template = template.replace("$BENCHMARK_ROOT/bin/", "$SIMULATOR_ROOT/benchmark/matmul_test/bin/")
    if worker3_kind == "npu":
        old_block = ('  - cmd: "$SIMULATOR_ROOT/benchmark/matmul_test/bin/matmul_cu"\n'
                     '    args: ["1", "1"]\n'
                     '    log: "gpgpusim.1.1.log"\n'
                     '    is_to_stdout: false\n'
                     '    clock_rate: 1\n'
                     '    pre_copy: "$SIMULATOR_ROOT/gpgpu-sim/configs/tested-cfgs/SM2_GTX480/*"\n')
        x, y = positions["worker3"]
        cpu_x, cpu_y = positions["cpu"]
        new_block = ('  - cmd: "$SIMULATOR_ROOT/benchmark/matmul_test/bin/matmul_npu"\n'
                     '    args: ["%d", "%d", "%d", "%d", "%d"]\n'
                     '    log: "npu.%d.%d.log"\n'
                     '    is_to_stdout: false\n'
                     '    clock_rate: 1\n') % (x, y, cpu_x, cpu_y, npu_compute_cycles, x, y)
        if template.count(old_block) != 1:
            raise ValueError("Expected one third-GPU YAML block to replace with NPU")
        template = template.replace(old_block, new_block)
    originals = {"worker1": (0, 1), "worker2": (1, 0), "worker3": (1, 1)}
    workers = ("worker1", "worker2") if worker3_kind == "npu" else tuple(originals)
    for role in workers:
        original = originals[role]
        old = 'args: ["%d", "%d"]' % original
        if template.count(old) != 1:
            raise ValueError("Expected one GPU YAML argument line: %s" % old)
        template = template.replace(old, "<%s_ARGS>" % role)
        old_log = 'log: "gpgpusim.%d.%d.log"' % original
        if template.count(old_log) != 1:
            raise ValueError("Expected one GPU YAML log line: %s" % old_log)
        template = template.replace(old_log, "<%s_LOG>" % role)
    for role in workers:
        x, y = positions[role]
        cpu_x, cpu_y = positions["cpu"]
        args = [x, y] + ([cpu_x, cpu_y] if (cpu_x, cpu_y) != (0, 0) else [])
        template = template.replace("<%s_ARGS>" % role,
                                    'args: [%s]' % ", ".join('"%d"' % value for value in args))
        log_x, log_y = originals[role] if positions == default_placements(width) else (x, y)
        template = template.replace("<%s_LOG>" % role,
                                    'log: "gpgpusim.%d.%d.log"' % (log_x, log_y))
    old_cpu = ('args: ["--", "$SIMULATOR_ROOT/benchmark/matmul_test/bin/'
               'matmul_checked_c", "0", "0"]')
    if template.count(old_cpu) != 1:
        raise ValueError("Expected one checked CPU YAML argument line")
    if width != 2 or positions != default_placements(width):
        coordinates = positions["cpu"] + sum((positions[role] for role in originals), [])
        new_cpu = ('args: ["--", "$SIMULATOR_ROOT/benchmark/matmul_test/bin/'
                   'matmul_checked_c", %s]' % ", ".join('"%d"' % value for value in coordinates))
        template = template.replace(old_cpu, new_cpu)
    cpu_x, cpu_y = positions["cpu"]
    template = template.replace('log: "sniper.0.0.log"',
                                'log: "sniper.%d.%d.log"' % (cpu_x, cpu_y))
    if width != 2:
        old_ary = '"-A", "4"'
        if template.count(old_ary) != 1:
            raise ValueError("Expected one Popnet ary-size argument")
        template = template.replace(old_ary, '"-A", "%d"' % (width * width))
    return template


def prepare(scenario_path, assumptions_path=ASSUMPTIONS):
    spec = load_spec(scenario_path)
    timing_path = None
    model_files = {}
    if "npu_timing" in spec:
        timing_path = (ROOT / spec["npu_timing"]["path"]).resolve()
        timing_path.relative_to(ROOT)
        model_files = verify_npu_model_files(verify_npu_timing(spec, timing_path))
    params = json.loads(assumptions_path.read_text())
    params["coordinate_width"] = spec.get("coordinate_width", 2)
    check_assumptions(params)
    edge_files = verify_edge_profile_sources(spec, params=params)
    output_parent = ROOT / "run_outputs"
    output_parent.mkdir(exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="mesh_%s." % spec["name"], dir=output_parent))
    (run_dir / "scenario.json").write_text(json.dumps(spec, indent=2) + "\n")
    if timing_path is not None:
        shutil.copy2(timing_path, run_dir / "npu_timing.json")
        for name, path in model_files.items():
            shutil.copy2(path, run_dir / name)
    for name, path in edge_files.items():
        shutil.copy2(path, run_dir / name)
    (run_dir / "link_assumptions.json").write_text(json.dumps(params, indent=2) + "\n")
    (run_dir / "topology.gv").write_text(graph_text(spec, params))
    (run_dir / "input.mesh.yml").write_text(render_yaml(
        TEMPLATE.read_text(), run_dir, params["coordinate_width"],
        spec.get("worker3_kind", "gpu"), spec.get("npu_compute_cycles"),
        resolve_placements(spec)))
    checked = BENCHMARK / "bin/matmul_checked_c"
    if checked.exists():
        shutil.copy2(checked, run_dir / "matmul_checked_c")
    gpu = BENCHMARK / "bin/matmul_cu"
    if gpu.exists():
        shutil.copy2(gpu, run_dir / "matmul_cu")
    return run_dir


def shortest_route(source, destination, spec, params):
    """Find shortest routes and reject ties with different E/P cost signatures."""
    width = params["coordinate_width"]
    adjacency = {node: set() for node in range(width * width)}
    for a, b in mesh_edges(width):
        adjacency[a].add(b)
        adjacency[b].add(a)
    stage_cost = (params["router_pipeline_stage_delay_cycles"] *
                  params["router_pipeline_stages"])
    distance = {source: stage_cost}
    states = {source: {(0, 0, Decimal(0)): (source,)}}
    queue = [(stage_cost, source)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance != distance[node]:
            continue
        for neighbor in sorted(adjacency[node]):
            edge = "%d-%d" % tuple(sorted((node, neighbor)))
            kind = spec["edges"][edge]
            profile = edge_profile(spec, params, edge)
            candidate = current_distance + stage_cost + profile["edge_delay_cycles"]
            neighbor_states = {}
            energy = Decimal(str(profile["dynamic_energy_j_per_transmitted_bit_hop"]))
            for signature, path in states[node].items():
                next_signature = (signature[0] + int(kind == KINDS[0]),
                                  signature[1] + int(kind == KINDS[1]),
                                  signature[2] + energy)
                neighbor_states[next_signature] = path + (neighbor,)
            if candidate < distance.get(neighbor, float("inf")):
                distance[neighbor] = candidate
                states[neighbor] = neighbor_states
                heapq.heappush(queue, (candidate, neighbor))
            elif candidate == distance.get(neighbor):
                for signature, path in neighbor_states.items():
                    previous = states[neighbor].get(signature)
                    if previous is None or path < previous:
                        states[neighbor][signature] = path
    if destination not in states:
        raise ValueError("No route from %s to %s" % (source, destination))
    if len(states[destination]) != 1:
        raise ValueError("Equal-delay routes have different E/P mixes or dynamic link energy for %s -> %s" %
                         (source, destination))
    signature, route = next(iter(states[destination].items()))
    return distance[destination], signature[:2], route


def read_scenario_trace(path, spec, params):
    width = params["coordinate_width"]
    adjacency = {node: set() for node in range(width * width)}
    for a, b in mesh_edges(width):
        adjacency[a].add(b)
        adjacency[b].add(a)
    transfers = read_trace(path, adjacency, params["flit_bits"])
    nodes = placement_nodes(spec)
    endpoints = set(nodes.values())
    for transfer in transfers:
        if ({transfer["source"], transfer["destination"]} - endpoints or
                (transfer["source"] != nodes["cpu"] and
                 transfer["destination"] != nodes["cpu"])):
            raise ValueError("Matmul traffic must connect the placed CPU to a worker")
        score, mix, route = shortest_route(transfer["source"], transfer["destination"], spec, params)
        transfer["hops"] = len(route) - 1
        transfer["route_cost_cycles"] = score
        transfer["optical_hops"], transfer["electrical_hops"] = mix
        transfer["route"] = "->".join(map(str, route))
        transfer["transmitted_bit_hops_including_header"] = (
            transfer["flits_including_header"] * params["flit_bits"] * (len(route) - 1)
        )
    return transfers


def validate_workload_traffic(transfers, spec, flit_bits):
    """Check the exact checked-matmul messages before reporting useful bytes."""
    if flit_bits <= 0 or flit_bits % 8:
        raise ValueError("Checked matmul requires positive whole-byte flits")
    bytes_per_flit = flit_bits // 8
    if MATRIX_INPUT_BYTES % bytes_per_flit or MATRIX_TILE_BYTES % bytes_per_flit:
        raise ValueError("Checked matmul payload sizes need whole flits")
    nodes = placement_nodes(spec)
    cpu = nodes["cpu"]
    expected = Counter()
    for role in ("worker1", "worker2", "worker3"):
        worker = nodes[role]
        expected[(cpu, worker, 1 + MATRIX_INPUT_BYTES // bytes_per_flit, 0)] = 2
        expected[(worker, cpu, 1 + MATRIX_TILE_BYTES // bytes_per_flit, 0)] = 1
    observed = Counter((item["source"], item["destination"],
                        item["flits_including_header"], item["descriptor"])
                       for item in transfers)
    if observed != expected:
        raise ValueError("Packet trace differs from the checked matmul message sizes/counts")
    return 3 * (2 * MATRIX_INPUT_BYTES + MATRIX_TILE_BYTES)


def calculate(summary, transfers, spec, params, network_reports):
    duration = summary["final_cycles"] / params["simulated_clock_hz"]
    roles_by_node = {node: role for role, node in placement_nodes(spec).items()}
    edge_metrics = {}
    for edge, kind in spec["edges"].items():
        profile = edge_profile(spec, params, edge)
        edge_metrics[edge] = {"kind": kind,
                              "parameters": {field: profile[field] for field in EDGE_COST_FIELDS},
                              "transmitted_bits_including_header": 0}
    for transfer in transfers:
        route = [int(node) for node in transfer["route"].split("->")]
        bits = transfer["flits_including_header"] * params["flit_bits"]
        for source, destination in zip(route, route[1:]):
            edge = "%d-%d" % tuple(sorted((source, destination)))
            edge_metrics[edge]["transmitted_bits_including_header"] += bits
    for metrics in edge_metrics.values():
        profile = metrics["parameters"]
        metrics["dynamic_energy_j"] = (metrics["transmitted_bits_including_header"] *
                                       profile["dynamic_energy_j_per_transmitted_bit_hop"])
        metrics["static_energy_j"] = (2 * profile["static_power_w_per_direction"] * duration)
        metrics["area_mm2"] = 2 * (
            profile["length_mm"] * profile["routing_pitch_um"] / 1000 +
            profile["interface_area_mm2_per_direction"])
    directed_counts = {
        kind: 2 * sum(edge_kind == kind for edge_kind in spec["edges"].values())
        for kind in KINDS
    }
    dynamic = {
        kind: math.fsum(item["dynamic_energy_j"] for item in edge_metrics.values()
                        if item["kind"] == kind)
        for kind in KINDS
    }
    static = {
        kind: math.fsum(item["static_energy_j"] for item in edge_metrics.values()
                        if item["kind"] == kind)
        for kind in KINDS
    }
    area = {
        kind: math.fsum(item["area_mm2"] for item in edge_metrics.values()
                        if item["kind"] == kind)
        for kind in KINDS
    }
    router_visits = sum(item["flits_including_header"] * params["flit_bits"] *
                        (item["hops"] + 1) for item in transfers)
    router_dynamic = router_visits * params["router"]["dynamic_energy_j_per_transmitted_bit_per_visit"]
    router_count = params["coordinate_width"] ** 2
    router_static = router_count * params["router"]["static_power_w_per_router"] * duration
    router_area = router_count * params["router"]["area_mm2_per_router"]
    payload_bound = sum(item["encoded_payload_bytes_upper_bound"] for item in transfers)
    latest = network_reports[-1]
    return {
        "scenario": spec["name"],
        "placements": resolve_placements(spec),
        "worker3_kind": spec.get("worker3_kind", "gpu"),
        "npu_compute_cycles": spec.get("npu_compute_cycles"),
        "npu_timing": spec.get("npu_timing"),
        "link_types": spec["edges"],
        "status": summary["status"],
        "workload_completion_cycles": summary["final_cycles"],
        "workload_completion_round": summary["rounds"][-1]["round"],
        "network_report_round": latest["round"],
        "network_average_packet_latency_cycles": latest["average_packet_latency_network_cycles"],
        "transfer_count": len(transfers),
        "encoded_payload_bytes_upper_bound": payload_bound,
        "encoded_payload_throughput_MBps_upper_bound": payload_bound / duration / 1e6,
        "router_count": router_count,
        "undirected_link_count": len(spec["edges"]),
        "directed_link_count": directed_counts,
        "edge_metrics": edge_metrics,
        "dynamic_link_energy_j": dynamic,
        "static_link_energy_j": static,
        "link_area_mm2": area,
        "router_dynamic_energy_j": router_dynamic,
        "router_static_energy_j": router_static,
        "router_area_mm2": router_area,
        "total_noc_energy_uJ": (sum(dynamic.values()) + sum(static.values()) +
                                router_dynamic + router_static) * 1e6,
        "total_noc_area_mm2": sum(area.values()) + router_area,
        "workload_validation": read_workload_validation(summary["run_dir"], len(summary["rounds"])),
        "traffic_signature": sorted([item["source"], item["destination"],
                                     item["flits_including_header"], item["descriptor"]]
                                    for item in transfers),
        "workload_signature": sorted([roles_by_node[item["source"]],
                                       roles_by_node[item["destination"]],
                                       item["flits_including_header"], item["descriptor"]]
                                      for item in transfers),
        "assumptions": params,
        "limitations": [
            "Single run of one workload; no uncertainty interval",
            ("Link energy, idle power, and area are isolated DSENT estimates; "
             "router costs and edge delays are illustrative" if "dsent_source" in params
             else "Per-edge delays and energy/area are illustrative values"),
            "Both link types use the same one-flit-per-cycle bandwidth",
            "Useful payload throughput divides checked matrix-transfer bytes by full simulated workload time",
            "No wavelength, thermal, laser-gating, or receiver contention model",
            "Area is routing footprint plus interface/router estimates, not a floorplan",
        ] + (["NPU compute cycles come from a matching SCALE-Sim GEMM model; NPU core energy and area are excluded"
              if "npu_timing" in spec else
              "NPU compute cycles are a placeholder; NPU core energy and area are excluded"]
             if spec.get("worker3_kind") == "npu" else []),
    }


def report(run_dir):
    run_dir = run_dir.resolve()
    spec = load_spec(run_dir / "scenario.json")
    if "npu_timing" in spec:
        verify_npu_model_files(verify_npu_timing(spec, run_dir / "npu_timing.json"), run_dir)
    params = json.loads((run_dir / "link_assumptions.json").read_text())
    check_assumptions(params)
    verify_edge_profile_sources(spec, run_dir, params)
    if (run_dir / "topology.gv").read_text() != graph_text(spec, params):
        raise ValueError("Topology file differs from scenario and assumptions")
    graph_reference = "$SIMULATOR_ROOT/" + (run_dir / "topology.gv").relative_to(ROOT).as_posix()
    input_text = (run_dir / "input.mesh.yml").read_text()
    if input_text.count(graph_reference) != 1:
        raise ValueError("YAML does not reference the saved graph exactly once")
    expected_input = render_yaml(TEMPLATE.read_text(), run_dir,
                                 params["coordinate_width"],
                                 spec.get("worker3_kind", "gpu"),
                                 spec.get("npu_compute_cycles"),
                                 resolve_placements(spec))
    if input_text != expected_input:
        raise ValueError("YAML differs from the generated scenario and template")
    controlled_input_sha = hashlib.sha256(
        input_text.replace(graph_reference, "<SCENARIO_TOPOLOGY>").encode()
    ).hexdigest()
    comparable_input = render_yaml(TEMPLATE.read_text(), run_dir,
                                   params["coordinate_width"],
                                   spec.get("worker3_kind", "gpu"),
                                   spec.get("npu_compute_cycles"))
    comparable_input_sha = hashlib.sha256(
        comparable_input.replace(graph_reference, "<SCENARIO_TOPOLOGY>").encode()
    ).hexdigest()
    summary = parse_coordinator((run_dir / "coordinator.log").read_text())
    summary["run_dir"] = run_dir
    transfers = read_scenario_trace(run_dir / "bench.txt", spec, params)
    useful_bytes = validate_workload_traffic(transfers, spec, params["flit_bits"])
    endpoints = set(placement_nodes(spec).values())
    observed = {endpoint for transfer in transfers
                for endpoint in (transfer["source"], transfer["destination"])}
    if observed != endpoints:
        raise ValueError("Matmul trace did not exercise every placed endpoint")
    reports = read_network_reports(run_dir)
    expected = len(summary["rounds"]) - int(summary["status"] == "converged")
    if len(reports) != expected:
        raise ValueError("Expected %s phase-2 reports, found %s" % (expected, len(reports)))
    for item in reports:
        log = (run_dir / item["log"]).read_text(errors="replace")
        for name, value in (("Routing algorithm", 4),
                            ("ary size", params["coordinate_width"] ** 2),
                            ("cube dimension", 1), ("flit size", 2)):
            if not re.search(r"^%s:\s*%s\s*$" % (re.escape(name), value), log, re.MULTILINE):
                raise ValueError("Popnet config mismatch in %s: %s" % (item["log"], name))
        if item["finished_packets"] != len(transfers):
            raise ValueError("Network packet count differs from trace")
        expected_graph = "/workspace/sim/" + (run_dir / "topology.gv").relative_to(ROOT).as_posix()
        if "-G %s" % expected_graph not in log:
            raise ValueError("Popnet did not run the saved topology in %s" % item["log"])
    result = calculate(summary, transfers, spec, params, reports)
    if result["encoded_payload_bytes_upper_bound"] != useful_bytes:
        raise ValueError("Encoded flit payload bound differs from known matmul bytes")
    result["useful_payload_bytes"] = useful_bytes
    result["useful_payload_throughput_MBps"] = (
        useful_bytes / (result["workload_completion_cycles"] /
                        params["simulated_clock_hz"]) / 1e6)
    result["controlled_input_sha256"] = controlled_input_sha
    result["comparable_input_sha256"] = comparable_input_sha
    if result["workload_validation"]["status"] != "passed":
        raise ValueError("The comparison requires the numerically checked CPU workload")
    if spec.get("worker3_kind") == "npu":
        npu_logs = sorted(run_dir.glob("proc_r*_p1_t*/npu.*.log"))
        if len(npu_logs) != len(summary["rounds"]):
            raise ValueError("Expected one NPU check log per simulation round")
        checks = []
        for log in npu_logs:
            text = log.read_text(errors="replace")
            matches = re.findall(r"NPU_CHECK PASS checksum=(\d+) compute_cycles=(\d+)", text)
            if len(matches) != 1 or int(matches[0][1]) != spec["npu_compute_cycles"]:
                raise ValueError("NPU functional/timing check missing or inconsistent in %s" % log)
            read_end_cycles = re.findall(
                r"\[INTERCMD\] READ [^\n]+\n\[RESPONSE\]\[INTERCMD\] SYNC (\d+)", text)
            write_cycles = re.findall(r"\[INTERCMD\] WRITE (\d+) \d+ \d+ \d+ \d+ 800 0", text)
            if (len(read_end_cycles) != 2 or len(write_cycles) != 1 or
                    int(write_cycles[0]) - int(read_end_cycles[1]) != spec["npu_compute_cycles"]):
                raise ValueError("NPU WRITE cycle did not include the configured compute delay in %s" % log)
            checks.append({"log": str(log.relative_to(run_dir)),
                           "checksum": int(matches[0][0]),
                           "compute_cycles": int(matches[0][1]),
                           "last_input_end_cycle": int(read_end_cycles[1]),
                           "output_write_cycle": int(write_cycles[0])})
        cpu_checksums = [item["checksum"] for item in
                         result["workload_validation"]["round_checks"]]
        if len(cpu_checksums) != len(checks) or any(
                3 * npu["checksum"] != cpu for npu, cpu in zip(checks, cpu_checksums)):
            raise ValueError("NPU tile checksum does not match one third of CPU reduction")
        result["npu_validation"] = {"status": "passed", "round_checks": checks}
    output = run_dir / "analysis"
    output.mkdir(exist_ok=True)
    (output / "mesh_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    snapshots = ("scenario.json", "link_assumptions.json", "topology.gv",
                 "input.mesh.yml", "matmul_checked_c", "matmul_cu")
    if spec.get("worker3_kind") == "npu":
        snapshots += ("matmul_npu",)
    if "npu_timing" in spec:
        snapshots += ("npu_timing.json",) + tuple(item[2] for item in NPU_MODEL_SNAPSHOTS)
    snapshots += tuple(edge_profile_snapshot_name(edge)
                       for edge in spec.get("edge_profile_sources", {}))
    (run_dir / "provenance.json").write_text(json.dumps({
        "snapshots_sha256": {
            name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
            for name in snapshots
        },
        "command_inside_container": (
            "/workspace/sim/interchiplet/bin/interchiplet /workspace/sim/" +
            (run_dir / "input.mesh.yml").relative_to(ROOT).as_posix() +
            " -w %d -f 2" % params["coordinate_width"]
        ),
        "note": "Hashes are of saved input and binary snapshots; the simulator binaries and source tree were not archived here.",
    }, indent=2) + "\n")
    with (output / "mesh_transfers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transfers[0]))
        writer.writeheader()
        writer.writerows(transfers)
    with (output / "mesh_edges.csv").open("w", newline="") as handle:
        fields = (["edge", "kind"] + list(EDGE_COST_FIELDS) +
                  ["transmitted_bits_including_header", "dynamic_energy_j",
                   "static_energy_j", "area_mm2"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for edge, metrics in result["edge_metrics"].items():
            writer.writerow({"edge": edge, "kind": metrics["kind"],
                             **metrics["parameters"],
                             **{field: metrics[field] for field in fields[-4:]}})
    return result


def row(result):
    return {name: result[name] for name in (
        "scenario", "status", "workload_completion_cycles", "workload_completion_round",
        "network_report_round",
        "network_average_packet_latency_cycles", "useful_payload_throughput_MBps",
        "encoded_payload_throughput_MBps_upper_bound",
        "total_noc_energy_uJ", "total_noc_area_mm2")}


def compare(run_dirs, output, make_plot):
    results = [report(path) for path in run_dirs]
    if len({result["scenario"] for result in results}) != len(results):
        raise ValueError("Scenario names must be unique")
    signatures = {json.dumps(result["workload_signature"]) for result in results}
    checksums = {tuple(check["checksum"] for check in result["workload_validation"]["round_checks"])
                 for result in results}
    assumptions = {json.dumps(result["assumptions"], sort_keys=True) for result in results}
    binaries = {
        tuple(hashlib.sha256((path / binary).read_bytes()).hexdigest()
              for binary in (("matmul_checked_c", "matmul_cu", "matmul_npu")
                             if load_spec(path / "scenario.json").get("worker3_kind") == "npu"
                             else ("matmul_checked_c", "matmul_cu")))
        for path in run_dirs
    }
    inputs = {result["comparable_input_sha256"] for result in results}
    npu_timings = {json.dumps(result["npu_timing"], sort_keys=True) for result in results}
    if any(len(group) != 1 for group in (signatures, checksums, assumptions,
                                         binaries, inputs, npu_timings)):
        raise ValueError("Runs differ in traffic, checksums, assumptions, binaries, NPU timing, or fixed YAML settings")
    if any(result["status"] != "converged" for result in results):
        raise ValueError("Compare only converged runs")
    results.sort(key=lambda result: result["scenario"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row(results[0])))
        writer.writeheader()
        writer.writerows(row(result) for result in results)
    cost_note = ("DSENT-derived isolated link costs; router costs and link delays remain illustrative"
                 if "dsent_source" in results[0]["assumptions"] else
                 "Shared placeholder link and router costs")
    (output / "comparison.json").write_text(json.dumps({
        "runs": [str(path) for path in run_dirs],
        "results": results,
        "interpretation": ("One validated workload and one run per architecture. " + cost_note +
                           ". Useful payload throughput is the checked 482400 bytes divided by "
                           "full workload duration. No statistical or physical validation claim."),
    }, indent=2, allow_nan=False, default=str) + "\n")
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        placement_varies = len({json.dumps(result["placements"], sort_keys=True)
                                for result in results}) > 1
        link_layout_varies = len({json.dumps(result["link_types"], sort_keys=True)
                                  for result in results}) > 1
        profile_varies = any(any(
            metrics["parameters"] != {field: result["assumptions"][metrics["kind"]][field]
                                      for field in EDGE_COST_FIELDS}
            for metrics in result["edge_metrics"].values()) for result in results)
        labels = []
        for result in results:
            if placement_varies:
                default = default_placements(result["assumptions"]["coordinate_width"])
                moved = []
                for role, position in result["placements"].items():
                    if position != default[role]:
                        name = "NPU" if role == "worker3" and result["worker3_kind"] == "npu" else (
                            "CPU" if role == "cpu" else role.replace("worker", "GPU "))
                        moved.append("%s (%d,%d)" % (name, *position))
                label = "\n".join(moved) if moved else "Default\nplacement"
                if link_layout_varies:
                    layout = re.sub(r"_\d+x\d+.*$", "", result["scenario"])
                    label = layout.replace("_", " ").title() + "\n" + label
            elif profile_varies:
                changed = [(edge, metrics["parameters"]) for edge, metrics in result["edge_metrics"].items()
                           if metrics["parameters"] != {
                               field: result["assumptions"][metrics["kind"]][field]
                               for field in EDGE_COST_FIELDS}]
                label = ("\n".join("Edge %s: %g mm, %d cycles" %
                                   (edge, profile["length_mm"], profile["edge_delay_cycles"])
                                   for edge, profile in changed) if changed else "Default\nlinks")
                if link_layout_varies:
                    layout = re.sub(r"_\d+x\d+.*$", "", result["scenario"])
                    label = layout.replace("_", " ").title() + "\n" + label
            else:
                label = re.sub(r"_\d+x\d+(?:_npu(?:_scalesim)?)?$", "", result["scenario"])
                label = label.replace("_", "\n").title()
            labels.append(label)
        metrics = (
            ("workload_completion_cycles", "Workload completion (cycles)"),
            ("useful_payload_throughput_MBps", "Useful payload per workload time (MB/s)"),
            ("total_noc_energy_uJ", "Illustrative NoC energy (µJ)"),
            ("total_noc_area_mm2", "Illustrative NoC area (mm²)"),
        )
        for index, (ax, (key, title)) in enumerate(zip(axes.flat, metrics)):
            values = [result[key] for result in results]
            if index < 2:
                ax.plot(range(len(values)), values, "o", color="#287b91", markersize=8)
                ax.set_xticks(range(len(values)), labels)
                span = max(values) - min(values)
                margin = max(span * 0.5, min(values) * 0.0002)
                ax.set_ylim(min(values) - margin, max(values) + margin)
            else:
                ax.bar(labels, values, color="#287b91")
            ax.set_title(title)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        fig.suptitle("One checked workload run per architecture; illustrative device costs")
        fig.tight_layout()
        fig.savefig(output / "comparison.png", dpi=180)
        fig.savefig(output / "comparison.svg")
        plt.close(fig)
    return results


def run_prepared(run_dir):
    container_dir = "/workspace/sim/" + run_dir.relative_to(ROOT).as_posix()
    params = json.loads((run_dir / "link_assumptions.json").read_text())
    spec = load_spec(run_dir / "scenario.json")
    check_assumptions(params)
    build = ("set -e; source /workspace/sim/setup_env.sh >/dev/null 2>&1; "
             "make -C /workspace/sim/benchmark/matmul_test -f makefile checked%s >/dev/null" %
             (" npu_checked" if spec.get("worker3_kind") == "npu" else ""))
    subprocess.run(["docker", "exec", "chiplet", "bash", "-c", build], check=True)
    binaries = ["matmul_checked_c", "matmul_cu"]
    if spec.get("worker3_kind") == "npu":
        binaries.append("matmul_npu")
    for binary in binaries:
        shutil.copy2(BENCHMARK / "bin" / binary, run_dir / binary)
    command = (
        "set -e; source /workspace/sim/setup_env.sh >/dev/null 2>&1; "
        "cd %s; /workspace/sim/interchiplet/bin/interchiplet %s -w %d -f 2 > coordinator.log 2>&1"
        % (shlex.quote(container_dir), shlex.quote(container_dir + "/input.mesh.yml"),
           params["coordinate_width"])
    )
    subprocess.run(["docker", "exec", "chiplet", "bash", "-c", command], check=True)
    return report(run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        sub = commands.add_parser(name)
        sub.add_argument("scenario", type=Path)
        sub.add_argument("--assumptions", type=Path, default=ASSUMPTIONS)
    sub = commands.add_parser("report")
    sub.add_argument("run_dir", type=Path)
    sub = commands.add_parser("compare")
    sub.add_argument("run_dirs", nargs="+", type=Path)
    sub.add_argument("--output", required=True, type=Path)
    sub.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        if args.command in ("prepare", "run"):
            run_dir = prepare(args.scenario, args.assumptions)
            print(run_dir, flush=True)
            if args.command == "run":
                result = run_prepared(run_dir)
                print("%s: %s cycles, %.3f µJ, %.3f mm²" % (
                    result["scenario"], result["workload_completion_cycles"],
                    result["total_noc_energy_uJ"], result["total_noc_area_mm2"]))
        elif args.command == "report":
            result = report(args.run_dir)
            print("%s: %s cycles" % (result["scenario"], result["workload_completion_cycles"]))
        else:
            results = compare(args.run_dirs, args.output, args.plot)
            print("Compared %s scenarios in %s" % (len(results), args.output))
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError,
            ImportError) as error:
        parser.exit(1, "Mesh study failed: %s\n" % error)


if __name__ == "__main__":
    main()
