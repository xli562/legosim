"""Main entry for the ONNX-driven GCN chiplet estimator."""

import json
import os

import onnx

from .cli import parse_args
from .common import parse_ops, resolve_memory_profile, resolve_onnx_path, to_abs
from .gem5_calibration import apply_gem5_calibration
from .graph import (
    assign_devices,
    build_dependency_graph,
    build_subgraphs,
    compute_subgraph_critical_path,
    infer_and_collect_tensors,
    merge_tiny_runs,
    summarize_memory_flow,
)
from .modeling import (
    apply_template_memory_profiles,
    build_tensor_flow_maps,
    estimate_node_profiles,
    estimate_subgraph_plan,
    schedule_subgraphs,
)
from .onnx_analysis import run_onnx_analysis
from .outputs import write_bench, write_partition_py, write_pim_pe_workload, write_runtime_plan
from .performance import ops_and_memory_cfg
from .planning import (
    build_interchiplet_bench,
    build_ir_pe_task_plan,
    build_phase1_runtime_plan,
    build_pim_intra_bench,
    resolve_phase1_runtime_modes,
)
from .reporting import write_sim_summary_csv


def main():
    """Run ONNX partitioning, timing estimation, and bench generation."""
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cfg_path = to_abs(base_dir, args.config)
    with open(cfg_path, "r", encoding="utf-8-sig") as handle:
        cfg = json.load(handle)

    onnx_path = resolve_onnx_path(base_dir, args.onnx)
    if onnx_path is None:
        raise SystemExit("[Estimator] ERROR: cannot find ONNX model.")

    model = onnx.load(onnx_path)
    model, tensor_info = infer_and_collect_tensors(model)

    nodes = list(model.graph.node)
    initializers = {initializer.name for initializer in model.graph.initializer}
    node_ids, predecessors, successors, topo = build_dependency_graph(nodes, initializers)
    graph_outputs = {value.name for value in model.graph.output if value.name}
    output_to_node, tensor_consumers = build_tensor_flow_maps(nodes, node_ids)

    cpu_ops = parse_ops(args.cpu_ops)
    pim_ops = parse_ops(args.pim_ops)
    io_ops = parse_ops(args.io_ops)
    node_to_device, _ = assign_devices(nodes, node_ids, cpu_ops, pim_ops, io_ops)
    merge_tiny_runs(topo, node_to_device, args.min_subgraph_size)

    device_subgraphs = {"CPU": [], "PIM": [], "IO": []}
    for node_id in topo:
        device_subgraphs[node_to_device[node_id]].append(node_id)
    subgraph_plan, node_to_sg = build_subgraphs(topo, node_to_device, successors)

    partition_info = {
        "total_nodes": len(nodes),
        "node_to_device": node_to_device,
        "device_subgraphs": device_subgraphs,
        "sorted_execution_order": topo,
        "subgraph_plan": subgraph_plan,
    }

    part_out = to_abs(base_dir, args.partition_info)
    write_partition_py(part_out, partition_info, onnx_path)

    print(f"[Estimator] ONNX source: {onnx_path}")
    print(f"[Estimator] Total nodes: {len(nodes)}")
    print(f"[Estimator] Partition saved: {part_out}")
    if args.partition_only:
        return 0

    est_cfg = cfg.get("estimation", {})
    _selected_profile_name, _selected_profile_cfg = resolve_memory_profile(est_cfg, args.memory_profile)

    dev_ops_base, _, _ = ops_and_memory_cfg(cfg)
    cfg_runtime, dev_ops, gem5_calibration = apply_gem5_calibration(cfg, dev_ops_base, args, base_dir)
    _, mem_bw, memory_levels = ops_and_memory_cfg(cfg_runtime)
    est_cfg_runtime = cfg_runtime.get("estimation", {})
    memory_profile_name, memory_profile_cfg = resolve_memory_profile(est_cfg_runtime, args.memory_profile)

    node_stats, node_meta = estimate_node_profiles(
        nodes,
        node_ids,
        node_to_device,
        tensor_info,
        initializers,
        cfg_runtime,
    )

    analysis_info = run_onnx_analysis(
        model,
        tensor_info,
        est_cfg_runtime,
        cfg_runtime,
        node_to_device=node_to_device,
        memory_profile_name=memory_profile_name,
        memory_profile_cfg=memory_profile_cfg,
    )
    apply_template_memory_profiles(analysis_info, node_stats, node_to_device)

    phase1_strategy, phase1_device_modes = resolve_phase1_runtime_modes(cfg_runtime)
    task_plan, node_task_map, pe_arch = build_ir_pe_task_plan(
        topo,
        predecessors,
        node_to_device,
        node_stats,
        cfg_runtime,
        dev_ops,
        int(args.pim_node_offset),
    )

    estimate_subgraph_plan(
        subgraph_plan,
        node_stats,
        node_meta,
        tensor_info,
        output_to_node,
        tensor_consumers,
        graph_outputs,
        cfg_runtime,
        dev_ops,
        mem_bw,
        memory_levels,
        memory_profile_cfg,
        analysis_info,
    )
    schedule_subgraphs(subgraph_plan)

    critical_path_cycles, critical_path_by_subgraph = compute_subgraph_critical_path(subgraph_plan)
    makespan_cycles = int(max(item["end_cycle"] for item in subgraph_plan) if subgraph_plan else 0)
    memory_flow_summary = summarize_memory_flow(subgraph_plan)

    flit_payload = int(cfg_runtime["estimation"]["flit_payload_bytes"])
    bench_lines = []
    if not args.no_inter_bench:
        bench_lines = build_interchiplet_bench(
            topo,
            predecessors,
            node_to_device,
            node_stats,
            node_task_map,
            flit_payload,
            phase1_strategy=phase1_strategy,
            phase1_device_modes=phase1_device_modes,
        )

    npu_count = int(cfg.get("pim", {}).get("npu_count", 16))
    bench_intra_lines, pim_mapping = build_pim_intra_bench(
        topo,
        node_to_device,
        node_stats,
        node_task_map,
        flit_payload,
        npu_count,
        int(args.pim_node_offset),
        cfg_runtime,
    )
    pim_mapping["mapping_mode"] = "onnx_node_split"

    bench_out = to_abs(base_dir, args.bench)
    bench_intra_out = to_abs(base_dir, args.bench_intra)
    if not args.no_inter_bench:
        bench_lines = write_bench(bench_out, bench_lines)
    else:
        bench_lines = []
    bench_intra_lines = write_bench(bench_intra_out, bench_intra_lines)

    pim_mapping_out = to_abs(base_dir, args.pim_mapping_report)
    os.makedirs(os.path.dirname(pim_mapping_out) or ".", exist_ok=True)
    with open(pim_mapping_out, "w", encoding="utf-8") as handle:
        json.dump(pim_mapping, handle, indent=2)

    pim_pe_workload_out = to_abs(base_dir, args.pim_pe_workload)
    pim_pe_workload_summary = write_pim_pe_workload(pim_pe_workload_out, npu_count, node_task_map)

    runtime_plan = build_phase1_runtime_plan(
        topo,
        successors,
        node_to_device,
        node_to_sg,
        node_stats,
        subgraph_plan,
        phase1_strategy=phase1_strategy,
        phase1_device_modes=phase1_device_modes,
    )
    runtime_pim_out = to_abs(base_dir, args.runtime_plan_pim)
    runtime_io_out = to_abs(base_dir, args.runtime_plan_io)
    write_runtime_plan(runtime_pim_out, "PIM", runtime_plan["events"]["PIM"])
    write_runtime_plan(runtime_io_out, "IO", runtime_plan["events"]["IO"])

    report = {
        "onnx_path": onnx_path,
        "chiplet_config": cfg,
        "chiplet_config_runtime": cfg_runtime,
        "memory_profile": {"name": memory_profile_name, "config": memory_profile_cfg},
        "partition": partition_info,
        "gem5_calibration": gem5_calibration,
        "analysis_summary": analysis_info,
        "subgraphs": subgraph_plan,
        "pe_arch": pe_arch,
        "task_plan": task_plan,
        "node_estimates": node_stats,
        "bench_files": {
            "inter_chiplet": None if args.no_inter_bench else bench_out,
            "pim_intra": bench_intra_out,
        },
        "pim_mapping_file": pim_mapping_out,
        "pim_pe_workload_file": pim_pe_workload_out,
        "pim_pe_workload_summary": pim_pe_workload_summary,
        "runtime_plan_files": {
            "PIM": runtime_pim_out,
            "IO": runtime_io_out,
        },
        "runtime_plan": runtime_plan,
        "phase1_runtime": {
            "strategy": phase1_strategy,
            "device_modes": phase1_device_modes,
        },
        "totals": {
            "cycles_by_device": {
                "CPU": int(sum(item["total_cycles"] for item in subgraph_plan if item["device"] == "CPU")),
                "PIM": int(sum(item["total_cycles"] for item in subgraph_plan if item["device"] == "PIM")),
                "IO": int(sum(item["total_cycles"] for item in subgraph_plan if item["device"] == "IO")),
            },
            "memory_flow": memory_flow_summary,
            "critical_path_cycles": int(critical_path_cycles),
            "makespan_cycles": int(makespan_cycles),
            "critical_path_by_subgraph": critical_path_by_subgraph,
            "pe_task_count": int(len(task_plan)),
            "pe_task_cycles_by_device": {
                "CPU": int(sum(task["total_cycles"] for task in task_plan if task["device"] == "CPU")),
                "PIM": int(sum(task["total_cycles"] for task in task_plan if task["device"] == "PIM")),
                "IO": int(sum(task["total_cycles"] for task in task_plan if task["device"] == "IO")),
            },
        },
    }

    report_out = to_abs(base_dir, args.report)
    os.makedirs(os.path.dirname(report_out) or ".", exist_ok=True)
    with open(report_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    csv_out = write_sim_summary_csv(report_out, subgraph_plan)

    print(f"[Estimator] Report saved: {report_out}")
    print(f"[Estimator] CSV saved: {csv_out}")
    if args.no_inter_bench:
        print("[Estimator] inter bench generation skipped (--no-inter-bench)")
    else:
        print(f"[Estimator] inter bench saved: {bench_out}, lines={len(bench_lines)}")
    print(f"[Estimator] pim intra bench saved: {bench_intra_out}, lines={len(bench_intra_lines)}")
    print(f"[Estimator] PIM mapping saved: {pim_mapping_out}")
    print(f"[Estimator] PIM PE workload saved: {pim_pe_workload_out}")
    print(f"[Estimator] runtime plan (PIM) saved: {runtime_pim_out}")
    print(f"[Estimator] runtime plan (IO) saved: {runtime_io_out}")

    if args.verbose:
        print("[Estimator] Device node counts:")
        for dev in ["CPU", "PIM", "IO"]:
            print(f"  {dev}: {len(device_subgraphs[dev])}")
        print(f"[Estimator] memory profile: {memory_profile_name}")
        print(f"[Estimator] analysis mode: {analysis_info.get('analysis_mode')}")
        node_memory = analysis_info.get("node_memory", {})
        print(
            "[Estimator] ONNX template memory coverage: "
            f"nodes={node_memory.get('node_profile_count', 0)}, "
            f"supported_templates={node_memory.get('supported_template_count', 0)}"
        )
        if gem5_calibration.get("enabled", False):
            print("[Estimator] gem5 calibration enabled")
            interchiplet_cmd = gem5_calibration.get("interchiplet_cmd")
            if isinstance(interchiplet_cmd, dict):
                print(
                    "[Estimator] interchiplet cmd returncode: "
                    f"{interchiplet_cmd.get('returncode')}"
                )
            for dev in ["CPU", "IO"]:
                scale_info = gem5_calibration.get("scales", {}).get(dev)
                if scale_info:
                    print(
                        f"  {dev}: scale={scale_info['ops_scale']:.4f}, "
                        f"ipc={scale_info['ipc']:.4f}, ref_ipc={scale_info['reference_ipc']:.4f}"
                    )
            print(
                "[Estimator] cache_miss_penalty: "
                f"{gem5_calibration.get('cache_miss_penalty_before')} -> "
                f"{gem5_calibration.get('cache_miss_penalty_after')}"
            )

    return 0
