from collections import defaultdict

from .common import (
    estimate_node_ops,
    estimate_streamed_weight_bytes,
    get_tensor_bytes_for_device,
    is_weight_initializer,
    safe_numel,
)
from .graph import estimate_subgraph_bytes_locality
from .performance import estimate_cycles, estimate_cycles_with_explicit_memory, ops_and_memory_cfg


def build_tensor_flow_maps(nodes, node_ids):
    """Build producer/consumer maps for tensors."""
    output_to_node = {}
    tensor_consumers = defaultdict(set)
    for index, node in enumerate(nodes):
        node_id = node_ids[index]
        seen_inputs = set()
        seen_outputs = set()
        for name in node.input:
            if name and name not in seen_inputs:
                seen_inputs.add(name)
                tensor_consumers[name].add(node_id)
        for name in node.output:
            if name and name not in seen_outputs:
                seen_outputs.add(name)
                output_to_node[name] = node_id
    return output_to_node, tensor_consumers


def _normalize_workload(workload):
    """Normalize estimate_node_ops output for compatibility."""
    if isinstance(workload, dict):
        ops_total = int(max(0, workload.get("ops", 0)))
        return {
            "ops": int(max(1, ops_total)),
            "int8_ops": int(max(0, workload.get("int8_ops", 0))),
            "int32_ops": int(max(0, workload.get("int32_ops", 0))),
            "compute_kind": str(workload.get("compute_kind", "generic")),
        }

    try:
        ops_total = int(workload)
    except Exception:
        ops_total = 0
    ops_total = int(max(1, ops_total))
    return {
        "ops": ops_total,
        "int8_ops": ops_total,
        "int32_ops": 0,
        "compute_kind": "generic",
    }


def estimate_node_profiles(nodes, node_ids, node_to_device, tensor_info, initializers, cfg_runtime):
    """Estimate per-node ops, bytes, and cycle-level timing."""
    est_cfg_runtime = cfg_runtime.get("estimation", {})
    dev_ops, mem_bw, memory_levels = ops_and_memory_cfg(cfg_runtime)
    node_stats = {}
    node_meta = {}

    for index, node in enumerate(nodes):
        node_id = node_ids[index]
        device = node_to_device[node_id]
        in_shapes = []
        out_shapes = []
        activation_in_bytes = 0
        weight_bytes = 0
        constant_initializer_bytes = 0
        out_bytes = 0
        seen_inputs = set()
        seen_outputs = set()
        activation_inputs = []
        output_tensors = []
        weight_tensors = []
        constant_tensors = []

        for name in node.input:
            if name in seen_inputs:
                continue
            seen_inputs.add(name)
            if name in tensor_info:
                shape, dtype = tensor_info[name]
                numel = safe_numel(shape)
                in_shapes.append(shape)
                if name in initializers:
                    nbytes = numel * 4
                    if is_weight_initializer(node.op_type, nbytes, est_cfg_runtime):
                        num_bytes = get_tensor_bytes_for_device(numel, dtype, "weight", cfg_runtime, device)
                        weight_bytes += num_bytes
                        weight_tensors.append(name)
                    else:
                        constant_initializer_bytes += numel * 4
                        constant_tensors.append(name)
                else:
                    num_bytes = get_tensor_bytes_for_device(numel, dtype, "activation", cfg_runtime, device)
                    activation_in_bytes += num_bytes
                    activation_inputs.append(name)
            elif name and name not in initializers:
                activation_inputs.append(name)

        for name in node.output:
            if name in seen_outputs:
                continue
            seen_outputs.add(name)
            if name:
                output_tensors.append(name)
            if name in tensor_info:
                shape, dtype = tensor_info[name]
                numel = safe_numel(shape)
                out_shapes.append(shape)
                out_bytes += get_tensor_bytes_for_device(numel, dtype, "output", cfg_runtime, device)

        workload = _normalize_workload(estimate_node_ops(node, in_shapes, out_shapes))
        ops_total = int(workload.get("ops", 0))
        streamed_weight_bytes = estimate_streamed_weight_bytes(weight_bytes, est_cfg_runtime)
        bytes_total = activation_in_bytes + out_bytes + streamed_weight_bytes
        cycles = estimate_cycles(device, workload, bytes_total, cfg_runtime, dev_ops, mem_bw, memory_levels)

        node_stats[node_id] = {
            "op_type": node.op_type,
            "device": device,
            "ops": ops_total,
            "int8_ops": int(workload.get("int8_ops", 0)),
            "int32_ops": int(workload.get("int32_ops", 0)),
            "compute_kind": str(workload.get("compute_kind", "generic")),
            "bytes": int(bytes_total),
            "bytes_breakdown": {
                "activation_in_bytes": int(activation_in_bytes),
                "weight_bytes": int(weight_bytes),
                "streamed_weight_bytes": int(streamed_weight_bytes),
                "constant_initializer_bytes": int(constant_initializer_bytes),
                "output_bytes": int(out_bytes),
            },
            **cycles,
        }
        node_meta[node_id] = {
            "op_type": node.op_type,
            "activation_inputs": activation_inputs,
            "outputs": output_tensors,
            "weight_tensors": weight_tensors,
            "constant_tensors": constant_tensors,
            "streamed_weight_bytes": int(streamed_weight_bytes),
            "constant_initializer_bytes": int(constant_initializer_bytes),
        }

    return node_stats, node_meta


def apply_template_memory_profiles(analysis_info, node_stats, node_to_device):
    """Attach ONNX-shape-driven explicit memory profiles onto node_stats."""
    node_memory = analysis_info.get("node_memory", {}) if isinstance(analysis_info, dict) else {}
    node_profiles = node_memory.get("node_profiles", {}) if isinstance(node_memory, dict) else {}
    if not isinstance(node_profiles, dict):
        return

    for node_id, profile in node_profiles.items():
        if node_id not in node_stats or not isinstance(profile, dict):
            continue
        if not profile.get("supported", False):
            continue

        stat = node_stats[node_id]
        device = str(node_to_device.get(node_id, stat.get("device", "CPU"))).upper()
        onchip_bytes = int(max(0, profile.get("onchip_bytes", 0)))
        offchip_bytes = int(max(0, profile.get("offchip_bytes", 0)))
        total_bytes = int(max(0, profile.get("total_bytes", onchip_bytes + offchip_bytes)))
        memory_cycles = int(max(0, profile.get("memory_cycles", 0)))

        stat["onnx_template_memory"] = {
            "enabled": True,
            "device": device,
            "template_kind": profile.get("template_kind"),
            "source": profile.get("source"),
            "onchip_bytes": onchip_bytes,
            "offchip_bytes": offchip_bytes,
            "total_bytes": total_bytes,
            "memory_cycles": memory_cycles,
        }
        stat["analytic_onchip_bytes"] = onchip_bytes
        stat["analytic_offchip_bytes"] = offchip_bytes
        stat["analytic_total_memory_bytes"] = total_bytes
        stat["analytic_memory_cycles"] = memory_cycles

        bytes_breakdown = stat.setdefault("bytes_breakdown", {})
        bytes_breakdown["onnx_template_memory"] = {
            "enabled": True,
            "template_kind": profile.get("template_kind"),
            "onchip_bytes": onchip_bytes,
            "offchip_bytes": offchip_bytes,
            "total_bytes": total_bytes,
            "memory_cycles": memory_cycles,
        }

        pim_tiling = profile.get("pim_sram_tiling")
        if device == "PIM" and isinstance(pim_tiling, dict) and pim_tiling.get("supported", False):
            stat["pim_sram_tiling"] = pim_tiling
            stat["pim_peak_sram_bytes"] = int(max(0, pim_tiling.get("peak_sram_bytes", 0)))
            stat["pim_fits_sram"] = bool(pim_tiling.get("fits_sram", False))
            bytes_breakdown["pim_sram_tiling"] = {
                "supported": True,
                "peak_sram_bytes": int(max(0, pim_tiling.get("peak_sram_bytes", 0))),
                "sram_capacity_bytes": int(max(0, pim_tiling.get("sram_capacity_bytes", 0))),
                "activation_tile_bytes": int(max(0, pim_tiling.get("activation_tile_bytes", 0))),
                "weight_tile_bytes": int(max(0, pim_tiling.get("weight_tile_bytes", 0))),
                "accumulator_tile_bytes": int(max(0, pim_tiling.get("accumulator_tile_bytes", 0))),
                "output_tile_bytes": int(max(0, pim_tiling.get("output_tile_bytes", 0))),
                "reuse_bytes": int(max(0, pim_tiling.get("reuse_bytes", 0))),
                "evict_bytes": int(max(0, pim_tiling.get("evict_bytes", 0))),
                "tile_plan": pim_tiling.get("tile_plan"),
                "memory_cycles_breakdown": pim_tiling.get("memory_cycles_breakdown", {}),
            }


def estimate_subgraph_plan(
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
):
    """Estimate timing for each subgraph, using explicit PIM memory detail when available."""
    est_cfg_runtime = cfg_runtime.get("estimation", {})
    node_memory = analysis_info.get("node_memory", {}) if isinstance(analysis_info, dict) else {}
    node_profiles = node_memory.get("node_profiles", {}) if isinstance(node_memory, dict) else {}

    for subgraph in subgraph_plan:
        subgraph_nodes = subgraph["nodes"]
        ops = sum(node_stats[node_id]["ops"] for node_id in subgraph_nodes)
        int8_ops = sum(node_stats[node_id].get("int8_ops", 0) for node_id in subgraph_nodes)
        int32_ops = sum(node_stats[node_id].get("int32_ops", 0) for node_id in subgraph_nodes)

        graph_bytes_total, graph_bytes_breakdown = estimate_subgraph_bytes_locality(
            subgraph,
            node_meta,
            tensor_info,
            output_to_node,
            tensor_consumers,
            graph_outputs,
            est_cfg_runtime,
            device=subgraph["device"],
            memory_profile=memory_profile_cfg,
            subgraph_io_scale=1.0,
        )
        bytes_breakdown = dict(graph_bytes_breakdown)

        supported_nodes = []
        unsupported_nodes = []
        explicit_onchip_bytes = 0
        explicit_offchip_bytes = 0
        peak_sram_bytes = 0
        pim_tiling_nodes = []

        if isinstance(node_profiles, dict) and node_memory.get("enabled", False):
            for node_id in subgraph_nodes:
                profile = node_profiles.get(node_id)
                if isinstance(profile, dict) and profile.get("supported", False):
                    supported_nodes.append(node_id)
                    explicit_onchip_bytes += int(max(0, profile.get("onchip_bytes", 0)))
                    explicit_offchip_bytes += int(max(0, profile.get("offchip_bytes", 0)))
                    peak_sram_bytes = max(
                        peak_sram_bytes,
                        int(max(0, profile.get("peak_sram_bytes", 0))),
                    )
                    if isinstance(profile.get("pim_sram_tiling"), dict):
                        pim_tiling_nodes.append(node_id)
                else:
                    unsupported_nodes.append(node_id)
        else:
            unsupported_nodes = list(subgraph_nodes)

        subgraph_workload = {
            "ops": ops,
            "int8_ops": int8_ops,
            "int32_ops": int32_ops,
            "compute_kind": "generic",
        }

        if supported_nodes:
            fallback_offchip_bytes = 0
            fallback_breakdown = {}
            if unsupported_nodes:
                fallback_offchip_bytes, fallback_breakdown = estimate_subgraph_bytes_locality(
                    {"nodes": unsupported_nodes},
                    node_meta,
                    tensor_info,
                    output_to_node,
                    tensor_consumers,
                    graph_outputs,
                    est_cfg_runtime,
                    device=subgraph["device"],
                    memory_profile=memory_profile_cfg,
                    subgraph_io_scale=1.0,
                )

            explicit_onchip_bytes += int(fallback_breakdown.get("estimated_onchip_reuse_bytes", 0))
            explicit_offchip_bytes += int(fallback_offchip_bytes)
            bytes_breakdown["onnx_template_memory"] = {
                "enabled": True,
                "supported_node_count": int(len(supported_nodes)),
                "unsupported_node_count": int(len(unsupported_nodes)),
                "supported_nodes": supported_nodes,
                "unsupported_nodes": unsupported_nodes,
                "onchip_bytes": int(explicit_onchip_bytes),
                "offchip_bytes": int(explicit_offchip_bytes),
                "max_peak_sram_bytes": int(peak_sram_bytes),
                "pim_tiling_nodes": pim_tiling_nodes,
                "profile_name": node_memory.get("profile_name"),
                "analysis_source": node_memory.get("analysis_source"),
            }
            bytes_breakdown["selected_memory_source"] = "onnx_template_explicit"
            subgraph.update(
                estimate_cycles_with_explicit_memory(
                    subgraph["device"],
                    subgraph_workload,
                    explicit_onchip_bytes,
                    explicit_offchip_bytes,
                    cfg_runtime,
                    dev_ops,
                    mem_bw,
                    memory_levels,
                    memory_source="onnx_template",
                )
            )
            bytes_total = int(explicit_offchip_bytes)
            extra_onchip_bytes = int(explicit_onchip_bytes)
        else:
            extra_onchip_bytes = int(graph_bytes_breakdown.get("estimated_onchip_reuse_bytes", 0))
            bytes_breakdown["selected_memory_source"] = "graph_locality"
            subgraph.update(
                estimate_cycles(
                    subgraph["device"],
                    subgraph_workload,
                    graph_bytes_total,
                    cfg_runtime,
                    dev_ops,
                    mem_bw,
                    memory_levels,
                    extra_onchip_bytes=extra_onchip_bytes,
                )
            )
            bytes_total = int(graph_bytes_total)

        subgraph["estimated_ops"] = int(ops)
        subgraph["estimated_int8_ops"] = int(int8_ops)
        subgraph["estimated_int32_ops"] = int(int32_ops)
        subgraph["estimated_bytes"] = int(bytes_total)
        subgraph["estimated_total_memory_bytes"] = int(bytes_total + extra_onchip_bytes)
        subgraph["estimated_bytes_breakdown"] = bytes_breakdown


def schedule_subgraphs(subgraph_plan):
    """Assign start/end cycles to each scheduled subgraph."""
    subgraph_end = {}
    device_ready = {"CPU": 0, "PIM": 0, "IO": 0}
    for subgraph in subgraph_plan:
        ready = 0
        for dep in subgraph.get("depends_on", []):
            ready = max(ready, subgraph_end.get(dep, 0))
        start = max(ready, device_ready[subgraph["device"]])
        end = start + subgraph["total_cycles"]
        subgraph["start_cycle"] = int(start)
        subgraph["end_cycle"] = int(end)
        subgraph_end[subgraph["id"]] = end
        device_ready[subgraph["device"]] = end

