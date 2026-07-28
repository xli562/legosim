import math
from collections import defaultdict, deque

from onnx import shape_inference

from .common import (
    clamp_ratio,
    estimate_streamed_weight_bytes,
    get_tensor_bytes_for_device,
    pick_device_entry,
    safe_numel,
    tensor_nbytes,
    tensor_shape_from_value_info,
)


def infer_and_collect_tensors(model):
    """
    Run ONNX shape inference and collect shape and dtype metadata for every known tensor.
    The returned tensor_info map covers graph inputs, value_info entries, outputs,
    and initializers.
    """
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    tensor_info = {}

    def add_value(value):
        if not value.name:
            return
        try:
            shape, dtype = tensor_shape_from_value_info(value)
            tensor_info[value.name] = (shape, dtype)
        except Exception:
            pass

    for value in model.graph.input:
        add_value(value)
    for value in model.graph.value_info:
        add_value(value)
    for value in model.graph.output:
        add_value(value)
    for initializer in model.graph.initializer:
        tensor_info[initializer.name] = (list(initializer.dims), initializer.data_type)

    return model, tensor_info


def build_dependency_graph(nodes, initializers):
    """
    Build the ONNX dependency graph and return a topological ordering.
    Returns node_ids, predecessor sets, successor sets, and the topological order.
    If a cycle is detected, the original node order is used as a fallback.
    """
    node_ids = []
    output_to_node = {}
    predecessors = defaultdict(set)
    successors = defaultdict(set)

    for index, node in enumerate(nodes):
        node_id = f"{node.op_type}_{index}"
        node_ids.append(node_id)
        for out_name in node.output:
            if out_name:
                output_to_node[out_name] = node_id

    for index, node in enumerate(nodes):
        current = node_ids[index]
        for input_name in node.input:
            if not input_name or input_name in initializers:
                continue
            prev = output_to_node.get(input_name)
            if prev and prev != current:
                predecessors[current].add(prev)
                successors[prev].add(current)

    for node_id in node_ids:
        predecessors[node_id]
        successors[node_id]

    indegree = {node_id: len(predecessors[node_id]) for node_id in node_ids}
    queue = deque([node_id for node_id in node_ids if indegree[node_id] == 0])
    topo = []
    while queue:
        node_id = queue.popleft()
        topo.append(node_id)
        for next_id in successors[node_id]:
            indegree[next_id] -= 1
            if indegree[next_id] == 0:
                queue.append(next_id)

    if len(topo) != len(node_ids):
        topo = node_ids

    return node_ids, predecessors, successors, topo


def assign_devices(nodes, node_ids, cpu_ops, pim_ops, io_ops):
    """
    Assign each node to CPU, PIM, or IO according to the configured op allowlists.
    Priority is PIM first, then IO, then CPU, with PIM used as the default fallback.
    """
    mapping = {}
    by_device = {"CPU": [], "PIM": [], "IO": []}
    for index, node in enumerate(nodes):
        node_id = node_ids[index]
        op = node.op_type
        if op in pim_ops:
            device = "PIM"
        elif op in io_ops:
            device = "IO"
        elif op in cpu_ops:
            device = "CPU"
        else:
            device = "PIM"
        mapping[node_id] = device
        by_device[device].append(node_id)
    return mapping, by_device


def merge_tiny_runs(topo, mapping, min_size):
    """
    Merge short same-device runs into a neighboring run.
    CPU runs are protected and will not be rewritten.
    """
    protected_devices = {"CPU"}
    runs = []
    start = 0
    while start < len(topo):
        device = mapping[topo[start]]
        end = start + 1
        while end < len(topo) and mapping[topo[end]] == device:
            end += 1
        runs.append((start, end))
        start = end

    for index, (start, end) in enumerate(runs):
        if (end - start) >= min_size:
            continue
        current = mapping[topo[start]]
        if current in protected_devices:
            continue
        prev = mapping[topo[runs[index - 1][0]]] if index > 0 else None
        next_dev = mapping[topo[runs[index + 1][0]]] if index + 1 < len(runs) else None
        target = None
        if prev and prev == next_dev:
            target = prev
        elif prev and prev != current:
            target = prev
        elif next_dev and next_dev != current:
            target = next_dev
        if target:
            for pos in range(start, end):
                mapping[topo[pos]] = target


def build_subgraphs(topo, mapping, successors):
    """
    Group adjacent same-device nodes into subgraphs and derive cross-subgraph dependencies.
    """
    groups = []
    current = []
    for node_id in topo:
        if not current:
            current = [node_id]
            continue
        if mapping[node_id] == mapping[current[-1]]:
            current.append(node_id)
        else:
            groups.append(current)
            current = [node_id]
    if current:
        groups.append(current)

    node_to_subgraph = {}
    plan = []
    for index, node_group in enumerate(groups):
        subgraph_id = f"SG{index}"
        for node_id in node_group:
            node_to_subgraph[node_id] = subgraph_id
        plan.append(
            {
                "id": subgraph_id,
                "device": mapping[node_group[0]],
                "nodes": node_group,
                "depends_on": [],
            }
        )

    dep_map = defaultdict(set)
    for src in topo:
        src_sg = node_to_subgraph[src]
        for dst in successors[src]:
            dst_sg = node_to_subgraph[dst]
            if src_sg != dst_sg:
                dep_map[dst_sg].add(src_sg)

    indexed_plan = {entry["id"]: entry for entry in plan}
    for subgraph_id, deps in dep_map.items():
        indexed_plan[subgraph_id]["depends_on"] = sorted(deps)

    return plan, node_to_subgraph


def estimate_subgraph_bytes_locality(
    subgraph,
    node_meta,
    tensor_info,
    output_to_node,
    tensor_consumers,
    graph_outputs,
    est_cfg,
    device="CPU",
    memory_profile=None,
    subgraph_io_scale=1.0,
):
    """
    Estimate subgraph memory traffic and summarize on-chip versus off-chip locality.
    """
    subgraph_nodes = set(subgraph["nodes"])
    external_in_tensors = set()
    external_out_tensors = set()
    reused_internal_tensors = set()

    for node_id in subgraph["nodes"]:
        meta = node_meta[node_id]
        for tensor_name in meta["activation_inputs"]:
            producer = output_to_node.get(tensor_name)
            if producer is None or producer not in subgraph_nodes:
                external_in_tensors.add(tensor_name)
            else:
                reused_internal_tensors.add(tensor_name)
        for tensor_name in meta["outputs"]:
            consumers = tensor_consumers.get(tensor_name, set())
            used_outside = any(consumer not in subgraph_nodes for consumer in consumers)
            if used_outside or (tensor_name in graph_outputs) or (not consumers):
                external_out_tensors.add(tensor_name)

    fusion_scale = max(0.05, float(subgraph_io_scale))
    
    def subgraph_tensor_bytes(name, role):
        """
        Return device-specific byte size for a tensor used inside this subgraph.
        """
        info = tensor_info.get(name)
        if info is None:
            return 0
        shape, dtype = info
        numel = int(safe_numel(shape))
        return get_tensor_bytes_for_device(numel, dtype, role, est_cfg, device)
    
    external_in_bytes_raw = int(sum(subgraph_tensor_bytes(name, "activation") for name in external_in_tensors))
    external_out_bytes_raw = int(sum(subgraph_tensor_bytes(name, "output") for name in external_out_tensors))
    external_in_bytes = int(math.ceil(external_in_bytes_raw * fusion_scale))
    external_out_bytes = int(math.ceil(external_out_bytes_raw * fusion_scale))
    reused_internal_bytes = int(sum(subgraph_tensor_bytes(name, "output") for name in reused_internal_tensors))

    mode = str(est_cfg.get("weight_stream_mode", "ratio")).strip().lower()
    if mode == "once_per_subgraph":
        unique_weight_tensors = set()
        for node_id in subgraph["nodes"]:
            for weight_name in node_meta[node_id]["weight_tensors"]:
                unique_weight_tensors.add(weight_name)
        streamed_weight_bytes = int(
            sum(
                estimate_streamed_weight_bytes(subgraph_tensor_bytes(weight_name, "weight"), est_cfg)
                for weight_name in unique_weight_tensors
            )
        )
    elif mode == "ignore":
        streamed_weight_bytes = 0
    else:
        streamed_weight_bytes = int(sum(node_meta[node_id]["streamed_weight_bytes"] for node_id in subgraph["nodes"]))

    constant_initializer_bytes = int(
        sum(node_meta[node_id]["constant_initializer_bytes"] for node_id in subgraph["nodes"])
    )

    raw_offchip_bytes = int(external_in_bytes + external_out_bytes + streamed_weight_bytes)
    dev_opt = pick_device_entry(memory_profile, device)
    if not dev_opt:
        dev_opt = pick_device_entry(est_cfg.get("device_memory_optimization", {}), device)

    offchip_reduction_ratio = clamp_ratio(dev_opt.get("offchip_reduction_ratio", 0.0))
    tile_reuse_ratio = clamp_ratio(dev_opt.get("tile_reuse_ratio", 0.0))
    cache_affinity_ratio = clamp_ratio(dev_opt.get("cache_affinity_ratio", 0.0))
    p2p_reuse_ratio = clamp_ratio(dev_opt.get("p2p_reuse_ratio", 0.0))
    combined_reduction = 1.0
    for ratio in [
        offchip_reduction_ratio,
        tile_reuse_ratio,
        cache_affinity_ratio,
        p2p_reuse_ratio,
    ]:
        combined_reduction *= 1.0 - ratio
    combined_reduction = max(0.0, min(1.0, 1.0 - combined_reduction))

    effective_offchip_bytes = int(math.ceil(raw_offchip_bytes * (1.0 - combined_reduction)))
    effective_offchip_bytes = int(max(0, min(raw_offchip_bytes, effective_offchip_bytes)))
    estimated_onchip_reuse_bytes = int(max(0, raw_offchip_bytes - effective_offchip_bytes))

    return int(effective_offchip_bytes), {
        "external_in_bytes": external_in_bytes,
        "external_out_bytes": external_out_bytes,
        "external_in_bytes_raw": external_in_bytes_raw,
        "external_out_bytes_raw": external_out_bytes_raw,
        "subgraph_io_scale": float(fusion_scale),
        "streamed_weight_bytes": streamed_weight_bytes,
        "reused_internal_bytes": reused_internal_bytes,
        "ignored_constant_initializer_bytes": constant_initializer_bytes,
        "raw_offchip_bytes": int(raw_offchip_bytes),
        "effective_offchip_bytes": int(effective_offchip_bytes),
        "estimated_onchip_reuse_bytes": int(estimated_onchip_reuse_bytes),
        "memory_profile_device": str(device),
        "memory_profile": {
            "offchip_reduction_ratio": float(offchip_reduction_ratio),
            "tile_reuse_ratio": float(tile_reuse_ratio),
            "cache_affinity_ratio": float(cache_affinity_ratio),
            "p2p_reuse_ratio": float(p2p_reuse_ratio),
            "combined_reduction_ratio": float(combined_reduction),
        },
    }


def compute_subgraph_critical_path(subgraph_plan):
    """
    Compute the critical path length for the subgraph plan in cycles.
    """
    if not subgraph_plan:
        return 0, {}
    cp_end = {}
    for subgraph in subgraph_plan:
        dep_ready = 0
        for dep in subgraph.get("depends_on", []):
            dep_ready = max(dep_ready, int(cp_end.get(dep, 0)))
        cp_end[subgraph["id"]] = int(dep_ready + int(subgraph.get("total_cycles", 0)))
    return int(max(cp_end.values()) if cp_end else 0), cp_end


def summarize_memory_flow(subgraph_plan):
    """
    Aggregate on-chip and off-chip memory traffic for the full subgraph plan.
    The result is reported both globally and per device.
    """
    by_device = {
        "CPU": {"onchip_bytes": 0, "offchip_bytes": 0, "onchip_cycles": 0, "offchip_cycles": 0},
        "PIM": {"onchip_bytes": 0, "offchip_bytes": 0, "onchip_cycles": 0, "offchip_cycles": 0},
        "IO": {"onchip_bytes": 0, "offchip_bytes": 0, "onchip_cycles": 0, "offchip_cycles": 0},
    }
    for subgraph in subgraph_plan:
        device = str(subgraph.get("device", "CPU")).upper()
        if device not in by_device:
            continue
        by_device[device]["onchip_bytes"] += int(max(0, subgraph.get("onchip_memory_bytes", 0)))
        by_device[device]["offchip_bytes"] += int(max(0, subgraph.get("offchip_memory_bytes", 0)))
        by_device[device]["onchip_cycles"] += int(max(0, subgraph.get("onchip_memory_cycles", 0)))
        by_device[device]["offchip_cycles"] += int(max(0, subgraph.get("offchip_memory_cycles", 0)))

    total_onchip_bytes = int(sum(value["onchip_bytes"] for value in by_device.values()))
    total_offchip_bytes = int(sum(value["offchip_bytes"] for value in by_device.values()))
    total_onchip_cycles = int(sum(value["onchip_cycles"] for value in by_device.values()))
    total_offchip_cycles = int(sum(value["offchip_cycles"] for value in by_device.values()))

    for device in ["CPU", "PIM", "IO"]:
        offchip_bytes = int(by_device[device]["offchip_bytes"])
        offchip_cycles = int(by_device[device]["offchip_cycles"])
        by_device[device]["onchip_offchip_byte_ratio"] = (
            float(by_device[device]["onchip_bytes"]) / float(offchip_bytes) if offchip_bytes > 0 else None
        )
        by_device[device]["onchip_offchip_cycle_ratio"] = (
            float(by_device[device]["onchip_cycles"]) / float(offchip_cycles) if offchip_cycles > 0 else None
        )

    return {
        "total_onchip_bytes": int(total_onchip_bytes),
        "total_offchip_bytes": int(total_offchip_bytes),
        "total_onchip_cycles": int(total_onchip_cycles),
        "total_offchip_cycles": int(total_offchip_cycles),
        "total_onchip_offchip_byte_ratio": (
            float(total_onchip_bytes) / float(total_offchip_bytes) if total_offchip_bytes > 0 else None
        ),
        "total_onchip_offchip_cycle_ratio": (
            float(total_onchip_cycles) / float(total_offchip_cycles) if total_offchip_cycles > 0 else None
        ),
        "by_device": by_device,
    }

