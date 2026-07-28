import math
from collections import defaultdict

from .performance import compose_compute_memory_cycles
from .pim_sram_tiling import scale_pim_tile_memory


def resolve_phase1_runtime_modes(cfg):
    """
    Resolve the phase1 runtime mode for each device.
    The default split-authority strategy uses real execution for CPU and IO,
    while PIM stays on the analytic model.
    """
    phase1_cfg = cfg.get("phase1_runtime", {})
    strategy = str(phase1_cfg.get("strategy", "split_authority")).strip().lower()
    device_modes = {"CPU": "analytic", "PIM": "analytic", "IO": "analytic"}
    if strategy == "split_authority":
        device_modes.update({"CPU": "real", "PIM": "analytic", "IO": "real"})

    overrides = phase1_cfg.get("devices", {})
    if isinstance(overrides, dict):
        for dev in ["CPU", "PIM", "IO"]:
            value = overrides.get(dev)
            if value is None:
                value = overrides.get(dev.lower())
            if value is None:
                continue
            mode = str(value).strip().lower()
            if mode in {"real", "analytic"}:
                device_modes[dev] = mode
    return strategy, device_modes


def estimate_pe_task_cycles(device, task_ops, task_bytes, pe_arch, cfg, task_memory_detail=None):
    """
    Estimate the cycle cost of one PE task.
    The return value breaks total cost into compute, memory, miss, and pipeline terms.
    """
    est = cfg["estimation"]
    memory_unit = int(est["memory_unit_bytes"])
    miss_penalty = int(est["cache_miss_penalty_cycles"])
    pipeline_cfg = est.get("pipeline_depth_cycles", {})
    if not isinstance(pipeline_cfg, dict):
        pipeline_cfg = {}

    if isinstance(task_ops, dict):
        workload = task_ops
        ops_total = int(workload.get("ops", 0))
    else:
        ops_total = int(task_ops)
        workload = {
            "ops": ops_total,
            "int8_ops": ops_total,
            "int32_ops": 0,
            "compute_kind": "generic",
        }

    if device == "PIM":
        pim_ops = pe_arch["ops_per_cycle"].get("PIM", {})
        int8_peak = float(pim_ops.get("int8", 400.0))
        int32_peak = float(pim_ops.get("int32", 100.0))
        
        int8_ops = int(workload.get("int8_ops", 0))
        int32_ops = int(workload.get("int32_ops", 0))
        compute_kind = str(workload.get("compute_kind", "generic"))
        
        if compute_kind == "fused_mac":
            int8_compute_cycles = int(math.ceil(int8_ops / max(1.0, int8_peak)))
            int32_compute_cycles = int(math.ceil(int32_ops / max(1.0, int32_peak))) if int32_ops > 0 else 0
            compute_cycles = max(int8_compute_cycles, int32_compute_cycles)
        elif compute_kind == "int32_only":
            compute_cycles = int(math.ceil(int32_ops / max(1.0, int32_peak))) if int32_ops > 0 else 0
        elif compute_kind == "int8_only":
            compute_cycles = int(math.ceil(int8_ops / max(1.0, int8_peak))) if int8_ops > 0 else 0
        else:
            compute_cycles = int(math.ceil(ops_total / max(1.0, int8_peak)))
        
        detail = task_memory_detail if isinstance(task_memory_detail, dict) else {}
        if detail:
            mem_cycles = int(max(0, detail.get("per_use_memory_cycles", 0)))
            if mem_cycles <= 0:
                offchip_bytes = int(max(0, detail.get("per_use_offchip_bytes", 0)))
                onchip_bytes = int(max(0, detail.get("per_use_onchip_bytes", 0)))
                ddr_bw = float(pe_arch["mem_bw_per_pe"]["PIM"])
                sram_bw = float(est.get("pim_sram_bandwidth_bytes_per_cycle", 64))
                ddr_setup = int(est.get("pim_ddr_setup_cycles", 20)) if offchip_bytes > 0 else 0
                sram_setup = int(est.get("pim_sram_setup_cycles", 4)) if onchip_bytes > 0 else 0
                wb_setup = int(est.get("pim_writeback_setup_cycles", 8)) if offchip_bytes > 0 else 0
                mem_cycles = (
                    int(math.ceil(offchip_bytes / max(1.0, ddr_bw)))
                    + ddr_setup
                    + int(math.ceil(onchip_bytes / max(1.0, sram_bw)))
                    + sram_setup
                    + wb_setup
                )
        else:
            ddr_bw = float(pe_arch["mem_bw_per_pe"]["PIM"])
            sram_bw = float(est.get("pim_sram_bandwidth_bytes_per_cycle", 64))
            ddr_setup = int(est.get("pim_ddr_setup_cycles", 20))
            sram_setup = int(est.get("pim_sram_setup_cycles", 4))
            wb_setup = int(est.get("pim_writeback_setup_cycles", 8))
            mem_cycles = (
                int(math.ceil(task_bytes / max(1.0, ddr_bw)))
                + ddr_setup
                + int(math.ceil(task_bytes / max(1.0, sram_bw)))
                + sram_setup
                + int(math.ceil(task_bytes / max(1.0, sram_bw)))
                + wb_setup
            )
    else:
        pe_ops = float(pe_arch["ops_per_cycle"][device])
        compute_cycles = int(math.ceil(ops_total / max(1.0, pe_ops)))
        mem_bw = float(pe_arch["mem_bw_per_pe"][device])
        mem_cycles = int(math.ceil(task_bytes / max(1.0, mem_bw)))

    cache_bytes = int(pe_arch["cache_bytes_per_pe"][device])
    if device == "PIM" and isinstance(task_memory_detail, dict):
        overflow_ref = int(max(0, task_memory_detail.get("peak_sram_bytes", task_bytes)))
    else:
        overflow_ref = int(task_bytes)
    overflow = max(0, overflow_ref - cache_bytes)
    miss_count = int(math.ceil(overflow / max(1, memory_unit)))
    miss_cycles = int(miss_count * miss_penalty)
    pipeline_cycles = int(max(0, pipeline_cfg.get(device, 0)))

    core_cycles, compose_mode = compose_compute_memory_cycles(
        compute_cycles, mem_cycles, est, key_prefix="pe_"
    )
    total_cycles = max(1, core_cycles + miss_cycles + pipeline_cycles)
    return {
        "compute_cycles": int(compute_cycles),
        "int8_ops": int(workload.get("int8_ops", 0)),
        "int32_ops": int(workload.get("int32_ops", 0)),
        "compute_kind": str(workload.get("compute_kind", "generic")),
        "memory_cycles": int(mem_cycles),
        "compute_memory_core_cycles": int(core_cycles),
        "compute_memory_compose_mode": compose_mode,
        "cache_miss_count": int(miss_count),
        "cache_miss_cycles": int(miss_cycles),
        "pipeline_cycles": int(pipeline_cycles),
        "total_cycles": int(total_cycles),
    }


def build_ir_pe_task_plan(topo, predecessors, node_to_device, node_stats, cfg, dev_ops, pim_node_offset):
    """
    Split each node into per-PE tasks and schedule them on CPU, PIM, or IO workers.
    Returns the task list, a node-to-task map, and derived PE architecture metadata.
    """
    est = cfg["estimation"]
    target_tile_cycles = int(est.get("ir_target_tile_cycles", 256))
    io_bw_total = cfg["io"]["memory_channels"] * cfg["io"]["channel_bandwidth_bytes_per_cycle"]
    pim_ddr_bw = float(est.get("pim_ddr_bandwidth_bytes_per_cycle", io_bw_total))

    pe_count = {
        "CPU": int(max(1, cfg["cpu"].get("cores", cfg["cpu"].get("pe_count", 1)))),
        "IO": int(max(1, cfg["io"].get("cores", 1))),
        "PIM": int(max(1, cfg["pim"].get("npu_count", 16))),
    }
    pe_arch = {
        "pe_count": pe_count,
        "ops_per_cycle": {
            "CPU": float(dev_ops["CPU"]) / pe_count["CPU"],
            "IO": float(dev_ops["IO"]) / pe_count["IO"],
            "PIM": {
                "int8": float(dev_ops["PIM"]["int8"]) / pe_count["PIM"],
                "int32": float(dev_ops["PIM"]["int32"]) / pe_count["PIM"],
            },
        },
        "cache_bytes_per_pe": {
            "CPU": int(cfg["cpu"].get("l2_bytes", cfg["cpu"].get("l1_bytes", 131072))),
            "IO": int(
                cfg["io"].get("local_caches", {}).get("l2_bytes", cfg["io"].get("system_llc_bytes", 131072))
            ),
            "PIM": int(cfg["pim"].get("sram_per_npu_bytes", 131072)),
        },
        "mem_bw_per_pe": {
            "CPU": float(io_bw_total) / pe_count["CPU"],
            "IO": float(io_bw_total) / pe_count["IO"],
            "PIM": float(pim_ddr_bw) / pe_count["PIM"],
        },
        "router_map": {"CPU": 0, "PIM": 1, "IO": 2},
        "pim_node_offset": int(pim_node_offset),
    }

    def choose_split(dev, ops, bytes_total):
        n_pe = pe_count[dev]
        if n_pe <= 1:
            return 1
        ops_per_cycle = pe_arch["ops_per_cycle"][dev]
        if isinstance(ops_per_cycle, dict):
            ops_per_cycle = ops_per_cycle.get("int8", 400.0)
        ops_tile = int(
            math.ceil(
                ops
                / max(
                    1.0,
                    float(ops_per_cycle) * max(1, target_tile_cycles),
                )
            )
        )
        cache_tile = int(
            math.ceil(bytes_total / max(1.0, float(pe_arch["cache_bytes_per_pe"][dev]) * 2.0))
        )
        return int(min(n_pe, max(1, ops_tile, cache_tile)))

    pe_ready = {dev: [0 for _ in range(pe_count[dev])] for dev in ["CPU", "PIM", "IO"]}
    node_finish = {}
    node_task_map = {}
    task_plan = []
    task_index = 0

    for node_id in topo:  # Split each node into PE tasks.
        stat = node_stats[node_id]
        dev = node_to_device[node_id]
        total_ops = int(max(1, stat["ops"]))
        total_int8_ops = int(max(0, stat.get("int8_ops", total_ops)))
        total_int32_ops = int(max(0, stat.get("int32_ops", 0)))
        compute_kind = str(stat.get("compute_kind", "generic"))
        total_bytes = int(max(1, stat["bytes"]))
        analytic_total_bytes = int(max(1, stat.get("analytic_total_memory_bytes", total_bytes))) if dev == "PIM" else total_bytes
        # Choose how many PEs should share this node.
        split = choose_split(dev, total_ops, analytic_total_bytes)

        dep_ready = 0
        for prev in predecessors.get(node_id, []):
            dep_ready = max(dep_ready, int(node_finish.get(prev, 0)))

        ops_base = total_ops // split
        ops_rem = total_ops % split
        int8_ops_base = total_int8_ops // split
        int8_ops_rem = total_int8_ops % split
        int32_ops_base = total_int32_ops // split
        int32_ops_rem = total_int32_ops % split
        bytes_base = analytic_total_bytes // split
        bytes_rem = analytic_total_bytes % split
        node_memory_detail = stat.get("pim_sram_tiling") if dev == "PIM" else None

        this_node_tasks = []
        for idx in range(split):
            pe_id = idx % pe_count[dev]
            task_ops = int(ops_base + (1 if idx < ops_rem else 0))
            task_int8_ops = int(int8_ops_base + (1 if idx < int8_ops_rem else 0))
            task_int32_ops = int(int32_ops_base + (1 if idx < int32_ops_rem else 0))
            task_bytes = int(bytes_base + (1 if idx < bytes_rem else 0))
            task_memory_detail = None
            if dev == "PIM" and isinstance(node_memory_detail, dict):
                ratio = float(task_bytes) / float(max(1, analytic_total_bytes))
                task_memory_detail = scale_pim_tile_memory(node_memory_detail, ratio)
            task_workload = {
                "ops": task_ops,
                "int8_ops": task_int8_ops,
                "int32_ops": task_int32_ops,
                "compute_kind": compute_kind,
            }
            cycles = estimate_pe_task_cycles(
                dev,
                task_workload,
                task_bytes,
                pe_arch,
                cfg,
                task_memory_detail=task_memory_detail,
            )
            start = int(max(dep_ready, pe_ready[dev][pe_id]))
            end = int(start + cycles["total_cycles"])
            pe_ready[dev][pe_id] = end

            task = {
                "task_id": f"T{task_index}",
                "node_id": node_id,
                "device": dev,
                "pe_id": int(pe_id),
                "router_id": int(pe_arch["router_map"][dev]),
                "pim_router_id": int(pe_arch["pim_node_offset"] + pe_id) if dev == "PIM" else None,
                "task_ops": task_ops,
                "task_int8_ops": task_int8_ops,
                "task_int32_ops": task_int32_ops,
                "task_compute_kind": compute_kind,
                "task_bytes": task_bytes,
                "task_memory_detail": task_memory_detail,
                "start_cycle": start,
                "end_cycle": end,
                **cycles,
            }
            task_index += 1
            task_plan.append(task)
            this_node_tasks.append(task)

        node_task_map[node_id] = this_node_tasks
        node_finish[node_id] = max(task["end_cycle"] for task in this_node_tasks) if this_node_tasks else dep_ready

    return task_plan, node_task_map, pe_arch


def split_by_weights(total, weights):
    """
    Split an integer total across buckets according to the provided weights.
    """
    if total <= 0 or not weights:
        return [0 for _ in weights]
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        base = total // len(weights)
        rem = total % len(weights)
        return [base + (1 if idx < rem else 0) for idx in range(len(weights))]
    raw = [float(total) * float(weight) / weight_sum for weight in weights]
    out = [int(math.floor(value)) for value in raw]
    rem = int(total - sum(out))
    frac_idx = sorted(range(len(raw)), key=lambda idx: (raw[idx] - out[idx]), reverse=True)
    for idx in range(rem):
        out[frac_idx[idx % len(frac_idx)]] += 1
    return out


def build_interchiplet_bench(
    topo,
    predecessors,
    node_to_device,
    node_stats,
    node_task_map,
    flit_payload,
    phase1_strategy="split_authority",
    phase1_device_modes=None,
):
    """
    Build the inter-chiplet PopNet bench trace.
    The generated lines describe CPU, PIM, and IO traffic at chiplet granularity.
    """
    dev_to_node = {"CPU": 0, "PIM": 1, "IO": 2}
    if not isinstance(phase1_device_modes, dict):
        phase1_device_modes = {"CPU": "analytic", "PIM": "analytic", "IO": "analytic"}
    topo_index = {node_id: idx for idx, node_id in enumerate(topo)}

    has_bootstrap_edge = False
    has_cpu_to_io_edge = False
    first_pim_node = next((node_id for node_id in topo if node_to_device.get(node_id) == "PIM"), None)
    for dst in topo:
        for src in predecessors.get(dst, []):
            if node_to_device.get(src) == "CPU" and node_to_device.get(dst) == "PIM":
                has_bootstrap_edge = True
            if node_to_device.get(src) == "CPU" and node_to_device.get(dst) == "IO":
                has_cpu_to_io_edge = True

    bench_lines = []
    if not has_bootstrap_edge:
        bootstrap_payload = max(64, int(node_stats.get(first_pim_node, {}).get("bytes", 64)))
        bootstrap_flits = int(math.ceil(bootstrap_payload / max(1, flit_payload)) + 1)
        bench_lines.append((0, 0, dev_to_node["CPU"], dev_to_node["PIM"], bootstrap_flits, 0))

    for dst in topo:
        for src in predecessors.get(dst, []):
            src_dev = node_to_device[src]
            dst_dev = node_to_device[dst]
            if src_dev == dst_dev:
                continue
            src_tasks = node_task_map.get(src, [])
            if phase1_strategy == "split_authority" and phase1_device_modes.get(src_dev) == "real":
                send_time = 0
            else:
                if not src_tasks:
                    continue
                send_time = int(max(task["end_cycle"] for task in src_tasks))
            task_count = max(1, len(src_tasks))
            payload = max(64, int(math.ceil(node_stats[src]["bytes"] / float(task_count))))
            flits = int(math.ceil(payload / max(1, flit_payload)) + 1)
            bench_lines.append((send_time, send_time, dev_to_node[src_dev], dev_to_node[dst_dev], flits, 0))

    pim_task_end = 0
    final_pim_node = None
    for node_id, tasks in node_task_map.items():
        if tasks and node_to_device[node_id] == "PIM":
            pim_task_end = max(pim_task_end, max(task["end_cycle"] for task in tasks))
            if final_pim_node is None or topo_index.get(node_id, -1) > topo_index.get(final_pim_node, -1):
                final_pim_node = node_id
    final_cycle = int(max(1, pim_task_end) + 1)
    final_payload = max(64, int(node_stats.get(final_pim_node, {}).get("bytes", 64)))
    final_flits = int(math.ceil(final_payload / max(1, flit_payload)) + 1)
    bench_lines.append((final_cycle, final_cycle, dev_to_node["PIM"], dev_to_node["CPU"], final_flits, 0))

    if not has_cpu_to_io_edge:
        cpu_task_end = 0
        cpu_payload = 64
        for node_id, tasks in node_task_map.items():
            if not tasks or node_to_device.get(node_id) != "CPU":
                continue
            cpu_task_end = max(cpu_task_end, max(task["end_cycle"] for task in tasks))
            cpu_payload = max(cpu_payload, int(node_stats.get(node_id, {}).get("bytes", 64)))
        if cpu_task_end > 0:
            flits = int(math.ceil(cpu_payload / max(1, flit_payload)) + 1)
            send_time = int(cpu_task_end + 1)
            bench_lines.append((send_time, send_time, dev_to_node["CPU"], dev_to_node["IO"], flits, 0))
        elif final_pim_node is not None:
            cpu_payload = max(cpu_payload, int(node_stats.get(final_pim_node, {}).get("bytes", 64)))
            flits = int(math.ceil(cpu_payload / max(1, flit_payload)) + 1)
            send_time = int(final_cycle + 1)
            bench_lines.append((send_time, send_time, dev_to_node["CPU"], dev_to_node["IO"], flits, 0))
    return bench_lines


def build_phase1_runtime_plan(
    topo,
    successors,
    node_to_device,
    node_to_sg,
    node_stats,
    subgraph_plan,
    phase1_strategy="split_authority",
    phase1_device_modes=None,
):
    """
    Build phase1 runtime events for CPU, PIM, and IO.
    Events are emitted as RECV, COMPUTE, and SEND records in execution order.
    """
    dev_to_coord = {"CPU": (0, 0), "PIM": (0, 1), "IO": (1, 0)}
    if not isinstance(phase1_device_modes, dict):
        phase1_device_modes = {"CPU": "analytic", "PIM": "analytic", "IO": "analytic"}
    id_to_sg = {subgraph["id"]: subgraph for subgraph in subgraph_plan}
    topo_index = {node_id: idx for idx, node_id in enumerate(topo)}
    sg_successors = defaultdict(set)
    sg_edge_bytes = defaultdict(int)
    sg_edge_nodes = defaultdict(list)

    for subgraph in subgraph_plan:
        for dep in subgraph.get("depends_on", []):
            sg_successors[dep].add(subgraph["id"])

    for src in topo:
        src_sg = node_to_sg[src]
        src_dev = node_to_device[src]
        payload = max(64, int(node_stats[src]["bytes"]))
        for dst in successors.get(src, []):
            dst_sg = node_to_sg[dst]
            dst_dev = node_to_device[dst]
            if src_sg == dst_sg or src_dev == dst_dev:
                continue
            edge_key = (src_sg, dst_sg)
            sg_edge_bytes[edge_key] += payload
            sg_edge_nodes[edge_key].append((src, dst, payload))

    events = {"CPU": [], "PIM": [], "IO": []}
    for subgraph in subgraph_plan:
        subgraph_id = subgraph["id"]
        dev = subgraph["device"]
        for dep in sorted(subgraph.get("depends_on", [])):
            dep_dev = id_to_sg[dep]["device"]
            if dep_dev == dev:
                continue
            bytes_total = int(max(64, sg_edge_bytes.get((dep, subgraph_id), 64)))
            events[dev].append(
                {
                    "type": "RECV",
                    "peer_dev": dep_dev,
                    "peer_xy": dev_to_coord[dep_dev],
                    "bytes": bytes_total,
                    "subgraph_id": subgraph_id,
                    "edge_from": dep,
                }
            )

        events[dev].append(
            {
                "type": "COMPUTE",
                "cycles": int(max(1, subgraph.get("total_cycles", 1))),
                "ops": int(max(0, subgraph.get("estimated_ops", 0))),
                "bytes": int(max(0, subgraph.get("estimated_bytes", 0))),
                "subgraph_id": subgraph_id,
                "start_cycle": int(subgraph.get("start_cycle", 0)),
                "end_cycle": int(subgraph.get("end_cycle", 0)),
            }
        )

        for succ in sorted(sg_successors.get(subgraph_id, [])):
            succ_dev = id_to_sg[succ]["device"]
            if succ_dev == dev:
                continue
            bytes_total = int(max(64, sg_edge_bytes.get((subgraph_id, succ), 64)))
            events[dev].append(
                {
                    "type": "SEND",
                    "peer_dev": succ_dev,
                    "peer_xy": dev_to_coord[succ_dev],
                    "bytes": bytes_total,
                    "subgraph_id": subgraph_id,
                    "edge_to": succ,
                }
            )

    has_cpu_to_io_send = any(
        str(event.get("type", "")).upper() == "SEND" and str(event.get("peer_dev", "")).upper() == "IO"
        for event in events["CPU"]
    )
    has_pim_to_cpu_send = any(
        str(event.get("type", "")).upper() == "SEND" and str(event.get("peer_dev", "")).upper() == "CPU"
        for event in events["PIM"]
    )
    has_cpu_to_pim_send = any(
        str(event.get("type", "")).upper() == "SEND" and str(event.get("peer_dev", "")).upper() == "PIM"
        for event in events["CPU"]
    )
    first_pim_subgraph = None
    final_pim_subgraph = None
    has_cpu_model_subgraph = False
    for subgraph in subgraph_plan:
        dev = str(subgraph.get("device", "")).upper()
        if dev == "CPU":
            has_cpu_model_subgraph = True
        if dev != "PIM":
            continue
        if first_pim_subgraph is None:
            first_pim_subgraph = subgraph
        if final_pim_subgraph is None:
            final_pim_subgraph = subgraph
            continue
        last_node = subgraph.get("nodes", [])[-1] if subgraph.get("nodes") else ""
        final_node = final_pim_subgraph.get("nodes", [])[-1] if final_pim_subgraph.get("nodes") else ""
        if topo_index.get(last_node, -1) > topo_index.get(final_node, -1):
            final_pim_subgraph = subgraph
    if not has_cpu_to_pim_send and first_pim_subgraph is not None:
        bootstrap_bytes = int(
            max(
                64,
                first_pim_subgraph.get(
                    "estimated_bytes",
                    first_pim_subgraph.get("estimated_total_memory_bytes", 64),
                ),
            )
        )
        bootstrap_id = str(first_pim_subgraph["id"])
        events["CPU"].insert(
            0,
            {
                "type": "SEND",
                "peer_dev": "PIM",
                "peer_xy": dev_to_coord["PIM"],
                "bytes": bootstrap_bytes,
                "subgraph_id": bootstrap_id,
                "edge_to": bootstrap_id,
            },
        )
        events["PIM"].insert(
            0,
            {
                "type": "RECV",
                "peer_dev": "CPU",
                "peer_xy": dev_to_coord["CPU"],
                "bytes": bootstrap_bytes,
                "subgraph_id": bootstrap_id,
                "edge_from": "CPU_LOADER",
            },
        )
    handoff_bytes = None
    handoff_id = None
    if final_pim_subgraph is not None:
        handoff_bytes = int(
            max(
                64,
                final_pim_subgraph.get(
                    "estimated_bytes",
                    final_pim_subgraph.get("estimated_total_memory_bytes", 64),
                ),
            )
        )
        handoff_id = str(final_pim_subgraph["id"])
    if not has_pim_to_cpu_send and handoff_bytes is not None and handoff_id is not None:
        events["PIM"].append(
            {
                "type": "SEND",
                "peer_dev": "CPU",
                "peer_xy": dev_to_coord["CPU"],
                "bytes": handoff_bytes,
                "subgraph_id": handoff_id,
                "edge_to": "CPU_RETURN",
            }
        )
        events["CPU"].append(
            {
                "type": "RECV",
                "peer_dev": "PIM",
                "peer_xy": dev_to_coord["PIM"],
                "bytes": handoff_bytes,
                "subgraph_id": handoff_id,
                "edge_from": handoff_id,
            }
        )
    if not has_cpu_to_io_send:
        for subgraph in subgraph_plan:
            if str(subgraph.get("device", "")).upper() != "CPU":
                continue
            subgraph_id = subgraph["id"]
            succs = sg_successors.get(subgraph_id, set())
            has_cross_device_succ = any(id_to_sg[succ]["device"] != "CPU" for succ in succs if succ in id_to_sg)
            if has_cross_device_succ:
                continue
            bytes_total = int(max(64, subgraph.get("estimated_bytes", subgraph.get("estimated_total_memory_bytes", 64))))
            events["CPU"].append(
                {
                    "type": "SEND",
                    "peer_dev": "IO",
                    "peer_xy": dev_to_coord["IO"],
                    "bytes": bytes_total,
                    "subgraph_id": subgraph_id,
                    "edge_to": "IO_SINK",
                }
            )
            events["IO"].append(
                {
                    "type": "RECV",
                    "peer_dev": "CPU",
                    "peer_xy": dev_to_coord["CPU"],
                    "bytes": bytes_total,
                    "subgraph_id": subgraph_id,
                    "edge_from": subgraph_id,
                }
            )
        if handoff_bytes is not None and handoff_id is not None and not has_cpu_model_subgraph:
            events["CPU"].append(
                {
                    "type": "SEND",
                    "peer_dev": "IO",
                    "peer_xy": dev_to_coord["IO"],
                    "bytes": handoff_bytes,
                    "subgraph_id": handoff_id,
                    "edge_to": "IO_SINK",
                }
            )
            events["IO"].append(
                {
                    "type": "RECV",
                    "peer_dev": "CPU",
                    "peer_xy": dev_to_coord["CPU"],
                    "bytes": handoff_bytes,
                    "subgraph_id": handoff_id,
                    "edge_from": handoff_id,
                }
            )

    for dev in ["CPU", "PIM", "IO"]:
        if not events[dev]:
            events[dev].append({"type": "COMPUTE", "cycles": 1, "subgraph_id": "IDLE"})
            continue
        if phase1_strategy == "split_authority" and phase1_device_modes.get(dev) == "real":
            for event in events[dev]:
                if str(event.get("type", "")).upper() != "COMPUTE":
                    continue
                event["cycles"] = 1
                event["ops"] = 0
                event["bytes"] = 0
                event["phase1_timing"] = "gem5_real_exec"

    return {
        "device_coords": dev_to_coord,
        "events": events,
        "sg_edge_bytes": {f"{src}->{dst}": int(value) for (src, dst), value in sorted(sg_edge_bytes.items())},
        "sg_edge_nodes": {f"{src}->{dst}": value for (src, dst), value in sorted(sg_edge_nodes.items())},
    }


def build_pim_intra_bench(
    topo,
    node_to_device,
    node_stats,
    node_task_map,
    flit_payload,
    npu_count,
    node_offset,
    cfg=None,
):
    """
    Build the PIM intra-chiplet PopNet bench trace.
    The model emits controller fanout traffic, column descent traffic,
    writeback traffic, and optional merge traffic for the 4x4 NPU array.
    """
    if npu_count <= 0:
        npu_count = 16
    cols = int(math.sqrt(npu_count))
    rows = npu_count // cols if cols > 0 else 0
    if rows * cols != npu_count or rows == 0:
        cols = 4
        rows = max(1, npu_count // cols)

    pim_cfg = cfg.get("pim", {}) if isinstance(cfg, dict) else {}
    ingress_cfg = pim_cfg.get("ingress_model", {}) if isinstance(pim_cfg, dict) else {}
    ddr_attached_npu = int(max(0, min(npu_count - 1, ingress_cfg.get("ddr_attached_npu", 0))))
    gateway_router_id = int(ingress_cfg.get("gateway_router_id", 1))
    top_row_targets = ingress_cfg.get("top_row_targets", list(range(min(cols, npu_count))))
    if not isinstance(top_row_targets, list):
        top_row_targets = list(range(min(cols, npu_count)))
    top_row_targets = [
        int(target)
        for target in top_row_targets
        if isinstance(target, (int, float)) and 0 <= int(target) < npu_count
    ]
    if not top_row_targets:
        top_row_targets = list(range(min(cols, npu_count)))
    merge_sources = ingress_cfg.get("controller_merge_sources", top_row_targets)
    if not isinstance(merge_sources, list):
        merge_sources = list(top_row_targets)
    merge_sources = [
        int(source)
        for source in merge_sources
        if isinstance(source, (int, float)) and 0 <= int(source) < npu_count
    ]
    if not merge_sources:
        merge_sources = list(top_row_targets)
    emit_gateway_transactions = bool(ingress_cfg.get("emit_gateway_transactions_in_bench", True))
    data_path = ingress_cfg.get("data_path", ["DDR", "local_SRAM", "in_memory_compute_unit"])
    if not isinstance(data_path, list):
        data_path = ["DDR", "local_SRAM", "in_memory_compute_unit"]

    column_paths = []
    reverse_column_paths = []
    for col in range(cols):
        down_path = [row * cols + col for row in range(rows)]
        column_paths.append(down_path)
        reverse_column_paths.append(list(reversed(down_path)))

    bench_lines = []
    mapping = {
        "npu_count": int(npu_count),
        "pim_node_offset": int(node_offset),
        "grid": {"rows": int(rows), "cols": int(cols)},
        "ingress_model": {
            "mode": str(ingress_cfg.get("mode", "single_ddr_entry_controller_fanout")),
            "ddr_attached_npu": int(ddr_attached_npu),
            "ddr_attached_global_router_id": int(node_offset + ddr_attached_npu),
            "gateway_router_id": int(gateway_router_id),
            "top_row_targets": [int(target) for target in top_row_targets],
            "top_row_global_router_ids": [int(node_offset + target) for target in top_row_targets],
            "controller_merge_sources": [int(source) for source in merge_sources],
            "controller_merge_global_router_ids": [int(node_offset + source) for source in merge_sources],
            "emit_gateway_transactions_in_bench": bool(emit_gateway_transactions),
            "data_path": [str(item) for item in data_path],
        },
        "column_paths": column_paths,
        "reverse_column_paths": reverse_column_paths,
        "subgraph_mapping": [],
    }

    pim_nodes = [node_id for node_id in topo if node_to_device.get(node_id) == "PIM" and node_task_map.get(node_id)]
    for node_id in pim_nodes:
        tasks = node_task_map[node_id]
        pe_to_task = {}
        for task in tasks:
            pe = int(task["pe_id"])
            if 0 <= pe < npu_count:
                pe_to_task[pe] = task
        if not pe_to_task:
            continue

        node_start = int(min(task["start_cycle"] for task in tasks))
        node_end = int(max(task["end_cycle"] for task in tasks))
        bytes_total = int(
            max(
                64,
                node_stats[node_id].get("analytic_total_memory_bytes", node_stats[node_id]["bytes"]),
            )
        )
        vertical_edges = max(1, cols * max(0, rows - 1))
        payload = max(64, int(math.ceil(bytes_total / float(vertical_edges))))
        flits = int(math.ceil(payload / max(1, flit_payload)) + 1)
        link_gap = max(1, int(math.ceil(flits / 2.0)))
        link_ready = defaultdict(int)

        fanout_payload = max(64, int(math.ceil(bytes_total / float(max(1, len(top_row_targets))))))
        fanout_flits = int(math.ceil(fanout_payload / max(1, flit_payload)) + 1)
        gateway_ready = int(node_start)
        gateway_gap = max(1, int(math.ceil(fanout_flits / 2.0)))

        active_top_targets = [
            int(target)
            for target in top_row_targets
            if target in pe_to_task
        ]
        if not active_top_targets:
            active_top_targets = sorted(
                set(
                    int(pe)
                    for pe in pe_to_task.keys()
                    if 0 <= int(pe) < min(cols, npu_count)
                )
            )

        active_merge_sources = [
            int(source)
            for source in merge_sources
            if source in pe_to_task
        ]
        if not active_merge_sources:
            active_merge_sources = list(active_top_targets)

        if emit_gateway_transactions:
            for target in active_top_targets:
                send_time = int(max(node_start, gateway_ready))
                bench_lines.append(
                    (
                        send_time,
                        send_time,
                        int(gateway_router_id),
                        int(node_offset + target),
                        int(fanout_flits),
                        0,
                    )
                )
                gateway_ready = send_time + gateway_gap

        for col in range(cols):
            for row in range(rows - 1):
                src_pe = row * cols + col
                dst_pe = (row + 1) * cols + col
                src_task = pe_to_task.get(src_pe)
                dst_task = pe_to_task.get(dst_pe)
                if src_task is None or dst_task is None:
                    continue
                link = (src_pe, dst_pe)
                send_time = int(max(node_start, src_task["end_cycle"], link_ready[link]))
                bench_lines.append((send_time, send_time, int(node_offset + src_pe), int(node_offset + dst_pe), int(flits), 0))
                link_ready[link] = send_time + link_gap

        wb_base = int(node_end)
        for col in range(cols):
            for row in range(rows - 1, 0, -1):
                src_pe = row * cols + col
                dst_pe = (row - 1) * cols + col
                src_task = pe_to_task.get(src_pe)
                dst_task = pe_to_task.get(dst_pe)
                if src_task is None or dst_task is None:
                    continue
                link = (src_pe, dst_pe)
                send_time = int(max(wb_base, src_task["end_cycle"], link_ready[link]))
                bench_lines.append((send_time, send_time, int(node_offset + src_pe), int(node_offset + dst_pe), int(flits), 0))
                link_ready[link] = send_time + link_gap

        if emit_gateway_transactions:
            merge_start = int(max(wb_base, gateway_ready))
            for source in active_merge_sources:
                source_task = pe_to_task.get(source)
                if source_task is None:
                    continue
                send_time = int(max(merge_start, source_task["end_cycle"], gateway_ready))
                bench_lines.append(
                    (
                        send_time,
                        send_time,
                        int(node_offset + source),
                        int(gateway_router_id),
                        int(fanout_flits),
                        0,
                    )
                )
                gateway_ready = send_time + gateway_gap

        mapping["subgraph_mapping"].append(
            {
                "subgraph_id": node_id,
                "start_cycle": int(node_start),
                "end_cycle": int(node_end),
                "estimated_bytes": int(bytes_total),
                "estimated_ops": int(node_stats[node_id]["ops"]),
                "logical_flow": {
                    "ingress": {
                        "ddr_attached_npu": int(ddr_attached_npu),
                        "gateway_router_id": int(gateway_router_id),
                        "top_row_targets": [int(target) for target in top_row_targets],
                        "active_top_row_targets": [int(target) for target in active_top_targets],
                        "data_path": [str(item) for item in data_path],
                        "fanout_payload_bytes": int(fanout_payload),
                        "fanout_flits": int(fanout_flits),
                        "transactions_emitted": bool(emit_gateway_transactions),
                    },
                    "column_descent": [
                        {
                            "column": int(col),
                            "path": [int(pe) for pe in column_paths[col]],
                            "global_router_path": [int(node_offset + pe) for pe in column_paths[col]],
                        }
                        for col in range(cols)
                    ],
                    "column_ascent": [
                        {
                            "column": int(col),
                            "path": [int(pe) for pe in reverse_column_paths[col]],
                            "global_router_path": [int(node_offset + pe) for pe in reverse_column_paths[col]],
                        }
                        for col in range(cols)
                    ],
                    "merge": {
                        "controller_merge_sources": [int(source) for source in merge_sources],
                        "active_controller_merge_sources": [int(source) for source in active_merge_sources],
                        "controller_merge_global_router_ids": [int(node_offset + source) for source in merge_sources],
                        "gateway_router_id": int(gateway_router_id),
                    },
                },
                "npu_slices": [
                    {
                        "npu_id": int(task["pe_id"]),
                        "global_router_id": int(node_offset + int(task["pe_id"])),
                        "start_cycle": int(task["start_cycle"]),
                        "end_cycle": int(task["end_cycle"]),
                        "assigned_bytes": int(task["task_bytes"]),
                        "task_memory_detail": task.get("task_memory_detail"),
                        "assigned_ops": int(task["task_ops"]),
                    }
                    for task in tasks
                ],
            }
        )

    if not bench_lines:
        bench_lines.append((0, 0, 0, 0, 2, 0))

    return bench_lines, mapping

