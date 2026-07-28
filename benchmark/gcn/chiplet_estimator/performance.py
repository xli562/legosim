"""
Performance estimation helpers.
"""
import math

from .common import merge_numeric_dict


def level_access_cycles(bytes_moved, bandwidth_bytes_per_cycle, access_latency_cycles, memory_unit):
    """
    Estimate cycles spent at one memory level.
    The total is transfer time plus per-access latency.
    """
    if bytes_moved <= 0:
        return 0, 0
    transfer_cycles = int(math.ceil(bytes_moved / max(1.0, bandwidth_bytes_per_cycle)))
    accesses = int(math.ceil(bytes_moved / max(1, memory_unit)))
    latency_cycles = int(accesses * max(0.0, access_latency_cycles))
    return transfer_cycles + latency_cycles, accesses


def estimate_cache_hierarchy_cycles(bytes_total, levels, memory_unit, miss_penalty):
    """
    Estimate memory cycles for a cache hierarchy by walking levels from L1 to MEM.
    Returns total cycles, miss count, miss cycles, and a per-level breakdown.
    """
    remaining = int(max(0, bytes_total))
    memory_cycles = 0
    miss_count = 0
    breakdown = []
    for level in levels:
        name = level["name"]
        capacity = level.get("capacity_bytes")
        hit_bytes = remaining if capacity is None else min(remaining, int(max(0, capacity)))
        cycles, accesses = level_access_cycles(
            hit_bytes,
            level.get("bandwidth_bytes_per_cycle", 1),
            level.get("access_latency_cycles", 0),
            memory_unit,
        )
        memory_cycles += cycles
        breakdown.append(
            {
                "level": name,
                "bytes": int(hit_bytes),
                "cycles": int(cycles),
                "accesses": int(accesses),
            }
        )
        if name == "MEM":
            miss_count = accesses
        remaining -= hit_bytes
        if remaining <= 0:
            break

    miss_cycles = int(miss_count * miss_penalty)
    return int(memory_cycles), int(miss_count), int(miss_cycles), breakdown


def estimate_pim_memory_cycles(bytes_total, cfg, io_bw, memory_unit, miss_penalty):
    """
    Estimate PIM memory cycles for the DDR -> SRAM -> PE data path.
    The result includes transfer/setup cost plus overflow-driven miss penalty.
    """
    pim = cfg["pim"]
    est = cfg["estimation"]
    npu_count = int(max(1, pim.get("npu_count", 16)))
    per_npu_bytes = int(math.ceil(max(0, bytes_total) / float(npu_count)))
    sram_bytes = int(max(1, pim["sram_per_npu_bytes"]))
    overflow = max(0, per_npu_bytes - sram_bytes)
    miss_count = int(math.ceil(overflow / max(1, memory_unit)))
    miss_cycles = int(miss_count * miss_penalty)

    ddr_bw = float(est.get("pim_ddr_bandwidth_bytes_per_cycle", io_bw))
    sram_bw = float(est.get("pim_sram_bandwidth_bytes_per_cycle", 64))
    ddr_setup = int(est.get("pim_ddr_setup_cycles", 20))
    sram_setup = int(est.get("pim_sram_setup_cycles", 4))
    wb_setup = int(est.get("pim_writeback_setup_cycles", 8))

    ddr_to_sram_cycles = int(math.ceil(max(0, bytes_total) / max(1.0, ddr_bw))) + ddr_setup
    sram_to_pe_cycles = int(math.ceil(per_npu_bytes / max(1.0, sram_bw))) + sram_setup
    pe_to_sram_cycles = int(math.ceil(per_npu_bytes / max(1.0, sram_bw))) + wb_setup
    total_memory_cycles = int(ddr_to_sram_cycles + sram_to_pe_cycles + pe_to_sram_cycles)

    return {
        "memory_cycles": total_memory_cycles,
        "cache_miss_count": miss_count,
        "cache_miss_cycles": miss_cycles,
        "pim_per_npu_bytes": int(per_npu_bytes),
        "pim_ddr_to_sram_cycles": int(ddr_to_sram_cycles),
        "pim_sram_to_pe_cycles": int(sram_to_pe_cycles),
        "pim_pe_writeback_cycles": int(pe_to_sram_cycles),
    }


def ops_and_memory_cfg(cfg):
    """
    Extract per-device compute throughput, memory bandwidth, and memory hierarchy data.
    PIM throughput is returned as an INT8/INT32 dictionary because the model uses
    INT8 multiply with INT32 accumulation.
    """
    cpu = cfg["cpu"]
    io = cfg["io"]
    pim = cfg["pim"]
    est = cfg["estimation"]

    cpu_total_ops_per_cycle = cpu["vector_peak_ops_per_sec"]["fp32"] / cpu["frequency_hz"]
    pim_int8_ops_per_cycle = pim["chiplet_int8_ops_per_sec"] / pim["frequency_hz"]
    pim_int32_ops_per_cycle = pim.get("chiplet_int32_acc_ops_per_sec", pim["chiplet_int8_ops_per_sec"]) / pim["frequency_hz"]
    io_total_ops_per_cycle = float(est.get("io_compute_ops_per_cycle", 16))

    io_bw = io["memory_channels"] * io["channel_bandwidth_bytes_per_cycle"]
    mem_bw = {"CPU": io_bw, "IO": io_bw, "PIM": io_bw}
    cpu_bw = merge_numeric_dict(
        est.get("cpu_cache_bandwidth_bytes_per_cycle"),
        {"l1": 128, "l2": 64, "l3": 32, "mem": io_bw},
    )
    cpu_lat = merge_numeric_dict(
        est.get("cpu_cache_access_latency_cycles"),
        {"l1": 1, "l2": 4, "l3": 12, "mem": 0},
    )
    io_bw_levels = merge_numeric_dict(
        est.get("io_cache_bandwidth_bytes_per_cycle"),
        {"l1": 64, "l2": 32, "l3": 16, "llc": 16, "mem": io_bw},
    )
    io_lat = merge_numeric_dict(
        est.get("io_cache_access_latency_cycles"),
        {"l1": 1, "l2": 3, "l3": 8, "llc": 12, "mem": 0},
    )

    memory_levels = {
        "CPU": [
            {
                "name": "L1",
                "capacity_bytes": int(cpu["l1_bytes"]),
                "bandwidth_bytes_per_cycle": cpu_bw["l1"],
                "access_latency_cycles": cpu_lat["l1"],
            },
            {
                "name": "L2",
                "capacity_bytes": int(cpu["l2_bytes"]),
                "bandwidth_bytes_per_cycle": cpu_bw["l2"],
                "access_latency_cycles": cpu_lat["l2"],
            },
            {
                "name": "L3",
                "capacity_bytes": int(cpu["l3_bytes"]),
                "bandwidth_bytes_per_cycle": cpu_bw["l3"],
                "access_latency_cycles": cpu_lat["l3"],
            },
            {
                "name": "MEM",
                "capacity_bytes": None,
                "bandwidth_bytes_per_cycle": cpu_bw["mem"],
                "access_latency_cycles": cpu_lat["mem"],
            },
        ],
        "IO": [
            {
                "name": "L1",
                "capacity_bytes": int(io["local_caches"]["l1_bytes"]),
                "bandwidth_bytes_per_cycle": io_bw_levels["l1"],
                "access_latency_cycles": io_lat["l1"],
            },
            {
                "name": "L2",
                "capacity_bytes": int(io["local_caches"]["l2_bytes"]),
                "bandwidth_bytes_per_cycle": io_bw_levels["l2"],
                "access_latency_cycles": io_lat["l2"],
            },
            {
                "name": "L3",
                "capacity_bytes": int(io["local_caches"]["l3_bytes"]),
                "bandwidth_bytes_per_cycle": io_bw_levels["l3"],
                "access_latency_cycles": io_lat["l3"],
            },
            {
                "name": "LLC",
                "capacity_bytes": int(io["system_llc_bytes"]),
                "bandwidth_bytes_per_cycle": io_bw_levels["llc"],
                "access_latency_cycles": io_lat["llc"],
            },
            {
                "name": "MEM",
                "capacity_bytes": None,
                "bandwidth_bytes_per_cycle": io_bw_levels["mem"],
                "access_latency_cycles": io_lat["mem"],
            },
        ],
    }
    dev_ops = {
        "CPU": cpu_total_ops_per_cycle,
        "IO": io_total_ops_per_cycle,
        "PIM": {"int8": pim_int8_ops_per_cycle, "int32": pim_int32_ops_per_cycle},
    }
    return dev_ops, mem_bw, memory_levels


def compose_compute_memory_cycles(compute_cycles, memory_cycles, est_cfg, key_prefix=""):
    """
    Combine compute and memory cycles according to the configured overlap policy.
    Supported modes are sum, max, and partial overlap.
    """
    mode_key = f"{key_prefix}compute_memory_overlap_mode" if key_prefix else "compute_memory_overlap_mode"
    alpha_key = f"{key_prefix}compute_memory_overlap_alpha" if key_prefix else "compute_memory_overlap_alpha"
    mode = str(est_cfg.get(mode_key, est_cfg.get("compute_memory_overlap_mode", "sum"))).strip().lower()
    compute = int(max(0, compute_cycles))
    memory = int(max(0, memory_cycles))
    if mode in {"max", "overlap"}:
        return int(max(compute, memory)), mode
    if mode == "partial":
        alpha = float(est_cfg.get(alpha_key, est_cfg.get("compute_memory_overlap_alpha", 0.3)))
        alpha = max(0.0, min(1.0, alpha))
        return int(max(compute, memory) + round(alpha * min(compute, memory))), mode
    return int(compute + memory), "sum"


def estimate_explicit_memory_traffic(device, onchip_bytes, offchip_bytes, cfg, mem_bw, memory_levels):
    """
    Estimate cycles for an explicit on-chip/off-chip memory split.
    This path is used when the analysis already provides separated traffic counts.
    """
    est = cfg["estimation"]
    memory_unit = int(est["memory_unit_bytes"])
    miss_penalty = int(est["cache_miss_penalty_cycles"])
    onchip_bytes = int(max(0, onchip_bytes))
    offchip_bytes = int(max(0, offchip_bytes))

    if device in {"CPU", "IO"}:
        onchip_level = next(
            (level for level in memory_levels[device] if str(level.get("name", "")).upper() != "MEM"),
            {"name": "ONCHIP", "bandwidth_bytes_per_cycle": 1, "access_latency_cycles": 0},
        )
        offchip_level = next(
            (level for level in memory_levels[device] if str(level.get("name", "")).upper() == "MEM"),
            {"name": "MEM", "bandwidth_bytes_per_cycle": mem_bw.get(device, 1), "access_latency_cycles": 0},
        )
        onchip_cycles, onchip_accesses = level_access_cycles(
            onchip_bytes,
            onchip_level.get("bandwidth_bytes_per_cycle", 1),
            onchip_level.get("access_latency_cycles", 0),
            memory_unit,
        )
        offchip_cycles, offchip_accesses = level_access_cycles(
            offchip_bytes,
            offchip_level.get("bandwidth_bytes_per_cycle", mem_bw.get(device, 1)),
            offchip_level.get("access_latency_cycles", 0),
            memory_unit,
        )
        hierarchy = [
            {
                "level": "EXPLICIT_ONCHIP",
                "bytes": int(onchip_bytes),
                "cycles": int(onchip_cycles),
                "accesses": int(onchip_accesses),
            },
            {
                "level": "EXPLICIT_OFFCHIP",
                "bytes": int(offchip_bytes),
                "cycles": int(offchip_cycles),
                "accesses": int(offchip_accesses),
            },
        ]
        miss_count = int(offchip_accesses)
        miss_cycles = int(miss_count * miss_penalty)
        return {
            "memory_cycles": int(onchip_cycles + offchip_cycles),
            "cache_miss_count": int(miss_count),
            "cache_miss_cycles": int(miss_cycles),
            "memory_hierarchy_breakdown": hierarchy,
            "onchip_memory_bytes": int(onchip_bytes),
            "offchip_memory_bytes": int(offchip_bytes),
            "onchip_memory_cycles": int(onchip_cycles),
            "offchip_memory_cycles": int(offchip_cycles),
            "extra_onchip_bytes": 0,
            "extra_onchip_cycles": 0,
        }

    ddr_bw = float(est.get("pim_ddr_bandwidth_bytes_per_cycle", mem_bw.get("PIM", 1)))
    sram_bw = float(est.get("pim_sram_bandwidth_bytes_per_cycle", 64))
    ddr_setup = int(est.get("pim_ddr_setup_cycles", 20)) if offchip_bytes > 0 else 0
    sram_setup = int(est.get("pim_sram_setup_cycles", 4)) if onchip_bytes > 0 else 0

    onchip_cycles, onchip_accesses = level_access_cycles(onchip_bytes, sram_bw, 0, memory_unit)
    offchip_cycles, offchip_accesses = level_access_cycles(offchip_bytes, ddr_bw, 0, memory_unit)
    onchip_cycles += sram_setup
    offchip_cycles += ddr_setup
    miss_count = int(offchip_accesses)
    miss_cycles = int(miss_count * miss_penalty)
    return {
        "memory_cycles": int(onchip_cycles + offchip_cycles),
        "cache_miss_count": int(miss_count),
        "cache_miss_cycles": int(miss_cycles),
        "memory_hierarchy_breakdown": [
            {
                "level": "EXPLICIT_ONCHIP",
                "bytes": int(onchip_bytes),
                "cycles": int(onchip_cycles),
                "accesses": int(onchip_accesses),
            },
            {
                "level": "EXPLICIT_OFFCHIP",
                "bytes": int(offchip_bytes),
                "cycles": int(offchip_cycles),
                "accesses": int(offchip_accesses),
            },
        ],
        "onchip_memory_bytes": int(onchip_bytes),
        "offchip_memory_bytes": int(offchip_bytes),
        "onchip_memory_cycles": int(onchip_cycles),
        "offchip_memory_cycles": int(offchip_cycles),
        "extra_onchip_bytes": 0,
        "extra_onchip_cycles": 0,
        "pim_ddr_to_sram_cycles": int(offchip_cycles),
        "pim_sram_to_pe_cycles": int(onchip_cycles),
        "pim_pe_writeback_cycles": 0,
    }


def estimate_cycles_with_explicit_memory(
    device,
    ops,
    onchip_bytes,
    offchip_bytes,
    cfg,
    dev_ops,
    mem_bw,
    memory_levels,
    memory_source="explicit",
):
    est = cfg["estimation"]
    pipeline_cfg = est.get("pipeline_depth_cycles", {})
    if not isinstance(pipeline_cfg, dict):
        pipeline_cfg = {}
    pipeline_depth = int(pipeline_cfg.get(device, 0))
    
    if isinstance(ops, dict):
        workload = ops
        ops_total = int(workload.get("ops", 0))
    else:
        ops_total = int(ops)
        workload = {
            "ops": ops_total,
            "int8_ops": ops_total,
            "int32_ops": 0,
            "compute_kind": "generic",
        }
    
    if device == "PIM":
        compute_cycles = estimate_pim_compute_cycles(workload, cfg)
    else:
        compute_cycles = int(math.ceil(ops_total / max(1.0, dev_ops[device])))

    memory_stats = estimate_explicit_memory_traffic(
        device,
        onchip_bytes,
        offchip_bytes,
        cfg,
        mem_bw,
        memory_levels,
    )
    memory_cycles = int(memory_stats["memory_cycles"])
    core_cycles, compose_mode = compose_compute_memory_cycles(compute_cycles, memory_cycles, est)
    total = max(1, core_cycles + int(memory_stats["cache_miss_cycles"]) + max(0, pipeline_depth))

    return {
        "compute_cycles": int(compute_cycles),
        "int8_ops": int(workload.get("int8_ops", 0)),
        "int32_ops": int(workload.get("int32_ops", 0)),
        "compute_kind": str(workload.get("compute_kind", "generic")),
        "memory_cycles": int(memory_cycles),
        "compute_memory_core_cycles": int(core_cycles),
        "compute_memory_compose_mode": compose_mode,
        "pipeline_cycles": int(max(0, pipeline_depth)),
        "total_cycles": int(total),
        "memory_model": "explicit_onchip_offchip",
        "memory_source": str(memory_source),
        **memory_stats,
    }


def estimate_pim_compute_cycles(workload, cfg):
    """
    Estimate PIM compute cycles for mixed-precision workloads.
    The model supports INT8 multiply and INT32 accumulation behavior.
    """
    pim = cfg["pim"]
    int8_rate = pim["chiplet_int8_ops_per_sec"] / pim["frequency_hz"]
    int32_rate = pim.get("chiplet_int32_acc_ops_per_sec", pim["chiplet_int8_ops_per_sec"]) / pim["frequency_hz"]
    
    kind = str(workload.get("compute_kind", "generic"))
    int8_ops = int(workload.get("int8_ops", 0))
    int32_ops = int(workload.get("int32_ops", 0))
    total_ops = int(workload.get("ops", int8_ops + int32_ops))
    
    if kind == "fused_mac":
        return int(max(
            math.ceil(int8_ops / max(1.0, int8_rate)),
            math.ceil(int32_ops / max(1.0, int32_rate)),
        ))
    
    if kind == "int32_only":
        return int(math.ceil(int32_ops / max(1.0, int32_rate)))
    
    if kind == "int8_only":
        return int(math.ceil(int8_ops / max(1.0, int8_rate)))
    
    return int(math.ceil(total_ops / max(1.0, int8_rate)))


def estimate_cycles(device, ops, bytes_total, cfg, dev_ops, mem_bw, memory_levels, extra_onchip_bytes=0):
    """
    Estimate total execution cycles for one node on CPU, IO, or PIM.
    The result combines compute, memory, cache-miss, and pipeline components.
    """
    est = cfg["estimation"]
    memory_unit = est["memory_unit_bytes"]
    miss_penalty = est["cache_miss_penalty_cycles"]
    pipeline_cfg = est.get("pipeline_depth_cycles", {})
    if not isinstance(pipeline_cfg, dict):
        pipeline_cfg = {}
    pipeline_depth = int(pipeline_cfg.get(device, 0))
    extra_onchip_bytes = int(max(0, extra_onchip_bytes))
    extra_onchip_cycles = 0
    
    if isinstance(ops, dict):
        workload = ops
        ops_total = int(workload.get("ops", 0))
    else:
        ops_total = int(ops)
        workload = {
            "ops": ops_total,
            "int8_ops": ops_total,
            "int32_ops": 0,
            "compute_kind": "generic",
        }

    if device in {"CPU", "IO"}:
        mem_cycles, miss_count, miss_cycles, hierarchy = estimate_cache_hierarchy_cycles(
            bytes_total,
            memory_levels[device],
            memory_unit,
            miss_penalty,
        )
        onchip_memory_bytes = int(
            sum(int(item.get("bytes", 0)) for item in hierarchy if str(item.get("level", "")).upper() != "MEM")
        )
        offchip_memory_bytes = int(
            sum(int(item.get("bytes", 0)) for item in hierarchy if str(item.get("level", "")).upper() == "MEM")
        )
        onchip_memory_cycles = int(
            sum(int(item.get("cycles", 0)) for item in hierarchy if str(item.get("level", "")).upper() != "MEM")
        )
        offchip_memory_cycles = int(
            sum(int(item.get("cycles", 0)) for item in hierarchy if str(item.get("level", "")).upper() == "MEM")
        )
        if extra_onchip_bytes > 0:
            l1_bw = float(memory_levels[device][0].get("bandwidth_bytes_per_cycle", 1))
            extra_onchip_cycles = int(math.ceil(extra_onchip_bytes / max(1.0, l1_bw)))
            onchip_memory_bytes += int(extra_onchip_bytes)
            onchip_memory_cycles += int(extra_onchip_cycles)
            mem_cycles += int(extra_onchip_cycles)
        pim_extra = {}
        compute_cycles = int(math.ceil(ops_total / max(1.0, dev_ops[device])))
    else:
        pim_extra = estimate_pim_memory_cycles(bytes_total, cfg, mem_bw["PIM"], memory_unit, miss_penalty)
        mem_cycles = int(pim_extra["memory_cycles"])
        miss_count = int(pim_extra["cache_miss_count"])
        miss_cycles = int(pim_extra["cache_miss_cycles"])
        hierarchy = []
        compute_cycles = estimate_pim_compute_cycles(workload, cfg)
        npu_count = int(max(1, cfg["pim"].get("npu_count", 16)))
        pim_extra["pim_per_npu_compute_cycles"] = int(
            math.ceil(compute_cycles / max(1, npu_count))
        )
        pim_extra["pim_per_npu_cycles"] = int(
            pim_extra["pim_per_npu_compute_cycles"]
            + pim_extra["pim_sram_to_pe_cycles"]
            + pim_extra["pim_pe_writeback_cycles"]
            + max(0, pipeline_depth)
        )
        offchip_memory_bytes = int(max(0, bytes_total))
        offchip_memory_cycles = int(max(0, pim_extra.get("pim_ddr_to_sram_cycles", 0)))
        per_npu_bytes = int(max(0, pim_extra.get("pim_per_npu_bytes", 0)))
        onchip_memory_bytes = int(per_npu_bytes * npu_count * 2)
        onchip_memory_cycles = int(
            max(0, pim_extra.get("pim_sram_to_pe_cycles", 0))
            + max(0, pim_extra.get("pim_pe_writeback_cycles", 0))
        )
        if extra_onchip_bytes > 0:
            sram_bw = float(est.get("pim_sram_bandwidth_bytes_per_cycle", 64))
            extra_onchip_cycles = int(math.ceil(extra_onchip_bytes / max(1.0, sram_bw)))
            onchip_memory_bytes += int(extra_onchip_bytes)
            onchip_memory_cycles += int(extra_onchip_cycles)
            mem_cycles += int(extra_onchip_cycles)

    core_cycles, compose_mode = compose_compute_memory_cycles(compute_cycles, mem_cycles, est)
    total = max(1, core_cycles + miss_cycles + max(0, pipeline_depth))

    return {
        "compute_cycles": int(compute_cycles),
        "int8_ops": int(workload.get("int8_ops", 0)),
        "int32_ops": int(workload.get("int32_ops", 0)),
        "compute_kind": str(workload.get("compute_kind", "generic")),
        "memory_cycles": int(mem_cycles),
        "compute_memory_core_cycles": int(core_cycles),
        "compute_memory_compose_mode": compose_mode,
        "cache_miss_count": miss_count,
        "cache_miss_cycles": miss_cycles,
        "pipeline_cycles": max(0, pipeline_depth),
        "memory_hierarchy_breakdown": hierarchy,
        "onchip_memory_bytes": int(onchip_memory_bytes),
        "offchip_memory_bytes": int(offchip_memory_bytes),
        "onchip_memory_cycles": int(onchip_memory_cycles),
        "offchip_memory_cycles": int(offchip_memory_cycles),
        "extra_onchip_bytes": int(extra_onchip_bytes),
        "extra_onchip_cycles": int(extra_onchip_cycles),
        "total_cycles": total,
        **pim_extra,
    }

