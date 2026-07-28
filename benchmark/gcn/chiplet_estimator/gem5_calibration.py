"""
gem5 calibration helpers for tuning estimator parameters from measured runs.
"""
import glob
import json
import math
import os
import re
import subprocess

from .common import to_abs


STAT_LINE_RE = re.compile(r"^\s*([^\s#]+)\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def parse_gem5_stats_file(path):
    """
    Parse a gem5 stats.txt file into a numeric key/value map.
    """
    stats = {}
    if not path or not os.path.exists(path):
        return stats
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = STAT_LINE_RE.match(line)
                if not match:
                    continue
                key = match.group(1).strip()
                try:
                    value = float(match.group(2))
                except Exception:
                    continue
                if math.isnan(value) or math.isinf(value):
                    continue
                stats[key] = value
    except Exception:
        pass
    return stats


def find_latest_existing(paths):
    """
    Return the newest existing path from a candidate list.
    """
    candidates = [path for path in paths if path and os.path.exists(path)]
    if not candidates:
        return None
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def run_shell_command(cmd, cwd):
    """
    Run a shell command in the requested directory and capture a compact result.
    """
    if not cmd:
        return None
    try:
        completed = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return {
            "cmd": cmd,
            "returncode": int(completed.returncode),
            "stdout_tail": completed.stdout[-2000:] if completed.stdout else "",
            "stderr_tail": completed.stderr[-2000:] if completed.stderr else "",
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": None, "error": str(exc)}


def auto_find_gem5_stats(stats_root, role):
    """
    Search stats_root for a gem5 stats.txt file that matches the requested role.
    CPU and IO use different search patterns because they occupy different phase1 slots.
    """
    if not stats_root:
        return None
    role = str(role).upper()
    if role == "CPU":
        patterns = [
            "proc_r*_p1_t0/m5out/stats.txt",
            "**/proc_r*_p1_t0/m5out/stats.txt",
            "**/m5out/stats.txt",
        ]
    elif role == "IO":
        patterns = [
            "proc_r*_p1_t2/m5out/stats.txt",
            "**/proc_r*_p1_t2/m5out/stats.txt",
            "**/m5out/stats.txt",
        ]
    else:
        patterns = ["**/m5out/stats.txt"]

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(stats_root, pattern), recursive=True))
    return find_latest_existing(candidates)


def pick_active_cpu_idx(stats):
    """
    Pick the CPU index with the largest numCycles value.
    """
    best_idx = 0
    best_cycles = -1.0
    for key, value in stats.items():
        match = re.match(r"^system\.cpu(\d+)\.numCycles$", key)
        if not match:
            continue
        idx = int(match.group(1))
        if value > best_cycles:
            best_cycles = float(value)
            best_idx = idx
    return best_idx


def parse_gem5_stats_summary(path, role_hint):
    """
    Parse a gem5 stats file and extract the metrics used by the estimator,
    including IPC, CPI, cycle count, and dcache miss behavior.
    """
    stats = parse_gem5_stats_file(path)
    if not stats:
        return None

    cpu_idx = pick_active_cpu_idx(stats)
    prefix = f"system.cpu{cpu_idx}"

    cpi = stats.get(f"{prefix}.cpi", stats.get(f"{prefix}.commitStats0.cpi"))
    ipc = stats.get(f"{prefix}.ipc", stats.get(f"{prefix}.commitStats0.ipc"))
    num_cycles = stats.get(f"{prefix}.numCycles")
    sim_insts = stats.get("simInsts", stats.get(f"{prefix}.commitStats0.numInsts"))
    sim_ticks = stats.get("simTicks")
    sim_seconds = stats.get("simSeconds")
    clk_tick = stats.get("system.cpu_clk_domain.clock", stats.get("system.clk_domain.clock"))
    miss_rate = stats.get(
        f"{prefix}.dcache.overallMissRate::total",
        stats.get(f"{prefix}.dcache.demandMissRate::total"),
    )
    miss_lat_tick = stats.get(
        f"{prefix}.dcache.overallAvgMissLatency::total",
        stats.get(f"{prefix}.dcache.demandAvgMissLatency::total"),
    )

    miss_lat_cycles = None
    if miss_lat_tick is not None and clk_tick is not None and clk_tick > 0:
        miss_lat_cycles = float(miss_lat_tick) / float(clk_tick)

    return {
        "role": role_hint,
        "path": path,
        "active_cpu_idx": int(cpu_idx),
        "cpi": float(cpi) if cpi is not None else None,
        "ipc": float(ipc) if ipc is not None else None,
        "num_cycles": float(num_cycles) if num_cycles is not None else None,
        "sim_insts": float(sim_insts) if sim_insts is not None else None,
        "sim_ticks": float(sim_ticks) if sim_ticks is not None else None,
        "sim_seconds": float(sim_seconds) if sim_seconds is not None else None,
        "dcache_miss_rate": float(miss_rate) if miss_rate is not None else None,
        "dcache_avg_miss_cycles": float(miss_lat_cycles) if miss_lat_cycles is not None else None,
    }


def apply_gem5_calibration(cfg, dev_ops, args, base_dir):
    """
    Apply gem5-based calibration to device throughput and cache miss penalty settings.
    Returns the updated configuration, updated dev_ops map, and a calibration report.
    """
    est_cfg = cfg.get("estimation", {})
    gcfg = est_cfg.get("gem5_calibration", {})
    enabled = bool(args.enable_gem5_calibration or gcfg.get("enabled", False))

    report = {
        "enabled": enabled,
        "cpu_stats": None,
        "io_stats": None,
        "interchiplet_cmd": None,
        "gem5_cmd": {"cpu": None, "io": None},
        "scales": {},
        "cache_miss_penalty_before": int(est_cfg.get("cache_miss_penalty_cycles", 120)),
        "cache_miss_penalty_after": int(est_cfg.get("cache_miss_penalty_cycles", 120)),
    }
    if not enabled:
        return cfg, dict(dev_ops), report

    cfg_runtime = json.loads(json.dumps(cfg))
    dev_ops_runtime = dict(dev_ops)
    est_runtime = cfg_runtime.setdefault("estimation", {})

    stats_root = args.gem5_stats_root or gcfg.get("stats_root") or base_dir
    if stats_root and not os.path.isabs(stats_root):
        stats_root = to_abs(base_dir, stats_root)

    interchiplet_cmd = args.interchiplet_cmd or gcfg.get("interchiplet_cmd")
    if interchiplet_cmd:
        report["interchiplet_cmd"] = run_shell_command(interchiplet_cmd, base_dir)
        interchiplet_result = report["interchiplet_cmd"] or {}
        require_success = bool(gcfg.get("require_success", True))
        returncode = interchiplet_result.get("returncode")
        if require_success and returncode not in (0, None):
            raise RuntimeError(
                f"[Estimator] interchiplet command failed (returncode={returncode}): {interchiplet_cmd}"
            )

    cpu_cmd = args.gem5_cpu_cmd or gcfg.get("cpu_cmd")
    io_cmd = args.gem5_io_cmd or gcfg.get("io_cmd")
    if cpu_cmd:
        report["gem5_cmd"]["cpu"] = run_shell_command(cpu_cmd, base_dir)
    if io_cmd:
        report["gem5_cmd"]["io"] = run_shell_command(io_cmd, base_dir)

    cpu_stats_path = args.gem5_stats_cpu or gcfg.get("cpu_stats")
    io_stats_path = args.gem5_stats_io or gcfg.get("io_stats")
    if cpu_stats_path and not os.path.isabs(cpu_stats_path):
        cpu_stats_path = to_abs(base_dir, cpu_stats_path)
    if io_stats_path and not os.path.isabs(io_stats_path):
        io_stats_path = to_abs(base_dir, io_stats_path)

    if not cpu_stats_path:
        cpu_stats_path = auto_find_gem5_stats(stats_root, "CPU")
    if not io_stats_path:
        io_stats_path = auto_find_gem5_stats(stats_root, "IO")

    if (
        cpu_stats_path
        and io_stats_path
        and os.path.normpath(cpu_stats_path) == os.path.normpath(io_stats_path)
        and not args.gem5_stats_io
        and not gcfg.get("io_stats")
    ):
        io_stats_path = None

    cpu_summary = parse_gem5_stats_summary(cpu_stats_path, "CPU") if cpu_stats_path else None
    io_summary = parse_gem5_stats_summary(io_stats_path, "IO") if io_stats_path else None
    report["cpu_stats"] = cpu_summary
    report["io_stats"] = io_summary

    ref_ipc = float(gcfg.get("reference_ipc", est_cfg.get("gem5_reference_ipc", 1.0)))
    min_scale = float(gcfg.get("min_scale", 0.05))
    max_scale = float(gcfg.get("max_scale", 2.0))

    def clamp(value):
        return max(min_scale, min(max_scale, float(value)))

    for dev, summary in [("CPU", cpu_summary), ("IO", io_summary)]:
        if not summary:
            continue
        ipc = summary.get("ipc")
        if ipc is None or ipc <= 0:
            continue
        scale = clamp(ipc / max(1e-6, ref_ipc))
        dev_ops_runtime[dev] = float(dev_ops_runtime[dev]) * scale
        report["scales"][dev] = {
            "ipc": float(ipc),
            "reference_ipc": float(ref_ipc),
            "ops_scale": float(scale),
            "ops_per_cycle_before": float(dev_ops[dev]),
            "ops_per_cycle_after": float(dev_ops_runtime[dev]),
        }

    miss_samples = []
    for summary in [cpu_summary, io_summary]:
        if not summary:
            continue
        value = summary.get("dcache_avg_miss_cycles")
        if value is not None and value > 0:
            miss_samples.append(float(value))
    if miss_samples:
        miss_measured = sum(miss_samples) / float(len(miss_samples))
        alpha = float(gcfg.get("miss_penalty_blend_alpha", 0.5))
        alpha = max(0.0, min(1.0, alpha))
        miss_before = float(est_runtime.get("cache_miss_penalty_cycles", 120))
        miss_after = int(round((1.0 - alpha) * miss_before + alpha * miss_measured))
        est_runtime["cache_miss_penalty_cycles"] = max(1, miss_after)
        report["cache_miss_penalty_after"] = int(est_runtime["cache_miss_penalty_cycles"])

    return cfg_runtime, dev_ops_runtime, report
