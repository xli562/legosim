import math

from .common import clamp_ratio


DEFAULT_TILE_CANDIDATES = [8, 16, 32, 64, 96, 128, 192, 256]


def _shape_list(shape):
    if not isinstance(shape, (list, tuple)):
        return []
    out = []
    for dim in shape:
        try:
            out.append(max(1, int(dim)))
        except Exception:
            out.append(1)
    return out


def _product(shape):
    total = 1
    for dim in shape:
        total *= max(1, int(dim))
    return int(total)


def _candidate_values(extent, est_cfg):
    extent = int(max(1, extent))
    raw = est_cfg.get("pim_tile_search_candidates", DEFAULT_TILE_CANDIDATES)
    values = {1, extent}

    if extent <= 64:
        values.update(range(1, extent + 1))
    else:
        if isinstance(raw, (list, tuple)):
            for value in raw:
                try:
                    value = int(value)
                except Exception:
                    continue
                if 1 <= value <= extent:
                    values.add(value)
        for div in [2, 3, 4, 5, 6, 8, 12, 16]:
            values.add(max(1, extent // div))
            values.add(max(1, int(math.ceil(extent / float(div)))))

    max_candidates = int(max(4, est_cfg.get("pim_tile_search_max_candidates_per_axis", 18)))
    ordered = sorted(value for value in values if 1 <= value <= extent)
    if len(ordered) <= max_candidates:
        return ordered

    selected = set()
    for idx in range(max_candidates):
        pos = int(round(idx * (len(ordered) - 1) / float(max(1, max_candidates - 1))))
        selected.add(ordered[pos])
    return sorted(selected)


def _fragment_penalty(extent, tile):
    extent = int(max(1, extent))
    tile = int(max(1, tile))
    remain = extent % tile
    if remain == 0:
        return 0.0
    return float(tile - remain) / float(tile)


def _apply_memory_optimizations(memory_stats, device_opt):
    device_opt = device_opt if isinstance(device_opt, dict) else {}
    offchip_reduction = clamp_ratio(device_opt.get("offchip_reduction_ratio", 0.0))
    tile_reuse = clamp_ratio(device_opt.get("tile_reuse_ratio", 0.0))
    cache_affinity = clamp_ratio(device_opt.get("cache_affinity_ratio", 0.0))
    p2p_reuse = clamp_ratio(device_opt.get("p2p_reuse_ratio", 0.0))

    offchip_scale = 1.0 - min(0.55, 0.45 * offchip_reduction + 0.20 * tile_reuse + 0.15 * p2p_reuse)
    onchip_scale = 1.0 - min(0.40, 0.25 * cache_affinity + 0.15 * tile_reuse)
    reuse_scale = 1.0 + min(0.50, 0.35 * tile_reuse + 0.15 * cache_affinity + 0.15 * p2p_reuse)

    stats = dict(memory_stats)
    stats["ddr_load_bytes"] = int(math.ceil(stats["ddr_load_bytes"] * offchip_scale))
    stats["ddr_store_bytes"] = int(math.ceil(stats["ddr_store_bytes"] * max(0.75, 1.0 - 0.20 * offchip_reduction)))
    stats["sram_fill_bytes"] = int(math.ceil(stats["sram_fill_bytes"] * onchip_scale))
    stats["sram_read_bytes"] = int(math.ceil(stats["sram_read_bytes"] * onchip_scale))
    stats["sram_write_bytes"] = int(math.ceil(stats["sram_write_bytes"] * onchip_scale))
    stats["reuse_bytes"] = int(math.ceil(stats["reuse_bytes"] * reuse_scale))
    return stats


def _finalize_cycles(memory_stats, est_cfg):
    ddr_bw = float(est_cfg.get("pim_ddr_bandwidth_bytes_per_cycle", 64))
    sram_bw = float(est_cfg.get("pim_sram_bandwidth_bytes_per_cycle", 64))
    ddr_setup = int(est_cfg.get("pim_ddr_setup_cycles", 20))
    sram_setup = int(est_cfg.get("pim_sram_setup_cycles", 4))
    wb_setup = int(est_cfg.get("pim_writeback_setup_cycles", 8))

    ddr_load_bytes = int(max(0, memory_stats.get("ddr_load_bytes", 0)))
    ddr_store_bytes = int(max(0, memory_stats.get("ddr_store_bytes", 0)))
    sram_fill_bytes = int(max(0, memory_stats.get("sram_fill_bytes", 0)))
    sram_read_bytes = int(max(0, memory_stats.get("sram_read_bytes", 0)))
    sram_write_bytes = int(max(0, memory_stats.get("sram_write_bytes", 0)))
    load_events = int(max(0, memory_stats.get("load_events", 0)))
    store_events = int(max(0, memory_stats.get("store_events", 0)))
    sram_events = int(max(0, memory_stats.get("sram_events", 0)))

    ddr_load_cycles = int(math.ceil(ddr_load_bytes / max(1.0, ddr_bw))) + load_events * ddr_setup
    ddr_store_cycles = int(math.ceil(ddr_store_bytes / max(1.0, ddr_bw))) + store_events * wb_setup
    sram_total_bytes = int(sram_fill_bytes + sram_read_bytes + sram_write_bytes)
    sram_cycles = int(math.ceil(sram_total_bytes / max(1.0, sram_bw))) + sram_events * sram_setup
    total_cycles = int(ddr_load_cycles + ddr_store_cycles + sram_cycles)

    return {
        "per_use_offchip_bytes": int(ddr_load_bytes + ddr_store_bytes),
        "per_use_onchip_bytes": int(sram_total_bytes),
        "per_use_total_bytes": int(ddr_load_bytes + ddr_store_bytes + sram_total_bytes),
        "per_use_memory_cycles": int(total_cycles),
        "memory_cycles_breakdown": {
            "ddr_load_cycles": int(ddr_load_cycles),
            "ddr_store_cycles": int(ddr_store_cycles),
            "sram_cycles": int(sram_cycles),
            "total_memory_cycles": int(total_cycles),
        },
    }


def _score_candidate(total_cycles, extent_triples, fragmentation_alpha, prefer_square):
    penalty = 0.0
    for extent, tile in extent_triples:
        penalty += _fragment_penalty(extent, tile)
    if prefer_square and len(extent_triples) >= 2:
        first = max(1, int(extent_triples[0][1]))
        second = max(1, int(extent_triples[1][1]))
        penalty += abs(math.log(float(first) / float(second), 2.0)) * 0.05
    return float(total_cycles) * (1.0 + fragmentation_alpha * penalty)


def _matmul_dims(signature):
    lhs = _shape_list(signature.get("lhs_shape"))
    rhs = _shape_list(signature.get("rhs_shape"))
    out = _shape_list(signature.get("out_shape"))
    if len(lhs) < 2 or len(rhs) < 2:
        return None
    m_dim = int(lhs[-2])
    k_dim = int(lhs[-1])
    n_dim = int(rhs[-1])
    if len(out) >= 2:
        m_dim = int(out[-2])
        n_dim = int(out[-1])
    if min(m_dim, n_dim, k_dim) <= 0:
        return None
    return m_dim, n_dim, k_dim


def _evaluate_matmul_candidate(m_dim, n_dim, k_dim, tm, tn, tk, cfg, est_cfg, device_opt):
    pim_cfg = cfg.get("pim", {})
    sram_capacity = int(max(1, pim_cfg.get("sram_per_npu_bytes", 131072)))
    reserved_bytes = int(max(0, est_cfg.get("pim_sram_reserved_bytes", 4096)))
    activation_bytes = int(max(1, pim_cfg.get("activation_bytes", 1)))
    weight_bytes = int(max(1, pim_cfg.get("weight_bytes", 1)))
    accumulator_bytes = int(max(1, pim_cfg.get("accumulator_bytes", 4)))
    output_bytes = int(max(1, pim_cfg.get("output_bytes", 4)))

    act_tile = int(tm * tk * activation_bytes)
    weight_tile = int(tk * tn * weight_bytes)
    acc_tile = int(tm * tn * accumulator_bytes)
    out_tile = int(tm * tn * output_bytes)
    peak_sram = int(act_tile + weight_tile + acc_tile + out_tile + reserved_bytes)
    fits = bool(peak_sram <= sram_capacity)

    m_tiles = int(math.ceil(m_dim / float(tm)))
    n_tiles = int(math.ceil(n_dim / float(tn)))
    k_tiles = int(math.ceil(k_dim / float(tk)))

    def build_stationary_plan(order_name):
        if order_name == "activation_stationary":
            activation_load_events = m_tiles * k_tiles
            weight_load_events = m_tiles * n_tiles * k_tiles
            activation_reuse_events = m_tiles * k_tiles * max(0, n_tiles - 1)
            weight_reuse_events = 0
        else:
            activation_load_events = m_tiles * n_tiles * k_tiles
            weight_load_events = n_tiles * k_tiles
            activation_reuse_events = 0
            weight_reuse_events = n_tiles * k_tiles * max(0, m_tiles - 1)

        output_store_events = m_tiles * n_tiles
        acc_read_events = m_tiles * n_tiles * max(0, k_tiles - 1)
        acc_write_events = m_tiles * n_tiles * max(1, k_tiles)

        raw_stats = {
            "ddr_load_bytes": int(activation_load_events * act_tile + weight_load_events * weight_tile),
            "ddr_store_bytes": int(output_store_events * out_tile),
            "sram_fill_bytes": int(activation_load_events * act_tile + weight_load_events * weight_tile),
            "sram_read_bytes": int(
                activation_load_events * act_tile
                + weight_load_events * weight_tile
                + acc_read_events * acc_tile
            ),
            "sram_write_bytes": int(acc_write_events * acc_tile + output_store_events * out_tile),
            "reuse_bytes": int(activation_reuse_events * act_tile + weight_reuse_events * weight_tile),
            "evict_bytes": int(
                activation_load_events * act_tile
                + weight_load_events * weight_tile
                + output_store_events * out_tile
                + m_tiles * n_tiles * acc_tile
            ),
            "load_events": int(activation_load_events + weight_load_events),
            "store_events": int(output_store_events),
            "sram_events": int(
                activation_load_events
                + weight_load_events
                + acc_read_events
                + acc_write_events
                + output_store_events
            ),
        }
        stats = _apply_memory_optimizations(raw_stats, device_opt)
        finalized = _finalize_cycles(stats, est_cfg)
        return {
            "loop_order": order_name,
            "load_events": int(raw_stats["load_events"]),
            "store_events": int(raw_stats["store_events"]),
            "activation_reuse_bytes": int(activation_reuse_events * act_tile),
            "weight_reuse_bytes": int(weight_reuse_events * weight_tile),
            "raw_memory": raw_stats,
            "optimized_memory": stats,
            **finalized,
        }

    stationaries = [
        build_stationary_plan("activation_stationary"),
        build_stationary_plan("weight_stationary"),
    ]
    best_stationary = min(stationaries, key=lambda item: item["per_use_memory_cycles"])

    return {
        "supported": True,
        "template_kind": "tile_matmul",
        "tile_plan": {
            "tile_m": int(tm),
            "tile_n": int(tn),
            "tile_k": int(tk),
            "tile_count_m": int(m_tiles),
            "tile_count_n": int(n_tiles),
            "tile_count_k": int(k_tiles),
            "tile_count_total": int(m_tiles * n_tiles * k_tiles),
            "loop_order": str(best_stationary["loop_order"]),
        },
        "sram_capacity_bytes": int(sram_capacity),
        "sram_reserved_bytes": int(reserved_bytes),
        "peak_sram_bytes": int(peak_sram),
        "fits_sram": bool(fits),
        "activation_tile_bytes": int(act_tile),
        "weight_tile_bytes": int(weight_tile),
        "accumulator_tile_bytes": int(acc_tile),
        "output_tile_bytes": int(out_tile),
        "ddr_load_bytes": int(best_stationary["optimized_memory"]["ddr_load_bytes"]),
        "ddr_store_bytes": int(best_stationary["optimized_memory"]["ddr_store_bytes"]),
        "sram_fill_bytes": int(best_stationary["optimized_memory"]["sram_fill_bytes"]),
        "sram_read_bytes": int(best_stationary["optimized_memory"]["sram_read_bytes"]),
        "sram_write_bytes": int(best_stationary["optimized_memory"]["sram_write_bytes"]),
        "reuse_bytes": int(best_stationary["optimized_memory"]["reuse_bytes"]),
        "evict_bytes": int(best_stationary["optimized_memory"]["evict_bytes"]),
        "activation_reuse_bytes": int(best_stationary["activation_reuse_bytes"]),
        "weight_reuse_bytes": int(best_stationary["weight_reuse_bytes"]),
        "loop_orders_considered": stationaries,
        "load_events": int(best_stationary["load_events"]),
        "store_events": int(best_stationary["store_events"]),
        **{
            key: value
            for key, value in best_stationary.items()
            if key
            in {
                "per_use_onchip_bytes",
                "per_use_offchip_bytes",
                "per_use_total_bytes",
                "per_use_memory_cycles",
                "memory_cycles_breakdown",
            }
        },
    }


def _evaluate_stream_candidate(
    template_kind,
    total_elems,
    tile_elems,
    cfg,
    est_cfg,
    device_opt,
    rhs_elems=0,
    reduce_extent=1,
):
    pim_cfg = cfg.get("pim", {})
    sram_capacity = int(max(1, pim_cfg.get("sram_per_npu_bytes", 131072)))
    reserved_bytes = int(max(0, est_cfg.get("pim_sram_reserved_bytes", 4096)))
    activation_bytes = int(max(1, pim_cfg.get("activation_bytes", 1)))
    accumulator_bytes = int(max(1, pim_cfg.get("accumulator_bytes", 4)))
    output_bytes = int(max(1, pim_cfg.get("output_bytes", 4)))

    input_tile = int(tile_elems * max(activation_bytes, output_bytes))
    rhs_tile = 0
    if template_kind == "tile_elementwise_binary":
        rhs_tile = int(min(tile_elems, max(1, rhs_elems)) * max(activation_bytes, output_bytes))
    scratch_tile = 0
    acc_tile = 0
    if template_kind == "tile_softmax":
        scratch_tile = int(tile_elems * output_bytes)
        acc_tile = int(max(1, tile_elems // max(1, reduce_extent)) * accumulator_bytes)
    elif template_kind == "tile_reduce":
        acc_tile = int(max(1, tile_elems // max(1, reduce_extent)) * accumulator_bytes)

    output_tile = int(tile_elems * output_bytes)
    peak_sram = int(input_tile + rhs_tile + scratch_tile + acc_tile + output_tile + reserved_bytes)
    fits = bool(peak_sram <= sram_capacity)

    tile_count = int(math.ceil(total_elems / float(max(1, tile_elems))))
    rhs_reuse = 0
    rhs_load_bytes = int(tile_count * rhs_tile)
    if template_kind == "tile_elementwise_binary" and rhs_elems > 0 and rhs_elems < total_elems:
        rhs_load_bytes = int(rhs_tile)
        rhs_reuse = int(max(0, tile_count - 1) * rhs_tile)

    passes = 1
    if template_kind == "tile_softmax":
        passes = 3
    elif template_kind == "tile_reduce":
        passes = 2

    raw_stats = {
        "ddr_load_bytes": int(tile_count * input_tile * passes + rhs_load_bytes),
        "ddr_store_bytes": int(tile_count * output_tile),
        "sram_fill_bytes": int(tile_count * input_tile * passes + rhs_load_bytes),
        "sram_read_bytes": int(tile_count * (input_tile * passes + scratch_tile + acc_tile)),
        "sram_write_bytes": int(tile_count * (output_tile + scratch_tile + acc_tile)),
        "reuse_bytes": int(rhs_reuse),
        "evict_bytes": int(tile_count * (input_tile + rhs_tile + output_tile + scratch_tile + acc_tile)),
        "load_events": int(tile_count * passes + (1 if rhs_load_bytes > 0 else 0)),
        "store_events": int(tile_count),
        "sram_events": int(tile_count * (2 * passes + 2)),
    }
    stats = _apply_memory_optimizations(raw_stats, device_opt)
    finalized = _finalize_cycles(stats, est_cfg)

    return {
        "supported": True,
        "template_kind": str(template_kind),
        "tile_plan": {
            "tile_elems": int(tile_elems),
            "tile_count_total": int(tile_count),
            "passes": int(passes),
        },
        "sram_capacity_bytes": int(sram_capacity),
        "sram_reserved_bytes": int(reserved_bytes),
        "peak_sram_bytes": int(peak_sram),
        "fits_sram": bool(fits),
        "activation_tile_bytes": int(input_tile + rhs_tile),
        "weight_tile_bytes": 0,
        "accumulator_tile_bytes": int(acc_tile),
        "output_tile_bytes": int(output_tile),
        "ddr_load_bytes": int(stats["ddr_load_bytes"]),
        "ddr_store_bytes": int(stats["ddr_store_bytes"]),
        "sram_fill_bytes": int(stats["sram_fill_bytes"]),
        "sram_read_bytes": int(stats["sram_read_bytes"]),
        "sram_write_bytes": int(stats["sram_write_bytes"]),
        "reuse_bytes": int(stats["reuse_bytes"]),
        "evict_bytes": int(stats["evict_bytes"]),
        "load_events": int(stats["load_events"]),
        "store_events": int(stats["store_events"]),
        **finalized,
    }


def search_pim_sram_tile_plan(template_kind, signature, cfg, est_cfg, device_opt=None):
    device_opt = device_opt if isinstance(device_opt, dict) else {}
    fragmentation_alpha = float(max(0.0, est_cfg.get("pim_tile_fragmentation_penalty_alpha", 0.08)))
    prefer_square = bool(est_cfg.get("pim_tile_search_prefer_square", True))

    if template_kind == "tile_matmul":
        dims = _matmul_dims(signature)
        if dims is None:
            return {"supported": False, "reason": "invalid matmul signature for SRAM tiling"}
        m_dim, n_dim, k_dim = dims
        best = None
        examined = 0
        for tm in _candidate_values(m_dim, est_cfg):
            for tn in _candidate_values(n_dim, est_cfg):
                for tk in _candidate_values(k_dim, est_cfg):
                    examined += 1
                    candidate = _evaluate_matmul_candidate(m_dim, n_dim, k_dim, tm, tn, tk, cfg, est_cfg, device_opt)
                    if not candidate.get("fits_sram", False):
                        continue
                    score = _score_candidate(
                        candidate.get("per_use_memory_cycles", 0),
                        [(m_dim, tm), (n_dim, tn), (k_dim, tk)],
                        fragmentation_alpha,
                        prefer_square,
                    )
                    candidate["search_score"] = float(score)
                    if best is None or score < best["search_score"]:
                        best = candidate
        if best is None:
            return {
                "supported": False,
                "reason": "no matmul tile fits into 128KB SRAM",
                "sram_capacity_bytes": int(cfg.get("pim", {}).get("sram_per_npu_bytes", 131072)),
                "examined_candidates": int(examined),
            }
        best["examined_candidates"] = int(examined)
        best["search_strategy"] = "candidate_grid"
        return best

    if template_kind in {"tile_elementwise_binary", "tile_elementwise_unary"}:
        total_elems = _product(_shape_list(signature.get("out_shape")))
        rhs_elems = _product(_shape_list(signature.get("rhs_shape"))) if template_kind == "tile_elementwise_binary" else 0
        best = None
        examined = 0
        for tile_elems in _candidate_values(total_elems, est_cfg):
            examined += 1
            candidate = _evaluate_stream_candidate(
                template_kind,
                total_elems,
                tile_elems,
                cfg,
                est_cfg,
                device_opt,
                rhs_elems=rhs_elems,
            )
            if not candidate.get("fits_sram", False):
                continue
            score = _score_candidate(
                candidate.get("per_use_memory_cycles", 0),
                [(total_elems, tile_elems)],
                fragmentation_alpha,
                False,
            )
            candidate["search_score"] = float(score)
            if best is None or score < best["search_score"]:
                best = candidate
        if best is None:
            return {"supported": False, "reason": "no elementwise tile fits into 128KB SRAM"}
        best["examined_candidates"] = int(examined)
        best["search_strategy"] = "candidate_1d"
        return best

    if template_kind in {"tile_softmax", "tile_reduce"}:
        input_shape = _shape_list(signature.get("input_shape") or signature.get("out_shape"))
        total_elems = int(_product(input_shape))
        reduce_extent = int(max(1, input_shape[-1] if input_shape else 1))
        best = None
        examined = 0
        for tile_elems in _candidate_values(total_elems, est_cfg):
            examined += 1
            candidate = _evaluate_stream_candidate(
                template_kind,
                total_elems,
                tile_elems,
                cfg,
                est_cfg,
                device_opt,
                reduce_extent=reduce_extent,
            )
            if not candidate.get("fits_sram", False):
                continue
            score = _score_candidate(
                candidate.get("per_use_memory_cycles", 0),
                [(total_elems, tile_elems)],
                fragmentation_alpha,
                False,
            )
            candidate["search_score"] = float(score)
            if best is None or score < best["search_score"]:
                best = candidate
        if best is None:
            return {"supported": False, "reason": "no reduction tile fits into 128KB SRAM"}
        best["examined_candidates"] = int(examined)
        best["search_strategy"] = "candidate_1d"
        return best

    return {"supported": False, "reason": f"{template_kind} does not have a PIM SRAM tiler"}


def scale_pim_tile_memory(detail, ratio):
    if not isinstance(detail, dict):
        return None
    ratio = max(0.0, float(ratio))
    scaled = {}
    passthrough = {
        "template_kind",
        "tile_plan",
        "sram_capacity_bytes",
        "sram_reserved_bytes",
        "peak_sram_bytes",
        "fits_sram",
        "search_strategy",
        "examined_candidates",
        "reason",
    }
    skip_keys = {"loop_orders_considered", "search_score"}
    scaled_keys = {
        "activation_tile_bytes",
        "weight_tile_bytes",
        "accumulator_tile_bytes",
        "output_tile_bytes",
        "ddr_load_bytes",
        "ddr_store_bytes",
        "sram_fill_bytes",
        "sram_read_bytes",
        "sram_write_bytes",
        "reuse_bytes",
        "evict_bytes",
        "per_use_onchip_bytes",
        "per_use_offchip_bytes",
        "per_use_total_bytes",
        "per_use_memory_cycles",
        "load_events",
        "store_events",
    }
    for key, value in detail.items():
        if key in skip_keys:
            continue
        if key in passthrough:
            scaled[key] = value
        elif key in scaled_keys:
            scaled[key] = int(max(1, math.ceil(float(value) * ratio))) if float(value) > 0 else 0
        elif key == "memory_cycles_breakdown" and isinstance(value, dict):
            scaled[key] = {
                sub_key: int(max(1, math.ceil(float(sub_value) * ratio))) if float(sub_value) > 0 else 0
                for sub_key, sub_value in value.items()
            }
        else:
            scaled[key] = value
    return scaled
