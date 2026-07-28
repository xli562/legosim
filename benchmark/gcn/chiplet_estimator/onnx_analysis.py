"""ONNX-shape-driven local analysis used by the GCN flow."""

import json

from .common import pick_device_entry
from .pim_sram_tiling import search_pim_sram_tile_plan


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


def _shape_product(shape):
    total = 1
    for dim in _shape_list(shape):
        total *= max(1, int(dim))
    return int(total)


def collect_node_io_shapes(node, tensor_info):
    """Collect input/output shapes for an ONNX node."""
    in_shapes = []
    out_shapes = []
    for name in node.input:
        info = tensor_info.get(name)
        if info:
            in_shapes.append(_shape_list(info[0]))
    for name in node.output:
        info = tensor_info.get(name)
        if info:
            out_shapes.append(_shape_list(info[0]))
    return in_shapes, out_shapes


def template_signature_for_node(node, in_shapes, out_shapes):
    """Build a reusable template signature from ONNX op type and shapes."""
    op_type = str(getattr(node, "op_type", "")).strip() or "Unknown"
    primary_out = out_shapes[0] if out_shapes else []

    if op_type in {"MatMul", "Gemm"} and len(in_shapes) >= 2:
        return {
            "template_kind": "tile_matmul",
            "signature": {
                "op_type": op_type,
                "lhs_shape": list(in_shapes[0]),
                "rhs_shape": list(in_shapes[1]),
                "out_shape": list(primary_out),
            },
        }

    if op_type in {"Add", "Sub", "Mul", "Div"}:
        rhs_shape = in_shapes[1] if len(in_shapes) >= 2 else []
        return {
            "template_kind": "tile_elementwise_binary",
            "signature": {
                "op_type": op_type,
                "rhs_shape": list(rhs_shape),
                "out_shape": list(primary_out),
            },
        }

    if op_type in {"Relu", "Sigmoid", "Tanh"}:
        return {
            "template_kind": "tile_elementwise_unary",
            "signature": {
                "op_type": op_type,
                "out_shape": list(primary_out),
            },
        }

    if op_type in {"Softmax", "LogSoftmax"}:
        return {
            "template_kind": "tile_softmax",
            "signature": {
                "op_type": op_type,
                "input_shape": list(in_shapes[0]) if in_shapes else list(primary_out),
                "out_shape": list(primary_out),
            },
        }

    if op_type in {"ReduceMean", "ReduceSum"}:
        return {
            "template_kind": "tile_reduce",
            "signature": {
                "op_type": op_type,
                "input_shape": list(in_shapes[0]) if in_shapes else list(primary_out),
                "out_shape": list(primary_out),
            },
        }

    return None


def _estimate_generic_profile(template_kind, signature):
    """Provide a lightweight explicit-memory estimate for non-PIM nodes if needed."""
    if template_kind == "tile_matmul":
        lhs_shape = _shape_list(signature.get("lhs_shape"))
        rhs_shape = _shape_list(signature.get("rhs_shape"))
        out_shape = _shape_list(signature.get("out_shape"))
        lhs_bytes = _shape_product(lhs_shape) * 4
        rhs_bytes = _shape_product(rhs_shape) * 4
        out_bytes = _shape_product(out_shape) * 4
        return {
            "supported": True,
            "template_kind": template_kind,
            "per_use_onchip_bytes": int(out_bytes),
            "per_use_offchip_bytes": int(lhs_bytes + rhs_bytes),
            "per_use_total_bytes": int(lhs_bytes + rhs_bytes + out_bytes),
            "per_use_memory_cycles": 0,
            "memory_cycles_breakdown": {},
            "peak_sram_bytes": int(out_bytes),
            "fits_sram": True,
        }

    out_shape = _shape_list(signature.get("out_shape"))
    out_bytes = _shape_product(out_shape) * 4
    return {
        "supported": True,
        "template_kind": template_kind,
        "per_use_onchip_bytes": int(out_bytes),
        "per_use_offchip_bytes": int(out_bytes),
        "per_use_total_bytes": int(out_bytes * 2),
        "per_use_memory_cycles": 0,
        "memory_cycles_breakdown": {},
        "peak_sram_bytes": int(out_bytes),
        "fits_sram": True,
    }


def run_onnx_analysis(
    model,
    tensor_info,
    est_cfg,
    cfg,
    node_to_device=None,
    memory_profile_name=None,
    memory_profile_cfg=None,
):
    """Run shape-driven reusable-template analysis for the current GCN flow."""
    templates = {}
    unsupported_nodes = []
    total_uses = 0
    node_profiles = {}
    by_device = {
        "CPU": {"onchip_bytes": 0, "offchip_bytes": 0, "supported_nodes": 0},
        "PIM": {"onchip_bytes": 0, "offchip_bytes": 0, "supported_nodes": 0},
        "IO": {"onchip_bytes": 0, "offchip_bytes": 0, "supported_nodes": 0},
    }
    partition_counts = {"CPU": 0, "PIM": 0, "IO": 0, "UNKNOWN": 0}
    partition_regions = {"CPU": [], "PIM": [], "IO": [], "UNKNOWN": []}

    for index, node in enumerate(model.graph.node):
        node_id = f"{node.op_type}_{index}"
        device = "UNKNOWN"
        if isinstance(node_to_device, dict):
            device = str(node_to_device.get(node_id, "UNKNOWN")).upper()
        if device not in partition_counts:
            device = "UNKNOWN"
        partition_counts[device] += 1
        partition_regions[device].append(node_id)

        in_shapes, out_shapes = collect_node_io_shapes(node, tensor_info)
        template = template_signature_for_node(node, in_shapes, out_shapes)
        if template is None:
            unsupported_nodes.append(node_id)
            continue

        key = json.dumps(
            {
                "assigned_device": device,
                "template_kind": template["template_kind"],
                "signature": template["signature"],
            },
            sort_keys=True,
        )
        entry = templates.setdefault(
            key,
            {
                "template_kind": template["template_kind"],
                "signature": template["signature"],
                "assigned_device": device,
                "use_count": 0,
                "node_ids": [],
            },
        )
        entry["use_count"] += 1
        entry["node_ids"].append(node_id)
        total_uses += 1

        if device == "PIM":
            device_opt = pick_device_entry(memory_profile_cfg, device)
            detail = search_pim_sram_tile_plan(
                template["template_kind"],
                template["signature"],
                cfg,
                est_cfg,
                device_opt=device_opt,
            )
        else:
            detail = _estimate_generic_profile(template["template_kind"], template["signature"])

        supported = bool(detail.get("supported", False))
        node_profiles[node_id] = {
            "supported": supported,
            "device": device,
            "template_kind": template["template_kind"],
            "source": "onnx_shape_rules",
            "onchip_bytes": int(max(0, detail.get("per_use_onchip_bytes", 0))),
            "offchip_bytes": int(max(0, detail.get("per_use_offchip_bytes", 0))),
            "total_bytes": int(max(0, detail.get("per_use_total_bytes", 0))),
            "memory_cycles": int(max(0, detail.get("per_use_memory_cycles", 0))),
            "pim_sram_tiling": detail if device == "PIM" else None,
            "peak_sram_bytes": int(max(0, detail.get("peak_sram_bytes", 0))),
            "fits_sram": bool(detail.get("fits_sram", False)),
            "reason": detail.get("reason"),
        }

        if supported and device in by_device:
            by_device[device]["onchip_bytes"] += int(max(0, detail.get("per_use_onchip_bytes", 0)))
            by_device[device]["offchip_bytes"] += int(max(0, detail.get("per_use_offchip_bytes", 0)))
            by_device[device]["supported_nodes"] += 1

    ordered_templates = sorted(
        templates.values(),
        key=lambda item: (
            -int(item["use_count"]),
            item["assigned_device"],
            item["template_kind"],
            json.dumps(item["signature"], sort_keys=True),
        ),
    )

    supported_template_count = 0
    for item in ordered_templates:
        if any(node_profiles.get(node_id, {}).get("supported", False) for node_id in item["node_ids"]):
            supported_template_count += 1

    return {
        "enabled": True,
        "analysis_mode": "onnx_template_reuse",
        "summary": (
            "Reuse ONNX shape templates for repeated operators and apply explicit 128KB PIM SRAM "
            "tiling to supported PIM nodes."
        ),
        "template_registry": {
            "template_count": int(len(ordered_templates)),
            "total_template_uses": int(total_uses),
            "unsupported_node_count": int(len(unsupported_nodes)),
            "unsupported_nodes": unsupported_nodes[:32],
            "templates": ordered_templates,
        },
        "node_memory": {
            "enabled": True,
            "analysis_source": "onnx_shape_rules + pim_sram_tiling",
            "profile_name": memory_profile_name,
            "supported_template_count": int(supported_template_count),
            "node_profile_count": int(len(node_profiles)),
            "total_onchip_bytes": int(sum(item["onchip_bytes"] for item in by_device.values())),
            "total_offchip_bytes": int(sum(item["offchip_bytes"] for item in by_device.values())),
            "by_device": by_device,
            "node_profiles": node_profiles,
        },
        "device_partition": {
            "method": "onnx_op_rules",
            "counts": partition_counts,
            "regions": partition_regions,
        },
    }

