import math
import os

from onnx import TensorProto


CPU_OPS_DEFAULT = "Softmax,LogSoftmax"
PIM_OPS_DEFAULT = (
    "MatMul,Gemm,Conv,Relu,Sigmoid,Tanh,Add,Sub,Mul,Div,"
    "BatchNormalization,ReduceMean,ReduceSum,"
    "Shape,Gather,Concat,LayerNorm"
)
IO_OPS_DEFAULT = "Transpose,Reshape,Unsqueeze,Squeeze,Split,Identity,Cast"


DTYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.INT8: 1,
    TensorProto.UINT8: 1,
    TensorProto.INT16: 2,
    TensorProto.UINT16: 2,
    TensorProto.INT32: 4,
    TensorProto.UINT32: 4,
    TensorProto.INT64: 8,
    TensorProto.UINT64: 8,
    TensorProto.BOOL: 1,
}


def to_abs(base_dir, path):
    """Resolve a path relative to base_dir."""
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def resolve_onnx_path(base_dir, user_path):
    """Resolve an ONNX path from user input or common benchmark defaults."""
    candidates = []
    if user_path:
        candidates.append(user_path)
    candidates.extend(
        [
            "gcn.onnx",
            "gcn_3chiplet.onnx",
            "gcn_complete.onnx",
            "gcn_graph.onnx",
            "gcn_full.onnx",
            "../gcn_v1/gcn_complete.onnx",
            "../gcn_v1.1/gcn_complete.onnx",
        ]
    )
    for path in candidates:
        abs_path = to_abs(base_dir, path)
        if os.path.exists(abs_path):
            return abs_path
    return None


def parse_ops(csv_ops):
    """Convert comma-separated op names into a set."""
    return {item.strip() for item in str(csv_ops).split(",") if item.strip()}


def parse_csv_set(value, fallback):
    """Normalize a string/list/set config entry into a set."""
    if isinstance(value, str) and value.strip():
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, (list, tuple, set)):
        out = {str(item).strip() for item in value if str(item).strip()}
        return out if out else set(fallback)
    return set(fallback)


def clamp_ratio(value):
    """Clamp a numeric ratio to [0.0, 1.0]."""
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, numeric))


def pick_device_entry(mapping, device):
    """Pick a per-device config entry using several key spellings."""
    if not isinstance(mapping, dict):
        return {}
    dev = str(device).strip()
    for key in (dev, dev.upper(), dev.lower()):
        if key in mapping and isinstance(mapping[key], dict):
            return mapping[key]
    return {}


def resolve_memory_profile(est_cfg, profile_name):
    """Resolve a named memory profile."""
    profiles = est_cfg.get("memory_profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}

    default_name = str(est_cfg.get("memory_profile_default", "baseline")).strip() or "baseline"
    selected_name = (str(profile_name).strip() if profile_name else default_name) or default_name

    selected_profile = profiles.get(selected_name)
    if not isinstance(selected_profile, dict):
        selected_profile = est_cfg.get("device_memory_optimization", {})
        if not isinstance(selected_profile, dict):
            selected_profile = {}
    return selected_name, selected_profile


def tensor_shape_from_value_info(value):
    """Extract tensor dims and dtype from ONNX ValueInfoProto."""
    ttype = value.type.tensor_type
    elem_type = ttype.elem_type
    dims = []
    for dim in ttype.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        else:
            dims.append(1)
    return dims, elem_type


def safe_numel(shape):
    """Safely compute the number of elements in a shape."""
    if not shape:
        return 1
    total = 1
    for dim in shape:
        total *= max(1, int(dim))
    return total


def tensor_nbytes(name, tensor_info):
    """Compute tensor bytes from tensor_info."""
    info = tensor_info.get(name)
    if not info:
        return 0
    shape, dtype = info
    return int(safe_numel(shape) * DTYPE_BYTES.get(dtype, 4))


def is_weight_initializer(op_type, nbytes, est_cfg):
    """Heuristically classify an initializer as a streamed weight tensor."""
    weighted_ops = parse_csv_set(
        est_cfg.get("weighted_ops"),
        {"MatMul", "Gemm", "Conv", "BatchNormalization"},
    )
    min_weight_bytes = int(est_cfg.get("min_weight_bytes", 64))
    return op_type in weighted_ops and int(nbytes) >= max(0, min_weight_bytes)


def estimate_streamed_weight_bytes(weight_bytes, est_cfg):
    """Estimate how many weight bytes should be treated as streamed traffic."""
    mode = str(est_cfg.get("weight_stream_mode", "ratio")).strip().lower()
    ratio = max(0.0, float(est_cfg.get("weight_stream_ratio", 0.25)))
    if mode == "ignore":
        return 0
    if mode == "full":
        return int(max(0, weight_bytes))
    return int(math.ceil(max(0, weight_bytes) * ratio))


def estimate_node_ops(node, in_shapes, out_shapes):
    """Estimate ops for a node, including PIM mixed-precision accounting."""
    op = node.op_type
    out_numel = sum(safe_numel(shape) for shape in out_shapes) if out_shapes else 1

    if op in {"MatMul", "Gemm"} and len(in_shapes) >= 2:
        lhs, rhs = in_shapes[0], in_shapes[1]
        if len(lhs) >= 2 and len(rhs) >= 2:
            m_dim = lhs[-2]
            k_dim = lhs[-1]
            n_dim = rhs[-1]
            mult_ops = m_dim * n_dim * k_dim
            acc_ops = m_dim * n_dim * max(0, k_dim - 1)
            return {
                "ops": max(1, mult_ops + acc_ops),
                "int8_ops": max(1, mult_ops),
                "int32_ops": max(0, acc_ops),
                "compute_kind": "fused_mac",
            }

    if op == "Conv" and len(in_shapes) >= 2 and out_shapes:
        weights = in_shapes[1]
        if len(weights) >= 4:
            cin = weights[1]
            kh = weights[2]
            kw = weights[3]
            mult_ops = out_numel * cin * kh * kw
            acc_ops = mult_ops
            return {
                "ops": max(1, mult_ops + acc_ops),
                "int8_ops": max(1, mult_ops),
                "int32_ops": max(1, acc_ops),
                "compute_kind": "fused_mac",
            }

    if op in {"Mul", "Div"}:
        return {
            "ops": max(1, out_numel),
            "int8_ops": max(1, out_numel),
            "int32_ops": 0,
            "compute_kind": "int8_only",
        }

    if op in {"Add", "Sub", "Relu", "BatchNormalization"}:
        return {
            "ops": max(1, out_numel),
            "int8_ops": 0,
            "int32_ops": max(1, out_numel),
            "compute_kind": "int32_only",
        }

    if op in {"ReduceMean", "ReduceSum"}:
        in_numel = sum(safe_numel(shape) for shape in in_shapes) if in_shapes else out_numel
        return {
            "ops": max(1, in_numel),
            "int8_ops": 0,
            "int32_ops": max(1, in_numel),
            "compute_kind": "int32_only",
        }

    if op == "Softmax":
        exp_ops = out_numel * 3
        sum_ops = out_numel * 2
        div_ops = out_numel
        return {
            "ops": max(1, out_numel * 5),
            "int8_ops": max(1, exp_ops + div_ops),
            "int32_ops": max(1, sum_ops),
            "compute_kind": "int32_only",
        }

    if op in {
        "Transpose",
        "Reshape",
        "Unsqueeze",
        "Squeeze",
        "Split",
        "Identity",
        "Cast",
        "Shape",
        "Gather",
        "Concat",
        "Sigmoid",
        "Tanh",
    }:
        return {
            "ops": max(1, out_numel // 8),
            "int8_ops": max(1, out_numel // 8),
            "int32_ops": 0,
            "compute_kind": "generic",
        }

    return {
        "ops": max(1, out_numel * 2),
        "int8_ops": max(1, out_numel),
        "int32_ops": max(1, out_numel),
        "compute_kind": "generic",
    }


def merge_numeric_dict(overrides, defaults):
    """Merge numeric overrides into a default dict."""
    out = dict(defaults)
    if not isinstance(overrides, dict):
        return out
    for key, value in overrides.items():
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def pim_tensor_bytes(numel, role, cfg):
    """Return PIM-role-specific tensor bytes."""
    pim = cfg.get("pim", {})
    if role == "weight":
        return int(numel * pim.get("weight_bytes", 1))
    if role == "activation":
        return int(numel * pim.get("activation_bytes", 1))
    if role == "output":
        return int(numel * pim.get("output_bytes", 4))
    if role == "accumulator":
        return int(numel * pim.get("accumulator_bytes", 4))
    return int(numel * pim.get("activation_bytes", 1))


def get_tensor_bytes_for_device(numel, dtype, role, cfg, device):
    """Return tensor bytes based on device and tensor role."""
    if str(device).upper() == "PIM":
        return pim_tensor_bytes(numel, role, cfg)
    return int(numel * DTYPE_BYTES.get(dtype, 4))
