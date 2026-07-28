#!/usr/bin/env python3
"""Generate a lightweight 2-layer GCN ONNX model for chiplet timing estimation."""

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_gcn_onnx(out_path, num_nodes, in_features, hidden_features, num_classes, opset, seed):
    rng = np.random.default_rng(seed)

    w0 = (rng.standard_normal((in_features, hidden_features), dtype=np.float32) * 0.05).astype(np.float32)
    b0 = np.zeros((hidden_features,), dtype=np.float32)
    w1 = (rng.standard_normal((hidden_features, num_classes), dtype=np.float32) * 0.05).astype(np.float32)
    b1 = np.zeros((num_classes,), dtype=np.float32)

    nodes = [
        helper.make_node("MatMul", ["Adj", "X"], ["AX0"], name="gcn_adj_x"),
        helper.make_node("MatMul", ["AX0", "W0"], ["Z0"], name="gcn_fc0"),
        helper.make_node("Add", ["Z0", "B0"], ["Z0B"], name="gcn_bias0"),
        helper.make_node("Relu", ["Z0B"], ["H0"], name="gcn_relu0"),
        helper.make_node("MatMul", ["Adj", "H0"], ["AX1"], name="gcn_adj_h"),
        helper.make_node("MatMul", ["AX1", "W1"], ["Z1"], name="gcn_fc1"),
        helper.make_node("Add", ["Z1", "B1"], ["Logits"], name="gcn_bias1"),
        helper.make_node("Softmax", ["Logits"], ["Prob"], axis=1, name="gcn_softmax"),
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="gcn_two_layer",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [num_nodes, in_features]),
            helper.make_tensor_value_info("Adj", TensorProto.FLOAT, [num_nodes, num_nodes]),
        ],
        outputs=[helper.make_tensor_value_info("Prob", TensorProto.FLOAT, [num_nodes, num_classes])],
        initializer=[
            numpy_helper.from_array(w0, name="W0"),
            numpy_helper.from_array(b0, name="B0"),
            numpy_helper.from_array(w1, name="W1"),
            numpy_helper.from_array(b1, name="B1"),
        ],
        value_info=[
            helper.make_tensor_value_info("AX0", TensorProto.FLOAT, [num_nodes, in_features]),
            helper.make_tensor_value_info("H0", TensorProto.FLOAT, [num_nodes, hidden_features]),
            helper.make_tensor_value_info("AX1", TensorProto.FLOAT, [num_nodes, hidden_features]),
            helper.make_tensor_value_info("Logits", TensorProto.FLOAT, [num_nodes, num_classes]),
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="gcn_builder",
        opset_imports=[helper.make_operatorsetid("", int(opset))],
    )
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    onnx.save(model, out_path)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Build a sample 2-layer GCN ONNX model.")
    parser.add_argument("--out", default="gcn_3chiplet.onnx", help="Output ONNX path")
    parser.add_argument("--num-nodes", type=int, default=2708, help="Graph node count")
    parser.add_argument("--in-features", type=int, default=1433, help="Input feature width")
    parser.add_argument("--hidden-features", type=int, default=128, help="Hidden feature width")
    parser.add_argument("--num-classes", type=int, default=7, help="Output class count")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    out = build_gcn_onnx(
        out_path=args.out,
        num_nodes=args.num_nodes,
        in_features=args.in_features,
        hidden_features=args.hidden_features,
        num_classes=args.num_classes,
        opset=args.opset,
        seed=args.seed,
    )
    print(f"[GCN-ONNX] model saved: {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

