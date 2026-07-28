"""Command-line argument parsing for the ONNX-driven chiplet estimator."""

import argparse

from .common import CPU_OPS_DEFAULT, IO_OPS_DEFAULT, PIM_OPS_DEFAULT


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ONNX partition + chiplet roofline timing + PopNet bench generation"
    )
    parser.add_argument("--onnx", default=None, help="ONNX model path")
    parser.add_argument("--config", default="chiplet_config.json", help="Chiplet config JSON")
    parser.add_argument("--partition-info", default="partition_info.py", help="Output partition info")
    parser.add_argument("--report", default="reports/execution_plan.json", help="Output report JSON")
    parser.add_argument("--bench", default="bench.txt", help="Output inter-chiplet PopNet bench trace")
    parser.add_argument(
        "--bench-intra",
        default="bench_pim_intra.txt",
        help="Output PIM intra-chiplet PopNet bench trace",
    )
    parser.add_argument(
        "--no-inter-bench",
        action="store_true",
        help="Do not generate or overwrite inter-chiplet bench.txt",
    )
    parser.add_argument(
        "--pim-node-offset",
        type=int,
        default=4,
        help="Global router-id offset for PIM intra nodes",
    )
    parser.add_argument(
        "--pim-mapping-report",
        default="reports/pim_mapping.json",
        help="Output PIM NPU mapping report",
    )
    parser.add_argument(
        "--pim-pe-workload",
        default="reports/pim_pe_workload.txt",
        help="Output per-PE workload summary for the PIM backend",
    )
    parser.add_argument(
        "--runtime-plan-pim",
        default="reports/runtime_plan_pim.txt",
        help="Output phase1 runtime event plan for PIM",
    )
    parser.add_argument(
        "--runtime-plan-io",
        default="reports/runtime_plan_io.txt",
        help="Output phase1 runtime event plan for IO",
    )
    parser.add_argument("--partition-only", action="store_true", help="Only emit partition_info.py")
    parser.add_argument(
        "--memory-profile",
        default=None,
        help="Named memory-optimization profile in chiplet_config.json",
    )
    parser.add_argument("--min-subgraph-size", type=int, default=2, help="Merge tiny contiguous device runs")
    parser.add_argument("--cpu-ops", default=CPU_OPS_DEFAULT, help="Comma-separated op types for CPU")
    parser.add_argument("--pim-ops", default=PIM_OPS_DEFAULT, help="Comma-separated op types for PIM")
    parser.add_argument("--io-ops", default=IO_OPS_DEFAULT, help="Comma-separated op types for IO")
    parser.add_argument(
        "--enable-gem5-calibration",
        action="store_true",
        help="Enable gem5 stats-based calibration for roofline parameters",
    )
    parser.add_argument(
        "--gem5-stats-root",
        default=None,
        help="Root directory used for auto-finding gem5 m5out/stats.txt",
    )
    parser.add_argument("--gem5-stats-cpu", default=None, help="Explicit CPU gem5 stats.txt path")
    parser.add_argument("--gem5-stats-io", default=None, help="Explicit IO gem5 stats.txt path")
    parser.add_argument(
        "--gem5-cpu-cmd",
        default=None,
        help="Optional shell command to run CPU gem5 before parsing stats",
    )
    parser.add_argument(
        "--gem5-io-cmd",
        default=None,
        help="Optional shell command to run IO gem5 before parsing stats",
    )
    parser.add_argument(
        "--interchiplet-cmd",
        default=None,
        help="Optional interchiplet command used to launch gem5 CPU/IO simulation before calibration",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logs")
    return parser.parse_args()
