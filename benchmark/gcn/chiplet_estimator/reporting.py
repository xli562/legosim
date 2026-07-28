"""
Simulation report writers.
"""
import csv
import os


def write_sim_summary_csv(report_out, subgraph_plan):
    """
    Write a CSV summary for the subgraph execution plan.
    The table includes node count, ops, bytes, and cycle breakdown fields.
    """
    csv_out = os.path.join(os.path.dirname(report_out), "sim_summary.csv")
    with open(csv_out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "subgraph_id",
                "device",
                "nodes",
                "ops",
                "bytes",
                "compute_cycles",
                "memory_cycles",
                "onchip_memory_bytes",
                "offchip_memory_bytes",
                "onchip_memory_cycles",
                "offchip_memory_cycles",
                "cache_miss_count",
                "cache_miss_cycles",
                "pipeline_cycles",
                "start_cycle",
                "end_cycle",
                "total_cycles",
            ]
        )
        for subgraph in subgraph_plan:
            writer.writerow(
                [
                    subgraph["id"],
                    subgraph["device"],
                    len(subgraph["nodes"]),
                    subgraph["estimated_ops"],
                    subgraph["estimated_bytes"],
                    subgraph["compute_cycles"],
                    subgraph["memory_cycles"],
                    subgraph.get("onchip_memory_bytes", 0),
                    subgraph.get("offchip_memory_bytes", 0),
                    subgraph.get("onchip_memory_cycles", 0),
                    subgraph.get("offchip_memory_cycles", 0),
                    subgraph["cache_miss_count"],
                    subgraph["cache_miss_cycles"],
                    subgraph.get("pipeline_cycles", 0),
                    subgraph["start_cycle"],
                    subgraph["end_cycle"],
                    subgraph["total_cycles"],
                ]
            )
    return csv_out
