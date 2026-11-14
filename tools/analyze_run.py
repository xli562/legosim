#!/usr/bin/env python3
"""Export a LEGOSim coordinator log to JSON/CSV and optionally plot convergence.

The exports use only Python's standard library. --plot additionally needs
matplotlib. This reader does not convert simulator cycles to physical seconds.
"""

import argparse
import csv
import json
from pathlib import Path
import re


def parse_coordinator(text):
    """Reject incomplete/failed runs; keep convergence separate from completion."""
    starts = re.findall(r"Start simulation process (\d+)\.", text)
    exits = re.findall(r"Simulation process (\d+) terminate with status = (\d+)", text)
    if not starts or sorted(starts) != sorted(pid for pid, _ in exits):
        raise ValueError("Missing child-process start/termination records")
    failed = [(pid, status) for pid, status in exits if int(status) != 0]
    if failed:
        raise ValueError("Child processes failed (PID, wait status): %s" % failed)
    if text.count("**** End of Simulation ****") != 1:
        raise ValueError("Expected exactly one completed run in coordinator.log")

    rounds = []
    current_round = None
    final_cycles = None
    ended = False
    converged = False
    for line in text.splitlines():
        phase = re.search(r"\*+ Round (\d+) Phase (\d+) \*+", line)
        if phase:
            current_round = int(phase[1]) if phase[2] == "1" else None
        if "simulation cycle has converged" in line:
            converged = True
        if "**** End of Simulation ****" in line:
            ended = True
        cycles = re.search(r"Benchmark elapses (\d+) cycle", line)
        if cycles:
            value = int(cycles[1])
            if value <= 0:
                raise ValueError("Nonpositive simulated cycle count")
            if ended:
                final_cycles = value
            elif current_round is not None:
                previous = rounds[-1]["cycles"] if rounds else None
                rounds.append({
                    "round": current_round,
                    "cycles": value,
                    "relative_change_percent": (
                        100 * abs(value - previous) / value if previous is not None else None
                    ),
                })
    if not rounds or final_cycles != rounds[-1]["cycles"]:
        raise ValueError("Missing or inconsistent final cycle summary")
    if [row["round"] for row in rounds] != list(range(1, len(rounds) + 1)):
        raise ValueError("Missing or duplicate round cycle records")
    elapsed = re.search(r"Simulation elapseds (\d+)d (\d+)h (\d+)m (\d+)s", text)
    if not elapsed:
        raise ValueError("Missing wall-clock duration summary")
    days, hours, minutes, seconds = map(int, elapsed.groups())
    return {
        "status": "converged" if converged else "completed_without_convergence",
        "final_cycles": final_cycles,
        "wall_seconds_reported": days * 86400 + hours * 3600 + minutes * 60 + seconds,
        "successful_child_processes": len(exits),
        "rounds": rounds,
    }


def read_network_reports(run_dir):
    """Read Popnet summaries without assigning unverified SI units to power."""
    reports = []
    for path in sorted(run_dir.glob("proc_r*_p2_t*/*.log")):
        text = path.read_text(errors="replace")
        latency = re.search(r"^average Delay:\s+(\S+)", text, re.MULTILINE)
        finished = re.search(r"^total finished:\s+(\d+)", text, re.MULTILINE)
        if not latency or not finished:
            continue
        round_id = int(re.search(r"proc_r(\d+)_p2", path.parent.name)[1])
        reports.append({
            "round": round_id,
            "log": str(path.relative_to(run_dir)),
            "finished_packets": int(finished[1]),
            "average_packet_latency_network_cycles": float(latency[1]),
            "power_as_reported_unvalidated": {
                name.strip(): float(value)
                for name, value in re.findall(
                    r"^(total(?: \w+)? power):\s+(\S+)", text, re.MULTILINE
                )
            },
        })
    return sorted(reports, key=lambda report: (report["round"], report["log"]))


def read_workload_validation(run_dir, round_count):
    """Find the optional checked CPU's per-round numerical result in Sniper logs."""
    inputs = list(run_dir.glob("input*.yml"))
    required = any("matmul_checked_c" in path.read_text() for path in inputs)
    checks = []
    for path in sorted(run_dir.glob("proc_r*_p1_t*/sniper*.log")):
        text = path.read_text(errors="replace")
        if "MATMUL_CHECK FAIL" in text:
            raise ValueError("Workload numerical check failed in %s" % path)
        markers = re.findall(r"MATMUL_CHECK PASS checksum=(\d+)", text)
        if not markers:
            continue
        if len(markers) != 1:
            raise ValueError("Expected one numerical check in %s" % path)
        checks.append({
            "round": int(re.search(r"proc_r(\d+)_p1", path.parent.name)[1]),
            "checksum": int(markers[0]),
            "log": str(path.relative_to(run_dir)),
        })
    if required or checks:
        if sorted(check["round"] for check in checks) != list(range(1, round_count + 1)):
            raise ValueError("Missing numerical check in one or more phase-1 rounds")
        return {"status": "passed", "round_checks": checks}
    return {"status": "not_requested", "round_checks": []}


def plot_rounds(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = summary["rounds"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["round"] for row in rows], [row["cycles"] for row in rows],
            "o-", color="#176b87")
    for row in rows:
        ax.annotate("{:,}".format(row["cycles"]), (row["round"], row["cycles"]),
                    xytext=(0, 10), textcoords="offset points", ha="center")
    ax.set(xlabel="Simulation round", ylabel="Workload completion (coordinator cycles)",
           title="LEGOSim iteration history — " + summary["status"].replace("_", " "))
    ax.set_xticks([row["round"] for row in rows])
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.margins(x=0.15, y=0.25)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=180)
    fig.savefig(output / "convergence.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing coordinator.log")
    parser.add_argument("--output", type=Path, help="Default: RUN_DIR/analysis")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    try:
        summary = parse_coordinator((args.run_dir / "coordinator.log").read_text())
        summary["network_reports"] = read_network_reports(args.run_dir)
        summary["workload_validation"] = read_workload_validation(args.run_dir, len(summary["rounds"]))
        summary["interpretation"] = {
            "cycles": "Coordinator and network cycles are separate clock domains; verify clock_rate.",
            "network_reports": "Phase 2 may end one round before the converged phase 1 result.",
            "power": "Raw Popnet/ORION report values; technology, units and time normalization need validation.",
            "scope": (
                "The checked benchmark validated its reduced 10x10 tile; physical network accuracy remains unverified."
                if summary["workload_validation"]["status"] == "passed" else
                "Process completion and convergence do not prove workload correctness or physical accuracy."
            ),
        }
        output = args.output or args.run_dir / "analysis"
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        with (output / "rounds.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["round", "cycles", "relative_change_percent"])
            writer.writeheader()
            writer.writerows(summary["rounds"])
        if args.plot:
            plot_rounds(summary, output)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, "Analysis failed: %s\n" % error)
    print("%s: %s cycles across %s rounds; outputs: %s" % (
        summary["status"], summary["final_cycles"], len(summary["rounds"]), output))


if __name__ == "__main__":
    main()
