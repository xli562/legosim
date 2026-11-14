"""Checks for misleading success cases in research result collection."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analyze_run import parse_coordinator, read_workload_validation


LOG = """
**** Round 1 Phase 1 ****
Start simulation process 10. Command: example
Simulation process 10 terminate with status = 0.
Benchmark elapses 1000 cycle.
**** Round 1 Phase 2 ***
Start simulation process 11. Command: network
Simulation process 11 terminate with status = 0.
**** Round 2 Phase 1 ****
Start simulation process 12. Command: example
Simulation process 12 terminate with status = 0.
Benchmark elapses 1001 cycle.
Quit simulation because simulation cycle has converged.
**** End of Simulation ****
Benchmark elapses 1001 cycle.
Simulation elapseds 0d 0h 1m 35s.
"""


class CoordinatorTests(unittest.TestCase):
    def test_final_summary_is_not_an_extra_round(self):
        result = parse_coordinator(LOG)
        self.assertEqual(result["status"], "converged")
        self.assertEqual(len(result["rounds"]), 2)
        self.assertEqual(result["wall_seconds_reported"], 95)
        self.assertAlmostEqual(result["rounds"][1]["relative_change_percent"], 100 / 1001)

    def test_round_limit_is_not_convergence(self):
        result = parse_coordinator(LOG.replace("Quit simulation because simulation cycle has converged.", ""))
        self.assertEqual(result["status"], "completed_without_convergence")

    def test_child_failure_despite_final_summary(self):
        with self.assertRaisesRegex(ValueError, "failed"):
            parse_coordinator(LOG.replace("status = 0", "status = 256", 1))

    def test_missing_child_status(self):
        with self.assertRaisesRegex(ValueError, "termination"):
            parse_coordinator(LOG.replace("Simulation process 11 terminate with status = 0.", ""))

    def test_truncated_or_combined_logs(self):
        for text in (LOG.split("**** End of Simulation ****")[0], LOG + LOG):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_coordinator(text)

    def test_inconsistent_final_cycles(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            parse_coordinator(LOG.replace("**** End of Simulation ****\nBenchmark elapses 1001", "**** End of Simulation ****\nBenchmark elapses 999"))

    def test_checked_workload_requires_pass_in_every_round(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.photonic.yml").write_text('cmd: "matmul_checked_c"\n')
            for round_id in (1, 2):
                log_dir = root / ("proc_r%d_p1_t3" % round_id)
                log_dir.mkdir()
                (log_dir / "sniper.0.0.log").write_text("MATMUL_CHECK PASS checksum=42\n")
            self.assertEqual(read_workload_validation(root, 2)["status"], "passed")
            (root / "proc_r2_p1_t3/sniper.0.0.log").write_text("No result\n")
            with self.assertRaisesRegex(ValueError, "Missing numerical check"):
                read_workload_validation(root, 2)

    def test_explicit_workload_failure_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            log_dir = root / "proc_r1_p1_t3"
            log_dir.mkdir()
            (log_dir / "sniper.0.0.log").write_text("MATMUL_CHECK FAIL index=0\n")
            with self.assertRaisesRegex(ValueError, "numerical check failed"):
                read_workload_validation(root, 1)


if __name__ == "__main__":
    unittest.main()
