"""Check that NPU timing comes from a matching, valid SCALE-Sim report."""
from pathlib import Path
from tempfile import TemporaryDirectory
import hashlib
import json
import unittest

from mesh_study import NPU_MODEL_SNAPSHOTS, verify_npu_model_files, verify_npu_timing
from npu_scalesim import DEFAULT_TOPOLOGY, check_tile, read_compute_cycles


class NpuScalesimTests(unittest.TestCase):
    def test_matching_tile_and_cycle_report(self):
        check_tile(DEFAULT_TOPOLOGY)
        with TemporaryDirectory() as directory:
            report = Path(directory) / "COMPUTE_REPORT.csv"
            report.write_text("LayerID, Total Cycles, Stall Cycles,\n0, 129, 0,\n")
            self.assertEqual(read_compute_cycles(report), 129)

    def test_invalid_or_unrelated_reports_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            topology = root / "other.csv"
            topology.write_text("Layer name, M, N, K,\nOther, 20, 10, 100,\n")
            with self.assertRaisesRegex(ValueError, "M=10"):
                check_tile(topology)
            report = root / "COMPUTE_REPORT.csv"
            report.write_text("LayerID, Total Cycles,\n0, NaN,\n")
            with self.assertRaisesRegex(ValueError, "positive integer"):
                read_compute_cycles(report)

    def test_saved_timing_inputs_are_checked(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"compute_cycles": 129, "gemm": {"m": 10, "n": 10, "k": 100}}
            for source_key, hash_key, snapshot_name in NPU_MODEL_SNAPSHOTS:
                source = root / snapshot_name
                content = {
                    "npu_scalesim_config.cfg": "[general]\nrun_name: test\n",
                    "npu_scalesim_topology.csv": "Layer name, M, N, K,\nTile, 10, 10, 100,\n",
                    "npu_scalesim_compute_report.csv": "LayerID, Total Cycles,\n0, 129,\n",
                }[snapshot_name]
                source.write_text(content)
                manifest[source_key] = str(source)
                manifest[hash_key] = hashlib.sha256(source.read_bytes()).hexdigest()
            timing = root / "npu_timing.json"
            timing.write_text(json.dumps(manifest))
            scenario = {"npu_compute_cycles": 129, "npu_timing": {
                "sha256": hashlib.sha256(timing.read_bytes()).hexdigest()}}
            self.assertEqual(verify_npu_timing(scenario, timing), manifest)
            self.assertEqual(len(verify_npu_model_files(manifest, root)), 3)
            (root / "npu_scalesim_config.cfg").write_text("changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_npu_model_files(manifest, root)
            report = root / "npu_scalesim_compute_report.csv"
            report.write_text("LayerID, Total Cycles,\n0, 128,\n")
            manifest["compute_report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
            config = root / "npu_scalesim_config.cfg"
            config.write_text("[general]\nrun_name: test\n")
            with self.assertRaisesRegex(ValueError, "cycles differ"):
                verify_npu_model_files(manifest, root)


if __name__ == "__main__":
    unittest.main()
