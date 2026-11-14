"""Checks topology and accounting assumptions that can change reported results."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from photonic_mesh_report import DEFAULT_ASSUMPTIONS, DEFAULT_TOPOLOGY, estimate, read_edges, read_trace


class PhotonicMeshTests(unittest.TestCase):
    def test_two_hop_trace_includes_header_and_static_interval(self):
        edges, adjacency = read_edges(DEFAULT_TOPOLOGY)
        params = json.loads(DEFAULT_ASSUMPTIONS.read_text())
        with TemporaryDirectory() as directory:
            trace = Path(directory) / "bench.txt"
            trace.write_text("10 0 0 3 3 0\n")
            transfers = read_trace(trace, adjacency, 128)
        result = estimate({"status": "converged", "final_cycles": 1000}, transfers, edges, params)
        self.assertEqual(transfers[0]["hops"], 2)
        self.assertEqual(result["number_of_directed_optical_links"], 8)
        self.assertEqual(result["encoded_payload_bytes_upper_bound"], 32)
        self.assertEqual(result["transmitted_bit_hops_including_headers"], 768)
        self.assertAlmostEqual(result["dynamic_optical_link_energy_j"], 768 * 1.5e-12)
        self.assertAlmostEqual(result["static_optical_link_energy_j"], 0.0096 * 1e-6)
        self.assertAlmostEqual(result["total_optical_link_area_mm2"], 0.56)
        self.assertEqual(result["transmitted_bit_router_visits_including_headers"], 1152)
        self.assertAlmostEqual(result["router_dynamic_energy_j"], 1152 * 4e-13)
        self.assertAlmostEqual(result["router_static_energy_j"], 0.002 * 1e-6)
        self.assertAlmostEqual(result["total_noc_area_mm2"], 0.72)

    def test_missing_link_fails_instead_of_under_counting_energy(self):
        with self.assertRaisesRegex(ValueError, "Disconnected"):
            with TemporaryDirectory() as directory:
                trace = Path(directory) / "bench.txt"
                trace.write_text("10 0 0 3 3 0\n")
                read_trace(trace, {0: {1}, 1: {0}, 2: {3}, 3: {2}}, 128)


if __name__ == "__main__":
    unittest.main()
