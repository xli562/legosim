"""Checks that mixed-link routing and cost accounting stay comparable."""
import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mesh_study import (ASSUMPTIONS, BENCHMARK, EDGE_COST_FIELDS, ROOT, TEMPLATE,
                        calculate, check_assumptions, graph_text, load_spec, mesh_edges,
                        read_scenario_trace, render_yaml, resolve_placements,
                        shortest_route, validate_workload_traffic,
                        verify_edge_profile_sources)


class MeshStudyTests(unittest.TestCase):
    def setUp(self):
        self.params = json.loads(ASSUMPTIONS.read_text())

    def test_all_optical_matches_existing_link_accounting(self):
        spec = load_spec(BENCHMARK / "scenarios/all_optical.json")
        score, mix, route = shortest_route(0, 3, spec, self.params)
        self.assertEqual(score, 23)
        self.assertEqual(mix, (2, 0))
        self.assertEqual(len(route), 3)
        self.assertIn('0--1 [weight=4, id="optical"]', graph_text(spec, self.params))
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            trace = run_dir / "bench.txt"
            trace.write_text("10 0 0 3 3 0\n")
            transfers = read_scenario_trace(trace, spec, self.params)
            result = calculate(
                {"final_cycles": 1000, "status": "converged", "run_dir": run_dir,
                 "rounds": [{"round": 1}]},
                transfers, spec, self.params,
                [{"round": 1, "average_packet_latency_network_cycles": 12.0}],
            )
        self.assertEqual(result["directed_link_count"], {"optical": 8, "electrical": 0})
        self.assertEqual(result["dynamic_link_energy_j"]["optical"], 768 * 1.5e-12)
        self.assertAlmostEqual(result["total_noc_area_mm2"], 0.72)

    def test_vertical_mixed_path_and_tie_safety(self):
        spec = load_spec(BENCHMARK / "scenarios/vertical_optical.json")
        score, mix, _ = shortest_route(0, 3, spec, self.params)
        self.assertEqual((score, mix), (27, (1, 1)))
        ambiguous = copy.deepcopy(spec)
        ambiguous["edges"] = {"0-1": "optical", "0-2": "electrical",
                              "1-3": "optical", "2-3": "electrical"}
        equal_delay = copy.deepcopy(self.params)
        equal_delay["electrical"]["edge_delay_cycles"] = 4
        with self.assertRaisesRegex(ValueError, "different E/P mixes"):
            shortest_route(0, 3, ambiguous, equal_delay)

    def test_bandwidth_mismatch_rejected(self):
        bad = copy.deepcopy(self.params)
        bad["electrical"]["assumed_bit_rate_bps_per_direction"] = 64e9
        with self.assertRaisesRegex(ValueError, "Electrical line rate"):
            check_assumptions(bad)

    def test_three_by_three_corner_mapping_and_route(self):
        spec = load_spec(BENCHMARK / "scenarios/all_optical_3x3.json")
        params = copy.deepcopy(self.params)
        params["coordinate_width"] = 3
        check_assumptions(params)
        self.assertEqual(len(mesh_edges(3)), 12)
        self.assertEqual(shortest_route(0, 8, spec, params)[:2], (41, (4, 0)))
        self.assertIn('7--8 [weight=4, id="optical"]', graph_text(spec, params))
        yaml = render_yaml(TEMPLATE.read_text(), ROOT / "run_outputs/test_only", 3)
        self.assertIn('args: ["-A", "9"', yaml)
        self.assertIn('matmul_checked_c", "0", "0", "0", "2", "2", "0", "2", "2"]', yaml)
        self.assertIn('args: ["2", "2"]', yaml)

    def test_three_by_three_mixed_link_hops(self):
        spec = load_spec(BENCHMARK / "scenarios/vertical_optical_3x3.json")
        params = copy.deepcopy(self.params)
        params["coordinate_width"] = 3
        self.assertEqual(shortest_route(0, 6, spec, params)[:2], (23, (2, 0)))
        self.assertEqual(shortest_route(0, 2, spec, params)[:2], (31, (0, 2)))
        self.assertEqual(shortest_route(0, 8, spec, params)[:2], (49, (2, 2)))

    def test_npu_worker_replaces_only_third_gpu(self):
        spec = load_spec(BENCHMARK / "scenarios/all_optical_3x3_npu.json")
        yaml = render_yaml(TEMPLATE.read_text(), ROOT / "run_outputs/test_npu", 3,
                           spec["worker3_kind"], spec["npu_compute_cycles"])
        self.assertEqual(yaml.count('cmd: "$SIMULATOR_ROOT/benchmark/matmul_test/bin/matmul_cu"'), 2)
        self.assertEqual(yaml.count('cmd: "$SIMULATOR_ROOT/benchmark/matmul_test/bin/matmul_npu"'), 1)
        self.assertIn('args: ["2", "2", "0", "0", "128"]', yaml)

    def test_center_cpu_placement_updates_workers_and_trace(self):
        spec = load_spec(BENCHMARK / "scenarios/all_optical_3x3_npu_center_cpu.json")
        params = copy.deepcopy(self.params)
        params["coordinate_width"] = 3
        yaml = render_yaml(TEMPLATE.read_text(), ROOT / "run_outputs/test_center_cpu", 3,
                           "npu", 129, resolve_placements(spec))
        self.assertIn('args: ["0", "0", "1", "1"]', yaml)
        self.assertIn('args: ["2", "2", "1", "1", "129"]', yaml)
        self.assertIn('matmul_checked_c", "1", "1", "0", "0", "2", "0", "2", "2"]', yaml)
        with TemporaryDirectory() as directory:
            trace = Path(directory) / "bench.txt"
            trace.write_text("10 0 4 0 3 0\n")
            self.assertEqual(read_scenario_trace(trace, spec, params)[0]["source"], 4)
            trace.write_text("10 0 0 2 3 0\n")
            with self.assertRaisesRegex(ValueError, "placed CPU"):
                read_scenario_trace(trace, spec, params)

    def test_duplicate_placement_rejected(self):
        spec = load_spec(BENCHMARK / "scenarios/all_optical_3x3.json")
        spec["placements"] = {"cpu": [1, 1], "worker1": [1, 1],
                              "worker2": [2, 0], "worker3": [2, 2]}
        with self.assertRaisesRegex(ValueError, "distinct"):
            resolve_placements(spec)

    def test_edge_profile_changes_delay_and_per_edge_cost(self):
        spec = load_spec(BENCHMARK / "scenarios/all_optical_3x3_npu_short_edge01.json")
        params = json.loads((ROOT / "run_outputs/dsent_link_costs/mesh_link_assumptions_dsent.json").read_text())
        params["coordinate_width"] = 3
        self.assertIn('0--1 [weight=2, id="optical"]', graph_text(spec, params))
        self.assertEqual(shortest_route(0, 8, spec, params)[:2], (39, (4, 0)))
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            trace = run_dir / "bench.txt"
            trace.write_text("10 0 0 2 3 0\n")
            transfers = read_scenario_trace(trace, spec, params)
            metrics = calculate(
                {"final_cycles": 1000, "status": "converged", "run_dir": run_dir,
                 "rounds": [{"round": 1}]}, transfers, spec, params,
                [{"round": 1, "average_packet_latency_network_cycles": 12.0}])
        edge = metrics["edge_metrics"]["0-1"]
        self.assertEqual(edge["parameters"]["length_mm"], 5.0)
        self.assertEqual(edge["transmitted_bits_including_header"], 3 * 128)
        self.assertAlmostEqual(edge["dynamic_energy_j"],
                               3 * 128 * edge["parameters"]["dynamic_energy_j_per_transmitted_bit_hop"])
        self.assertAlmostEqual(metrics["total_noc_area_mm2"],
                               sum(value["area_mm2"] for value in metrics["edge_metrics"].values()) +
                               9 * params["router"]["area_mm2_per_router"])

    def test_equal_delay_routes_with_different_edge_energy_rejected(self):
        spec = load_spec(BENCHMARK / "scenarios/all_optical.json")
        profile = {field: self.params["optical"][field] for field in EDGE_COST_FIELDS}
        profile["dynamic_energy_j_per_transmitted_bit_hop"] *= 2
        spec["edge_parameters"] = {"0-1": profile}
        with self.assertRaisesRegex(ValueError, "dynamic link energy"):
            shortest_route(0, 3, spec, self.params)

    def test_saved_edge_profile_source_hash_is_checked(self):
        spec = load_spec(BENCHMARK / "scenarios/all_optical.json")
        profile = {field: self.params["optical"][field] for field in EDGE_COST_FIELDS}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "edge_profile_0_1.json"
            snapshot.write_text(json.dumps(profile))
            spec["edge_parameters"] = {"0-1": profile}
            spec["edge_profile_sources"] = {"0-1": {
                "path": "unused", "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest()}}
            self.assertEqual(verify_edge_profile_sources(spec, root),
                             {snapshot.name: snapshot})
            snapshot.write_text(json.dumps({**profile, "length_mm": 5}))
            with self.assertRaisesRegex(ValueError, "source hash differs"):
                verify_edge_profile_sources(spec, root)

    def test_checked_matmul_traffic_has_exact_useful_bytes(self):
        spec = load_spec(BENCHMARK / "scenarios/all_optical_3x3_npu_center_cpu.json")
        cpu = 4
        transfers = []
        for worker in (0, 2, 8):
            transfers += [{"source": cpu, "destination": worker,
                           "flits_including_header": 5001, "descriptor": 0}] * 2
            transfers.append({"source": worker, "destination": cpu,
                              "flits_including_header": 51, "descriptor": 0})
        self.assertEqual(validate_workload_traffic(transfers, spec, 128), 482400)
        with self.assertRaisesRegex(ValueError, "message sizes/counts"):
            validate_workload_traffic(transfers[:-1], spec, 128)


if __name__ == "__main__":
    unittest.main()
