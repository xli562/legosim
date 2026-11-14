"""Check generated mesh edges and targeted E/P overrides."""
import unittest

from generate_mesh_scenario import make_scenario


class GenerateMeshScenarioTests(unittest.TestCase):
    def test_vertical_optical_three_by_three(self):
        spec = make_scenario(3, "vertical-optical", "vertical_optical_3x3")
        self.assertEqual(len(spec["edges"]), 12)
        self.assertEqual(spec["edges"]["0-3"], "optical")
        self.assertEqual(spec["edges"]["0-1"], "electrical")

    def test_custom_edge_override(self):
        spec = make_scenario(4, "all-electrical", "custom_4x4",
                             ("0-1=optical", "6-10=optical"))
        self.assertEqual(len(spec["edges"]), 24)
        self.assertEqual(sum(kind == "optical" for kind in spec["edges"].values()), 2)
        with self.assertRaisesRegex(ValueError, "Invalid edge override"):
            make_scenario(4, "all-electrical", "custom_4x4", ("0-15=optical",))

    def test_npu_worker_requires_explicit_latency(self):
        spec = make_scenario(3, "all-optical", "all_optical_3x3_npu",
                             worker3_kind="npu", npu_compute_cycles=128)
        self.assertEqual(spec["worker3_kind"], "npu")
        self.assertEqual(spec["npu_compute_cycles"], 128)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            make_scenario(3, "all-optical", "bad", worker3_kind="npu")

    def test_custom_placements_are_saved_and_checked(self):
        positions = {"cpu": [1, 1], "worker1": [0, 0],
                     "worker2": [2, 0], "worker3": [2, 2]}
        spec = make_scenario(3, "all-optical", "center_cpu", placements=positions)
        self.assertEqual(spec["placements"], positions)
        positions["worker3"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "distinct"):
            make_scenario(3, "all-optical", "bad", placements=positions)


if __name__ == "__main__":
    unittest.main()
