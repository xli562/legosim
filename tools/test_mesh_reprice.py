"""Guard the boundary between cost-only changes and rerun-required changes."""
import copy
import json
import unittest

from mesh_reprice import timing_signature
from mesh_study import ASSUMPTIONS, check_assumptions


class RepriceSafetyTests(unittest.TestCase):
    def setUp(self):
        self.original = json.loads(ASSUMPTIONS.read_text())

    def test_cost_changes_keep_timing_signature(self):
        changed = copy.deepcopy(self.original)
        changed["optical"]["static_power_w_per_direction"] *= 0.01
        changed["optical"]["dynamic_energy_j_per_transmitted_bit_hop"] *= 2
        changed["optical"]["interface_area_mm2_per_direction"] *= 0.5
        changed["router"]["area_mm2_per_router"] *= 2
        self.assertEqual(timing_signature(changed), timing_signature(self.original))

    def test_timing_geometry_and_bandwidth_changes_require_rerun(self):
        changes = (
            ("optical", "edge_delay_cycles", 5),
            ("optical", "length_mm", 20),
            ("optical", "bit_rate_per_wavelength_bps", 8e9),
            ("electrical", "edge_delay_cycles", 10),
            ("electrical", "length_mm", 20),
        )
        for kind, field, value in changes:
            with self.subTest(kind=kind, field=field):
                changed = copy.deepcopy(self.original)
                changed[kind][field] = value
                self.assertNotEqual(timing_signature(changed), timing_signature(self.original))
        wider = copy.deepcopy(self.original)
        wider["coordinate_width"] = 3
        self.assertNotEqual(timing_signature(wider), timing_signature(self.original))

    def test_nonfinite_cost_rejected(self):
        changed = copy.deepcopy(self.original)
        changed["optical"]["static_power_w_per_direction"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid optical cost"):
            check_assumptions(changed)


if __name__ == "__main__":
    unittest.main()
