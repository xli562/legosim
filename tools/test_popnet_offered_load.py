"""Check deterministic offered traffic and completion-window accounting."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from popnet_offered_load import (generate_trace, parse_completions,
                                summarize_seeds, summarize_window)


class OfferedLoadTests(unittest.TestCase):
    def test_same_random_stream_supports_matched_rate_sweeps(self):
        low = generate_trace(9, 0.1, 40, 4, 17)
        high = generate_trace(9, 0.4, 40, 4, 17)
        self.assertTrue(set(low) < set(high))
        self.assertTrue(all(source != destination for _, source, destination, _ in high))
        hotspot = generate_trace(9, 0.4, 40, 4, 17, "hotspot", 4)
        self.assertTrue(all(destination == 4 for _, source, destination, _ in hotspot
                            if source != 4))

    def test_tail_completion_records_define_delivery_window(self):
        packets = [(0, 0, 2, 4), (10, 1, 2, 4), (15, 2, 0, 4)]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "delayInfo.txt"
            path.write_text("0 0 2 8\n10 1 2 5\n15 2 0 8\n")
            completed = parse_completions(path, packets, 3)
            result = summarize_window(packets, completed, 10, 10, 3, 4)
            self.assertEqual(result["offered_packets_in_window"], 2)
            self.assertEqual(result["delivered_packets_in_window"], 1)
            self.assertAlmostEqual(result["delivered_flits_per_node_cycle"], 4 / 30)
            self.assertEqual(result["completed_packet_mean_latency_cycles"], 5)
            with self.assertRaisesRegex(ValueError, "finished count differs"):
                parse_completions(path, packets, 2)

    def test_seed_summary_keeps_mean_and_range_separate(self):
        rows = [
            {"scenario": "optical", "seed": 17, "requested_packets_per_node_cycle": 0.2,
             "offered_flits_per_node_cycle": 0.8,
             "delivered_flits_per_node_cycle": 0.7,
             "completed_packet_mean_latency_cycles": 30,
             "undelivered_packets_at_horizon": 10},
            {"scenario": "optical", "seed": 23, "requested_packets_per_node_cycle": 0.2,
             "offered_flits_per_node_cycle": 0.9,
             "delivered_flits_per_node_cycle": 0.75,
             "completed_packet_mean_latency_cycles": 40,
             "undelivered_packets_at_horizon": 20},
        ]
        summary = summarize_seeds(rows)[0]
        self.assertEqual(summary["seed_count"], 2)
        self.assertAlmostEqual(summary["delivered_flits_per_node_cycle_mean"], 0.725)
        self.assertEqual(summary["delivered_flits_per_node_cycle_min"], 0.7)
        self.assertEqual(summary["delivered_flits_per_node_cycle_max"], 0.75)
        self.assertEqual(summary["completed_packet_mean_latency_cycles_seed_mean"], 35)
        self.assertEqual(summary["undelivered_packets_at_horizon_mean"], 15)
        with self.assertRaisesRegex(ValueError, "Repeated seed"):
            summarize_seeds([rows[0], rows[0]])


if __name__ == "__main__":
    unittest.main()
