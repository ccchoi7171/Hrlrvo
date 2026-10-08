import unittest

from layout import (
    INTERVAL_COUNT,
    OUTPUT_FRAMES,
    assembly_occupancy,
    body_plans,
    bridge_plans,
    validate_contract,
)


class LayoutTest(unittest.TestCase):
    def test_exact_output_coverage(self):
        validate_contract()
        occupancy = assembly_occupancy()
        self.assertEqual(len(occupancy), OUTPUT_FRAMES)
        self.assertEqual(set(occupancy), {1})

    def test_tail_and_operator(self):
        bridges = bridge_plans()
        bodies = body_plans()
        self.assertEqual(len(bridges), 15)
        self.assertEqual(len(bodies), INTERVAL_COUNT)
        self.assertEqual(bridges[-1].global_start, 1022)
        self.assertEqual(bridges[-1].pad_right, 37)
        self.assertTrue(all(row.global_stop - row.global_start == 81 for row in bodies))
        self.assertTrue(all(row.middle_global_stop - row.middle_global_start == 64 for row in bodies))


if __name__ == "__main__":
    unittest.main()
