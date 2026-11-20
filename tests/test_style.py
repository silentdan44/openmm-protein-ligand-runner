import unittest

import numpy as np

from prot_lig_runner.report.style import moving_average


class MovingAverageTests(unittest.TestCase):
    def test_constant_series_stays_constant_at_edges(self):
        values = np.full(40, 3.0)
        smoothed = moving_average(values, 7)
        self.assertTrue(np.allclose(smoothed, 3.0))

    def test_edges_are_not_pulled_towards_zero(self):
        values = np.linspace(10.0, 20.0, 30)
        smoothed = moving_average(values, 9)
        self.assertGreater(smoothed[0], 10.0)
        self.assertLess(smoothed[-1], 20.0)
        self.assertGreater(smoothed[0], 10.5)
        self.assertLess(smoothed[-1], 19.5)

    def test_nan_is_ignored(self):
        values = np.array([1.0, np.nan, 1.0, 1.0, 1.0])
        smoothed = moving_average(values, 3)
        self.assertTrue(np.allclose(smoothed[[0, 2, 3, 4]], 1.0))


if __name__ == "__main__":
    unittest.main()
