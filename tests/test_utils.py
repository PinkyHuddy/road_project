"""Regression tests for the reusable closure-labeling helpers."""

import unittest

import numpy as np
import pandas as pd

from main.utils import (
    apply_closure_to_weather,
    build_closure_intervals,
    make_future_road_status_target,
)


class ClosureIntervalTests(unittest.TestCase):
    def test_missing_reopening_time_remains_censored(self):
        records = pd.DataFrame(
            {
                "closure_time": ["2025-01-01T03:00:00Z"],
                "reopening_time": [None],
            }
        )

        result = build_closure_intervals(records)

        self.assertFalse(bool(result.loc[0, "has_reopening_time"]))
        self.assertTrue(pd.isna(result.loc[0, "closure_end"]))

        weather = pd.DataFrame(
            {
                "datetime": pd.date_range(
                    "2025-01-01", periods=6, freq="h", tz="UTC"
                )
            }
        )
        labeled = apply_closure_to_weather(weather, result, ambiguous_hours=3)
        self.assertEqual(labeled.loc[3, "closure"], 1)
        self.assertTrue(np.isnan(labeled.loc[4, "closure"]))

    def test_confirmed_closure_overrides_ambiguous_window(self):
        hours = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
        weather = pd.DataFrame({"datetime": hours})
        intervals = pd.DataFrame(
            {
                "closure_start": [hours[1], hours[3]],
                "closure_end": [pd.NaT, hours[5]],
                "has_reopening_time": [False, True],
            }
        )

        result = apply_closure_to_weather(weather, intervals, ambiguous_hours=6)

        self.assertEqual(result.loc[1, "closure"], 1)
        self.assertTrue(np.isnan(result.loc[2, "closure"]))
        self.assertTrue((result.loc[3:5, "closure"] == 1).all())


class FutureTargetTests(unittest.TestCase):
    def test_target_requires_exact_complete_future_window(self):
        complete = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC"),
                "closure": [0, 0, 1, 0],
            }
        )

        result = make_future_road_status_target(complete, horizon_hours=2)

        self.assertEqual(result.loc[0, "road_closed_within_2h"], 1)
        self.assertEqual(result.loc[1, "road_closed_within_2h"], 1)
        self.assertTrue(np.isnan(result.loc[2, "road_closed_within_2h"]))

    def test_gap_makes_negative_target_unknown(self):
        gapped = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2025-01-01T00:00:00Z", "2025-01-01T02:00:00Z"]
                ),
                "closure": [0, 0],
            }
        )

        result = make_future_road_status_target(gapped, horizon_hours=1)

        self.assertTrue(np.isnan(result.loc[0, "road_closed_within_1h"]))


if __name__ == "__main__":
    unittest.main()
