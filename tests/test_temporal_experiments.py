"""Unit tests for 10-class taxonomy mapping, dual-threshold hysteresis, and SED boundary metrics."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from class_mapping import URBAN_SED_MAPPING, extract_class_scores
from temporal_experiments import detect_hysteresis, detect_threshold, evaluate_predictions


class TemporalExperimentTests(unittest.TestCase):
    def test_10class_mapping_completeness(self):
        expected_classes = {
            "air_conditioner",
            "car_horn",
            "children_playing",
            "dog_bark",
            "drilling",
            "engine_idling",
            "gun_shot",
            "jackhammer",
            "siren",
            "street_music",
        }
        self.assertEqual(set(URBAN_SED_MAPPING.keys()), expected_classes)
        dummy_scores = np.zeros((4, 521), dtype=np.float32)
        dummy_scores[:, 346] = 0.2
        dummy_scores[:, 337] = 0.6
        primary = extract_class_scores(dummy_scores, ["engine_idling"], aggregation="primary")
        hier = extract_class_scores(dummy_scores, ["engine_idling"], aggregation="max")
        np.testing.assert_allclose(primary[:, 0], 0.2)
        np.testing.assert_allclose(hier[:, 0], 0.6)

    def test_hysteresis_bridges_mid_event_dips(self):
        scores = np.array([0.0, 0.25, 0.09, 0.29, 0.08, 0.39, 0.01, 0.0], dtype=np.float32)
        raw_intervals = detect_threshold(scores, 0.20, 3.84)
        hyst_intervals = detect_hysteresis(scores, 0.20, 0.06, 3.84)
        self.assertEqual(len(raw_intervals), 3)
        self.assertEqual(hyst_intervals, [(0.48, 2.88)])

    def test_sed_fragmentation_and_delay_metrics(self):
        scenes = [
            {
                "name": "s1",
                "duration_s": 4.8,
                "events": [{"label": "engine_idling", "onset_s": 0.48, "offset_s": 2.88}],
            }
        ]
        pred_raw = {"s1": {"engine_idling": [(0.48, 0.96), (1.44, 1.92), (2.40, 2.88)]}}
        pred_hyst = {"s1": {"engine_idling": [(0.48, 2.88)]}}

        m_raw = evaluate_predictions(scenes, pred_raw, ["engine_idling"])
        m_hyst = evaluate_predictions(scenes, pred_hyst, ["engine_idling"])

        self.assertEqual(m_raw["fragmentation_per_event"], 3.0)
        self.assertEqual(m_hyst["fragmentation_per_event"], 1.0)
        self.assertGreater(m_hyst["segment_f1"], m_raw["segment_f1"])
        self.assertAlmostEqual(m_hyst["mean_onset_delay_s"], 0.0)
        self.assertAlmostEqual(m_hyst["mean_offset_delay_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
