"""Check recording-level counts with false positives and missed labels."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_pilot import evaluate_pilot


class PilotMetricTests(unittest.TestCase):
    def test_multilabel_counts_ignore_repeated_intervals(self):
        manifest = {"scenes": [
            {"name": "mixed", "source_placements": [
                {"label": "dog_bark"}, {"label": "dog_bark"}, {"label": "siren"},
            ]},
            {"name": "silence", "source_placements": []},
        ]}
        run = {
            "backend": "Test", "threshold": 0.2, "ema_alpha": 0.5,
            "scenes": [
                {"scene": "mixed", "events": {
                    "raw": {"dog_bark": [[0, 1], [2, 3]], "siren": [], "car_horn": [[3, 4]]},
                    "ema": {"dog_bark": [[0, 3]], "siren": [[3, 4]], "car_horn": []},
                }},
                {"scene": "silence", "events": {
                    "raw": {"dog_bark": [], "siren": [], "car_horn": []},
                    "ema": {"dog_bark": [], "siren": [], "car_horn": [[0, 1]]},
                }},
            ],
        }

        metrics = evaluate_pilot(manifest, run)

        self.assertEqual(metrics["methods"]["raw"],
                         {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5})
        self.assertEqual(metrics["methods"]["ema"],
                         {"tp": 2, "fp": 1, "fn": 0, "precision": 2 / 3, "recall": 1.0})
        self.assertEqual(metrics["scenes"][0]["expected"], ["dog_bark", "siren"])
        self.assertEqual(metrics["scenes"][0]["raw"]["predicted"], ["car_horn", "dog_bark"])
        self.assertEqual(metrics["scene_label_pairs"], 6)

    def test_empty_silence_metrics_are_undefined(self):
        manifest = {"scenes": [{"name": "silence", "source_placements": []}]}
        run = {
            "backend": "Test", "threshold": 0.2, "ema_alpha": 0.5,
            "scenes": [{"scene": "silence", "events": {
                "raw": {"dog_bark": [], "siren": [], "car_horn": []},
                "ema": {"dog_bark": [], "siren": [], "car_horn": []},
            }}],
        }

        metrics = evaluate_pilot(manifest, run)

        for method in ["raw", "ema"]:
            self.assertEqual(metrics["methods"][method],
                             {"tp": 0, "fp": 0, "fn": 0, "precision": None, "recall": None})


if __name__ == "__main__":
    unittest.main()
