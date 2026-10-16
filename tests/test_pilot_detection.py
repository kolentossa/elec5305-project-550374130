"""Unit tests for sliding-window YAMNet inference, EMA smoothing, and interval grouping."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pilot_detection import find_events, infer_scene, smooth_scores


class MockWindowRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, window):
        self.calls.append(window.copy())
        return (
            np.full((6, 521), len(self.calls), dtype=np.float32)
            + np.arange(6, dtype=np.float32)[:, None] / 10
        )


class PilotDetectionTests(unittest.TestCase):
    def test_event_interval_grouping(self):
        events = find_events([0.0, 0.2, 0.8, 0.0, 0.3], 0.2, 2.2)
        self.assertEqual(events, [(0.48, 1.44), (1.92, 2.2)])

    def test_empty_detection(self):
        self.assertEqual(find_events([0.0, 0.1], 0.2, 0.96), [])

    def test_causal_ema(self):
        scores = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(
            smooth_scores(scores, 0.5),
            [[0.0, 1.0], [0.5, 0.5], [0.25, 0.25]],
        )

    def test_sliding_window_coverage(self):
        runner = MockWindowRunner()
        audio = np.arange(8 * 16000, dtype=np.float32)
        scores = infer_scene(runner, audio)
        self.assertEqual(scores.shape, (17, 521))
        self.assertEqual(len(runner.calls), 4)
        np.testing.assert_allclose(scores[:5, 0], [1, 1.1, 1.2, 1.3, 1.4])
        np.testing.assert_allclose(scores[-2:, 0], [4, 4.1])
        self.assertEqual(runner.calls[1][0, 0], audio[38400])
        self.assertTrue(np.all(runner.calls[-1][0, 12800:] == 0))


if __name__ == "__main__":
    unittest.main()
