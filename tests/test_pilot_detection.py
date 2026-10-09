"""Small checks for temporal decisions and coverage of the model windows."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pilot_detection import find_events, infer_scene, smooth_scores


class WindowSession:
    """Return identifiable rows so the retained window patches can be checked."""

    def __init__(self):
        self.calls = []

    def get_inputs(self):
        return [type("Input", (), {"name": "audio"})()]

    def run(self, names, inputs):
        self.calls.append(inputs["audio"].copy())
        return [np.full((6, 521), len(self.calls), dtype=np.float32)
                + np.arange(6, dtype=np.float32)[:, None] / 10]


class PilotTests(unittest.TestCase):
    def test_separate_events_and_open_tail(self):
        events = find_events([0.0, 0.2, 0.8, 0.0, 0.3], 0.2, 2.2)
        self.assertEqual(events, [(0.48, 1.44), (1.92, 2.2)])

    def test_no_event(self):
        self.assertEqual(find_events([0.0, 0.1], 0.2, 0.96), [])

    def test_smoothing_uses_previous_scores(self):
        scores = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_allclose(smooth_scores(scores, 0.5),
                                   [[0.0, 1.0], [0.5, 0.5], [0.25, 0.25]])

    def test_long_audio_retains_nonduplicated_patches_and_pads_tail(self):
        session = WindowSession()
        audio = np.arange(8 * 16000, dtype=np.float32)
        scores = infer_scene(session, audio)
        self.assertEqual(scores.shape, (17, 521))
        self.assertEqual(len(session.calls), 4)
        np.testing.assert_allclose(scores[:5, 0], [1, 1.1, 1.2, 1.3, 1.4])
        np.testing.assert_allclose(scores[-2:, 0], [4, 4.1])
        self.assertEqual(session.calls[1][0, 0], audio[38400])
        self.assertTrue(np.all(session.calls[-1][0, 12800:] == 0))


if __name__ == "__main__":
    unittest.main()
