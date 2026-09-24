import unittest

import numpy as np
import pandas as pd

from src.p1.grid import build_grid, load_tokenizer
from src.p1.pool import (
    VISION_COLUMNS,
    copy_word_features_to_grid,
    pool_audio_lld,
    pool_vision_frames,
)


def words(*intervals):
    return [
        {
            "word_id": index,
            "word_text": f"word{index}",
            "t_s": start,
            "t_e": end,
            "confidence": confidence,
        }
        for index, (start, end, confidence) in enumerate(intervals)
    ]


class PoolTest(unittest.TestCase):
    def test_audio_uses_10ms_centers_and_half_open_interval(self) -> None:
        values = np.arange(5 * 25, dtype=np.float32).reshape(5, 25)
        result = pool_audio_lld(values, words((0.0, 0.02, 1.0), (0.02, 0.03, 1.0)))
        np.testing.assert_allclose(result["vectors"][0], values[:2].mean(axis=0))
        np.testing.assert_allclose(result["vectors"][1], values[2])
        np.testing.assert_array_equal(result["observed_mask"], [1, 1])
        self.assertEqual(result["missing_reason"], ["observed", "observed"])

    def test_audio_silent_zero_is_observed_but_nan_is_missing(self) -> None:
        values = np.zeros((3, 25), dtype=np.float32)
        values[2] = np.nan
        result = pool_audio_lld(values, words((0.0, 0.02, 1.0), (0.02, 0.03, 1.0)))
        np.testing.assert_array_equal(result["observed_mask"], [1, 0])
        self.assertEqual(result["missing_reason"], ["observed", "all_nan"])

    def test_vision_no_face_and_pts_half_open(self) -> None:
        values = np.ones((3, len(VISION_COLUMNS)), dtype=np.float32)
        values[1] = np.nan
        frame_features = {
            "values": values,
            "observed_mask": np.array([1, 0, 1], dtype=np.uint8),
            "pts_time": np.array([0.01, 0.02, 0.03]),
        }
        result = pool_vision_frames(frame_features, words((0.0, 0.02, 1.0), (0.02, 0.03, 1.0)))
        np.testing.assert_array_equal(result["observed_mask"], [1, 0])
        self.assertEqual(result["missing_reason"], ["observed", "no_face"])
        np.testing.assert_allclose(result["vectors"][0], np.ones(len(VISION_COLUMNS)))

    def test_copy_repeats_word_vector_to_subwords_and_punctuation(self) -> None:
        tokenizer = load_tokenizer()
        grid = build_grid("I don't know.", tokenizer)
        word_values = {
            "vectors": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            "observed_mask": np.array([1, 0, 1], dtype=np.uint8),
            "quality": np.array([0.9, 0.0, 0.8], dtype=np.float32),
            "missing_reason": ["observed", "alignment_failed", "observed"],
            "feature_names": ["x"],
        }
        result = copy_word_features_to_grid(word_values, grid)
        np.testing.assert_array_equal(result["values"][[2, 3, 4]], [[2.0], [2.0], [2.0]])
        np.testing.assert_array_equal(result["observed_mask"][[2, 3, 4]], [0, 0, 0])
        self.assertEqual(result["missing_reason"][4], "alignment_failed")


if __name__ == "__main__":
    unittest.main()
