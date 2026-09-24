import unittest

import numpy as np

from src.p1.assemble import (
    SCHEMA_HASH,
    assemble_sample,
    compute_normalization_stats,
    validate_sample,
)
from src.p1.grid import build_grid, load_tokenizer


class AssembleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = build_grid("I don't know.", load_tokenizer())
        cls.words = [
            {"word_id": index, "t_s": index * 0.1 + 0.02, "t_e": index * 0.1 + 0.1,
             "confidence": 0.9, "alignment_status": "aligned", "failure_reason": None}
            for index in range(3)
        ]

    def sample(self, identifier="sample"):
        content = self.grid["content_mask"].astype(np.uint8)
        audio = np.tile(np.arange(25, dtype=np.float32), (50, 1))
        vision = np.tile(np.arange(23, dtype=np.float32), (50, 1))
        pooled_audio = {
            "values": audio, "observed_mask": content.copy(), "quality": content.astype(np.float32),
            "missing_reason": ["observed" if value else None for value in content], "feature_names": [],
        }
        pooled_vision = {
            "values": vision, "observed_mask": content.copy(), "quality": content.astype(np.float32),
            "missing_reason": ["observed" if value else None for value in content], "feature_names": [],
        }
        return assemble_sample(
            sample_id=identifier, raw_text="I don't know.", grid=self.grid,
            alignment={"words": self.words, "status": "ok", "failure_rate": 0.0},
            text=np.ones((50, 768), dtype=np.float32),
            audio_grid=pooled_audio, vision_grid=pooled_vision,
        )

    def test_schema_and_time_mapping(self) -> None:
        sample = self.sample()
        validate_sample(sample)
        self.assertEqual(sample["schema_hash"], SCHEMA_HASH)
        self.assertEqual(sample["wp_word_time_map"][2]["t_s"], self.words[1]["t_s"])
        self.assertEqual(sample["audio"].shape, (50, 25))
        self.assertEqual(sample["vision"].shape, (50, 23))

    def test_normalization_uses_only_explicit_train_ids(self) -> None:
        train = self.sample("train")
        holdout = self.sample("holdout")
        holdout["audio"][:] = 9999
        stats = compute_normalization_stats([train, holdout], {"train"})
        np.testing.assert_allclose(stats["audio"]["mean"], np.arange(25, dtype=np.float32))
        self.assertEqual(stats["train_ids"], ["train"])


if __name__ == "__main__":
    unittest.main()
