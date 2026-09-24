import pickle
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.p1.grid import build_grid, load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
LABEL_PATH = (
    ROOT
    / "data"
    / "附件1-数据集原始多模态样本"
    / "MOSEI数据集部分原始视频-100条"
    / "label-100.xlsx"
)
ALIGNED_PATH = ROOT / "data" / "附件2-数据集特征文件" / "aligned_50.pkl"


class GridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = load_tokenizer()

    def test_train_overlap_11_matches_attachment2_bitwise(self) -> None:
        labels = pd.read_excel(LABEL_PATH, sheet_name="label")
        label_text = {
            f"{row.video_id}$_${int(row.clip_id)}": row.text
            for row in labels.itertuples()
        }
        with ALIGNED_PATH.open("rb") as handle:
            attachment = pickle.load(handle)
        train = attachment["train"]
        overlap = [index for index, sample_id in enumerate(train["id"]) if sample_id in label_text]
        self.assertEqual(len(overlap), 11)

        for index in overlap:
            sample_id = train["id"][index]
            grid = build_grid(label_text[sample_id], self.tokenizer)
            actual = np.stack([
                grid["input_ids"],
                grid["attention_mask"],
                grid["token_type_ids"],
            ])
            np.testing.assert_array_equal(actual, train["text_bert"][index], err_msg=sample_id)
            self.assertFalse(grid["anomalies"], sample_id)

    def test_cross_boundary_word_uses_only_retained_pieces_for_alpha(self) -> None:
        text = " ".join(["hello"] * 47 + ["unaffordability", "after"])
        grid = build_grid(text, self.tokenizer)

        self.assertTrue(grid["truncated"])
        self.assertEqual(grid["sep_position"], 49)
        self.assertEqual(grid["input_ids"][49], self.tokenizer.sep_token_id)
        boundary = grid["wp_word_map"][48]
        self.assertEqual(boundary["token"], "una")
        self.assertEqual(boundary["word_text"], "unaffordability")
        self.assertTrue(boundary["trunc_flag"])
        self.assertEqual(boundary["wp_count_in_word"], 1)
        self.assertEqual(boundary["alpha"], 1.0)

    def test_more_than_48_pieces_drops_only_tail(self) -> None:
        grid = build_grid(" ".join(["hello"] * 49), self.tokenizer)
        self.assertEqual(grid["retained_wp_count"], 48)
        self.assertEqual(grid["dropped_word_count"], 1)
        self.assertTrue(grid["full_word_map"][-1]["trunc_flag"])
        self.assertTrue(grid["full_word_map"][-1]["dropped"])
        self.assertEqual(grid["content_mask"].sum(), 48)
        self.assertEqual(grid["special_mask"].sum(), 2)
        self.assertEqual(grid["padding_mask"].sum(), 0)
        np.testing.assert_array_equal(
            grid["content_mask"] + grid["special_mask"] + grid["padding_mask"],
            np.ones(50, dtype=np.uint8),
        )

    def test_punctuation_inheritance_follows_boundary_table(self) -> None:
        grid = build_grid('(Hello), world - okay.', self.tokenizer)
        rows = {row["token"]: row for row in grid["wp_word_map"] if row["word_id"] is not None}

        self.assertEqual(rows["("]["word_text"], "hello")
        self.assertEqual(rows["("]["attach_type"], "head")
        self.assertEqual(rows[")"]["word_text"], "hello")
        self.assertEqual(rows[")"]["attach_type"], "tail")
        self.assertEqual(rows[","]["word_text"], "hello")
        self.assertEqual(rows["-"]["word_text"], "world")
        self.assertEqual(rows["-"]["attach_type"], "ambi")
        self.assertEqual(rows["."]["word_text"], "okay")

    def test_contraction_is_one_pronounced_word(self) -> None:
        grid = build_grid("I don't know.", self.tokenizer)
        rows = [row for row in grid["wp_word_map"] if row["word_text"] == "don't"]
        self.assertEqual([row["token"] for row in rows], ["don", "'", "t"])
        self.assertEqual({row["word_id"] for row in rows}, {1})
        self.assertAlmostEqual(sum(row["alpha"] for row in rows), 1.0)
        self.assertFalse(grid["anomalies"])

    def test_ambiguous_quotes_default_to_previous_word(self) -> None:
        grid = build_grid('the word "power"', self.tokenizer)
        quote_rows = [row for row in grid["wp_word_map"] if row["token"] == '"']
        self.assertEqual(len(quote_rows), 2)
        self.assertEqual(quote_rows[0]["word_text"], "word")
        self.assertEqual(quote_rows[1]["word_text"], "power")
        self.assertEqual({row["attach_type"] for row in quote_rows}, {"ambi"})
        self.assertFalse(grid["anomalies"])


if __name__ == "__main__":
    unittest.main()
