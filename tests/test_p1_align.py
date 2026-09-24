import unittest

from src.p1.align import (
    HOP_SECONDS,
    _summarize,
    build_alignment_plan,
    interpolate_failed_words,
)

LABELS = tuple("-|ETAONIHSRDLUMWCFGYPBVK'XJQZ")


def direct_word(word_id: int, word: str, start: int, end: int, confidence: float = 0.9):
    return {
        "word_id": word_id,
        "word_text": word,
        "ctc_text": word.upper(),
        "raw_start_frame": start,
        "raw_end_frame": end,
        "raw_t_s": start * HOP_SECONDS,
        "raw_t_e": end * HOP_SECONDS,
        "start_frame": start,
        "end_frame": end,
        "t_s": start * HOP_SECONDS,
        "t_e": end * HOP_SECONDS,
        "confidence": confidence,
        "failure_reason": None,
        "alignment_status": "aligned",
    }


class AlignmentTest(unittest.TestCase):
    def test_numeric_words_are_excluded_from_ctc_target(self) -> None:
        records, target = build_alignment_plan("In 2008 there were 10th prizes", LABELS)
        self.assertEqual(target, "IN|THERE|WERE|PRIZES")
        self.assertEqual(records[1]["failure_reason"], "numeric_oov")
        self.assertEqual(records[4]["failure_reason"], "numeric_oov")
        self.assertNotIn("2", target)

    def test_numeric_gap_is_interpolated_on_whole_20ms_frames(self) -> None:
        records = [
            direct_word(0, "in", 5, 10),
            {
                **direct_word(1, "2008", 0, 0, 0.0),
                "ctc_text": None,
                "failure_reason": "numeric_oov",
                "alignment_status": "excluded",
            },
            direct_word(2, "there", 20, 25),
        ]
        result = interpolate_failed_words(records, duration=1.0)
        number = result[1]
        self.assertEqual(number["alignment_status"], "interpolated")
        self.assertEqual((number["start_frame"], number["end_frame"]), (10, 20))
        self.assertEqual((number["t_s"], number["t_e"]), (0.2, 0.4))
        self.assertEqual(number["confidence"], 0.0)

    def test_consecutive_failures_split_gap_without_overlap(self) -> None:
        records = [
            direct_word(0, "more", 5, 10),
            {**direct_word(1, "100", 0, 0, 0.0), "failure_reason": "numeric_oov"},
            {**direct_word(2, "000", 0, 0, 0.0), "failure_reason": "numeric_oov"},
            direct_word(3, "people", 20, 25),
        ]
        result = interpolate_failed_words(records, duration=1.0)
        self.assertEqual((result[1]["start_frame"], result[1]["end_frame"]), (10, 15))
        self.assertEqual((result[2]["start_frame"], result[2]["end_frame"]), (15, 20))
        self.assertEqual(result[1]["t_e"], result[2]["t_s"])

    def test_leading_failure_starts_after_zero(self) -> None:
        records = [
            {**direct_word(0, "2008", 0, 0, 0.0), "failure_reason": "numeric_oov"},
            direct_word(1, "began", 10, 15),
        ]
        result = interpolate_failed_words(records, duration=1.0)
        self.assertEqual(result[0]["start_frame"], 1)
        self.assertGreater(result[0]["t_s"], 0.0)
        self.assertEqual(result[0]["t_e"], result[1]["t_s"])

    def test_failure_rate_uses_strict_5_and_20_percent_thresholds(self) -> None:
        at_five = [direct_word(i, f"w{i}", i + 1, i + 2) for i in range(20)]
        at_five[0]["failure_reason"] = "low_confidence"
        summary = _summarize(at_five, None)
        self.assertEqual(summary["failure_rate"], 0.05)
        self.assertFalse(summary["needs_review"])

        at_ten = at_five[:10]
        summary = _summarize(at_ten, None)
        self.assertTrue(summary["needs_review"])
        self.assertFalse(summary["rollback_required"])
        self.assertEqual(summary["status"], "review")

        at_twenty_five = at_five[:4]
        summary = _summarize(at_twenty_five, None)
        self.assertTrue(summary["rollback_required"])
        self.assertEqual(summary["status"], "rollback")


if __name__ == "__main__":
    unittest.main()
