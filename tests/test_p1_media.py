import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

from src.p1.media import FF, extract_audio_16k, extract_frames_pts, probe_duration


class MediaExtractionTest(unittest.TestCase):
    def test_extracts_pcm_audio_and_preserves_variable_pts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            source = tmp / "variable_pts.mp4"
            generated = subprocess.run(
                [
                    FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=32x32:rate=10:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2",
                    "-filter:v", "setpts=PTS+if(gte(N\\,5)\\,0.1/TB\\,0)",
                    "-fps_mode", "vfr", "-c:v", "libx264", "-c:a", "aac",
                    "-shortest", str(source),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)

            wav_path = extract_audio_16k(source, tmp / "audio.wav")
            with wave.open(str(wav_path), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.getsampwidth(), 2)

            frames = extract_frames_pts(source, tmp / "frames")
            self.assertGreaterEqual(len(frames), 9)
            self.assertEqual(
                [row["frame_idx"] for row in frames], list(range(len(frames)))
            )
            self.assertTrue(all(Path(row["path"]).is_file() for row in frames))
            deltas = [
                round(right["pts_time"] - left["pts_time"], 6)
                for left, right in zip(frames, frames[1:])
            ]
            self.assertGreater(max(deltas), min(deltas))
            self.assertGreater(probe_duration(source), 0.0)

    def test_missing_source_is_reported_before_ffmpeg(self) -> None:
        with self.assertRaises(FileNotFoundError):
            extract_frames_pts("does-not-exist.mp4", "unused")


if __name__ == "__main__":
    unittest.main()
