"""Assert-тесты для src/sync.py (задача 04a: frame_time).

Запуск: python -m unittest discover -s tests -t . -v
Тесты на реальных данных пропускаются, если нет data/trackingData_*.txt
(файл данных не коммитится) или output/video_meta.json (генерируется задачей 02).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parse_tracking import read_header, timestamp_range  # noqa: E402
from src.sync import FrameTime, duration_ns, frame_time, iter_frames  # noqa: E402

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
META_PATH = ROOT / "output" / "video_meta.json"
NO_DATA = "нет data/trackingData_*.txt или output/video_meta.json (данные не коммитятся)"


class FrameTimeTest(unittest.TestCase):
    """Синтетические метаданные: без больших файлов и без cv2."""

    META = {"frame_count": 100, "duration": 3.336}  # 33.36 мс на кадр
    HEADER = 1_700_000_000_000_000_000
    STEP_NS = 33_360_000

    def _time(self, k: int, end_ns: int | None = None) -> FrameTime:
        end = self.HEADER + 10**12 if end_ns is None else end_ns
        return frame_time(k, self.META, self.HEADER, end)

    def test_first_frame_is_header_time(self) -> None:
        frame = self._time(0)
        self.assertEqual(frame.index, 0)
        self.assertEqual(frame.ts_ns, self.HEADER)
        self.assertEqual(frame.t_rel_s, 0.0)
        self.assertTrue(frame.in_tracking)

    def test_offsets_follow_duration_over_frame_count(self) -> None:
        for k in (0, 1, 50, 99):
            self.assertEqual(self._time(k).ts_ns, self.HEADER + k * self.STEP_NS)

    def test_integer_rounding_with_non_divisible_duration(self) -> None:
        meta = {"frame_count": 3, "duration": 1.0}
        offsets = [frame_time(k, meta, self.HEADER, self.HEADER + 10**12).ts_ns - self.HEADER for k in range(3)]
        self.assertEqual(offsets, [0, 333_333_333, 666_666_667])
        # каждый офсет отличается от точного k/3 с не более чем на 1 нс
        for k, offset in enumerate(offsets):
            self.assertLessEqual(abs(offset - k * 1e9 / 3), 1.0)

    def test_index_bounds_raise(self) -> None:
        for bad in (-1, 100, 101, True, 3.0, "3"):
            with self.subTest(k=bad), self.assertRaises(ValueError):
                self._time(bad)  # type: ignore[arg-type]

    def test_bad_meta_raises(self) -> None:
        end = self.HEADER + 10**12
        for meta in (
            {"frame_count": 0, "duration": 1.0},
            {"frame_count": 100},
            {"frame_count": 100, "duration": 0},
            {"frame_count": 100, "duration": -1.0},
            {"frame_count": 100, "duration": float("nan")},
            {"frame_count": 100, "duration": "3.336"},
            {"duration": 1.0},
        ):
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                frame_time(0, meta, self.HEADER, end)  # type: ignore[arg-type]

    def test_bad_header_and_tracking_end_raise(self) -> None:
        for header in (True, 1.0, "0"):
            with self.subTest(header=header), self.assertRaises(ValueError):
                frame_time(0, self.META, header, self.HEADER + 10**9)  # type: ignore[arg-type]
        for end in (True, 1.0, "0"):
            with self.subTest(end=end), self.assertRaises(ValueError):
                frame_time(0, self.META, self.HEADER, end)  # type: ignore[arg-type]

    def test_tail_of_video_is_flagged(self) -> None:
        end = self.HEADER + 98 * self.STEP_NS  # ровно время кадра k = 98
        self.assertTrue(self._time(98, end).in_tracking)  # граница включительна
        self.assertFalse(self._time(99, end).in_tracking)
        frames = list(iter_frames(self.META, self.HEADER, end))
        self.assertEqual(len(frames), 100)
        self.assertEqual([f.index for f in frames if not f.in_tracking], [99])

    def test_duration_ns_is_exact(self) -> None:
        self.assertEqual(duration_ns({"duration": 242.661}), 242_661_000_000)
        self.assertEqual(duration_ns({"duration": 0.1}), 100_000_000)


@unittest.skipIf(DATA is None or not META_PATH.is_file(), NO_DATA)
class RealDataTest(unittest.TestCase):
    """Реальные 7279 кадров и 21 787 сэмплов (при наличии data/ и output/)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        cls.header_ns = read_header(DATA)["timeStampNs"]
        cls.first_ns, cls.last_ns = timestamp_range(DATA)

    def test_frame_zero_equals_header(self) -> None:
        frame = frame_time(0, self.meta, self.header_ns, self.last_ns)
        self.assertEqual(frame.ts_ns, 1777989460861108224)
        self.assertEqual(frame.ts_ns, self.header_ns)

    def test_last_frame_time(self) -> None:
        frame = frame_time(7278, self.meta, self.header_ns, self.last_ns)
        self.assertEqual(frame.ts_ns - self.header_ns, 242_627_662_866)
        self.assertAlmostEqual(frame.t_rel_s, 242.627663, places=6)
        # В карточке округлено до «242.63 с»; точная формула даёт 242.6277 с, разница 2.3 мс.
        self.assertLess(abs(frame.t_rel_s - 242.63), 0.005)

    def test_step_is_33_337_ms(self) -> None:
        first = frame_time(1, self.meta, self.header_ns, self.last_ns)
        self.assertEqual(first.ts_ns - self.header_ns, 33_337_134)
        self.assertAlmostEqual((first.ts_ns - self.header_ns) / 1e6, 33.337134, places=6)

    def test_last_three_frames_have_no_tracking(self) -> None:
        frames = list(iter_frames(self.meta, self.header_ns, self.last_ns))
        self.assertEqual(len(frames), 7279)
        self.assertTrue(frames[7275].in_tracking)
        self.assertEqual([f.index for f in frames if not f.in_tracking], [7276, 7277, 7278])
        self.assertEqual(sum(f.in_tracking for f in frames), 7276)

    def test_tracking_end_is_last_sample(self) -> None:
        # span first→last = 242.542 с, но относительно header последний сэмпл — 242.5607 с
        self.assertEqual((self.last_ns - self.header_ns) / 1e9, 242.560678912)
        self.assertEqual(self.meta["tracking_duration_s"], 242.542)

    def test_tail_margin_is_sub_millisecond(self) -> None:
        frame = frame_time(7276, self.meta, self.header_ns, self.last_ns)
        margin_ns = frame.ts_ns - self.last_ns
        self.assertGreater(margin_ns, 0)
        self.assertLess(margin_ns, 1_000_000, "зазор k=7276 должен быть меньше 1 мс")


if __name__ == "__main__":
    unittest.main()
