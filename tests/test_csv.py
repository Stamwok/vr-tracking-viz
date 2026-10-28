"""Assert-тесты write_tracking_csv / frame_stats (задача 03d).

Запуск: python -m unittest discover -s tests -t . -v
Тесты на реальном файле пропускаются, если data/trackingData_*.txt отсутствует.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parse_tracking import CSV_COLUMNS, read_header, write_tracking_csv  # noqa: E402

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
NO_DATA = "нет data/trackingData_*.txt — файл данных не коммитится (см. README)"

EXPECTED_HEAD = ["t_ns", "t_rel_s", "head_x", "head_y", "head_z", "head_qx", "head_qy", "head_qz", "head_qw"]
EXPECTED_TAIL = ["left_active", "right_active"]


@unittest.skipIf(DATA is None, NO_DATA)
class TrackingCsvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import pandas as pd

        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / "tracking.csv"
        cls.header = read_header(DATA)
        cls.stats = write_tracking_csv(DATA, cls.path, header=cls.header)
        cls.frame = pd.read_csv(cls.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_schema_is_stable(self) -> None:
        self.assertEqual(list(self.frame.columns), list(CSV_COLUMNS))
        self.assertEqual(list(CSV_COLUMNS)[: len(EXPECTED_HEAD)], EXPECTED_HEAD)
        self.assertEqual(list(CSV_COLUMNS)[-len(EXPECTED_TAIL) :], EXPECTED_TAIL)
        self.assertEqual(len(CSV_COLUMNS), 41)  # 2 времени + 7 головы + 30 рук + 2 флага

    def test_rows_and_timing(self) -> None:
        self.assertEqual(len(self.frame), 21787)
        self.assertEqual(self.stats["rows"], 21787)
        self.assertAlmostEqual(self.frame["t_rel_s"].iloc[0], 0.018, places=3)
        self.assertAlmostEqual(self.stats["duration_s"], 242.54, places=2)
        self.assertAlmostEqual(self.stats["median_dt_ms"], 11.12, places=2)

    def test_nan_only_for_invalid_right_hand(self) -> None:
        hand_columns = [column for column in CSV_COLUMNS if column.endswith(("_x", "_y", "_z"))]
        right_columns = [column for column in hand_columns if column.startswith("right_")]
        left_columns = [column for column in hand_columns if column.startswith("left_")]
        self.assertEqual(len(right_columns), 15)
        self.assertEqual(int(self.frame[right_columns].isna().sum().sum()), 135 * 15)
        self.assertEqual(int(self.frame[left_columns].isna().sum().sum()), 0)
        self.assertEqual(self.stats["right_invalid"], 135)
        self.assertEqual(self.stats["left_invalid"], 0)

    def test_active_flags_are_integers(self) -> None:
        self.assertEqual(set(self.frame["left_active"].unique()), {1})
        self.assertEqual(set(self.frame["right_active"].unique()), {0, 1})
        self.assertEqual(int((self.frame["right_active"] == 0).sum()), 135)


if __name__ == "__main__":
    unittest.main()
