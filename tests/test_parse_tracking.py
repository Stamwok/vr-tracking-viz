"""Assert-тесты для src/parse_tracking.py (задача 03a).

Запуск: python -m unittest discover -s tests -t . -v
Тесты на реальном файле пропускаются, если data/trackingData_*.txt отсутствует
(файл данных не коммитится).
"""
from __future__ import annotations

import json
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parse_tracking import count_samples, iter_samples, read_header  # noqa: E402

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
NO_DATA = "нет data/trackingData_*.txt — файл данных не коммитится (см. README)"


@unittest.skipIf(DATA is None, NO_DATA)
class HeaderTest(unittest.TestCase):
    """Header: timeStampNs, 4 интринсика, 2 матрицы 4×4."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.header = read_header(DATA)

    def test_timestamp_and_intrinsics(self) -> None:
        self.assertEqual(self.header["timeStampNs"], 1777989460861108224)
        cx, cy, fx, fy = self.header["cameraIntrinsics"]
        self.assertEqual(len(self.header["cameraIntrinsics"]), 4)
        self.assertAlmostEqual(cx, 1079.5, places=6)
        self.assertAlmostEqual(cy, 404.5, places=6)
        self.assertAlmostEqual(fx, 1373.6681390725917, places=6)
        self.assertAlmostEqual(fy, 686.8680648828482, places=6)

    def test_extrinsics_are_two_4x4_matrices(self) -> None:
        matrices = self.header["cameraExtrinsics"]
        self.assertEqual(len(matrices), 2)
        for matrix in matrices:
            self.assertEqual(len(matrix), 4)
            for row in matrix:
                self.assertEqual(len(row), 4)
            self.assertEqual(matrix[3], [0.0, 0.0, 0.0, 1.0])
        # левый и правый глаз отличаются трансляцией (IPD ≈ 64 мм)
        self.assertNotEqual(matrices[0][0][3], matrices[1][0][3])


@unittest.skipIf(DATA is None, NO_DATA)
class SamplesTest(unittest.TestCase):
    """Итератор сэмплов: ровно 21 787 и стриминговое чтение."""

    def test_counts_exactly_21787_samples(self) -> None:
        self.assertEqual(count_samples(DATA), 21787)

    def test_first_sample_has_head_and_hand(self) -> None:
        first = next(iter_samples(DATA))
        for key in ("timeStampNs", "Head", "Hand"):
            self.assertIn(key, first)

    def test_memory_does_not_grow_with_file(self) -> None:
        tracemalloc.start()
        try:
            taken = 0
            for taken, _ in enumerate(iter_samples(DATA), start=1):
                if taken == 1000:
                    break
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(taken, 1000)
        # 1000 строк по ~11.8 КБ — это ~12 МБ данных; в памяти держим одну строку.
        self.assertLess(peak, 20 * 2**20, f"пик памяти {peak / 2**20:.1f} МиБ при 1000 сэмплах")


class FormatTest(unittest.TestCase):
    """Краевые случаи формата — на временных файлах, без больших данных."""

    def test_skips_header_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.txt"
            path.write_text('{"timeStampNs": 0}\n\n{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
            self.assertEqual([sample["a"] for sample in iter_samples(path)], [1, 2])

    def test_header_without_fields_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.txt"
            path.write_text('{"timeStampNs": 1}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_header(path)

    def test_single_extrinsics_matrix_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.txt"
            matrix = json.dumps([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            path.write_text(
                json.dumps({"timeStampNs": 1, "cameraIntrinsics": "[1,2,3,4]", "cameraExtrinsics": matrix}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                read_header(path)


if __name__ == "__main__":
    unittest.main()
