"""Assert-тесты для src/camera_model.py (задача 05a).

Запуск: python -m unittest discover -s tests -t . -v
Тесты на реальном header пропускаются, если data/trackingData_*.txt отсутствует
(файл данных не коммитится). Синтетические тесты работают всегда.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_model import SIZE_TOL, Intrinsics  # noqa: E402
from src.parse_tracking import read_header  # noqa: E402

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
NO_DATA = "нет data/trackingData_*.txt — файл данных не коммитится (см. README)"

#: cameraIntrinsics из реального header (файловый порядок [cx, cy, fx, fy]).
RAW = [1079.5, 404.5, 1373.6681390725917, 686.8680648828482]


def header(values: list[float]) -> dict:
    """Синтетический header с заданным cameraIntrinsics."""
    return {"cameraIntrinsics": list(values)}


@unittest.skipIf(DATA is None, NO_DATA)
class RealHeaderTest(unittest.TestCase):
    """Реальный header: полный кадр 2160×810, per-eye 1080×810, FOV ≈ 89° по диагонали."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.intrinsics = Intrinsics.from_header(read_header(DATA))

    def test_full_frame_from_derived_size(self) -> None:
        intr = self.intrinsics
        self.assertEqual((intr.full_w, intr.full_h), (2160, 810))
        self.assertEqual((intr.eye_w, intr.eye_h), (1080, 810))
        self.assertAlmostEqual(intr.fx, 1373.6681390725917, places=9)
        self.assertAlmostEqual(intr.fy, 686.8680648828482, places=9)
        self.assertAlmostEqual(intr.cx, 1079.5, places=9)
        self.assertAlmostEqual(intr.cy, 404.5, places=9)
        self.assertEqual(2.0 * intr.cx + 1.0, 2160.0)
        self.assertEqual(2.0 * intr.cy + 1.0, 810.0)

    def test_per_eye_left(self) -> None:
        eye = self.intrinsics.per_eye("left")
        self.assertAlmostEqual(eye.fx, 686.8680648828482, places=9)
        self.assertAlmostEqual(eye.fy, 686.8680648828482, places=9)
        self.assertAlmostEqual(eye.cx, 539.5, places=9)
        self.assertAlmostEqual(eye.cy, 404.5, places=9)
        self.assertEqual((eye.full_w, eye.full_h), (1080, 810))

    def test_per_eye_right(self) -> None:
        eye = self.intrinsics.per_eye("right")
        self.assertEqual(eye.fx, eye.fy)
        self.assertAlmostEqual(eye.cx, 1619.5, places=9)
        self.assertAlmostEqual(eye.cy, 404.5, places=9)
        self.assertEqual((eye.full_w, eye.full_h), (1080, 810))
        # правый глаз сдвинут ровно на ширину одного глаза
        self.assertAlmostEqual(eye.cx - self.intrinsics.per_eye("left").cx, 1080.0, places=9)

    def test_fov_per_eye(self) -> None:
        h, v, diag = self.intrinsics.per_eye("left").fov_deg()
        self.assertAlmostEqual(h, 76.34724276124926, places=6)
        self.assertAlmostEqual(v, 61.04999923706055, places=6)
        self.assertAlmostEqual(diag, 89.00141161465035, places=6)
        # критерий карточки: диагональный FOV в пределах 88–90°
        self.assertLess(88.0, diag)
        self.assertGreater(90.0, diag)

    def test_per_eye_fov_matches_full_frame(self) -> None:
        # fx полного кадра ≈ 2·fy (не ровно: разница 0.068 px), поэтому FOV полного кадра
        # и одного глаза совпадают в пределах 0.01° (h: 76.3500° vs 76.3472°).
        full = self.intrinsics.fov_deg()
        eye = self.intrinsics.per_eye("right").fov_deg()
        for full_value, eye_value in zip(full, eye):
            self.assertLess(abs(full_value - eye_value), 0.01)

    def test_square_pixels_one_eye(self) -> None:
        eye = self.intrinsics.per_eye("left")
        self.assertAlmostEqual(eye.fx, eye.fy, places=12)
        half_w = eye.full_w / 2.0
        half_h = eye.full_h / 2.0
        # 540/f = 1.333·405/f — отношение полусторон равно 4/3
        self.assertAlmostEqual(half_w / eye.fx, (4.0 / 3.0) * half_h / eye.fy, places=9)

    def test_default_matches_explicit_frame_size(self) -> None:
        explicit = Intrinsics.from_header(read_header(DATA), frame_size=(2160, 810))
        self.assertEqual(explicit, self.intrinsics)

    def test_eye_w_is_half_full_w(self) -> None:
        for side in ("left", "right"):
            eye = self.intrinsics.per_eye(side)
            self.assertEqual(eye.eye_w, eye.full_w // 2)


class WrongOrderTest(unittest.TestCase):
    """Порядок [fx, fy, cx, cy] обязан падать, а не приниматься молча."""

    def test_wrong_order_rejected_by_derived_size(self) -> None:
        # [fx, fy, cx, cy]: 2·cx + 1 = 2748.336 — не целое
        with self.assertRaises(ValueError):
            Intrinsics.from_header(header([1373.6681390725917, 686.8680648828482, 1079.5, 404.5]))

    def test_wrong_order_rejected_with_explicit_frame_size(self) -> None:
        # тот же массив, но размер известен: центр «cx» = 1373.668 ≠ 2160/2 − 0.5
        with self.assertRaises(ValueError):
            Intrinsics.from_header(
                header([1373.6681390725917, 686.8680648828482, 1079.5, 404.5]),
                frame_size=(2160, 810),
            )

    def test_wrong_order_center_outside_frame(self) -> None:
        # cy = 686.868 вместо 404.5 — тоже отсекается проверкой центра
        with self.assertRaises(ValueError):
            Intrinsics.from_header(
                header([1079.5, 686.8680648828482, 1373.6681390725917, 404.5]),
                frame_size=(2160, 810),
            )


class ValidationTest(unittest.TestCase):
    """from_header: frame_size, отрицательные/нефинитные, длина, per_eye(side)."""

    def test_frame_size_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Intrinsics.from_header(header(RAW), frame_size=(2160, 808))
        with self.assertRaises(ValueError):
            Intrinsics.from_header(header(RAW), frame_size=(2159, 810))
        with self.assertRaises(ValueError):
            Intrinsics.from_header(header(RAW), frame_size=(2160, 811))

    def test_odd_frame_width_rejected(self) -> None:
        # центр для 2159 и 810 «правильный», но SBS требует чётной ширины
        with self.assertRaises(ValueError):
            Intrinsics.from_header(header([1079.0, 404.5, 686.868, 686.868]), frame_size=(2159, 810))

    def test_center_tolerance(self) -> None:
        # 2e-6 от центра — уже ошибка (допуск SIZE_TOL = 1e-6)
        shifted = [1079.5 + 2e-6, 404.5, 1373.6681390725917, 686.8680648828482]
        with self.assertRaises(ValueError):
            Intrinsics.from_header(header(shifted), frame_size=(2160, 810))
        # 1e-7 — в пределах допуска
        aligned = [1079.5 + 1e-7, 404.5, 1373.6681390725917, 686.8680648828482]
        self.assertEqual(Intrinsics.from_header(header(aligned), frame_size=(2160, 810)).cx, aligned[0])

    def test_non_positive_and_non_finite_rejected(self) -> None:
        bad_values = [
            [0.0, 404.5, 1373.6681390725917, 686.8680648828482],
            [-1079.5, 404.5, 1373.6681390725917, 686.8680648828482],
            [1079.5, -404.5, 1373.6681390725917, 686.8680648828482],
            [1079.5, 404.5, -1373.6681390725917, 686.8680648828482],
            [1079.5, 404.5, 1373.6681390725917, -686.8680648828482],
            [math.nan, 404.5, 1373.6681390725917, 686.8680648828482],
            [math.inf, 404.5, 1373.6681390725917, 686.8680648828482],
            [1079.5, 404.5, 1373.6681390725917, math.nan],
            [1079.5, 404.5, "fx", 686.8680648828482],
        ]
        for values in bad_values:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    Intrinsics.from_header(header(values))

    def test_wrong_intrinsics_length_rejected(self) -> None:
        for length in (0, 3, 5):
            with self.subTest(length=length):
                with self.assertRaises(ValueError):
                    Intrinsics.from_header(header([1079.5] * length))

    def test_header_without_intrinsics_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Intrinsics.from_header({"timeStampNs": 0})
        with self.assertRaises(ValueError):
            Intrinsics.from_header({})

    def test_per_eye_rejects_unknown_side(self) -> None:
        intr = Intrinsics.from_header(header(RAW))
        for side in ("up", "Right", "left ", ""):
            with self.subTest(side=side):
                with self.assertRaises(ValueError):
                    intr.per_eye(side)

    def test_dataclass_validation(self) -> None:
        good = dict(fx=686.868, fy=686.868, cx=539.5, cy=404.5, full_w=1080, full_h=810, eye_w=540, eye_h=810)
        Intrinsics(**good)  # валидный объект создаётся
        with self.assertRaises(ValueError):
            Intrinsics(**{**good, "eye_w": 541})
        with self.assertRaises(ValueError):
            Intrinsics(**{**good, "full_w": 1079, "eye_w": 539})
        with self.assertRaises(ValueError):
            Intrinsics(**{**good, "fx": 0.0})
        with self.assertRaises(ValueError):
            Intrinsics(**{**good, "full_h": 0})

    def test_size_tolerance_value(self) -> None:
        self.assertEqual(SIZE_TOL, 1e-6)


if __name__ == "__main__":
    unittest.main()
