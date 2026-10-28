"""Assert-тесты parse_pose (задача 03b).

Запуск: python -m unittest discover -s tests -t . -v
Тест на реальном файле пропускается, если data/trackingData_*.txt отсутствует.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parse_tracking import QUAT_NORM_TOL, iter_samples, parse_pose  # noqa: E402

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
NO_DATA = "нет data/trackingData_*.txt — файл данных не коммитится (см. README)"

# Head.pose первого сэмпла: x, y, z, qx, qy, qz, qw — с десятичной запятой.
HEAD_POSE = "-0,08517717,-0,02745685,-0,0058761,0,162562191,-0,2861428,-0,116477557,-0,9370853"
HEAD_POSE_EXPECTED = [-0.08517717, -0.02745685, -0.0058761, 0.162562191, -0.2861428, -0.116477557, -0.9370853]


class ParsePoseTest(unittest.TestCase):
    def test_decimal_comma_is_parsed(self) -> None:
        pose = parse_pose(HEAD_POSE)
        self.assertEqual(len(pose), 7)
        for got, expected in zip(pose, HEAD_POSE_EXPECTED):
            self.assertAlmostEqual(got, expected, places=9)
        self.assertAlmostEqual(pose[0], -0.08517717, places=9)

    def test_quaternion_norm_is_one(self) -> None:
        q = parse_pose(HEAD_POSE)[3:]
        self.assertEqual(len(q), 4)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in q)), 1.0, places=6)

    def test_body_va_style_whole_numbers(self) -> None:
        # Нечётное число токенов: документированное правило — все токены целые, берём как есть.
        self.assertEqual(parse_pose("0,0,0"), [0.0, 0.0, 0.0])
        self.assertEqual(parse_pose("1,-2,3"), [1.0, -2.0, 3.0])

    def test_odd_tokens_with_fraction_raise(self) -> None:
        # Пять токенов, но "5.5" не целое → неоднозначно (в файле так не пишут).
        with self.assertRaises(ValueError):
            parse_pose("1,2,3,4,5.5")

    def test_bad_quaternion_norm_raises_with_context(self) -> None:
        with self.assertRaises(ValueError) as caught:
            parse_pose("0,0,0,1,0,0,1", where="строка 7, Head.pose")  # |q| = √2
        message = str(caught.exception)
        self.assertIn("|q|", message)
        self.assertIn("строка 7", message)

    def test_empty_pose_raises(self) -> None:
        for broken in ("", "   "):
            with self.assertRaises(ValueError):
                parse_pose(broken)


@unittest.skipIf(DATA is None, NO_DATA)
class AllHeadPosesTest(unittest.TestCase):
    def test_all_21787_head_poses_parse_with_unit_norm(self) -> None:
        count = 0
        worst = 0.0
        for line_no, sample in enumerate(iter_samples(DATA), start=2):
            pose = parse_pose(sample["Head"]["pose"], where=f"строка {line_no}, Head.pose")
            self.assertEqual(len(pose), 7)
            norm = math.sqrt(sum(q * q for q in pose[3:]))
            worst = max(worst, abs(norm - 1.0))
            count += 1
        self.assertEqual(count, 21787)
        self.assertLessEqual(worst, QUAT_NORM_TOL, f"худшее отклонение |q| = {worst:.3e}")


if __name__ == "__main__":
    unittest.main()
