"""Assert-тесты extract_sample (задача 03c).

Запуск: python -m unittest discover -s tests -t . -v
Тесты на реальном файле пропускаются, если data/trackingData_*.txt отсутствует.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parse_tracking import HAND_JOINTS, extract_sample, iter_samples, parse_pose  # noqa: E402

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
NO_DATA = "нет data/trackingData_*.txt — файл данных не коммитится (см. README)"

HEAD_POSE = "-0,08517717,-0,02745685,-0,0058761,0,162562191,-0,2861428,-0,116477557,-0,9370853"
FIRST_RIGHT_HAND_SAMPLE = 66  # первый сэмпл с isActive = 1 у правой руки, t ≈ 0.83 с


def _joints(count: int = 26, valid_flags: float = 15.0, zero_index: int | None = None) -> list[dict]:
    return [
        {"p": HEAD_POSE, "s": 0.0 if index == zero_index else valid_flags, "r": 0.0}
        for index in range(count)
    ]


@unittest.skipIf(DATA is None, NO_DATA)
class FirstSampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = next(iter_samples(DATA))
        cls.record = extract_sample(cls.sample, where="строка 2")

    def test_head_pose_is_three_plus_four(self) -> None:
        self.assertAlmostEqual(self.record["head_x"], -0.08517717, places=9)
        self.assertEqual(sum(1 for key in self.record if key.startswith("head_")), 7)
        expected = parse_pose(self.sample["Head"]["pose"])
        for axis, value in zip(("x", "y", "z", "qx", "qy", "qz", "qw"), expected):
            self.assertEqual(self.record[f"head_{axis}"], value)

    def test_five_points_per_hand(self) -> None:
        for side in ("left", "right"):
            for name, _ in HAND_JOINTS:
                for axis in "xyz":
                    self.assertIn(f"{side}_{name}_{axis}", self.record)
        self.assertEqual(sum(1 for key in self.record if key.startswith("left_") and key[5:].count("_") == 1), 15)

    def test_first_sample_right_hand_is_nan(self) -> None:
        self.assertEqual(self.record["right_active"], 0)
        for name, _ in HAND_JOINTS:
            for axis in "xyz":
                value = self.record[f"right_{name}_{axis}"]
                self.assertTrue(math.isnan(value), f"right_{name}_{axis} = {value!r}, ожидался NaN")

    def test_left_hand_is_valid(self) -> None:
        self.assertEqual(self.record["left_active"], 1)
        for name, _ in HAND_JOINTS:
            self.assertTrue(math.isfinite(self.record[f"left_{name}_x"]))
            self.assertTrue(math.isfinite(self.record[f"left_{name}_y"]))
            self.assertTrue(math.isfinite(self.record[f"left_{name}_z"]))

    def test_inactive_positions_are_not_zeros_but_nan(self) -> None:
        # В файле у неактивной руки p — не нули, а «замороженная» поза: проверяем, что
        # валидация решает по флагам, а не по значениям.
        raw = parse_pose(self.sample["Hand"]["rightHand"]["HandJointLocations"][0]["p"])
        self.assertTrue(any(abs(value) > 1e-6 for value in raw))
        self.assertTrue(math.isnan(self.record["right_palm_x"]))


@unittest.skipIf(DATA is None, NO_DATA)
class ActivityTest(unittest.TestCase):
    def test_right_hand_becomes_valid(self) -> None:
        for index, sample in enumerate(iter_samples(DATA)):
            record = extract_sample(sample)
            if record["right_active"]:
                self.assertEqual(index, FIRST_RIGHT_HAND_SAMPLE)
                self.assertTrue(math.isfinite(record["right_palm_x"]))
                self.assertTrue(math.isfinite(record["right_middle_z"]))
                return
        self.fail("правая рука так и не стала активной")

    def test_exactly_135_inactive_right_hand_samples(self) -> None:
        inactive = sum(1 for record in map(extract_sample, iter_samples(DATA)) if not record["right_active"])
        self.assertEqual(inactive, 135)


class SyntheticTest(unittest.TestCase):
    def test_s_zero_makes_point_nan_even_if_hand_active(self) -> None:
        sample = {
            "timeStampNs": 1,
            "Head": {"pose": HEAD_POSE, "status": 3},
            "Hand": {
                "leftHand": {"isActive": 1, "HandJointLocations": _joints()},
                "rightHand": {"isActive": 1, "HandJointLocations": _joints(zero_index=0)},
            },
        }
        record = extract_sample(sample)
        self.assertEqual(record["right_active"], 1)
        self.assertTrue(math.isnan(record["right_palm_x"]), "s = 0 должен давать NaN")
        self.assertTrue(math.isfinite(record["right_wrist_x"]), "s = 15 должен давать координату")

    def test_inactive_hand_gives_all_nan(self) -> None:
        sample = {
            "timeStampNs": 2,
            "Head": {"pose": HEAD_POSE, "status": 3},
            "Hand": {
                "leftHand": {"isActive": 0, "HandJointLocations": _joints()},
                "rightHand": {"isActive": 0, "HandJointLocations": _joints()},
            },
        }
        record = extract_sample(sample)
        for side in ("left", "right"):
            self.assertEqual(record[f"{side}_active"], 0)
            for name, _ in HAND_JOINTS:
                for axis in "xyz":
                    self.assertTrue(math.isnan(record[f"{side}_{name}_{axis}"]))


if __name__ == "__main__":
    unittest.main()
