"""Assert-тесты для src/config.py, задача 07c: константы калибровки и их единственность.

Критерии карточки 07c, которые проверяются:

* константы зафиксированы значениями 07b: оси ``+x −y −z`` (камера = знак·ось головы),
  поворот ``(0, 180, 0)``, смещение ``(0, 0, 0)``, ``INVERT_QUAT``/``Y_DOWN_FLIP`` — ``False``;
* ``camera_model.check_config`` ловит рассогласованный конфиг (запись осей ≠ матрица
  поворота, ``det ≠ +1``, флаг не ``bool``), а не «подрисовывает» по нему;
* смена констант в ``src/config.py`` полностью определяет результат: флаги, поворот и
  смещение по умолчанию читаются из конфига, код камеры не правится;
* рендер одного кадра воспроизводится без правок кода: проекция точки при
  ``IDENTITY_POSE`` даёт ожидаемые пиксели (golden-числа), а метрика на эталоне 07a —
  39/39 без чужих рук и промахов (режим ``calib_search --verify-config``).

Тесты на реальных данных пропускаются, если нет ``data/trackingData_*.txt`` или
``output/video_meta.json`` (эти файлы не коммитятся, см. README).
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import calib_search, camera_model, config  # noqa: E402
from src.camera_model import (  # noqa: E402
    Intrinsics,
    axis_convention,
    check_config,
    head_cam_matrix,
    project_eye,
    world_to_camera,
    world_to_camera_matrix,
)

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
META = ROOT / "output" / "video_meta.json"
NO_DATA = "нет data/trackingData_*.txt — файл данных не коммитится (см. README)"
NO_META = "нет output/video_meta.json — метаданные видео не сгенерированы (см. README)"

#: Единичная поза: голова в начале координат, без поворота.
IDENTITY_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

#: Интринсики полного SBS-кадра из header записи (05a): [cx, cy, fx, fy].
FULL_INTRINSICS = Intrinsics(
    fx=1373.6681390725917,
    fy=686.8680648828482,
    cx=1079.5,
    cy=404.5,
    full_w=2160,
    full_h=810,
    eye_w=1080,
    eye_h=810,
)

#: Матрица победителя 07b: ``A = diag(1, −1, −1)`` (камера от головы), ``T_head_cam`` — та же.
WINNER_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
)


class FrozenConfigTest(unittest.TestCase):
    """Значения из 07b зафиксированы в config.py, а не «плавают» по коду."""

    def test_axis_convention_value(self) -> None:
        self.assertEqual(config.AXIS_CONVENTION, "+x −y −z")

    def test_rotation_and_translation(self) -> None:
        self.assertEqual(tuple(config.HEAD_CAM_ROTATION_RPY_DEG), (0.0, 180.0, 0.0))
        self.assertEqual(tuple(config.HEAD_CAM_TRANSLATION_M), (0.0, 0.0, 0.0))

    def test_flags_are_off(self) -> None:
        self.assertIs(config.INVERT_QUAT, False)
        self.assertIs(config.Y_DOWN_FLIP, False)

    def test_video_time_offset_is_frozen(self) -> None:
        # диагностика 29.10 (src/drift_diag.py): минимум ошибки наложения на −140 мс,
        # визуальная сверка −80…−140; зафиксировано −100 с точностью ±40 мс
        self.assertEqual(config.VIDEO_TIME_OFFSET_MS, -100.0)

    def test_head_cam_matrix_is_winner(self) -> None:
        matrix = head_cam_matrix()
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(matrix[i][j], WINNER_ROTATION[i][j], places=12)
            self.assertAlmostEqual(matrix[i][3], 0.0, places=12)
        self.assertEqual(
            axis_convention(), "+x −y −z"
        )


class CheckConfigTest(unittest.TestCase):
    """check_config — живая проверка конфига, а не константа в коде."""

    def test_summary_contains_winner(self) -> None:
        summary = check_config()
        self.assertEqual(summary["axis_convention"], "+x −y −z")
        self.assertFalse(summary["invert_quat"])
        self.assertFalse(summary["y_down_flip"])
        self.assertEqual(summary["head_cam_rotation_rpy_deg"], (0.0, 180.0, 0.0))
        self.assertEqual(summary["head_cam_translation_m"], (0.0, 0.0, 0.0))
        self.assertEqual(len(summary["head_cam_matrix"]), 4)

    def test_axis_string_must_match_matrix(self) -> None:
        # запись осей врёт о матрице — конфиг противоречив, падать нужно сразу
        with mock.patch.object(config, "AXIS_CONVENTION", "+x +y +z"):
            with self.assertRaises(ValueError):
                check_config()

    def test_y_down_flag_alone_is_allowed(self) -> None:
        # добавочный переворот y — отдельный флаг; сам T_head_cam остаётся поворотом
        with mock.patch.object(config, "Y_DOWN_FLIP", True):
            self.assertTrue(check_config()["y_down_flip"])

    def test_reflection_in_head_cam_rejected(self) -> None:
        # det = −1 в самом T_head_cam — это отражение, а не поворот камеры
        reflected = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        with mock.patch.object(camera_model, "head_cam_matrix", return_value=reflected), \
                mock.patch.object(config, "AXIS_CONVENTION", "+x −y +z"):
            with self.assertRaises(ValueError):
                check_config()

    def test_non_bool_flags_rejected(self) -> None:
        for name, value in (("Y_DOWN_FLIP", "False"), ("INVERT_QUAT", 1), ("Y_DOWN_FLIP", None)):
            with self.subTest(name=name, value=value):
                with mock.patch.object(config, name, value):
                    with self.assertRaises(ValueError):
                        check_config()

    def test_axis_convention_tracks_rotation(self) -> None:
        with mock.patch.object(config, "HEAD_CAM_ROTATION_RPY_DEG", (0.0, 0.0, 0.0)):
            self.assertEqual(axis_convention(), "+x +y +z")
        with mock.patch.object(config, "HEAD_CAM_ROTATION_RPY_DEG", (30.0, 20.0, 10.0)):
            self.assertEqual(axis_convention(), "custom")


class ConfigDrivesResultTest(unittest.TestCase):
    """Смена констант в config.py меняет результат без правок кода камеры."""

    def test_default_flags_come_from_config(self) -> None:
        pose = (0.3, -0.2, 0.1, 0.1, -0.2, 0.3, 0.92)
        self.assertEqual(
            world_to_camera_matrix(pose),
            world_to_camera_matrix(
                pose, opencv_conventions=config.Y_DOWN_FLIP, invert_quat=config.INVERT_QUAT
            ),
        )

    def test_y_down_flip_changes_default_result(self) -> None:
        pose = IDENTITY_POSE
        point = (0.0, 0.2, -1.0)
        before = world_to_camera(point, pose)
        self.assertAlmostEqual(before[1], -0.2, places=12)
        with mock.patch.object(config, "Y_DOWN_FLIP", True):
            after = world_to_camera(point, pose)
        self.assertAlmostEqual(after[1], 0.2, places=12)
        self.assertAlmostEqual(after[2], before[2], places=12)

    def test_invert_quat_comes_from_config(self) -> None:
        pose = (0.0, 0.0, 0.0, 0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5))  # yaw 90°
        point = (1.0, 0.0, 0.0)
        with mock.patch.object(config, "INVERT_QUAT", True):
            self.assertEqual(
                world_to_camera(point, pose),
                world_to_camera(point, pose, invert_quat=True),
            )

    def test_rotation_and_translation_from_config(self) -> None:
        # конфиг-заглушка: поворот нулевой и смещение 5 см — точка (0, 0, 1) даёт z = 0.95
        with mock.patch.object(config, "HEAD_CAM_ROTATION_RPY_DEG", (0.0, 0.0, 0.0)), \
                mock.patch.object(config, "AXIS_CONVENTION", "+x +y +z"), \
                mock.patch.object(config, "HEAD_CAM_TRANSLATION_M", (0.0, 0.0, 0.05)):
            camera = world_to_camera((0.0, 0.0, 1.0), IDENTITY_POSE)
            self.assertAlmostEqual(camera[2], 0.95, places=12)
            self.assertEqual(check_config()["axis_convention"], "+x +y +z")
        # без патча — снова калибровка 07c: точка перед головой даёт z = 1
        self.assertAlmostEqual(world_to_camera((0.0, 0.0, -1.0), IDENTITY_POSE)[2], 1.0, places=12)


class GoldenFrameTest(unittest.TestCase):
    """Рендер одного кадра: конфиг + интринсики дают конкретные пиксели (без данных)."""

    def test_point_in_front_projects_to_expected_pixels(self) -> None:
        point = (0.1, 0.2, -1.0)  # 10 см вправо, 20 см вверх, 1 м перед головой (−z)
        camera = world_to_camera(point, IDENTITY_POSE)
        self.assertAlmostEqual(camera[0], 0.1, places=12)
        self.assertAlmostEqual(camera[1], -0.2, places=12)  # y вниз: верх → −y
        self.assertAlmostEqual(camera[2], 1.0, places=12)
        pixel = project_eye(camera, FULL_INTRINSICS, "left")
        self.assertTrue(pixel.visible)
        self.assertAlmostEqual(pixel.u, 608.1868064882848, places=9)
        self.assertAlmostEqual(pixel.v, 267.1263870234303, places=9)

    def test_gaze_centre_and_backward_point(self) -> None:
        # перед головой → главная точка кадра; «за затылком» (+z головы) — за камерой
        centre = project_eye(world_to_camera((0.0, 0.0, -1.0), IDENTITY_POSE),
                             FULL_INTRINSICS, "left")
        self.assertTrue(centre.visible)
        self.assertAlmostEqual(centre.u, 539.5, places=9)
        self.assertAlmostEqual(centre.v, 404.5, places=9)
        behind = project_eye(world_to_camera((0.0, 0.0, 1.0), IDENTITY_POSE),
                             FULL_INTRINSICS, "left")
        self.assertFalse(behind.visible)

    def test_head_rotation_still_puts_gaze_centre(self) -> None:
        # поворот головы не сдвигает точку перед головой с главной точки кадра
        for quat in ((0.0, 0.0, 0.0, 1.0), (0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)),
                     (0.1626, -0.2861, -0.1165, 0.9300)):
            pose = (1.2, -0.4, 0.3, *quat)
            forward = camera_model.transform_point(
                camera_model.head_to_world(pose), (0.0, 0.0, -1.0)
            )
            pixel = project_eye(world_to_camera(forward, pose), FULL_INTRINSICS, "left")
            with self.subTest(pose=pose[:3]):
                self.assertTrue(pixel.visible)
                self.assertAlmostEqual(pixel.u, 539.5, places=9)
                self.assertAlmostEqual(pixel.v, 404.5, places=9)


@unittest.skipIf(DATA is None, NO_DATA)
@unittest.skipIf(not META.is_file(), NO_META)
class CalibratedMetricTest(unittest.TestCase):
    """Метрика конвенции из config.py на эталоне 07a — ровно результат 07b."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = calib_search.load_calib_data(DATA, META)
        cls.convention = calib_search.convention_from_config()

    def test_convention_matches_winner(self) -> None:
        # поворот из config.py: та же матрица 07b с точностью до float-шума sin(180°)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(
                    self.convention.rotation[i][j], WINNER_ROTATION[i][j], places=12
                )
        self.assertFalse(self.convention.invert_quat)
        self.assertEqual(self.convention.camera_offset, (0.0, 0.0, 0.0))

    def test_metric_reproduces_07b(self) -> None:
        # с поправкой времени кадра (``config.VIDEO_TIME_OFFSET_MS``, доработка 08a) в кадр
        # попадает 40 суставов вместо 39 — на k=30 ещё один; все 40 — в рамках своих рук
        metrics = calib_search.evaluate_convention(self.data, self.convention)
        self.assertEqual(metrics.good, 40)
        self.assertEqual(metrics.in_frame, 40)
        self.assertEqual(metrics.wrong, 0)
        self.assertEqual(metrics.miss, 0)
        self.assertEqual(metrics.mean_error_px, 0.0)
        self.assertEqual(metrics.total, 45)

    def test_metric_worse_without_calibration(self) -> None:
        # историческая заглушка 05b: ни один сустав не попадает в кадр
        stub = calib_search.Convention(
            rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        metrics = calib_search.evaluate_convention(self.data, stub)
        self.assertEqual(metrics.in_frame, 0)


if __name__ == "__main__":
    unittest.main()
