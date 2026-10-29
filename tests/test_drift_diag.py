"""Assert-тесты для src/drift_diag.py (доработка 08a: диагностика слёта скелета).

Запуск: python -m unittest discover -s tests -t . -v
Синтетика: генерируется маленький трек и сцена, картинка — прямоугольник «кожи»; больших
файлов и OpenCV-видео не нужно. Тесты картинки пропускаются без OpenCV.
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import camera_model, drift_diag as diag, overlay_video as ov, sync  # noqa: E402

try:  # pragma: no cover — окружение, а не логика
    import cv2
except ImportError:
    cv2 = None

MS = 1_000_000
INTRINSICS = [1079.5, 404.5, 1373.6681390725917, 686.8680648828482]
EYE_FULL = camera_model.Intrinsics.from_header({"cameraIntrinsics": INTRINSICS})
#: BGR «кожи» из записи (проходит HSV-порог диагностики: H ≈ 6, S ≈ 135, V ≈ 117).
SKIN_BGR = (55, 68, 117)
#: BGR стола (H ≈ 15 — за порогом кожи).
TABLE_BGR = (104, 126, 148)

JOINTS = ov.JOINTS


def _quat_yaw(deg: float) -> tuple[float, float, float, float]:
    """Кватернион поворота вокруг Y на ``deg`` градусов (порядок xyzw)."""
    half = math.radians(deg) / 2.0
    return (0.0, math.sin(half), 0.0, math.cos(half))


def make_track(
    times_ms: list[float],
    right_palm: list[tuple[float, float, float]],
    head_quats: list[tuple[float, float, float, float]] | None = None,
    head_x: list[float] | None = None,
) -> sync.PoseTrack:
    """Синтетический трек: правая кисть (5 точек в одной позиции) и голова в начале координат."""
    ts = [int(round(value * MS)) for value in times_ms]
    count = len(ts)
    values = {name: [0.0] * count for name in (*sync.POSITION_FIELDS, *sync.QUAT_FIELDS)}
    for index in range(count):
        values["head_x"][index] = 0.0 if head_x is None else head_x[index]
        values["head_qx"][index] = 0.0
        values["head_qy"][index] = 0.0
        values["head_qz"][index] = 0.0
        values["head_qw"][index] = 1.0
        if head_quats is not None:
            qx, qy, qz, qw = head_quats[index]
            values["head_qx"][index] = qx
            values["head_qy"][index] = qy
            values["head_qz"][index] = qz
            values["head_qw"][index] = qw
        for side in ("left", "right"):
            point = right_palm[index] if side == "right" else (0.0, 0.0, -0.4)
            for joint in JOINTS:
                for axis, value in zip(("x", "y", "z"), point):
                    values[f"{side}_{joint}_{axis}"][index] = value
    return sync.PoseTrack(ts, values, {"left": [True] * count, "right": [True] * count})


def make_scene(
    track: sync.PoseTrack, frame_count: int, duration: float, offset_ns: int = 0
) -> ov.Scene:
    """Сцена диагностики: метаданные видео + трек + интринсики + поправка времени."""
    return ov.Scene(
        meta={"frame_count": frame_count, "duration": duration},
        header_ns=int(track.ts[0]),
        tracking_end_ns=int(track.ts[-1]),
        track=track,
        intrinsics=EYE_FULL,
        time_offset_ns=offset_ns,
    )


def world_for_pixel(u: float, v: float, depth: float = 1.0) -> tuple[float, float, float]:
    """Мир (голова в нуле, калибровка 07c) для пикселя левого глаза на глубине ``depth``.

    Камера: ``x_cam = x_head``, ``y_cam = −y_head``, ``z_cam = −z_head``; поэтому
    ``v = fy·(−y)/z + cy``, т.е. ``y = (cy − v)·z/fy``.
    """
    eye = EYE_FULL.per_eye("left")
    x = (u - eye.cx) * depth / eye.fy
    y = (eye.cy - v) * depth / eye.fy
    return (x, y, -depth)


#: Пиксель, вокруг которого строится «кожа» в тестовой картинке.
SKIN_CENTER = (540.0, 405.0)


def skin_image(width: int = 1080, height: int = 810) -> np.ndarray:
    """Картинка в координатах левого глаза: «кожа» прямоугольником, вокруг — «стол»."""
    image = np.full((height, width, 3), TABLE_BGR, np.uint8)
    cx, cy = (int(value) for value in SKIN_CENTER)
    image[cy - 35 : cy + 35, cx - 60 : cx + 60] = SKIN_BGR
    return image


class SpeedAndTimingTest(unittest.TestCase):
    """Скорость по сырому треку и расстояние до ближайшего сэмпла."""

    def test_speed_of_uniform_motion(self) -> None:
        # 0.1 м за 100 мс → 1 м/с
        track = make_track([0, 100, 200], [(0.0, 0.0, -1.0), (0.1, 0.0, -1.0), (0.2, 0.0, -1.0)])
        speeds = diag.sample_speeds(track)
        self.assertAlmostEqual(speeds[1], 1.0, places=9)
        self.assertTrue(math.isnan(speeds[0]) and math.isnan(speeds[-1]))

    def test_single_missing_joint_does_not_zero_speed(self) -> None:
        track = make_track([0, 100, 200], [(0.0, 0.0, -1.0), (0.1, 0.0, -1.0), (0.2, 0.0, -1.0)])
        for axis in ("x", "y", "z"):
            track._values[f"right_thumb_{axis}"] = (math.nan, math.nan, math.nan)
        self.assertAlmostEqual(diag.sample_speeds(track)[1], 1.0, places=9)

    def test_speed_all_nan_is_nan(self) -> None:
        track = make_track([0, 100, 200], [(math.nan, math.nan, math.nan)] * 3)
        speeds = diag.sample_speeds(track)
        self.assertTrue(math.isnan(speeds[1]))

    def test_dt_to_nearest_is_distance_to_sample(self) -> None:
        track = make_track([0, 100, 200], [(0.0, 0.0, -1.0)] * 3)
        self.assertAlmostEqual(diag.dt_to_nearest_ms(track, 30 * MS), 30.0)
        self.assertAlmostEqual(diag.dt_to_nearest_ms(track, 130 * MS), 30.0)
        self.assertAlmostEqual(diag.dt_to_nearest_ms(track, 150 * MS), 50.0)  # ничья → раньше
        self.assertAlmostEqual(diag.dt_to_nearest_ms(track, 100 * MS), 0.0)

    def test_track_speed_uses_nearest_sample(self) -> None:
        track = make_track([0, 100, 200], [(0.0, 0.0, -1.0), (0.1, 0.0, -1.0), (0.2, 0.0, -1.0)])
        speeds = diag.sample_speeds(track)
        self.assertAlmostEqual(diag.track_speed_at_ms(track, speeds, 110 * MS), 1.0)

    def test_hand_depth_in_camera(self) -> None:
        # перед головой (0, 0, −0.4) → в камере Z = 0.4
        track = make_track([0], [(0.0, 0.0, -0.4)])
        pose = track.sample_pose(0)
        self.assertAlmostEqual(diag.hand_depth_m(pose), 0.4, places=9)

    def test_hand_depth_without_points_is_nan(self) -> None:
        track = make_track([0], [(math.nan, math.nan, math.nan)])
        self.assertTrue(math.isnan(diag.hand_depth_m(track.sample_pose(0))))


class HeadAnglesTest(unittest.TestCase):
    """Рысканье и размах поворота головы."""

    def test_yaw_of_rotated_head(self) -> None:
        track = make_track(
            [0, 100], [(0.0, 0.0, -1.0)] * 2, head_quats=[(0.0, 0.0, 0.0, 1.0), _quat_yaw(30.0)]
        )
        reference = track.sample_pose(0).head_quat
        yaw = diag.head_yaw_deg(track.sample_pose(1), reference)
        self.assertAlmostEqual(abs(yaw), 30.0, places=6)
        opposite = track.sample_pose(0)
        yaw_back = diag.head_yaw_deg(opposite, _quat_yaw(30.0))
        self.assertAlmostEqual(yaw, -yaw_back, places=6)

    def test_rotation_range_over_window(self) -> None:
        # поворот на 10° за сэмпл, сэмплы через 50 мс; окно ±50 мс = ±1 сэмпл → 20°
        quats = [_quat_yaw(10.0 * index) for index in range(5)]
        track = make_track([0, 50, 100, 150, 200], [(0.0, 0.0, -1.0)] * 5, head_quats=quats)
        scene = make_scene(track, frame_count=3, duration=0.3)  # кадр k=1 → 100 мс
        self.assertAlmostEqual(diag.head_rotation_range_deg(scene, 1), 20.0, places=6)

    def test_rotation_range_zero_for_static_head(self) -> None:
        track = make_track([0, 50, 100], [(0.0, 0.0, -1.0)] * 3)
        scene = make_scene(track, frame_count=3, duration=0.3)
        self.assertAlmostEqual(diag.head_rotation_range_deg(scene, 1), 0.0, places=9)


@unittest.skipIf(cv2 is None, "нет OpenCV")
class ImageMetricTest(unittest.TestCase):
    """Ошибка наложения по картинке: маска кожи, расстояние, лучший сдвиг времени."""

    def test_distance_map_inside_skin_is_zero(self) -> None:
        distance = diag.skin_distance_map(skin_image())
        self.assertEqual(float(distance[405, 540]), 0.0)  # внутри прямоугольника
        self.assertGreater(float(distance[10, 10]), 50.0)  # далеко от кожи
        self.assertGreater(float(distance[405, 460]), 10.0)  # левее прямоугольника
        self.assertGreater(float(distance[50, 540]), 10.0)  # «стол» кожей не считается

    def test_joint_error_zero_on_skin(self) -> None:
        track = make_track([0], [world_for_pixel(*SKIN_CENTER, depth=1.0)])
        scene = make_scene(track, frame_count=1, duration=0.1)
        distance = diag.skin_distance_map(skin_image())
        errors = diag.joint_errors_px(track.sample_pose(0), scene, distance)
        self.assertEqual(len(errors), 5)
        self.assertTrue(all(value == 0.0 for value in errors))

    def test_joint_error_grows_with_offset(self) -> None:
        inside = SKIN_CENTER
        shifted = (SKIN_CENTER[0] + 100.0, SKIN_CENTER[1])  # правее прямоугольника
        distance = diag.skin_distance_map(skin_image())
        track_in = make_track([0], [world_for_pixel(*inside, depth=1.0)])
        track_out = make_track([0], [world_for_pixel(*shifted, depth=1.0)])
        scene_in = make_scene(track_in, frame_count=1, duration=0.1)
        scene_out = make_scene(track_out, frame_count=1, duration=0.1)
        error_in = diag.error_px(track_in.sample_pose(0), scene_in, distance)
        error_out = diag.error_px(track_out.sample_pose(0), scene_out, distance)
        self.assertEqual(error_in, 0.0)
        self.assertGreater(error_out, 20.0)  # правый край кожи на 600, точка на 640

    def test_error_without_points_is_nan(self) -> None:
        track = make_track([0], [(math.nan, math.nan, math.nan)])
        scene = make_scene(track, frame_count=1, duration=0.1)
        distance = diag.skin_distance_map(skin_image())
        self.assertTrue(math.isnan(diag.error_px(track.sample_pose(0), scene, distance)))

    def test_best_offset_finds_time_shift(self) -> None:
        # кисть едет вправо на 0.2 м за 50 мс (4 м/с); «кожа» — там, где кисть была 100 мс назад
        positions = [(0.2 * index, 0.0, -1.0) for index in range(9)]  # 0…1.6 м
        track = make_track([50 * index for index in range(9)], positions)
        scene = make_scene(track, frame_count=2, duration=0.3)  # кадр k=1 → 150 мс
        eye = EYE_FULL.per_eye("left")
        corrected = scene.track.pose_at(
            int(sync.frame_time(1, scene.meta, scene.header_ns, scene.tracking_end_ns).ts_ns)
            + int(-100 * MS)
        )
        target = ov.project_hands(corrected, scene.intrinsics)["right"]["palm"]
        self.assertTrue(target.visible)
        image = np.full((810, 1080, 3), TABLE_BGR, np.uint8)
        u, v = int(round(target.u)), int(round(target.v))
        image[v - 15 : v + 15, u - 15 : u + 15] = SKIN_BGR
        distance = diag.skin_distance_map(image)
        error, delta = diag.best_offset_ms(scene, 1, distance)
        self.assertAlmostEqual(delta, -100.0, places=6)
        self.assertEqual(error, 0.0)

    def test_offset_scan_has_constant_minimum(self) -> None:
        positions = [(0.2 * index / 4.0, 0.0, -1.0) for index in range(5)]
        track = make_track([0, 50, 100, 150, 200], positions)
        scene = make_scene(track, frame_count=3, duration=0.3)
        image = skin_image()
        distance = diag.skin_distance_map(image)
        rows = [
            {"k": 0, "err_px": 1.0},
            {"k": 1, "err_px": 2.0},
            {"k": 2, "err_px": 3.0},
        ]
        curves = diag.offset_scan(scene, rows, {0: distance, 1: distance, 2: distance})
        self.assertEqual(set(curves), {"all", "early", "mid", "late"})
        self.assertEqual(len(curves["all"]), len(diag.OFFSET_GRID_MS))
        self.assertTrue(all(not math.isnan(value) for _, value, _ in curves["all"]))


class StatisticsTest(unittest.TestCase):
    """Ранги, корреляции, nan-safe среднее."""

    def test_rankdata_average_ranks(self) -> None:
        self.assertEqual(list(diag.rankdata([10, 10, 20])), [1.5, 1.5, 3.0])
        self.assertEqual(list(diag.rankdata([3, 1, 2])), [3.0, 1.0, 2.0])

    def test_spearman_extremes(self) -> None:
        self.assertAlmostEqual(diag.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=9)
        self.assertAlmostEqual(diag.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0, places=9)
        self.assertTrue(math.isnan(diag.spearman([1], [2])))
        self.assertTrue(math.isnan(diag.spearman([1, 1, 1], [1, 2, 3])))

    def test_spearman_skips_nan_pairs(self) -> None:
        self.assertAlmostEqual(
            diag.spearman([1, math.nan, 3, 4], [10, 20, 30, 40]), 1.0, places=9
        )
        self.assertTrue(math.isnan(diag.spearman([math.nan, math.nan], [1, 2])))

    def test_pearson(self) -> None:
        self.assertAlmostEqual(diag.pearson([1, 2, 3], [2, 4, 6]), 1.0, places=9)
        self.assertTrue(math.isnan(diag.pearson([], [])))

    def test_nanmean(self) -> None:
        self.assertAlmostEqual(diag.nanmean([1.0, math.nan, 3.0]), 2.0)
        self.assertTrue(math.isnan(diag.nanmean([math.nan, math.nan])))


class CohortTest(unittest.TestCase):
    """Отбор когорты и сводка."""

    def test_select_fast_frames_picks_the_moving_one(self) -> None:
        # ровное движение 4 м/с: кадр k=1 (t = 100 мс) — в середине быстрого участка
        positions = [(0.2 * index, 0.0, -1.0) for index in range(5)]
        track = make_track([0, 50, 100, 150, 200], positions)
        scene = make_scene(track, frame_count=3, duration=0.3)
        frames, frame_speed, speeds = diag.select_fast_frames(scene, count=1)
        self.assertEqual(frames, [1])
        self.assertAlmostEqual(float(frame_speed[1]), 4.0, places=6)
        self.assertEqual(len(speeds), len(track))

    def test_select_fast_frames_skips_nan(self) -> None:
        track = make_track([0, 50, 100], [(math.nan, math.nan, math.nan)] * 3)
        scene = make_scene(track, frame_count=3, duration=0.3)
        frames, frame_speed, _ = diag.select_fast_frames(scene, count=2)
        self.assertEqual(frames, [])
        self.assertTrue(np.all(np.isnan(frame_speed)))

    def test_describe_contains_correlations(self) -> None:
        rows = [
            {
                "k": index,
                "t_s": index * 0.033,
                "dt_to_nearest_ms": 1.0,
                "speed_mps": 1.0 + 0.1 * index,
                "depth_z_m": 0.4,
                "head_yaw_deg": 10.0,
                "head_rotation_range_deg": 2.0,
                "err_px": 70.0 - 5.0 * index,
                "on_skin_frac": 0.2,
                "err_palm_wrist_px": 40.0,
                "joints_in_frame": 5,
                "err_at_best_px": 30.0,
                "best_offset_ms": -120.0,
            }
            for index in range(5)
        ]
        frame_speed = np.array([0.4, 0.5, math.nan, 1.6, 1.7])
        curves = {
            "all": [(float(delta), 60.0 + 0.1 * abs(delta), len(rows))
                    for delta in diag.OFFSET_GRID_MS],
            "early": [],
            "mid": [],
            "late": [],
        }
        text = diag.describe(rows, curves, frame_speed)
        self.assertIn("Корреляции", text)
        self.assertIn("speed_mps", text)
        self.assertIn("depth_z_m", text)
        self.assertIn("Когорта: 5", text)

    def test_write_table_roundtrip(self) -> None:
        rows = [
            {
                "k": 10,
                "t_s": 0.3,
                "dt_to_nearest_ms": 2.0,
                "speed_mps": 1.5,
                "depth_z_m": 0.4,
                "head_yaw_deg": 12.0,
                "head_rotation_range_deg": 3.0,
                "err_px": 55.0,
                "on_skin_frac": 0.2,
                "err_palm_wrist_px": 30.0,
                "joints_in_frame": 5,
                "err_at_best_px": 20.0,
                "best_offset_ms": -100.0,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.csv"
            diag.write_table(rows, path)
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        for column in ("k", "dt_to_nearest_ms", "speed_mps", "depth_z_m", "head_yaw_deg",
                       "head_rotation_range_deg", "err_px"):
            self.assertIn(column, lines[0])
        self.assertTrue(lines[1].startswith("10,"))


if __name__ == "__main__":
    unittest.main()
