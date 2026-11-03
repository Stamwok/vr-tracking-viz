"""Assert-тесты для src/overlay_video.py (задача 08a: рендер скелета кистей).

Запуск: python -m unittest discover -s tests -t . -v
Тесты на реальных данных пропускаются, если нет data/ или output/video_meta.json
(данные не коммитятся). Тесты с OpenCV пропускаются, если cv2 недоступен.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src import calib_frames, camera_model, config, overlay_video as ov, sync  # noqa: E402
from src.parse_tracking import HAND_JOINTS_FULL  # noqa: E402

try:  # pragma: no cover — проверка окружения, а не логики
    import cv2
except ImportError:
    cv2 = None

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
VIDEO = next(iter(sorted((ROOT / "data").glob("CameraRecord_*.mp4"))), None)
META_PATH = ROOT / "output" / "video_meta.json"
NO_DATA = "нет data/trackingData_*.txt или output/video_meta.json (данные не коммитятся)"
NO_CV2 = "нет OpenCV (opencv-python-headless ставится из requirements.txt)"
NO_VIDEO = "нет data/CameraRecord_*.mp4 (видео не коммитится)"

# --- синтетические данные: тот же формат, что у реального файла трекинга --------------

BASE_NS = 1_700_000_000_000_000_000
INTRINSICS = [1079.5, 404.5, 1373.6681390725917, 686.8680648828482]
EYE = camera_model.Intrinsics.from_header({"cameraIntrinsics": INTRINSICS})
IDENTITY4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
#: Индексы 5 отрисовываемых суставов в HandJointLocations (как в parse_tracking).
JOINT_INDICES = {"palm": 0, "wrist": 1, "thumb": 5, "index": 10, "middle": 15}

HEADER = {
    "notice": "синтетический header для тестов 08a",
    "timeStampNs": BASE_NS,
    "cameraIntrinsics": json.dumps(INTRINSICS),
    "cameraExtrinsics": f"{json.dumps(IDENTITY4)}|{json.dumps(IDENTITY4)}",
}

#: Точки левой/правой руки в мире: перед головой (в OpenXR вперёд — это −z).
LEFT_POINTS = {
    "palm": (0.10, 0.00, -1.0),
    "wrist": (0.11, 0.00, -1.0),
    "thumb": (0.12, 0.01, -1.0),
    "index": (0.13, -0.01, -1.0),
    "middle": (0.14, 0.00, -1.0),
}
RIGHT_POINTS = {
    "palm": (0.20, 0.10, -1.0),
    "wrist": (0.21, 0.10, -1.0),
    "thumb": (0.22, 0.11, -1.0),
    "index": (0.23, 0.09, -1.0),
    "middle": (0.24, 0.10, -1.0),
}


def _num(value: float) -> str:
    """Число с десятичной запятой — как в реальном файле трекинга (03b)."""
    return f"{value:.8f}".replace(".", ",")


def _pose_string(values) -> str:
    return ",".join(_num(float(value)) for value in values)


def _joint(position) -> dict:
    return {"p": _pose_string([*position, 0.0, 0.0, 0.0, 1.0]), "s": 15, "r": 0.0}


def _hand(points: dict) -> dict:
    """26 суставов: нужные 5 — по местам OpenXR, остальные копия ладони (валидные)."""
    joints = [_joint(points["palm"]) for _ in range(26)]
    for name, index in JOINT_INDICES.items():
        joints[index] = _joint(points[name])
    return {"isActive": 1, "count": 26, "scale": 1.0, "HandJointLocations": joints}


def _write_tracking(
    path: Path, times_ns, head_pos=(0.0, 0.0, 0.0), left_series=None
) -> None:
    """Файл формата 03a: header + по одному JSON на сэмпл.

    ``left_series`` — по одному набору точек левой руки на сэмпл (для кадров с движением).
    """
    lines = [json.dumps(HEADER)]
    for index, ts_ns in enumerate(times_ns):
        left = LEFT_POINTS if left_series is None else left_series[index]
        sample = {
            "timeStampNs": int(ts_ns),
            "predictTime": 0.0,
            "Head": {"pose": _pose_string([*head_pos, 0.0, 0.0, 0.0, 1.0]), "status": 3},
            "Hand": {"leftHand": _hand(left), "rightHand": _hand(RIGHT_POINTS)},
        }
        lines.append(json.dumps(sample))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pose(points: dict | None = None) -> sync.Pose:
    """Синтетическая поза: голова в начале координат, руки перед ней."""
    points = points or {"left": LEFT_POINTS, "right": RIGHT_POINTS}
    hands = {}
    for side, joints in points.items():
        for joint, position in joints.items():
            hands[f"{side}_{joint}"] = tuple(position)
    return sync.Pose(
        ts_ns=BASE_NS,
        head_pos=(0.0, 0.0, 0.0),
        head_quat=(0.0, 0.0, 0.0, 1.0),
        hands=hands,
        active={"left": True, "right": True},
    )


#: Невырожденная «ладонь»: точки разнесены так, что линии не накладываются друг на друга.
PIXEL_SHAPE = {
    "palm": (100.0, 100.0),
    "wrist": (100.0, 200.0),
    "thumb": (200.0, 100.0),
    "index": (100.0, 20.0),
    "middle": (200.0, 20.0),
}
RIGHT_OFFSET = 600.0


def _points(visible: bool = True) -> dict:
    """Готовые пиксели для draw_skeleton: 5 точек каждой руки, руки разнесены по кадру."""
    return {
        side: {
            joint: camera_model.ProjectedPoint(
                x + (RIGHT_OFFSET if side == "right" else 0.0), y, visible
            )
            for joint, (x, y) in PIXEL_SHAPE.items()
        }
        for side in ("left", "right")
    }


# --- 08b: след и HUD ---------------------------------------------------------------------

#: Синтетический трек: левая ладонь идёт по заданным точкам, правая — «нет данных».
TRAIL_STEP_NS = 100_000_000


def _track_with_left_palm(positions, step_ns: int = TRAIL_STEP_NS) -> sync.PoseTrack:
    """PoseTrack с заданным движением левой ладони (остальные точки — в нуле, правая — NaN)."""
    from src.sync import POINT_KEYS, POSITION_FIELDS, QUAT_FIELDS

    count = len(positions)
    values = {name: [0.0] * count for name in (*POSITION_FIELDS, *QUAT_FIELDS)}
    values["head_qw"] = [1.0] * count
    for key in POINT_KEYS:
        if key == "left_palm":
            continue
        for axis in "xyz":
            values[f"{key}_{axis}"] = [math.nan if key.startswith("right_") else 0.0] * count
    for index, position in enumerate(positions):
        for axis, value in zip("xyz", position):
            values[f"left_palm_{axis}"][index] = value
    return sync.PoseTrack(
        [index * step_ns for index in range(count)],
        values,
        {"left": [True] * count, "right": [False] * count},
    )


def _frame_time(index: int = 0, t_rel_s: float = 0.0, in_tracking: bool = True) -> sync.FrameTime:
    return sync.FrameTime(index=index, ts_ns=BASE_NS, t_rel_s=t_rel_s, in_tracking=in_tracking)


class TrailSamplingTest(unittest.TestCase):
    """Выборка следа: по времени, а не по номерам кадров."""

    def test_times_are_ascending_and_time_based(self) -> None:
        times = ov.trail_times(10_000_000_000, seconds=0.2, step_s=0.05)
        self.assertEqual(len(times), 5)  # 0.2/0.05 + текущий момент
        self.assertEqual(times, tuple(sorted(times)))
        self.assertEqual(times[-1], 10_000_000_000)
        self.assertEqual([b - a for a, b in zip(times, times[1:])], [50_000_000] * 4)
        # окно задано временем: удвоение шага вдвое уменьшает число выборок, а не окно
        self.assertEqual(len(ov.trail_times(0, seconds=0.2, step_s=0.1)), 3)

    def test_window_validation(self) -> None:
        for seconds, step_s in ((0.0, 0.1), (-0.1, 0.1), (0.1, 0.0), (0.1, 0.2), (math.nan, 0.1)):
            with self.subTest(seconds=seconds, step_s=step_s), self.assertRaises(ValueError):
                ov.trail_times(0, seconds=seconds, step_s=step_s)
        for bad in (True, "1", None):
            with self.subTest(ts=bad), self.assertRaises(ValueError):
                ov.trail_times(bad)  # type: ignore[arg-type]
        with self.subTest(step_bigger_than_window=True), self.assertRaises(ValueError):
            ov.trail_times(0, seconds=0.2, step_s=0.5)

    def test_points_are_clipped_to_tracking_range(self) -> None:
        track = _track_with_left_palm([(0.0, 0.0, -1.0)] * 3)  # 0, 0.1, 0.2 с
        # след от t = 0.05 с: момент «−0.15 с» уже до первого сэмпла — его быть не должно
        points = ov.trail_points(track, 50_000_000, seconds=0.2, step_s=0.05)
        self.assertLessEqual(len(points["left"]), 3)
        for position in points["left"]:
            self.assertEqual(position, (0.0, 0.0, -1.0))
        # за концом трекинга ничего не добавляется: линия не «прилипает» к последнему кадру
        tail = ov.trail_points(track, 250_000_000, seconds=0.2, step_s=0.05)
        self.assertEqual(len(tail["left"]), 4)  # 50, 100, 150 и 200 мс; 250 мс уже нет
        self.assertEqual(ov.trail_points(track, 5_000_000_000)["left"], [])

    def test_nan_hand_keeps_nan_in_points(self) -> None:
        track = _track_with_left_palm([(0.0, 0.0, -1.0), (0.1, 0.0, -1.0)])
        points = ov.trail_points(track, 100_000_000, seconds=0.1, step_s=0.1)
        self.assertEqual(len(points["right"]), 2)
        for position in points["right"]:
            self.assertTrue(all(math.isnan(value) for value in position))

    def test_unknown_joint_is_rejected(self) -> None:
        track = _track_with_left_palm([(0.0, 0.0, -1.0)])
        with self.assertRaises(ValueError):
            ov.trail_points(track, 0, joint="elbow")
        with self.assertRaises(ValueError):
            ov.trail_points(object(), 0)  # type: ignore[arg-type]


@unittest.skipIf(cv2 is None, NO_CV2)
class DrawTrailTest(unittest.TestCase):
    """Отрисовка следа: видимые отрезки, разрывы на «нет данных», затухание."""

    #: Ладонь едет по x: u = 686.868·x + 539.5 при глубине 1 м (поза головы единичная).
    POSITIONS = ((-0.2, 0.0, -1.0), (-0.1, 0.0, -1.0), (0.0, 0.0, -1.0))

    def _image(self) -> np.ndarray:
        return np.zeros((810, 1080, 3), np.uint8)

    def _draw(self, positions=None, **kwargs) -> tuple[int, np.ndarray]:
        track = _track_with_left_palm(list(positions or self.POSITIONS))
        image = self._image()
        pose = track.pose_at(track.ts[-1])
        drawn = ov.draw_trail(image, pose, track, EYE, ov.style_for(1080), seconds=0.2, step_s=0.1, **kwargs)
        return drawn, image

    def test_draws_one_segment_per_step(self) -> None:
        drawn, image = self._draw()
        self.assertEqual(drawn, 2)  # 3 выборки → 2 отрезка; правая рука — NaN, отрезков нет
        self.assertGreater(int(image[..., 0].sum()), 0)  # цвет левой руки (BGR: B)

    def test_nan_breaks_the_line(self) -> None:
        # 4 выборки → 3 отрезка; «нет данных» во второй выборке рвёт два отрезка из трёх
        nan = (math.nan, math.nan, math.nan)
        track = _track_with_left_palm(
            [(-0.2, 0.0, -1.0), nan, (0.0, 0.0, -1.0), (0.1, 0.0, -1.0)]
        )
        image = self._image()
        pose = track.pose_at(track.ts[-1])
        drawn = ov.draw_trail(image, pose, track, EYE, ov.style_for(1080), seconds=0.3, step_s=0.1)
        self.assertEqual(drawn, 1, "линия через NaN не проводится, но остальные отрезки рисуются")

    def test_invisible_end_breaks_the_line(self) -> None:
        # средняя точка за камерой (z > 0 в мире при повороте камеры на 180° вокруг x)
        drawn, _image = self._draw([(0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.1, 0.0, -1.0)])
        self.assertEqual(drawn, 0)

    def test_newest_segment_is_more_opaque(self) -> None:
        drawn, image = self._draw()
        self.assertEqual(drawn, 2)
        # отрезки: старый (u ≈ 402→471) и новый (u ≈ 471→539); B-канал = 255·alpha
        old = int(image[404, 436, 0])
        new = int(image[404, 505, 0])
        self.assertGreater(new, old + 40)
        self.assertAlmostEqual(new / 255.0, ov.TRAIL_ALPHA_NEW, delta=0.06)

    def test_bad_arguments_raise(self) -> None:
        track = _track_with_left_palm([(0.0, 0.0, -1.0)])
        pose = track.pose_at(0)
        with self.assertRaises(ValueError):
            ov.draw_trail(self._image(), pose, track, EYE, ov.style_for(1080), alpha_old=0.9, alpha_new=0.1)
        with self.assertRaises(ValueError):
            ov.draw_trail(self._image(), pose, track, EYE, ov.style_for(1080), alpha_new=0.0)
        with self.assertRaises(ValueError):
            ov.draw_trail(None, pose, track, EYE, ov.style_for(1080))
        with self.assertRaises(ValueError):
            ov.draw_trail(self._image(), pose, track, EYE, style=None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ov._blend_line(self._image(), (0, 0), (10, 10), (0, 0, 255), 2, 0.0)  # noqa: SLF001


class HudTest(unittest.TestCase):
    """HUD: фиксированная панель, состояние рук, вписывание текста."""

    def test_rect_is_fixed_and_scales(self) -> None:
        self.assertEqual(ov.hud_rect((1080, 810)), (12, 12, 342, 148))
        self.assertEqual(ov.hud_rect((540, 405)), (6, 6, 171, 74))
        self.assertEqual(ov.hud_rect((2160, 1620)), (24, 24, 684, 296))

    def test_rect_validation(self) -> None:
        for bad in ((0, 810), (1080, 0), (1080,), "1080x810", (1080.0, 810)):
            with self.subTest(size=bad), self.assertRaises(ValueError):
                ov.hud_rect(bad)  # type: ignore[arg-type]

    def test_rows_report_hands_and_trail(self) -> None:
        pose = _pose()
        rows = ov.hud_items(_frame_time(3, 1.5), pose, 100)
        texts = [text for text, _color in rows]
        self.assertEqual(texts[0], "t =    1.500 s")
        self.assertEqual(texts[1], "frame 3 / 99")
        self.assertEqual(texts[2], "left: OK")
        self.assertEqual(texts[3], "right: OK")
        self.assertEqual(texts[4], "trail 0.6 s")
        self.assertEqual(rows[2][1], ov.HUD_TEXT_COLOR)

    def test_no_data_row_is_warning_coloured(self) -> None:
        inactive = _pose()
        inactive.active["right"] = False
        rows = ov.hud_items(_frame_time(), inactive, 100)
        self.assertIn(("right: NO DATA", ov.HUD_WARNING_COLOR), rows)
        self.assertIn(("left: OK", ov.HUD_TEXT_COLOR), rows)

    def test_frozen_pose_and_trail_off_are_reported(self) -> None:
        rows = ov.hud_items(_frame_time(in_tracking=False), _pose(), 10, trail_seconds=None)
        self.assertEqual(rows[-1][0], "trail off | pose frozen")

    def test_items_validation(self) -> None:
        with self.assertRaises(ValueError):
            ov.hud_items("moment", _pose(), 10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ov.hud_items(_frame_time(), _pose(), 0)
        with self.assertRaises(ValueError):
            ov.hud_items(_frame_time(index=10), _pose(), 10)
        with self.assertRaises(ValueError):
            ov.hud_items(_frame_time(), _pose(), 10, trail_seconds=0.0)

    @unittest.skipIf(cv2 is None, NO_CV2)
    def test_draw_paints_only_panel(self) -> None:
        pose = _pose()
        pose.active["right"] = False
        image = np.full((810, 1080, 3), 255, np.uint8)
        before = image.copy()
        rect = ov.draw_hud(image, _frame_time(3, 1.5), pose, 100)
        self.assertEqual(rect, ov.hud_rect((1080, 810)))
        x0, y0, x1, y1 = rect
        outside = np.ones(image.shape[:2], bool)
        outside[y0:y1, x0:x1] = False
        self.assertTrue(np.array_equal(image[outside], before[outside]), "HUD не должен задевать кадр вне панели")
        self.assertLess(int(image[y0 + 3, x0 + 3].mean()), 250)
        blue, green, red = (image[..., i].astype(int) for i in range(3))
        warning = (red > 170) & (green > 120) & (blue < 90)
        self.assertGreater(int(warning[y0:y1, x0:x1].sum()), 0, "нет предупреждения «нет данных»")

    @unittest.skipIf(cv2 is None, NO_CV2)
    def test_text_fits_panel(self) -> None:
        image = np.zeros((810, 1080, 3), np.uint8)
        # длинные строки (кадр 7000+, оба состояния, заморозка) — все должны вписываться
        pose = _pose()
        for inactive in (False, True):
            pose.active["right"] = not inactive
            for in_tracking in (True, False):
                with self.subTest(inactive=inactive, in_tracking=in_tracking):
                    ov.draw_hud(image, _frame_time(7200, 240.027, in_tracking), pose, 7279)


class StyleTest(unittest.TestCase):
    """Радиус и толщина: пропорциональны кадру и множителю ``--scale``."""

    def test_default_style_reference_frame(self) -> None:
        style = ov.style_for(1080)
        self.assertEqual((style.point_radius, style.line_thickness), (4, 2))

    def test_style_scales_with_frame_width(self) -> None:
        style = ov.style_for(2160)
        self.assertEqual((style.point_radius, style.line_thickness), (8, 4))

    def test_cli_scale_multiplies(self) -> None:
        style = ov.style_for(1080, 2.5)
        self.assertEqual((style.point_radius, style.line_thickness), (10, 5))

    def test_style_never_below_one_pixel(self) -> None:
        style = ov.style_for(1080, 0.01)
        self.assertEqual((style.point_radius, style.line_thickness), (1, 1))

    def test_invalid_width_and_scale(self) -> None:
        for bad in (0, -1080, 1080.0, True, "1080"):
            with self.subTest(width=bad), self.assertRaises(ValueError):
                ov.style_for(bad)  # type: ignore[arg-type]
        for bad in (0, -1, math.nan, math.inf, True, "1"):
            with self.subTest(scale=bad), self.assertRaises(ValueError):
                ov.style_for(1080, bad)  # type: ignore[arg-type]

    def test_style_rejects_non_positive_thickness(self) -> None:
        for kwargs in ({"point_radius": 0, "line_thickness": 2}, {"point_radius": 2, "line_thickness": -1}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                ov.Style(**kwargs)
        with self.assertRaises(ValueError):
            ov.Style(point_radius=2.0, line_thickness=2)  # type: ignore[arg-type]


class FramesTest(unittest.TestCase):
    """Диапазон кадров и разбор ``--frames``."""

    META = {"frame_count": 10}

    def test_full_range_by_default(self) -> None:
        self.assertEqual(ov.frame_range(self.META), (0, 9))

    def test_explicit_range(self) -> None:
        self.assertEqual(ov.frame_range(self.META, 2, 7), (2, 7))

    def test_invalid_ranges(self) -> None:
        for start, end in ((-1, 3), (5, 4), (0, 10), (11, 12)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                ov.frame_range(self.META, start, end)
        for bad in (1.0, True, "3"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                ov.frame_range(self.META, bad)  # type: ignore[arg-type]

    def test_missing_frame_count(self) -> None:
        for meta in ({}, {"frame_count": 0}, {"frame_count": 2.0}, {"frame_count": True}):
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                ov.frame_range(meta)

    def test_parse_frames(self) -> None:
        self.assertEqual(ov.parse_frames("30, 1800,3600"), (30, 1800, 3600))
        self.assertEqual(ov.parse_frames("7"), (7,))

    def test_parse_frames_rejects_garbage(self) -> None:
        for bad in ("", "  ", "a", "1,1", "1.5"):
            with self.subTest(text=bad), self.assertRaises(ValueError):
                ov.parse_frames(bad)
        with self.assertRaises(ValueError):
            ov.parse_frames(None)  # type: ignore[arg-type]


class ProjectHandsTest(unittest.TestCase):
    """3D → пиксели левого глаза: те же правила видимости, что в 05c."""

    def test_identity_head_centres_points(self) -> None:
        pose = _pose()
        pixels = ov.project_hands(pose, EYE)
        self.assertEqual(sorted(pixels), ["left", "right"])
        for side in ("left", "right"):
            self.assertEqual(set(pixels[side]), set(ov.JOINTS))
        # camera: x = x_head, y = −y_head, z = −z_head (калибровка 07c), голова в (0, 0, 0)
        palm = pixels["left"]["palm"]
        self.assertTrue(palm.visible)
        self.assertAlmostEqual(palm.u, EYE.fy * 0.10 + 539.5, places=9)
        self.assertAlmostEqual(palm.v, 404.5, places=9)

    def test_right_hand_is_offset_by_its_own_coordinates(self) -> None:
        palm = ov.project_hands(_pose(), EYE)["right"]["palm"]
        self.assertAlmostEqual(palm.u, EYE.fy * 0.20 + 539.5, places=9)
        self.assertAlmostEqual(palm.v, 404.5 - EYE.fy * 0.10, places=9)

    def test_point_behind_head_is_invisible(self) -> None:
        points = {side: dict(joints) for side, joints in (("left", LEFT_POINTS), ("right", RIGHT_POINTS))}
        points["left"]["thumb"] = (0.1, 0.0, 1.0)  # за затылком (в OpenXR назад — +z)
        thumb = ov.project_hands(_pose(points), EYE)["left"]["thumb"]
        self.assertFalse(thumb.visible)
        self.assertTrue(math.isnan(thumb.u) and math.isnan(thumb.v))

    def test_nan_joint_is_invisible(self) -> None:
        points = {side: dict(joints) for side, joints in (("left", LEFT_POINTS), ("right", RIGHT_POINTS))}
        points["right"]["index"] = (math.nan, math.nan, math.nan)
        index = ov.project_hands(_pose(points), EYE)["right"]["index"]
        self.assertFalse(index.visible)

    def test_right_eye_differs_by_eye_width(self) -> None:
        left = ov.project_hands(_pose(), EYE, "left")["left"]["palm"]
        right = ov.project_hands(_pose(), EYE, "right")["left"]["palm"]
        self.assertAlmostEqual(right.u - left.u, EYE.eye_w, places=9)

    def test_rejects_per_eye_intrinsics(self) -> None:
        with self.assertRaises(ValueError):
            ov.project_hands(_pose(), EYE.per_eye("left"))


@unittest.skipIf(cv2 is None, NO_CV2)
class DrawSkeletonTest(unittest.TestCase):
    """Отрисовка: точки, линии, цвета, невидимые точки и валидация входа."""

    SIZE = (810, 1080, 3)

    def setUp(self) -> None:
        self.image = np.zeros(self.SIZE, np.uint8)
        self.style = ov.Style(point_radius=4, line_thickness=2)

    def test_draws_five_points_per_hand(self) -> None:
        drawn = ov.draw_skeleton(self.image, _points(), self.style)
        self.assertEqual(drawn, 10)
        pixel = self.image[100, 100]  # ладонь левой руки: BGR (255, 140, 0)
        self.assertEqual((int(pixel[0]), int(pixel[1]), int(pixel[2])), ov.COLORS["left"])

    def test_right_hand_has_its_own_colour(self) -> None:
        ov.draw_skeleton(self.image, _points(), self.style)
        right_palm = _points()["right"]["palm"]
        pixel = self.image[int(right_palm.v), int(right_palm.u)]
        self.assertEqual((int(pixel[0]), int(pixel[1]), int(pixel[2])), ov.COLORS["right"])

    def test_lines_connect_bones(self) -> None:
        ov.draw_skeleton(self.image, _points(), self.style)
        # ладонь (100, 100) → запястье (100, 200): середина отрезка закрашена
        self.assertGreater(int(self.image[150, 100].max()), 0)
        # ладонь → большой палец (200, 100): середина тоже
        self.assertGreater(int(self.image[100, 150].max()), 0)

    def test_invisible_point_is_skipped_with_its_bone(self) -> None:
        pixels = _points()
        pixels["left"]["thumb"] = camera_model.ProjectedPoint(200.0, 100.0, False)
        drawn = ov.draw_skeleton(self.image, pixels, self.style)
        self.assertEqual(drawn, 9)
        # ладонь (100, 100) → большой палец (200, 100): линия НЕ нарисована
        self.assertEqual(int(self.image[100, 150].max()), 0)

    def test_off_frame_point_still_drawn(self) -> None:
        pixels = _points()
        for joint in ov.JOINTS:
            pixels["left"][joint] = camera_model.ProjectedPoint(-40.0, -40.0, True)
        drawn = ov.draw_skeleton(self.image, pixels, self.style)
        self.assertEqual(drawn, 10)  # точки вне кадра не теряются, cv2 их обрезает

    def test_rejects_bad_image_or_style(self) -> None:
        for image in (np.zeros((10, 10), np.uint8), None):
            with self.subTest(image=image), self.assertRaises(ValueError):
                ov.draw_skeleton(image, _points(), self.style)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ov.draw_skeleton(self.image, _points(), (4, 2))  # type: ignore[arg-type]

    def test_rejects_missing_hand(self) -> None:
        pixels = _points()
        del pixels["right"]
        with self.assertRaises(ValueError):
            ov.draw_skeleton(self.image, pixels, self.style)


@unittest.skipIf(cv2 is None, NO_CV2)
class SyntheticVideoCase(unittest.TestCase):
    """Общая заготовка: 6-кадровое SBS-видео и файл трекинга формата 03a.

    Отдельный класс без тестов, чтобы ``SyntheticRenderTest`` и ``DenseRenderTest`` (08f)
    не наследовали тесты друг друга.
    """

    FRAMES = 6
    FPS = 30.0

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.video = self.dir / "CameraRecord_test.mp4"
        self.meta_path = self.dir / "video_meta.json"
        self.tracking = self.dir / "trackingData_test.txt"
        self.out = self.dir / "overlay.mp4"
        self._write_video(self.FRAMES, self.FRAMES)

    def _write_video(
        self,
        frames: int,
        meta_frames: int,
        tracking_samples: int | None = None,
        left_series=None,
    ) -> None:
        writer = cv2.VideoWriter(
            str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (2160, 810)
        )
        self.assertTrue(writer.isOpened())
        image = np.full((810, 2160, 3), 40, np.uint8)
        for _ in range(frames):
            writer.write(image)
        writer.release()
        duration = meta_frames / self.FPS
        self.meta_path.write_text(
            json.dumps(
                {
                    "frame_count": meta_frames,
                    "duration": duration,
                    "width": 2160,
                    "height": 810,
                }
            ),
            encoding="utf-8",
        )
        step_ns = duration / meta_frames
        samples = frames if tracking_samples is None else tracking_samples
        _write_tracking(
            self.tracking,
            [BASE_NS + int(round(k * step_ns * 1e9)) for k in range(samples)],
            left_series=left_series,
        )

    def _scene(self) -> ov.Scene:
        return ov.load_scene(self.meta_path, self.tracking)


class SyntheticRenderTest(SyntheticVideoCase):
    """Рендер на синтетическом SBS-видео: те же файлы, что у реальной записи, но 6 кадров."""

    def test_scene_reads_time_offset_from_config(self) -> None:
        with mock.patch.object(config, "VIDEO_TIME_OFFSET_MS", -40.0):
            scene = ov.load_scene(self.meta_path, self.tracking)
        self.assertEqual(scene.time_offset_ns, -40_000_000)

    def test_time_offset_moves_moving_hand(self) -> None:
        # левая кисть едет вправо на 0.05 м за сэмпл (сэмплы идут по кадрам, 33.3 мс)
        left_series = [
            {joint: (x + 0.05 * index, y, z) for joint, (x, y, z) in LEFT_POINTS.items()}
            for index in range(self.FRAMES)
        ]
        self._write_video(self.FRAMES, self.FRAMES, left_series=left_series)
        scene = self._scene()
        columns = {}
        for label, offset in (("zero", 0), ("shift", -100 * 10**6)):
            paths = ov.render_stills(
                self.video,
                replace(scene, time_offset_ns=offset),
                self.dir / label,
                frames=(5,),
                log=lambda _: None,
            )
            image = cv2.imread(str(paths[0]))
            blue = (
                (image[..., 0] > 235)
                & (np.abs(image[..., 1].astype(int) - 140) < 12)
                & (image[..., 2] < 20)
            )
            self.assertGreater(int(blue.sum()), 0)
            columns[label] = float(np.argwhere(blue)[:, 1].mean())
        # сдвиг на 100 мс = 3 сэмпла × 0.05 м = 0.15 м; на глубине 1 м это 0.15·fx ≈ 103 px
        self.assertAlmostEqual(columns["zero"] - columns["shift"], 0.15 * EYE.fy, delta=10)

    def test_scene_fps_follows_frame_count_over_duration(self) -> None:
        scene = self._scene()
        self.assertEqual(scene.frame_count, 6)
        self.assertEqual(scene.fps, 30.0)
        self.assertEqual(scene.eye_size, (1080, 810))
        self.assertEqual(len(scene.track), 6)

    def test_render_fragment_writes_calibrated_skeleton(self) -> None:
        stats = ov.render_video(
            self.video, self.out, self._scene(), start=1, end=4, progress_every=0, log=lambda _: None
        )
        self.assertEqual((stats.first_frame, stats.last_frame), (1, 4))
        self.assertEqual(stats.frames, 4)
        self.assertEqual(stats.size, (1080, 810))
        self.assertAlmostEqual(stats.duration_s, 4 / 30.0, places=3)
        self.assertGreater(stats.joints_drawn, 0)
        self.assertIn(stats.codec, ("H.264 (avc1)", "H.264 (avc3)", "MPEG-4 part 2 (mp4v)"))
        self.assertTrue(self.out.is_file())

        cap = cv2.VideoCapture(str(self.out))
        try:
            ok, frame = cap.read()
        finally:
            cap.release()
        self.assertTrue(ok)
        u = int(round(EYE.fy * 0.10 + 539.5))
        point = frame[404, u]
        # голубая точка ладони левой руки; допуск — двойное кодирование H.264 (цвет «плывёт»)
        self.assertGreaterEqual(int(point[0]), 235)
        self.assertLessEqual(abs(int(point[1]) - 140), 12)
        self.assertLessEqual(int(point[2]), 12)

    def test_full_render_freezes_pose_past_tracking(self) -> None:
        stats = ov.render_video(
            self.video, self.out, self._scene(), progress_every=0, log=lambda _: None
        )
        self.assertEqual(stats.frames, 6)
        self.assertEqual(stats.frames_after_tracking, 0)  # последний сэмпл = последний кадр
        self.assertEqual(stats.joints_drawn, 60)  # 6 кадров × 10 точек

    def test_last_frame_without_samples_uses_frozen_pose(self) -> None:
        self._write_video(self.FRAMES, self.FRAMES, tracking_samples=self.FRAMES - 1)
        stats = ov.render_video(
            self.video, self.out, self._scene(), progress_every=0, log=lambda _: None
        )
        self.assertEqual(stats.frames, 6)
        self.assertEqual(stats.frames_after_tracking, 1)  # хвост без трекинга (04a)
        self.assertEqual(stats.joints_drawn, 60)  # поза заморожена, точки нарисованы

    def test_render_logs_progress(self) -> None:
        lines: list[str] = []
        ov.render_video(
            self.video, self.out, self._scene(), start=0, end=5, progress_every=2, log=lines.append
        )
        self.assertTrue(any("кадр" in line for line in lines))
        self.assertGreaterEqual(len([line for line in lines if "кадр" in line]), 3)

    def test_missing_frames_raise_and_leave_no_output(self) -> None:
        scene = self._scene()
        scene.meta["frame_count"] = 10  # врём: в видео 6 кадров
        scene.meta["duration"] = 10 / self.FPS
        with self.assertRaises(ValueError):
            ov.render_video(self.video, self.out, scene, progress_every=0, log=lambda _: None)
        self.assertFalse(self.out.exists())

    def test_render_reports_trail_and_hud(self) -> None:
        stats = ov.render_video(
            self.video, self.out, self._scene(), progress_every=0, log=lambda _: None
        )
        self.assertGreater(stats.trail_segments, 0, "след не нарисован ни на одном кадре")
        self.assertEqual(stats.trail_seconds, ov.TRAIL_SECONDS)
        self.assertTrue(stats.hud)

    def test_no_trail_no_hud_renders_plain_frame(self) -> None:
        scene = self._scene()
        on = ov.render_stills(self.video, scene, self.dir / "on", frames=(5,), log=lambda _: None)
        off = ov.render_stills(
            self.video, scene, self.dir / "off", frames=(5,), trail=False, hud=False, log=lambda _: None
        )
        image_on = cv2.imread(str(on[0]))
        image_off = cv2.imread(str(off[0]))
        self.assertFalse(np.array_equal(image_on, image_off))
        x0, y0, x1, y1 = ov.hud_rect((1080, 810))
        # фон синтетического видео — серый 40 (после кодека ±5); панель HUD его затемняет,
        # а текст панели ярче фона — обе проверки возможны только с включённым HUD
        corner = (slice(y0, y1), slice(x0, x1))
        self.assertLess(
            int(image_on[corner].mean()), int(image_off[corner].mean()), "панель HUD не затемнила кадр"
        )
        bright_on = int((image_on[corner].min(axis=2) > 150).sum())
        bright_off = int((image_off[corner].min(axis=2) > 150).sum())
        self.assertGreater(bright_on, bright_off)

    def test_cli_flags_disable_trail_and_hud(self) -> None:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            code = ov.main(
                [
                    str(self.video),
                    "--stills",
                    str(self.dir / "cli"),
                    "--frames",
                    "5",
                    "--tracking",
                    str(self.tracking),
                    "--meta",
                    str(self.meta_path),
                    "--no-trail",
                    "--no-hud",
                ]
            )
        self.assertEqual(code, 0)
        self.assertTrue((self.dir / "cli" / "frame_00005.png").is_file())

    def test_render_stills_saves_left_eye_png(self) -> None:
        out_dir = self.dir / "stills"
        paths = ov.render_stills(
            self.video, self._scene(), out_dir, frames=(1, 3), log=lambda _: None
        )
        self.assertEqual([path.name for path in paths], ["frame_00001.png", "frame_00003.png"])
        image = cv2.imread(str(paths[0]))
        self.assertEqual(image.shape, (810, 1080, 3))

    def test_render_stills_rejects_unknown_frame(self) -> None:
        with self.assertRaises(ValueError):
            ov.render_stills(self.video, self._scene(), self.dir, frames=(6,), log=lambda _: None)

    def test_probe_rejects_wrong_frame_count_and_size(self) -> None:
        with self.assertRaises(ValueError):
            ov.probe_output(self.video, 5, (1080, 810))  # в видео 6 кадров
        with self.assertRaises(ValueError):
            ov.probe_output(self.video, 6, (1080, 810))  # размер не тот (видео 2160×810)
        with self.assertRaises(ValueError):
            ov.probe_output(self.dir / "nope.mp4", 6, (1080, 810))

    def test_missing_video_and_bad_start(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ov.render_video(self.dir / "nope.mp4", self.out, self._scene())
        with self.assertRaises(ValueError):
            ov.render_video(self.video, self.out, self._scene(), start=9)


class SceneLoadTest(unittest.TestCase):
    """``load_scene``: поиск трекинга и проверки метаданных."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        (self.dir / "video_meta.json").write_text(
            json.dumps({"frame_count": 3, "duration": 0.1, "width": 2160, "height": 810}),
            encoding="utf-8",
        )

    def test_missing_tracking_dir(self) -> None:
        with self.assertRaises(ValueError):
            ov.load_scene(self.dir / "video_meta.json", data_dir=self.dir / "empty")

    def test_ambiguous_tracking_files(self) -> None:
        data = self.dir / "data"
        data.mkdir()
        _write_tracking(data / "trackingData_a.txt", [BASE_NS])
        _write_tracking(data / "trackingData_b.txt", [BASE_NS])
        with self.assertRaises(ValueError) as ctx:
            ov.load_scene(self.dir / "video_meta.json", data_dir=data)
        self.assertIn("несколько", str(ctx.exception))

    def test_single_tracking_found_automatically(self) -> None:
        data = self.dir / "data"
        data.mkdir()
        _write_tracking(data / "trackingData_a.txt", [BASE_NS, BASE_NS + 33_333_333])
        scene = ov.load_scene(self.dir / "video_meta.json", data_dir=data)
        self.assertEqual(len(scene.track), 2)
        self.assertEqual((scene.intrinsics.full_w, scene.intrinsics.full_h), (2160, 810))

    def test_bad_meta(self) -> None:
        for meta in ({"duration": 1.0}, {"frame_count": 0, "duration": 1.0}, {"frame_count": 3}):
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                path = self.dir / "bad.json"
                path.write_text(json.dumps(meta), encoding="utf-8")
                ov.load_scene(path, data_dir=self.dir / "data")

    def test_intrinsics_must_match_video_size(self) -> None:
        data = self.dir / "data"
        data.mkdir()
        _write_tracking(data / "trackingData_a.txt", [BASE_NS])
        meta = self.dir / "other.json"
        meta.write_text(
            json.dumps({"frame_count": 3, "duration": 0.1, "width": 1080, "height": 810}),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            ov.load_scene(meta, data_dir=data)


@unittest.skipUnless(DATA is not None and META_PATH.is_file(), NO_DATA)
class RealDataTest(unittest.TestCase):
    """На реальной записи: синхронизация, метрика 07a через код рендера и PNG."""

    #: Суставы в кадре левого глаза (левая, правая) на 5 кадрах 07a — с поправкой времени
    #: кадра (``config.VIDEO_TIME_OFFSET_MS``). 43 из 45; в метрике 07c по рамкам 40 из 45.
    IN_FRAME = {30: (5, 3), 1800: (4, 5), 3600: (3, 5), 5400: (3, 5), 7200: (5, 5)}

    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = ov.load_scene(META_PATH, DATA)

    def test_scene_matches_recording(self) -> None:
        self.assertEqual(len(self.scene.track), 21787)
        self.assertEqual((self.scene.intrinsics.full_w, self.scene.intrinsics.full_h), (2160, 810))
        self.assertEqual(self.scene.eye_size, (1080, 810))
        self.assertAlmostEqual(self.scene.fps, 29.9966, places=3)
        self.assertAlmostEqual(self.scene.duration_s, 242.661, places=3)
        self.assertEqual(self.scene.time_offset_ns, sync.video_time_offset_ns())

    def test_calibrated_projection_on_debug_frames(self) -> None:
        for k, expected in self.IN_FRAME.items():
            with self.subTest(frame=k):
                moment = sync.frame_time(k, self.scene.meta, self.scene.header_ns, self.scene.tracking_end_ns)
                pose = sync.pose_at_frame(self.scene.track, moment, self.scene.time_offset_ns)
                pixels = ov.project_hands(pose, self.scene.intrinsics)
                for side, want in zip(("left", "right"), expected):
                    self.assertEqual(
                        sum(1 for joint in ov.JOINTS if pixels[side][joint].visible), 5
                    )
                    got = sum(
                        1 for joint in ov.JOINTS if camera_model.in_frame(pixels[side][joint], self.scene.intrinsics)
                    )
                    self.assertEqual(got, want)

    def test_hud_does_not_cover_labeled_hands(self) -> None:
        x0, y0, x1, y1 = ov.hud_rect(self.scene.eye_size)
        for item in calib_frames.CALIB_FRAMES:
            for box in item.hands:
                bx0, by0, bx1, by1 = box.bbox
                overlap = x0 < bx1 and bx0 < x1 and y0 < by1 and by0 < y1
                self.assertFalse(
                    overlap, f"HUD пересекает рамку руки {box.hand} на кадре {item.index}"
                )

    def test_no_projected_joint_inside_hud(self) -> None:
        x0, y0, x1, y1 = ov.hud_rect(self.scene.eye_size)
        for k in ov.DEBUG_FRAMES:
            with self.subTest(frame=k):
                moment = sync.frame_time(
                    k, self.scene.meta, self.scene.header_ns, self.scene.tracking_end_ns
                )
                pose = sync.pose_at_frame(self.scene.track, moment, self.scene.time_offset_ns)
                pixels = ov.project_hands(pose, self.scene.intrinsics)
                for side in ("left", "right"):
                    for joint in ov.JOINTS:
                        point = pixels[side][joint]
                        if not point.visible:
                            continue
                        self.assertFalse(
                            x0 <= point.u < x1 and y0 <= point.v < y1,
                            f"сустав {side} {joint} попал в HUD на кадре {k}",
                        )

    def test_hud_reports_inactive_right_hand(self) -> None:
        moment = sync.frame_time(
            5, self.scene.meta, self.scene.header_ns, self.scene.tracking_end_ns
        )
        pose = sync.pose_at_frame(self.scene.track, moment, self.scene.time_offset_ns)
        rows = ov.hud_items(moment, pose, self.scene.frame_count)
        self.assertIn(("right: NO DATA", ov.HUD_WARNING_COLOR), rows)
        self.assertIn(("left: OK", ov.HUD_TEXT_COLOR), rows)

    @unittest.skipIf(cv2 is None, NO_CV2)
    def test_trail_is_drawn_on_debug_frames(self) -> None:
        style = ov.style_for(self.scene.eye_size[0])
        for k in ov.DEBUG_FRAMES:
            with self.subTest(frame=k):
                moment = sync.frame_time(
                    k, self.scene.meta, self.scene.header_ns, self.scene.tracking_end_ns
                )
                pose = sync.pose_at_frame(self.scene.track, moment, self.scene.time_offset_ns)
                image = np.zeros((810, 1080, 3), np.uint8)
                drawn = ov.draw_trail(
                    image, pose, self.scene.track, self.scene.intrinsics, style
                )
                self.assertGreater(drawn, 0, f"след не нарисован на кадре {k}")

    @unittest.skipIf(cv2 is None or VIDEO is None, f"{NO_CV2} или {NO_VIDEO}")
    def test_still_is_drawn_on_real_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = ov.render_stills(VIDEO, self.scene, tmp, frames=(1800,), log=lambda _: None)
            self.assertEqual([path.name for path in paths], ["frame_01800.png"])
            rendered = cv2.imread(str(paths[0]))
            self.assertEqual(rendered.shape, (810, 1080, 3))

            # цвет точки в пикселе проекции: рендер пишет ровно COLORS[side] (PNG без потерь)
            moment = sync.frame_time(
                1800, self.scene.meta, self.scene.header_ns, self.scene.tracking_end_ns
            )
            pose = sync.pose_at_frame(self.scene.track, moment, self.scene.time_offset_ns)
            pixels = ov.project_hands(pose, self.scene.intrinsics)
            checked = 0
            for side in ("left", "right"):
                for joint in ov.JOINTS:
                    point = pixels[side][joint]
                    if not camera_model.in_frame(point, self.scene.intrinsics):
                        continue
                    pixel = rendered[int(round(point.v)), int(round(point.u))]
                    if tuple(int(channel) for channel in pixel) != ov.COLORS[side]:
                        continue
                    checked += 1
            self.assertGreaterEqual(checked, 5)



# --- 08f: густой скелет ---------------------------------------------------------------


#: Укладка суставов густого скелета на 1080×810: пальцы вверх, большой палец влево.
#: Точки и середины рёбер разнесены, чтобы проверять отдельные кости по пикселям.
DENSE_PIXELS = {
    "palm": (500, 400),
    "wrist": (500, 480),
    "thumb_proximal": (430, 390),
    "thumb_distal": (390, 350),
    "thumb": (360, 310),
    "index_proximal": (455, 330),
    "index_intermediate": (450, 270),
    "index_distal": (445, 220),
    "index": (440, 180),
    "middle_proximal": (500, 320),
    "middle_intermediate": (500, 260),
    "middle_distal": (500, 210),
    "middle": (500, 170),
    "ring_proximal": (545, 330),
    "ring": (565, 190),
    "little_proximal": (585, 345),
    "little": (615, 215),
}

#: Середины рёбер, которые проверяются по цвету (см. :data:`DENSE_PIXELS`).
BONE_MIDPOINTS = {
    ("palm", "wrist"): (500, 440),
    ("index_proximal", "index_intermediate"): (452.5, 300),
    ("index_intermediate", "index_distal"): (447.5, 245),
    ("middle_proximal", "middle_intermediate"): (500, 290),
    ("middle_intermediate", "middle_distal"): (500, 235),
}


def _dense_pixels(visible: bool = True) -> dict:
    """Пиксели густого скелета для обеих рук: правая сдвинута на 400 px вправо."""
    return {
        side: {
            joint: camera_model.ProjectedPoint(float(x) + offset, float(y), visible)
            for joint, (x, y) in DENSE_PIXELS.items()
        }
        for side, offset in (("left", 0.0), ("right", 400.0))
    }


def _dense_track(extra_keys=(), values=None) -> sync.PoseTrack:
    """Трек из двух сэмплов: 5 точек как :data:`LEFT_POINTS`, остальные — из ``values``.

    ``values`` — ``{ключ: (x0, x1)}`` для нужных осей: так тест задаёт позицию
    дополнительного сустава в двух сэмплах и проверяет, что рендер читает трек,
    а не берёт сустав из позы.
    """
    ts = [BASE_NS, BASE_NS + 33_000_000]
    fields = {name: [0.0, 0.0] for name in (*sync.POSITION_FIELDS, *sync.QUAT_FIELDS)}
    fields["head_qw"] = [1.0, 1.0]
    for side, points in (("left", LEFT_POINTS), ("right", RIGHT_POINTS)):
        for joint, position in points.items():
            for axis, value in zip(sync.AXES, position):
                fields[f"{side}_{joint}_{axis}"] = [value, value]
    for key in extra_keys:
        for axis in sync.AXES:
            fields.setdefault(f"{key}_{axis}", [0.0, 0.0])
    for name, pair in (values or {}).items():
        if name not in fields:
            raise ValueError(f"нет колонки {name}")
        fields[name] = [float(pair[0]), float(pair[1])]
    active = {"left": [True, True], "right": [True, True]}
    return sync.PoseTrack(ts, fields, active, extra_keys=extra_keys)


class SkeletonDefinitionTest(unittest.TestCase):
    """Скелеты 08f: имена/индексы OpenXR, полнота цепочек, бюджет точек карточки."""

    def test_full_skeleton_matches_openxr_table(self) -> None:
        self.assertEqual(ov.FULL_JOINTS, tuple(name for name, _ in HAND_JOINTS_FULL))
        self.assertEqual(len(ov.ALL_JOINTS.joints), 26)
        self.assertEqual(len(ov.ALL_JOINTS.bones), 25)

    def test_default_skeleton_fits_card_budget(self) -> None:
        self.assertIs(ov.DEFAULT_SKELETON, ov.DENSE)
        self.assertTrue(15 <= len(ov.DENSE.joints) <= 18)
        self.assertEqual(ov.SKELETONS["dense"], ov.DENSE)

    def test_finger_chains_are_complete(self) -> None:
        chains = (
            ("thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb"),
            ("index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index"),
            ("middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle"),
            ("ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring"),
            ("little_metacarpal", "little_proximal", "little_intermediate", "little_distal", "little"),
        )
        for chain in chains:
            for start, end in zip(chain, chain[1:]):
                self.assertIn((start, end), ov.FULL_BONES)
            self.assertIn(("palm", chain[0]), ov.FULL_BONES)
        self.assertIn(("palm", "wrist"), ov.FULL_BONES)

    def test_every_bone_endpoint_belongs_to_joint_set(self) -> None:
        for skeleton in ov.SKELETONS.values():
            with self.subTest(skeleton=skeleton.name):
                for start, end in skeleton.bones:
                    self.assertIn(start, skeleton.joints)
                    self.assertIn(end, skeleton.joints)
                self.assertEqual(
                    len({frozenset(bone) for bone in skeleton.bones}), len(skeleton.bones)
                )

    def test_extra_keys_exclude_render_points(self) -> None:
        self.assertEqual(ov.FIVE_POINT.extra_keys, ())
        dense = ov.DENSE.extra_keys
        self.assertEqual(len(dense), 2 * (len(ov.DENSE.joints) - len(ov.JOINTS)))
        for key in dense:
            self.assertNotIn(key, sync.POINT_KEYS)
        self.assertEqual(len(ov.ALL_JOINTS.extra_keys), 2 * 21)
        self.assertIn("left_index_metacarpal", ov.ALL_JOINTS.extra_keys)

    def test_validation_rejects_bad_definitions(self) -> None:
        with self.assertRaises(ValueError):
            ov.Skeleton("x", "x", ("palm", "nope"), (("palm", "nope"),))
        with self.assertRaises(ValueError):
            ov.Skeleton("x", "x", ("palm", "palm"), ())
        with self.assertRaises(ValueError):
            ov.Skeleton("x", "x", ("palm", "wrist"), (("palm", "thumb"),))
        with self.assertRaises(ValueError):
            ov.Skeleton("x", "x", ("palm", "wrist"), (("palm", "wrist"), ("wrist", "palm")))
        with self.assertRaises(ValueError):
            ov.Skeleton("x", "x", ("palm", "wrist"), (("palm", "wrist"), ("palm", "wrist")))
        with self.assertRaises(ValueError):
            ov.Skeleton("x", "x", (), ())


@unittest.skipIf(cv2 is None, NO_CV2)
class DenseDrawingTest(unittest.TestCase):
    """``draw_skeleton`` на густом скелете: точки, рёбра и разрывы на невидимом конце."""

    def setUp(self) -> None:
        self.image = np.zeros((810, 1080, 3), np.uint8)
        self.style = ov.style_for(1080)
        self.pixels = _dense_pixels()

    def _draw(self, pixels=None) -> int:
        return ov.draw_skeleton(
            self.image,
            self.pixels if pixels is None else pixels,
            self.style,
            joints=ov.DENSE.joints,
            bones=ov.DENSE.bones,
        )

    def test_draws_all_dense_points(self) -> None:
        self.assertEqual(self._draw(), 2 * len(ov.DENSE.joints))
        # цвета рук те же, что у 5-точечного скелета (BGR)
        self.assertEqual(tuple(int(c) for c in self.image[400, 500]), ov.COLORS["left"])
        self.assertEqual(tuple(int(c) for c in self.image[400, 900]), ov.COLORS["right"])

    def test_all_dense_bones_are_drawn(self) -> None:
        self._draw()
        for (start, end), (x, y) in BONE_MIDPOINTS.items():
            with self.subTest(bone=f"{start}→{end}"):
                for offset in (0, 400):
                    pixel = self.image[int(round(y)), int(round(x)) + offset]
                    self.assertGreater(int(pixel.max()), 0, "середина ребра не закрашена")

    def test_invisible_endpoint_skips_both_touching_bones(self) -> None:
        self.pixels["left"]["index_intermediate"] = camera_model.ProjectedPoint(
            math.nan, math.nan, False
        )
        self.assertEqual(self._draw(), 2 * len(ov.DENSE.joints) - 1)
        x0 = DENSE_PIXELS["index_proximal"][0]
        x1 = DENSE_PIXELS["index_intermediate"][0]
        y = DENSE_PIXELS["index_intermediate"][1]
        self.assertEqual(int(self.image[y, int((x0 + x1) / 2)].max()), 0)
        # соседнее ребро до дистальной фаланги тоже не нарисовано (второй конец — тот же)
        self.assertEqual(int(self.image[245, 447].max()), 0)
        # остальные кости пальца на месте
        self.assertGreater(int(self.image[200, 442].max()), 0)

    def test_bone_to_off_frame_point_is_clipped_not_dropped(self) -> None:
        self.pixels["left"]["little"] = camera_model.ProjectedPoint(5000.0, 215.0, True)
        self.assertEqual(self._draw(), 2 * len(ov.DENSE.joints))
        self.assertGreater(int(self.image[:, 1070].max()), 0, "ребро обрезано краем кадра")

    def test_missing_joint_or_bad_bone_raises(self) -> None:
        pixels = _dense_pixels()
        del pixels["right"]["little"]
        with self.assertRaises(ValueError) as ctx:
            ov.draw_skeleton(
                self.image, pixels, self.style, joints=ov.DENSE.joints, bones=ov.DENSE.bones
            )
        self.assertIn("little", str(ctx.exception))
        with self.assertRaises(ValueError):
            ov.draw_skeleton(
                self.image,
                _dense_pixels(),
                self.style,
                joints=("palm", "wrist"),
                bones=(("palm", "thumb"),),
            )
        with self.assertRaises(ValueError):
            ov.draw_skeleton(self.image, _dense_pixels(), self.style, joints=())


class DenseProjectionTest(unittest.TestCase):
    """``project_skeleton``: 5 точек из позы, остальные — из трека с ``extra_keys``."""

    def test_extra_joints_are_read_from_track(self) -> None:
        track = _dense_track(
            ov.DENSE.extra_keys,
            values={
                "left_index_proximal_x": (0.30, 0.30),
                "left_index_proximal_y": (0.40, 0.40),
                "left_index_proximal_z": (-1.00, -1.00),
            },
        )
        pose = track.pose_at(BASE_NS)
        pixels = ov.project_skeleton(pose, EYE, ov.DENSE, track=track)
        self.assertEqual(set(pixels["left"]), set(ov.DENSE.joints))
        expected = camera_model.project(
            camera_model.transform_point(
                camera_model.world_to_camera_matrix(pose), (0.30, 0.40, -1.00)
            ),
            camera_model.eye_intrinsics(EYE, "left"),
        )
        got = pixels["left"]["index_proximal"]
        self.assertAlmostEqual(got.u, expected.u, places=9)
        self.assertAlmostEqual(got.v, expected.v, places=9)
        self.assertTrue(got.visible)

    def test_five_points_match_project_hands(self) -> None:
        track = _dense_track(ov.DENSE.extra_keys)
        pose = track.pose_at(BASE_NS)
        dense = ov.project_skeleton(pose, EYE, ov.DENSE, track=track)
        five = ov.project_hands(pose, EYE)
        for side in ("left", "right"):
            for joint in ov.JOINTS:
                self.assertAlmostEqual(dense[side][joint].u, five[side][joint].u, places=9)
                self.assertAlmostEqual(dense[side][joint].v, five[side][joint].v, places=9)
                self.assertEqual(dense[side][joint].visible, five[side][joint].visible)

    def test_nan_extra_joint_is_invisible(self) -> None:
        track = _dense_track(
            ov.DENSE.extra_keys,
            values={
                "right_middle_proximal_x": (0.20, 0.20),
                "right_middle_proximal_y": (0.10, 0.10),
                "right_middle_proximal_z": (-1.00, -1.00),
                "right_middle_intermediate_x": (math.nan, math.nan),
                "right_middle_intermediate_y": (math.nan, math.nan),
                "right_middle_intermediate_z": (math.nan, math.nan),
            },
        )
        pixels = ov.project_skeleton(track.pose_at(BASE_NS), EYE, ov.DENSE, track=track)
        self.assertFalse(pixels["right"]["middle_intermediate"].visible)
        # NaN в треке не портит соседние суставы
        self.assertTrue(pixels["right"]["middle_proximal"].visible)

    def test_track_without_extras_raises(self) -> None:
        track = _dense_track(())
        with self.assertRaises(ValueError) as ctx:
            ov.project_skeleton(track.pose_at(BASE_NS), EYE, ov.DENSE, track=track)
        self.assertIn("extra_keys", str(ctx.exception))
        with self.assertRaises(ValueError):
            ov.project_skeleton(track.pose_at(BASE_NS), EYE, ov.DENSE, track=None)

    def test_five_point_skeleton_needs_no_extras(self) -> None:
        pixels = ov.project_skeleton(_pose(), EYE, ov.FIVE_POINT)
        self.assertEqual(set(pixels["left"]), set(ov.JOINTS))


class DenseRenderTest(SyntheticVideoCase):
    """Рендер густого скелета на синтетическом видео (та же заготовка, что у 08a)."""

    def _dense_scene(self, skeleton: ov.Skeleton = ov.DENSE) -> ov.Scene:
        return ov.load_scene(self.meta_path, self.tracking, extra_keys=skeleton.extra_keys)

    def test_dense_render_draws_more_joints(self) -> None:
        stats = ov.render_video(
            self.video,
            self.out,
            self._dense_scene(),
            progress_every=0,
            log=lambda _: None,
            skeleton=ov.DENSE,
        )
        self.assertEqual(stats.skeleton, "dense")
        self.assertEqual(stats.joints_per_hand, len(ov.DENSE.joints))
        self.assertEqual(stats.joints_drawn, 6 * 2 * len(ov.DENSE.joints))

    def test_all_joints_render_has_26_per_hand(self) -> None:
        stats = ov.render_video(
            self.video,
            self.out,
            self._dense_scene(ov.ALL_JOINTS),
            start=0,
            end=2,
            progress_every=0,
            log=lambda _: None,
            skeleton=ov.ALL_JOINTS,
        )
        self.assertEqual(stats.skeleton, "all")
        self.assertEqual(stats.joints_per_hand, 26)
        self.assertGreater(stats.joints_drawn, 3 * 2 * len(ov.DENSE.joints))

    def test_dense_render_without_extra_keys_fails_before_writing(self) -> None:
        scene = self._scene()  # трек без extra_keys
        with self.assertRaises(ValueError) as ctx:
            ov.render_video(
                self.video, self.out, scene, progress_every=0, log=lambda _: None,
                skeleton=ov.DENSE,
            )
        self.assertIn("extra_keys", str(ctx.exception))
        self.assertFalse(self.out.exists(), "файл не должен появляться до проверки скелета")

    def test_cli_selects_skeleton_mode(self) -> None:
        import contextlib
        import io

        out = self.dir / "cli_dense"
        with contextlib.redirect_stdout(io.StringIO()):
            code = ov.main(
                [
                    str(self.video), "--stills", str(out), "--frames", "5",
                    "--tracking", str(self.tracking), "--meta", str(self.meta_path),
                ]
            )
        self.assertEqual(code, 0)
        self.assertTrue((out / "frame_00005.png").is_file())

        with contextlib.redirect_stdout(io.StringIO()):
            code = ov.main(
                [
                    str(self.video), "--stills", str(self.dir / "cli_five"), "--frames", "5",
                    "--tracking", str(self.tracking), "--meta", str(self.meta_path),
                    "--five-points",
                ]
            )
        self.assertEqual(code, 0)
        self.assertTrue((self.dir / "cli_five" / "frame_00005.png").is_file())

        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as ctx,
        ):
            ov.main(
                [
                    str(self.video), "--stills", str(self.dir / "cli_bad"), "--frames", "5",
                    "--tracking", str(self.tracking), "--meta", str(self.meta_path),
                    "--five-points", "--all-joints",
                ]
            )
        self.assertEqual(ctx.exception.code, 2)

    def test_render_reports_depth_shading(self) -> None:
        """08g: авторежим включается для густого скелета и выключен для 5-точечного 08a."""
        dense = ov.render_video(
            self.video, self.out, self._dense_scene(), progress_every=0, log=lambda _: None,
            skeleton=ov.DENSE,
        )
        self.assertTrue(dense.depth_shading)
        five = ov.render_video(
            self.video, self.out, self._scene(), progress_every=0, log=lambda _: None,
            skeleton=ov.FIVE_POINT,
        )
        self.assertFalse(five.depth_shading)
        forced_off = ov.render_video(
            self.video, self.out, self._dense_scene(), progress_every=0, log=lambda _: None,
            skeleton=ov.DENSE, depth_shading=False,
        )
        self.assertFalse(forced_off.depth_shading)

    def test_cli_depth_flag(self) -> None:
        """08g: ``--depth-color``/``--no-depth-color`` — по умолчанию цвет по глубине."""
        import contextlib
        import io

        def still(name: str, *flags: str) -> np.ndarray:
            out = self.dir / name
            with contextlib.redirect_stdout(io.StringIO()):
                code = ov.main(
                    [
                        str(self.video), "--stills", str(out), "--frames", "5",
                        "--tracking", str(self.tracking), "--meta", str(self.meta_path), *flags,
                    ]
                )
            self.assertEqual(code, 0)
            image = cv2.imread(str(out / "frame_00005.png"))
            self.assertIsNotNone(image)
            return image

        dense_auto = still("depth_auto")
        dense_flat = still("depth_flat", "--no-depth-color")
        self.assertFalse(np.array_equal(dense_auto, dense_flat))
        # 5-точечный кадр 08a по умолчанию плоский: флаг ничего не меняет
        five_auto = still("depth_five", "--five-points")
        five_flat = still("depth_five_flat", "--five-points", "--no-depth-color")
        self.assertTrue(np.array_equal(five_auto, five_flat))
        # явный флаг перекрывает авторежим и для 5 точек
        five_on = still("depth_five_on", "--five-points", "--depth-color")
        self.assertFalse(np.array_equal(five_auto, five_on))


# --- 08g: глубина → интенсивность цвета --------------------------------------------------


class DepthFactorTest(unittest.TestCase):
    """08g: множитель по глубине — окно, зажим, монотонность, неизвестная глубина."""

    def test_defaults_map_to_full_colour_and_floor(self) -> None:
        self.assertAlmostEqual(ov.depth_factor(ov.DEPTH_NEAR_M), 1.0, places=12)
        self.assertAlmostEqual(ov.depth_factor(ov.DEPTH_FAR_M), ov.DEPTH_MIN_FACTOR, places=12)
        # за границами окна — зажим, а не экстраполяция
        self.assertAlmostEqual(ov.depth_factor(0.01), 1.0, places=12)
        self.assertAlmostEqual(ov.depth_factor(5.0), ov.DEPTH_MIN_FACTOR, places=12)

    def test_monotone_non_increasing_with_depth(self) -> None:
        depths = (0.0, 0.1, ov.DEPTH_NEAR_M, 0.25, 0.30, 0.34, 0.40, 0.45, 0.50,
                  ov.DEPTH_FAR_M, 0.7, 1.0)
        factors = [ov.depth_factor(depth) for depth in depths]
        for previous, current in zip(factors, factors[1:]):
            self.assertGreaterEqual(previous + 1e-12, current)
        # внутри окна строго убывает: ближе — ярче
        self.assertGreater(factors[3], factors[4])
        self.assertGreater(factors[4], factors[5])
        # и не выходит из диапазона множителя
        self.assertTrue(all(ov.DEPTH_MIN_FACTOR - 1e-12 <= f <= 1.0 + 1e-12 for f in factors))

    def test_unknown_depth_keeps_full_colour(self) -> None:
        # «нет данных» — не повод выцветать: невидимая точка всё равно не рисуется
        for depth in (math.nan, math.inf, -math.inf):
            with self.subTest(depth=depth):
                self.assertEqual(ov.depth_factor(depth), 1.0)

    def test_behind_camera_keeps_full_colour(self) -> None:
        for depth in (-1.0, -0.001, 0.0):
            with self.subTest(depth=depth):
                self.assertEqual(ov.depth_factor(depth), 1.0)

    def test_window_and_factor_validation(self) -> None:
        with self.assertRaises(ValueError):
            ov.depth_factor(0.4, near=0.5, far=0.5)
        with self.assertRaises(ValueError):
            ov.depth_factor(0.4, near=0.6, far=0.58)
        with self.assertRaises(ValueError):
            ov.depth_factor(0.4, near=0.0)
        with self.assertRaises(ValueError):
            ov.depth_factor(0.4, near=-0.1)
        with self.assertRaises(ValueError):
            ov.depth_factor(0.4, near=math.nan)
        with self.assertRaises(ValueError):
            ov.depth_factor(0.4, far="0.58")
        with self.assertRaises(ValueError):
            ov.depth_factor(0.4, min_factor=0.0)
        with self.assertRaises(ValueError):
            ov.depth_factor(0.4, min_factor=1.5)
        with self.assertRaises(ValueError):
            ov.depth_factor(True)
        with self.assertRaises(ValueError):
            ov.depth_factor("0.4")

    def test_shade_colour_scales_channels_and_keeps_hue(self) -> None:
        colour = ov.COLORS["left"]
        self.assertEqual(ov.shade_color(colour, 1.0), colour)
        dimmed = ov.shade_color(colour, ov.DEPTH_MIN_FACTOR)
        self.assertEqual(dimmed, (76, 42, 0))
        # оттенок тот же: отношение каналов сохраняется (допуск — округление до int)
        self.assertAlmostEqual(colour[0] / colour[1], dimmed[0] / dimmed[1], delta=0.02)
        # зажим сверху и снизу
        self.assertEqual(ov.shade_color((200, 200, 200), 1.0), (200, 200, 200))
        self.assertEqual(ov.shade_color((255, 255, 255), 1.0), (255, 255, 255))
        self.assertEqual(ov.shade_color((0, 0, 0), 0.5), (0, 0, 0))

    def test_shade_colour_validation(self) -> None:
        with self.assertRaises(ValueError):
            ov.shade_color((255, 140), 0.5)
        with self.assertRaises(ValueError):
            ov.shade_color((255, 140, 300), 0.5)
        with self.assertRaises(ValueError):
            ov.shade_color((255.0, 140, 0), 0.5)
        with self.assertRaises(ValueError):
            ov.shade_color((255, 140, 0), 1.5)
        with self.assertRaises(ValueError):
            ov.shade_color((255, 140, 0), -0.1)
        with self.assertRaises(ValueError):
            ov.shade_color((255, 140, 0), math.nan)
        with self.assertRaises(ValueError):
            ov.shade_color((255, 140, 0), True)

    def test_auto_mode_follows_skeleton(self) -> None:
        self.assertTrue(ov.resolve_depth_shading(None, ov.DENSE))
        self.assertTrue(ov.resolve_depth_shading(None, ov.ALL_JOINTS))
        self.assertFalse(ov.resolve_depth_shading(None, ov.FIVE_POINT))
        # тот же 5-точечный набор, собранный заново, — тоже плоский кадр
        clone = ov.Skeleton("five", "5 точек", ov.JOINTS, ov.HAND_CHAINS)
        self.assertFalse(ov.resolve_depth_shading(None, clone))

    def test_explicit_flag_overrides_auto_mode(self) -> None:
        self.assertTrue(ov.resolve_depth_shading(True, ov.FIVE_POINT))
        self.assertFalse(ov.resolve_depth_shading(False, ov.DENSE))

    def test_auto_mode_validation(self) -> None:
        with self.assertRaises(ValueError):
            ov.resolve_depth_shading(1, ov.DENSE)  # int, не bool
        with self.assertRaises(ValueError):
            ov.resolve_depth_shading("yes", ov.DENSE)
        with self.assertRaises(ValueError):
            ov.resolve_depth_shading(None, "dense")


class DepthProjectionTest(unittest.TestCase):
    """08g: глубина — ``z`` точки в системе камеры, а не мировая координата."""

    def test_depth_is_camera_z_matching_the_matrix(self) -> None:
        pose = _pose()
        pixels, depths = ov.project_skeleton_depth(pose, EYE, ov.FIVE_POINT)
        matrix = camera_model.world_to_camera_matrix(pose)
        eye = camera_model.eye_intrinsics(EYE, "left")
        for side, joints in (("left", LEFT_POINTS), ("right", RIGHT_POINTS)):
            for joint, world in joints.items():
                with self.subTest(side=side, joint=joint):
                    p_cam = camera_model.transform_point(matrix, world)
                    self.assertAlmostEqual(depths[side][joint], p_cam[2], places=12)
                    expected = camera_model.project(p_cam, eye)
                    self.assertAlmostEqual(pixels[side][joint].u, expected.u, places=12)
                    self.assertAlmostEqual(pixels[side][joint].v, expected.v, places=12)
                    self.assertTrue(pixels[side][joint].visible)

    def test_nearer_hand_has_smaller_camera_z(self) -> None:
        far = dict(LEFT_POINTS)
        far["palm"] = (0.10, 0.0, -1.0)
        near = dict(RIGHT_POINTS)
        near["palm"] = (0.20, 0.10, -0.25)
        pixels, depths = ov.project_skeleton_depth(
            _pose({"left": far, "right": near}), EYE, ov.FIVE_POINT
        )
        self.assertLess(depths["right"]["palm"], depths["left"]["palm"])
        self.assertGreater(
            ov.depth_factor(depths["right"]["palm"]), ov.depth_factor(depths["left"]["palm"])
        )

    def test_depths_match_project_skeleton(self) -> None:
        track = _dense_track(ov.DENSE.extra_keys)
        pose = track.pose_at(BASE_NS)
        pixels, depths = ov.project_skeleton_depth(pose, EYE, ov.DENSE, track=track)
        plain = ov.project_skeleton(pose, EYE, ov.DENSE, track=track)
        for side in ("left", "right"):
            self.assertEqual(set(depths[side]), set(ov.DENSE.joints))
            for joint in ov.DENSE.joints:
                with self.subTest(side=side, joint=joint):
                    self.assertEqual(pixels[side][joint], plain[side][joint])

    def test_nan_joint_has_nan_depth(self) -> None:
        points = {"left": LEFT_POINTS, "right": dict(RIGHT_POINTS)}
        points["right"]["palm"] = (math.nan, math.nan, math.nan)
        pose = _pose(points)
        pixels, depths = ov.project_skeleton_depth(pose, EYE, ov.FIVE_POINT)
        self.assertFalse(pixels["right"]["palm"].visible)
        self.assertTrue(math.isnan(depths["right"]["palm"]))
        self.assertTrue(math.isfinite(depths["left"]["palm"]))

    def test_point_behind_camera_has_finite_depth_but_is_invisible(self) -> None:
        points = {"left": dict(LEFT_POINTS), "right": RIGHT_POINTS}
        points["left"]["palm"] = (0.10, 0.00, 1.0)  # за головой: вперёд смотрит −z
        pixels, depths = ov.project_skeleton_depth(_pose(points), EYE, ov.FIVE_POINT)
        self.assertFalse(pixels["left"]["palm"].visible)
        self.assertTrue(math.isnan(pixels["left"]["palm"].u))
        self.assertTrue(math.isfinite(depths["left"]["palm"]))
        self.assertEqual(ov.depth_factor(depths["left"]["palm"]), 1.0)

    def test_projection_is_deterministic(self) -> None:
        track = _dense_track(ov.DENSE.extra_keys)
        pose = track.pose_at(BASE_NS)
        first = ov.project_skeleton_depth(pose, EYE, ov.DENSE, track=track)
        second = ov.project_skeleton_depth(pose, EYE, ov.DENSE, track=track)
        self.assertEqual(first, second)


class BoneGradientTest(unittest.TestCase):
    """08g: градиент вдоль кости — число отрезков, монотонность, зажим, NaN-фолбэк."""

    COLOR = ov.COLORS["left"]   # BGR (255, 140, 0)

    def test_segment_count_and_monotone_along_bone(self) -> None:
        colors = ov.bone_gradient_colors(self.COLOR, 0.24, 0.44)
        self.assertEqual(len(colors), ov.DEPTH_BONE_SEGMENTS)
        lum = [sum(color) for color in colors]
        for previous, current in zip(lum, lum[1:]):
            self.assertGreater(previous, current)
        # первый отрезок ближе и заметно ярче последнего
        self.assertGreater(lum[0] - lum[-1], 20)

    def test_segment_colors_match_midpoint_factors(self) -> None:
        colors = ov.bone_gradient_colors(self.COLOR, 0.24, 0.44)
        start = ov.depth_factor(0.24)
        step = ov.depth_factor(0.44) - start
        for index, color in enumerate(colors):
            expected = ov.shade_color(self.COLOR, start + step * ((index + 0.5) / len(colors)))
            self.assertEqual(color, expected)

    def test_gradient_direction_follows_depth(self) -> None:
        # оба конца в окне: ближний конец (z меньше) ярче, дальний — темнее
        forward = ov.bone_gradient_colors(self.COLOR, 0.24, 0.44)
        backward = ov.bone_gradient_colors(self.COLOR, 0.44, 0.24)
        self.assertEqual(forward, tuple(reversed(backward)))

    def test_clamped_depths_give_uniform_colour(self) -> None:
        deep = ov.bone_gradient_colors(self.COLOR, 0.70, 0.90)
        floor = ov.shade_color(self.COLOR, ov.DEPTH_MIN_FACTOR)
        self.assertEqual(set(deep), {floor})
        shallow = ov.bone_gradient_colors(self.COLOR, 0.10, 0.18)
        self.assertEqual(set(shallow), {self.COLOR})

    def test_unknown_depth_falls_back_to_full_colour(self) -> None:
        for start, end in ((math.nan, 0.44), (0.24, math.nan), (math.inf, -math.inf)):
            with self.subTest(start=start, end=end):
                colors = ov.bone_gradient_colors(self.COLOR, start, end)
                self.assertEqual(set(colors), {self.COLOR})

    def test_typical_span_meets_intra_hand_target(self) -> None:
        """Целевой ориентир доработки: дальняя кость ≤ 0.6 × ближней на разбросе 0.10 м."""
        for near, far in ((0.34, 0.44), (0.36, 0.46)):
            with self.subTest(near=near, far=far):
                self.assertGreaterEqual(ov.depth_factor(near) / ov.depth_factor(far), 1.6)

    def test_segments_validation(self) -> None:
        for segments in (0, -1, 2.5, True, "5"):
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                ov.bone_gradient_colors(self.COLOR, 0.24, 0.44, segments=segments)


@unittest.skipIf(cv2 is None, NO_CV2)
class DepthShadingDrawTest(unittest.TestCase):
    """08g: ``draw_skeleton`` гасит дальние кости/точки, градиент вдоль кости, плоский фолбэк."""

    def setUp(self) -> None:
        self.style = ov.style_for(1080)
        self.pixels = {
            "left": {
                "palm": camera_model.ProjectedPoint(100.0, 100.0, True),
                "thumb": camera_model.ProjectedPoint(300.0, 100.0, True),
            },
            "right": {
                "palm": camera_model.ProjectedPoint(600.0, 100.0, True),
                "thumb": camera_model.ProjectedPoint(800.0, 100.0, True),
            },
        }
        self.joints = ("palm", "thumb")
        self.bones = (("palm", "thumb"),)
        # ладонь — на дальней границе окна, кончик большого — на ближней
        self.depths = {
            "left": {"palm": ov.DEPTH_FAR_M, "thumb": ov.DEPTH_NEAR_M},
            "right": {"palm": ov.DEPTH_FAR_M, "thumb": ov.DEPTH_NEAR_M},
        }

    def _draw(self, depths) -> np.ndarray:
        image = np.zeros((810, 1080, 3), np.uint8)
        ov.draw_skeleton(
            image, self.pixels, self.style, joints=self.joints, bones=self.bones, depths=depths
        )
        return image

    def test_nearer_point_is_full_colour_and_farther_is_dimmed(self) -> None:
        image = self._draw(self.depths)
        near = image[100, 300]
        far = image[100, 100]
        self.assertEqual(tuple(int(v) for v in near), ov.COLORS["left"])
        self.assertEqual(
            tuple(int(v) for v in far), ov.shade_color(ov.COLORS["left"], ov.DEPTH_MIN_FACTOR)
        )
        # оттенок руки сохранился: синий канал по-прежнему доминирует
        self.assertGreater(int(far[0]), int(far[1]))
        self.assertEqual(int(far[2]), 0)

    def test_bone_is_shaded_with_a_visible_gradient(self) -> None:
        """Нарисованная кость идёт от яркого ближнего конца к тёмному дальнему."""
        image = self._draw(self.depths)
        # начало ребра (palm, x=100) — дальняя граница окна, конец (thumb, x=300) — ближняя
        samples = [int(image[100, x][0]) for x in (120, 160, 200, 240, 280)]
        for previous, current in zip(samples, samples[1:]):
            self.assertGreater(current, previous)
        self.assertGreater(samples[-1] - samples[0], 20)
        # цвет середины первого отрезка — по множителю его середины, не средний по кости
        expected = ov.shade_color(
            ov.COLORS["left"], ov.depth_factor(ov.DEPTH_FAR_M)
            + (ov.depth_factor(ov.DEPTH_NEAR_M) - ov.depth_factor(ov.DEPTH_FAR_M)) * 0.1
        )
        self.assertEqual(tuple(int(v) for v in image[100, 120]), expected)

    def test_outline_stays_dark_on_both_depths(self) -> None:
        """Обводка точки остаётся тёмной при любом множителе — контраст на светлом фоне."""
        shaded = self._draw(self.depths)
        for u in (100, 300):
            with self.subTest(u=u):
                fill = shaded[100, u].astype(int)  # центр точки — цвет руки (с множителем)
                ring = shaded[100, u + 4].astype(int)  # ядро чёрной обводки
                self.assertTrue(all(int(channel) < 60 for channel in ring))
                for index, channel in enumerate(fill):
                    if channel:
                        self.assertLess(ring[index], channel * 0.35)

    def test_flat_mode_reproduces_previous_frame(self) -> None:
        flat = self._draw(None)
        unknown = self._draw(
            {side: {joint: math.nan for joint in self.joints} for side in ("left", "right")}
        )
        # глубины нет — полный цвет: ровно та же картинка, что без ``depths``
        self.assertTrue(np.array_equal(flat, unknown))
        self.assertEqual(tuple(int(v) for v in flat[100, 100]), ov.COLORS["left"])
        # кость без глубин — одна линия полного цвета, без градиента
        self.assertEqual(tuple(int(v) for v in flat[100, 200]), ov.COLORS["left"])

    def test_mixed_nan_endpoint_keeps_full_colour(self) -> None:
        depths = {
            "left": {"palm": ov.DEPTH_FAR_M, "thumb": math.nan},
            "right": {"palm": math.nan, "thumb": ov.DEPTH_NEAR_M},
        }
        image = self._draw(depths)
        # линия (100,100)→(300,100): один конец без глубины — полный цвет
        self.assertEqual(tuple(int(v) for v in image[100, 200]), ov.COLORS["left"])
        # точка с глубиной всё равно приглушается
        self.assertEqual(
            tuple(int(v) for v in image[100, 100]),
            ov.shade_color(ov.COLORS["left"], ov.DEPTH_MIN_FACTOR),
        )

    def test_drawing_is_deterministic(self) -> None:
        first = self._draw(self.depths)
        second = self._draw(self.depths)
        self.assertTrue(np.array_equal(first, second))

    def test_depths_validation(self) -> None:
        with self.assertRaises(ValueError):
            ov.draw_skeleton(
                np.zeros((810, 1080, 3), np.uint8), self.pixels, self.style,
                joints=self.joints, bones=self.bones, depths=["nope"],
            )
        with self.assertRaises(ValueError):
            ov.draw_skeleton(
                np.zeros((810, 1080, 3), np.uint8), self.pixels, self.style,
                joints=self.joints, bones=self.bones, depths={"left": self.depths["left"]},
            )
        with self.assertRaises(ValueError):
            ov.draw_skeleton(
                np.zeros((810, 1080, 3), np.uint8), self.pixels, self.style,
                joints=self.joints, bones=self.bones, depths={"left": {}, "right": {}},
            )


@unittest.skipUnless(DATA is not None and META_PATH.is_file(), NO_DATA)
class DenseRealDataTest(unittest.TestCase):
    """Реальные 5 кадров 07a: густой скелет не сдвигает 5 опорных точек."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = ov.load_scene(META_PATH, DATA, extra_keys=ov.DENSE.extra_keys)

    def _pixels(self, k: int, skeleton: ov.Skeleton) -> dict:
        moment = sync.frame_time(k, self.scene.meta, self.scene.header_ns, self.scene.tracking_end_ns)
        pose = sync.pose_at_frame(self.scene.track, moment, self.scene.time_offset_ns)
        return ov.project_skeleton(pose, self.scene.intrinsics, skeleton, track=self.scene.track)

    def _depth_pixels(self, k: int):
        moment = sync.frame_time(
            k, self.scene.meta, self.scene.header_ns, self.scene.tracking_end_ns
        )
        pose = sync.pose_at_frame(self.scene.track, moment, self.scene.time_offset_ns)
        return ov.project_skeleton_depth(
            pose, self.scene.intrinsics, ov.DENSE, track=self.scene.track
        )

    def test_dense_keeps_five_point_projection(self) -> None:
        for k in ov.DEBUG_FRAMES:
            with self.subTest(frame=k):
                five = self._pixels(k, ov.FIVE_POINT)
                dense = self._pixels(k, ov.DENSE)
                for side in ("left", "right"):
                    for joint in ov.JOINTS:
                        self.assertAlmostEqual(dense[side][joint].u, five[side][joint].u, places=9)
                        self.assertAlmostEqual(dense[side][joint].v, five[side][joint].v, places=9)
                        self.assertEqual(dense[side][joint].visible, five[side][joint].visible)

    def test_dense_projection_covers_debug_frames(self) -> None:
        for k in ov.DEBUG_FRAMES:
            with self.subTest(frame=k):
                pixels = self._pixels(k, ov.DENSE)
                for side in ("left", "right"):
                    self.assertEqual(set(pixels[side]), set(ov.DENSE.joints))
                    self.assertGreater(
                        sum(1 for joint in ov.DENSE.joints if pixels[side][joint].visible),
                        sum(1 for joint in ov.JOINTS if pixels[side][joint].visible),
                    )

    def test_dense_joints_do_not_enter_hud(self) -> None:
        """Густой скелет не рисуется поверх панели HUD (проверка 08b расширена на 17 точек)."""
        x0, y0, x1, y1 = ov.hud_rect(self.scene.eye_size)
        for k in ov.DEBUG_FRAMES:
            pixels = self._pixels(k, ov.DENSE)
            for side in ("left", "right"):
                for joint in ov.DENSE.joints:
                    point = pixels[side][joint]
                    if not point.visible:
                        continue
                    self.assertFalse(
                        x0 <= point.u < x1 and y0 <= point.v < y1,
                        f"сустав {side} {joint} попал в HUD на кадре {k}",
                    )

    def test_depth_is_deterministic_and_consistent_on_debug_frames(self) -> None:
        """08g: на 5 кадрах 07a глубина детерминирована и согласована с видимостью."""
        for k in ov.DEBUG_FRAMES:
            with self.subTest(frame=k):
                first = self._depth_pixels(k)
                self.assertEqual(first, self._depth_pixels(k))
                pixels, depths = first
                for side in ("left", "right"):
                    self.assertEqual(set(depths[side]), set(ov.DENSE.joints))
                    for joint in ov.DENSE.joints:
                        point = pixels[side][joint]
                        depth = depths[side][joint]
                        if point.visible:
                            self.assertTrue(math.isfinite(depth))
                            self.assertGreater(depth, camera_model.MIN_DEPTH_M)
                            self.assertGreaterEqual(ov.depth_factor(depth), ov.DEPTH_MIN_FACTOR)
                        else:
                            self.assertTrue(math.isnan(depth) or depth <= camera_model.MIN_DEPTH_M)

    @unittest.skipIf(cv2 is None or VIDEO is None, f"{NO_CV2} или {NO_VIDEO}")
    def test_dense_still_has_more_skeleton_pixels(self) -> None:
        """На реальном кадре густой скелет закрашивает больше пикселей, чем 5-точечный."""
        with tempfile.TemporaryDirectory() as tmp:
            five = ov.render_stills(
                VIDEO, self.scene, Path(tmp) / "five", frames=(1800,),
                skeleton=ov.FIVE_POINT, log=lambda _: None, depth_shading=False,
            )
            dense = ov.render_stills(
                VIDEO, self.scene, Path(tmp) / "dense", frames=(1800,),
                skeleton=ov.DENSE, log=lambda _: None, depth_shading=False,
            )
            image_five = cv2.imread(str(five[0]))
            image_dense = cv2.imread(str(dense[0]))

        def overlay_pixels(image) -> int:
            # левая рука цвета COLORS["left"] = BGR (255, 140, 0): PNG без потерь кодека
            blue = (image[..., 0] > 235) & (image[..., 1] > 120) & (image[..., 2] < 40)
            red = (image[..., 2] > 235) & (image[..., 1] < 30) & (image[..., 0] < 40)
            return int(blue.sum() + red.sum())

        self.assertGreater(overlay_pixels(image_dense), overlay_pixels(image_five))


if __name__ == "__main__":
    unittest.main()
