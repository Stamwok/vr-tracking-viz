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

from src import camera_model, config, overlay_video as ov, sync  # noqa: E402

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
class SyntheticRenderTest(unittest.TestCase):
    """Рендер на синтетическом SBS-видео: те же файлы, что у реальной записи, но 6 кадров."""

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


if __name__ == "__main__":
    unittest.main()
