"""Assert-тесты для src/camera_model.py, задача 05c: проекция камеры в пиксели одного глаза.

Запуск: python -m unittest discover -s tests -t . -v
Тесты на реальном файле трекинга пропускаются, если data/trackingData_*.txt или
output/video_meta.json отсутствуют.

Что проверяется (критерии карточки 05c):

* ``u = fx·X/Z + cx``, ``v = fy·Y/Z + cy`` — сверяется с формулой на сетке точек и с
  независимой реализацией ``cv2.projectPoints`` (если opencv установлен);
* точка ``(0, 0, 1)`` проецируется в главную точку ``(539.5, 404.5)``, а ``(0.5, 0, 1)``
  даёт ``u > 539.5`` (плюс ``y`` вниз: ``(0, 0.5, 1)`` → ``v > 404.5``);
* ``Z <= 0.01`` → ``visible = False`` (и координаты NaN, чтобы такое не нарисовалось);
* SBS: левый глаз — обрезанная половина ``0…1080``, правый — те же пиксели плюс 1080;
  лучи внутри FOV попадают в ``1080×810`` (это и есть «центр сцены не выходит за кадр»);
* кривые входы (длина точки, NaN, не-Intrinsics, per-eye интринсики вместо полного кадра,
  неверный ``side``) — ``ValueError``.
"""
from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_model import (  # noqa: E402
    DEBUG_FRAMES,
    MIN_DEPTH_M,
    Intrinsics,
    ProjectedPoint,
    eye_intrinsics,
    head_to_world,
    in_frame,
    project,
    project_eye,
    project_points,
    transform_point,
    world_to_camera,
    world_to_camera_matrix,
)

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
META = next(iter(sorted((ROOT / "output").glob("video_meta.json"))), None)
NO_DATA = "нет data/trackingData_*.txt — файл данных не коммитится (см. README)"
NO_META = "нет output/video_meta.json — он создаётся задачей 02 (см. README)"

#: cameraIntrinsics из header'а записи: [cx, cy, fx, fy] полного SBS-кадра 2160×810.
RAW_INTRINSICS = [1079.5, 404.5, 1373.6681390725917, 686.8680648828482]


def full_intrinsics() -> Intrinsics:
    """Интринсики полного SBS-кадра из значений header'а (файл данных для этого не нужен)."""
    return Intrinsics.from_header({"cameraIntrinsics": RAW_INTRINSICS})


class ProjectedPointContractTest(unittest.TestCase):
    """Возврат ровно ``(u, v, visible)`` — как записано в карточке 05c."""

    def test_is_three_tuple(self) -> None:
        point = project((0.0, 0.0, 1.0), eye_intrinsics(full_intrinsics(), "left"))
        self.assertIsInstance(point, tuple)
        self.assertEqual(len(point), 3)
        self.assertEqual(tuple(point), (point.u, point.v, point.visible))
        self.assertIsInstance(point, ProjectedPoint)

    def test_left_eye_is_default_side(self) -> None:
        self.assertEqual(
            project_eye((0.1, -0.2, 1.0), full_intrinsics()),
            project_eye((0.1, -0.2, 1.0), full_intrinsics(), "left"),
        )


class PrincipalPointTest(unittest.TestCase):
    """Критерий: ``(0, 0, 1)`` → (539.5, 404.5) у левого глаза, +1080 у правого."""

    def test_left_eye_axis_point(self) -> None:
        point = project_eye((0.0, 0.0, 1.0), full_intrinsics(), "left")
        self.assertAlmostEqual(point.u, 539.5, places=12)
        self.assertAlmostEqual(point.v, 404.5, places=12)
        self.assertTrue(point.visible)

    def test_right_eye_axis_point_is_shifted_by_eye_width(self) -> None:
        intrinsics = full_intrinsics()
        left = project_eye((0.0, 0.0, 1.0), intrinsics, "left")
        right = project_eye((0.0, 0.0, 1.0), intrinsics, "right")
        self.assertAlmostEqual(right.u, 1619.5, places=12)
        self.assertAlmostEqual(right.v, 404.5, places=12)
        self.assertAlmostEqual(right.u - left.u, intrinsics.eye_w, places=12)

    def test_eye_intrinsics_values(self) -> None:
        intrinsics = full_intrinsics()
        left = eye_intrinsics(intrinsics, "left")
        right = eye_intrinsics(intrinsics, "right")
        self.assertEqual((left.fx, left.fy, left.cx, left.cy), (intrinsics.fy, intrinsics.fy, 539.5, 404.5))
        self.assertEqual(right.cx, 1619.5)
        self.assertEqual((left.full_w, left.full_h), (1080, 810))


class FormulaTest(unittest.TestCase):
    """``u = fx·X/Z + cx``, ``v = fy·Y/Z + cy``; оси и масштаб ведут себя как ждёт OpenCV."""

    def test_matches_formula_on_grid(self) -> None:
        eye = eye_intrinsics(full_intrinsics(), "left")
        random.seed(5)
        for _ in range(200):
            z = random.uniform(0.05, 5.0)
            x = random.uniform(-1.0, 1.0) * z
            y = random.uniform(-1.0, 1.0) * z
            point = project((x, y, z), eye)
            with self.subTest(x=x, y=y, z=z):
                self.assertAlmostEqual(point.u, eye.fx * x / z + eye.cx, places=12)
                self.assertAlmostEqual(point.v, eye.fy * y / z + eye.cy, places=12)
                self.assertTrue(point.visible)

    def test_positive_x_moves_right(self) -> None:
        point = project_eye((0.5, 0.0, 1.0), full_intrinsics(), "left")
        self.assertGreater(point.u, 539.5)
        self.assertAlmostEqual(point.u, 539.5 + 0.5 * 686.8680648828482, places=9)
        # отрицательный X симметрично уводит влево
        self.assertLess(project_eye((-0.5, 0.0, 1.0), full_intrinsics()).u, 539.5)

    def test_positive_y_moves_down(self) -> None:
        # в OpenCV ось y изображения смотрит вниз, поэтому точка ниже оси взгляда (y > 0
        # в координатах камеры) проецируется ниже центра кадра (v > cy)
        point = project_eye((0.0, 0.5, 1.0), full_intrinsics(), "left")
        self.assertGreater(point.v, 404.5)
        self.assertAlmostEqual(point.v, 404.5 + 0.5 * 686.8680648828482, places=9)
        self.assertLess(project_eye((0.0, -0.5, 1.0), full_intrinsics()).v, 404.5)

    def test_lateral_offset_shrinks_with_depth(self) -> None:
        # смещение в пикселях = fx·X/Z: вдвое дальше — вдвое меньше, вдвое меньше X — вдвое меньше
        offset = 0.5 * 686.8680648828482
        for x, z, expected in ((0.5, 1.0, offset), (0.5, 2.0, offset / 2.0), (0.25, 1.0, offset / 2.0)):
            point = project_eye((x, 0.0, z), full_intrinsics(), "left")
            with self.subTest(x=x, z=z):
                self.assertAlmostEqual(point.u - 539.5, expected, places=9)

    def test_axis_point_does_not_depend_on_depth(self) -> None:
        for z in (0.02, 0.5, 1.0, 10.0, 1000.0):
            point = project_eye((0.0, 0.0, z), full_intrinsics(), "left")
            with self.subTest(z=z):
                self.assertAlmostEqual(point.u, 539.5, places=9)
                self.assertAlmostEqual(point.v, 404.5, places=9)
                self.assertTrue(point.visible)

    def test_offset_is_constant_1080_for_any_point(self) -> None:
        intrinsics = full_intrinsics()
        for p_cam in ((0.3, -0.2, 1.5), (-1.0, 2.0, 3.0), (0.0, 0.0, 0.02)):
            left = project_eye(p_cam, intrinsics, "left")
            right = project_eye(p_cam, intrinsics, "right")
            with self.subTest(p_cam=p_cam):
                self.assertAlmostEqual(right.u - left.u, 1080.0, places=9)
                self.assertAlmostEqual(right.v, left.v, places=12)

    def test_full_frame_intrinsics_give_sbs_pixels(self) -> None:
        # если передать полный кадр (fx = 1373.668, cx = 1079.5), пиксель — в координатах
        # 2160×810: ось взгляда попадает в центр SBS-кадра
        point = project((0.0, 0.0, 1.0), full_intrinsics())
        self.assertAlmostEqual(point.u, 1079.5, places=12)
        self.assertAlmostEqual(point.v, 404.5, places=12)


class DepthClipTest(unittest.TestCase):
    """Критерий: ``Z <= 0.01`` → ``visible = False``; координаты при этом NaN."""

    def test_min_depth_is_one_centimetre(self) -> None:
        self.assertEqual(MIN_DEPTH_M, 0.01)

    def test_zero_and_negative_depth_invisible(self) -> None:
        eye = eye_intrinsics(full_intrinsics(), "left")
        for z in (0.0, -1e-12, -0.5, -1.0, -100.0):
            point = project((0.5, 0.5, z), eye)
            with self.subTest(z=z):
                self.assertFalse(point.visible)
                self.assertTrue(math.isnan(point.u))
                self.assertTrue(math.isnan(point.v))

    def test_depth_up_to_min_is_invisible(self) -> None:
        eye = eye_intrinsics(full_intrinsics(), "left")
        for z in (0.0, 0.005, MIN_DEPTH_M):
            with self.subTest(z=z):
                self.assertFalse(project((0.0, 0.0, z), eye).visible)
        self.assertTrue(project((0.0, 0.0, MIN_DEPTH_M + 1e-9), eye).visible)

    def test_non_finite_coordinate_is_no_data(self) -> None:
        # NaN — «нет данных» (неактивная рука в 03c/04b), а не ошибка входа: рендер может
        # звать project на всех точках подряд; inf — тот же случай
        eye = eye_intrinsics(full_intrinsics(), "left")
        for p_cam in ((0.0, 0.0, math.nan), (math.nan, 0.0, 1.0), (0.0, math.nan, 1.0),
                      (math.inf, 0.0, 1.0), (0.0, 0.0, -math.inf)):
            point = project(p_cam, eye)
            with self.subTest(p_cam=p_cam):
                self.assertFalse(point.visible)
                self.assertTrue(math.isnan(point.u) and math.isnan(point.v))

    def test_overflow_is_invisible_not_infinite(self) -> None:
        eye = eye_intrinsics(full_intrinsics(), "left")
        point = project((1e308, 0.0, 0.02), eye)
        self.assertFalse(point.visible)
        self.assertTrue(math.isnan(point.u) and math.isnan(point.v))

    def test_far_point_stays_visible(self) -> None:
        point = project_eye((0.0, 0.0, 1e6), full_intrinsics(), "left")
        self.assertTrue(point.visible)
        self.assertAlmostEqual(point.u, 539.5, places=9)


class FrameBoundsTest(unittest.TestCase):
    """Критерий: после обрезки SBS координаты сравниваются с кадром 1080×810."""

    def setUp(self) -> None:
        self.intrinsics = full_intrinsics()

    def test_centre_is_inside_both_eyes_at_their_own_offsets(self) -> None:
        axis = project_eye((0.0, 0.0, 1.0), self.intrinsics, "left")
        self.assertTrue(in_frame(axis, self.intrinsics, "left"))
        right = project_eye((0.0, 0.0, 1.0), self.intrinsics, "right")
        self.assertTrue(in_frame(right, self.intrinsics, "right"))
        # центр правого глаза лежит вне кадра левого (его u = 1619.5 > 1080)
        self.assertFalse(in_frame(right, self.intrinsics, "left"))

    def test_horizontal_bounds_left_eye(self) -> None:
        eye = eye_intrinsics(self.intrinsics, "left")
        good = (
            ProjectedPoint(0.0, 404.5, True),
            ProjectedPoint(1079.999, 404.5, True),
        )
        bad = (
            ProjectedPoint(-0.5, 404.5, True),  # левый край FOV — half-pixel, уже за кадром
            ProjectedPoint(-1.0, 404.5, True),
            ProjectedPoint(1080.0, 404.5, True),
        )
        for point in good:
            with self.subTest(point=point):
                self.assertTrue(in_frame(point, self.intrinsics, "left"))
        for point in bad:
            with self.subTest(point=point):
                self.assertFalse(in_frame(point, self.intrinsics, "left"))
        self.assertEqual((eye.full_w, eye.full_h), (1080, 810))

    def test_vertical_bounds(self) -> None:
        for v, expected in ((0.0, True), (809.999, True), (810.0, False), (-0.001, False)):
            point = ProjectedPoint(539.5, v, True)
            with self.subTest(v=v):
                self.assertEqual(in_frame(point, self.intrinsics, "left"), expected)

    def test_right_eye_band(self) -> None:
        inside = ProjectedPoint(1080.0, 404.5, True)
        self.assertTrue(in_frame(inside, self.intrinsics, "right"))
        self.assertFalse(in_frame(inside, self.intrinsics, "left"))
        self.assertFalse(in_frame(ProjectedPoint(2160.0, 404.5, True), self.intrinsics, "right"))

    def test_invisible_point_is_never_in_frame(self) -> None:
        point = project_eye((0.0, 0.0, -1.0), self.intrinsics, "left")
        self.assertFalse(in_frame(point, self.intrinsics, "left"))
        self.assertFalse(in_frame(ProjectedPoint(math.nan, math.nan, False), self.intrinsics, "left"))


class FovCoverageTest(unittest.TestCase):
    """Критерий: для центра сцены (лучи внутри FOV) ни одна точка не выходит за 1080×810."""

    def setUp(self) -> None:
        self.intrinsics = full_intrinsics()
        self.eye = eye_intrinsics(self.intrinsics, "left")
        fov_h, fov_v, _ = self.eye.fov_deg()
        self.half_h = math.radians(fov_h / 2.0)
        self.half_v = math.radians(fov_v / 2.0)

    def test_all_rays_inside_fov_stay_in_frame(self) -> None:
        steps = 21
        for i in range(steps):
            angle_h = -self.half_h * 0.99 + 2.0 * self.half_h * 0.99 * i / (steps - 1)
            for j in range(steps):
                angle_v = -self.half_v * 0.99 + 2.0 * self.half_v * 0.99 * j / (steps - 1)
                point = project_eye((math.tan(angle_h), math.tan(angle_v), 1.0), self.intrinsics)
                with self.subTest(angle_h=angle_h, angle_v=angle_v):
                    self.assertTrue(point.visible)
                    self.assertTrue(in_frame(point, self.intrinsics))

    def test_border_ray_lands_on_half_pixel(self) -> None:
        # край FOV по горизонтали: u = 539.5 ± 540 → 1079.5 (в кадре) и −0.5 (слева за краем)
        right = project_eye((math.tan(self.half_h), 0.0, 1.0), self.intrinsics)
        left = project_eye((-math.tan(self.half_h), 0.0, 1.0), self.intrinsics)
        self.assertAlmostEqual(right.u, 1079.5, places=9)
        self.assertAlmostEqual(left.u, -0.5, places=9)
        self.assertAlmostEqual(project_eye((0.0, math.tan(self.half_v), 1.0), self.intrinsics).v, 809.5, places=9)
        self.assertTrue(in_frame(right, self.intrinsics))
        self.assertFalse(in_frame(left, self.intrinsics))

    def test_projection_grows_monotonically_with_angle(self) -> None:
        previous = -math.inf
        for step in range(30):
            angle = self.half_h * step / 29.0
            point = project_eye((math.tan(angle), 0.0, 1.0), self.intrinsics)
            with self.subTest(step=step):
                self.assertGreater(point.u, previous)
            previous = point.u


class PointBatchTest(unittest.TestCase):
    """``project_points`` — пачка точек за один вызов, порядок и NaN сохраняются."""

    def test_batch_matches_single_calls(self) -> None:
        intrinsics = full_intrinsics()
        points = [(0.0, 0.0, 1.0), (0.1, 0.2, 0.5), (0.0, 0.0, -1.0), (0.4, -0.3, 2.0)]
        batch = project_points(points, intrinsics, "left")
        self.assertEqual(len(batch), len(points))
        for single, batched in zip(points, batch):
            self.assertEqual(project_eye(single, intrinsics, "left"), batched)
        self.assertFalse(batch[2].visible)
        self.assertTrue(math.isnan(batch[2].u))

    def test_empty_batch(self) -> None:
        self.assertEqual(project_points([], full_intrinsics()), [])


class CrossCheckCv2Test(unittest.TestCase):
    """Независимая сверка с ``cv2.projectPoints`` (если opencv установлен)."""

    def test_matches_opencv(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:  # pragma: no cover - opencv ставится через requirements.txt
            self.skipTest("opencv не установлен")
        intrinsics = full_intrinsics()
        random.seed(11)
        points = [
            (random.uniform(-0.6, 0.6) * z, random.uniform(-0.6, 0.6) * z, z)
            for z in (0.2, 0.35, 1.0, 2.5, 6.0)
        ]
        for side in ("left", "right"):
            eye = eye_intrinsics(intrinsics, side)
            camera_matrix = np.array(
                [[eye.fx, 0.0, eye.cx], [0.0, eye.fy, eye.cy], [0.0, 0.0, 1.0]], dtype=np.float64
            )
            expected, _ = cv2.projectPoints(
                np.array(points, dtype=np.float64).reshape(-1, 1, 3),
                np.zeros(3),
                np.zeros(3),
                camera_matrix,
                None,
            )
            for p_cam, (u, v) in zip(points, expected.reshape(-1, 2)):
                actual = project_eye(p_cam, intrinsics, side)
                with self.subTest(side=side, p_cam=p_cam):
                    self.assertAlmostEqual(actual.u, float(u), places=9)
                    self.assertAlmostEqual(actual.v, float(v), places=9)


class ValidationTest(unittest.TestCase):
    """Кривые входы — ``ValueError`` с понятным сообщением, а не тихий мусор."""

    def test_bad_point(self) -> None:
        eye = eye_intrinsics(full_intrinsics(), "left")
        for p_cam in ((0.0, 0.0), (0.0, 0.0, 0.0, 0.0), "point", (0.0, "y", 0.0), (0.0, True, 1.0)):
            with self.subTest(p_cam=p_cam):
                with self.assertRaises(ValueError):
                    project(p_cam, eye)

    def test_non_intrinsics(self) -> None:
        for intrinsics in (None, "intrinsics", (686.868, 686.868, 539.5, 404.5)):
            with self.subTest(intrinsics=intrinsics):
                with self.assertRaises(ValueError):
                    project((0.0, 0.0, 1.0), intrinsics)  # type: ignore[arg-type]

    def test_per_eye_intrinsics_rejected_where_full_frame_expected(self) -> None:
        intrinsics = full_intrinsics()
        for side in ("left", "right"):
            with self.subTest(side=side):
                with self.assertRaises(ValueError):
                    project_eye((0.0, 0.0, 1.0), intrinsics.per_eye(side), "left")

    def test_bad_side(self) -> None:
        intrinsics = full_intrinsics()
        for side in ("LEFT", "", "both"):
            with self.subTest(side=side):
                with self.assertRaises(ValueError):
                    project_eye((0.0, 0.0, 1.0), intrinsics, side)
                with self.assertRaises(ValueError):
                    eye_intrinsics(intrinsics, side)
                with self.assertRaises(ValueError):
                    in_frame(ProjectedPoint(539.5, 404.5, True), intrinsics, side)

    def test_bad_batch(self) -> None:
        intrinsics = full_intrinsics()
        for points in ("abc", 5, None, [(0.0, 0.0, 1.0), (0.0, 0.0)]):
            with self.subTest(points=points):
                with self.assertRaises(ValueError):
                    project_points(points, intrinsics)  # type: ignore[arg-type]


@unittest.skipIf(DATA is None, NO_DATA)
@unittest.skipIf(META is None, NO_META)
class RealDataTest(unittest.TestCase):
    """Проекция на реальных позах: главная точка, видимость кистей и «центр сцены» в кадре."""

    @classmethod
    def setUpClass(cls) -> None:
        import json

        from src.parse_tracking import read_header, timestamp_range
        from src.sync import frame_time, load_poses

        with open(META, encoding="utf-8") as handle:  # type: ignore[arg-type]
            meta = json.load(handle)
        header = read_header(DATA)  # type: ignore[arg-type]
        header_ns = int(header["timeStampNs"])
        cls.intrinsics = Intrinsics.from_header(header)
        tracking_end_ns = timestamp_range(DATA)[1]  # type: ignore[arg-type]
        track = load_poses(DATA)  # type: ignore[arg-type]
        cls.frames = []
        for k in DEBUG_FRAMES:
            frame = frame_time(k, meta, header_ns, tracking_end_ns)
            cls.frames.append((k, track.pose_at(frame.ts_ns)))

    def test_gaze_projects_to_principal_point(self) -> None:
        for k, pose in self.frames:
            for invert in (False, True):
                matrix = world_to_camera_matrix(pose, invert_quat=invert)
                gaze_world = transform_point(head_to_world(pose, invert_quat=invert), (0.0, 0.0, 1.0))
                point = project_eye(transform_point(matrix, gaze_world), self.intrinsics)
                with self.subTest(k=k, invert=invert):
                    self.assertTrue(point.visible)
                    self.assertAlmostEqual(point.u, 539.5, places=9)
                    self.assertAlmostEqual(point.v, 404.5, places=9)
                    self.assertTrue(in_frame(point, self.intrinsics))

    def test_centre_of_scene_box_stays_in_frame(self) -> None:
        # геометрическая часть критерия «центр сцены»: лучи в пределах ±10° от оси взгляда
        # на глубине 0.3…3 м проецируются внутрь кадра 1080×810
        for depth in (0.3, 1.0, 3.0):
            for angle_h in (-10.0, -5.0, 0.0, 5.0, 10.0):
                for angle_v in (-10.0, 0.0, 10.0):
                    point = project_eye(
                        (
                            math.tan(math.radians(angle_h)) * depth,
                            math.tan(math.radians(angle_v)) * depth,
                            depth,
                        ),
                        self.intrinsics,
                    )
                    with self.subTest(depth=depth, angle_h=angle_h, angle_v=angle_v):
                        self.assertTrue(point.visible)
                        self.assertTrue(in_frame(point, self.intrinsics))

    def test_hands_behind_camera_are_invisible(self) -> None:
        # замер 05b: при invert_quat=False все 10 точек кистей оказываются за камерой,
        # поэтому проекция обязана вернуть visible=False, а не пиксели «наоборот»
        for k, pose in self.frames:
            matrix = world_to_camera_matrix(pose, invert_quat=False)
            finite = visible = 0
            for point_world in pose.hands.values():
                if any(math.isnan(value) for value in point_world):
                    continue
                finite += 1
                point = project_eye(transform_point(matrix, point_world), self.intrinsics)
                if point.visible:
                    visible += 1
                    self.assertTrue(math.isfinite(point.u) and math.isfinite(point.v))
                else:
                    self.assertTrue(math.isnan(point.u) and math.isnan(point.v))
                    self.assertFalse(in_frame(point, self.intrinsics))
            with self.subTest(k=k):
                self.assertEqual(finite, 10)
                self.assertEqual(visible, 0)

    def test_world_to_camera_then_project_matches_project_eye(self) -> None:
        # полный тракт «мир → камера → пиксель» через публичные функции совпадает с project_eye
        for k, pose in self.frames:
            for side in ("left", "right"):
                eye = eye_intrinsics(self.intrinsics, side)
                for point_world in pose.hands.values():
                    if any(math.isnan(value) for value in point_world):
                        continue
                    p_cam = world_to_camera(point_world, pose, invert_quat=True)
                    with self.subTest(k=k, side=side, point_world=point_world):
                        self.assertEqual(project(p_cam, eye), project_eye(p_cam, self.intrinsics, side))


if __name__ == "__main__":
    unittest.main()
