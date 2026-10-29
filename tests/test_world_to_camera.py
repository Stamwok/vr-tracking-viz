"""Assert-тесты для src/camera_model.py, задача 05b: world → camera.

Запуск: python -m unittest discover -s tests -t . -v
Тесты на реальном файле трекинга пропускаются, если data/trackingData_*.txt отсутствует.

Что проверяется (критерии карточки 05b):

* композиция ``T_cam_world = F · T_head_cam⁻¹ · T_head_world⁻¹``: точка перед головой
  ``(0, 0, −1)`` (в OpenXR вперёд смотрит ``−z``) даёт ``x ≈ y ≈ 0``, ``z > 0``;
* чистый поворот головы на 90° по yaw: точка в направлении взгляда остаётся на оси,
  точка в стороне уходит за камеру (``z < 0``);
* ``T_head_cam`` читается из ``src/config.py`` и меняется без правки кода;
* флаги ``opencv_conventions`` (Unity → OpenCV) и ``invert_quat`` (конвенция кватерниона)
  действительно включены в тракт, а не игнорируются;
* кривые входы (длина позы, нулевой кватернион, NaN, вырожденная матрица) — ``ValueError``.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.camera_model import (  # noqa: E402
    IDENTITY4,
    OPENCV_FLIP_Y,
    head_cam_matrix,
    head_pose_parts,
    head_to_world,
    quat_to_rotation,
    rotation_rpy,
    transform_point,
    world_to_camera,
    world_to_camera_matrix,
)

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
NO_DATA = "нет data/trackingData_*.txt — файл данных не коммитится (см. README)"

#: Единичная поза: голова в начале координат, без поворота.
IDENTITY_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def yaw90_quat() -> tuple[float, float, float, float]:
    """Кватернион поворота на 90° вокруг ``y`` (``x, y, z, w``)."""
    half = math.sqrt(0.5)
    return (0.0, half, 0.0, half)


def matrix_product(a, b):
    """Произведение 3×3/4×4 для проверок (независимо от внутренних хелперов модуля)."""
    n = len(a)
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0])))
        for i in range(n)
    )


def transpose3(matrix):
    """Транспонирование 3×3."""
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


def apply3(matrix, point):
    """Применить 3×3 к точке."""
    return tuple(sum(matrix[i][k] * point[k] for k in range(3)) for i in range(3))


class GazeAxisTest(unittest.TestCase):
    """Композиция сохраняет ось: у стаба-камеры (``T_head_cam = I``) это ``+z`` головы."""

    def test_gaze_axis_identity_pose(self) -> None:
        # точка вдоль +z системы головы; при стабе-камере (I) это её оптическая ось
        camera = world_to_camera((0.0, 0.0, 1.0), IDENTITY_POSE, t_head_cam=IDENTITY4)
        self.assertAlmostEqual(camera[0], 0.0, places=12)
        self.assertAlmostEqual(camera[1], 0.0, places=12)
        self.assertGreater(camera[2], 0.0)
        self.assertAlmostEqual(camera[2], 1.0, places=12)

    def test_gaze_axis_any_pose(self) -> None:
        # точка взгляда задаётся в системе головы, поэтому при повороте головы она
        # остаётся на оси камеры: проверяется, что композиция не «теряет» поворот.
        quat = (0.1826, -0.2861, -0.1165, 0.9300)  # нормируется внутри
        poses = [
            (1.2, -0.4, 0.3, *quat),
            (-0.8, 0.7, -1.5, *yaw90_quat()),
            (0.0, 0.0, 0.0, 0.0, 0.0, math.sin(0.7), math.cos(0.7)),  # pitch 80°
        ]
        for pose in poses:
            with self.subTest(pose=pose[:3]):
                gaze_world = transform_point(head_to_world(pose), (0.0, 0.0, 1.0))
                camera = world_to_camera(gaze_world, pose, t_head_cam=IDENTITY4)
                self.assertAlmostEqual(camera[0], 0.0, places=9)
                self.assertAlmostEqual(camera[1], 0.0, places=9)
                self.assertAlmostEqual(camera[2], 1.0, places=9)

    def test_gaze_offset_comes_from_config_translation(self) -> None:
        # камера без поворота, смещение из config.py (патч): точка (0, 0, 1) даёт z = 1 − tz
        with mock.patch.object(config, "HEAD_CAM_ROTATION_RPY_DEG", (0.0, 0.0, 0.0)), \
                mock.patch.object(config, "HEAD_CAM_TRANSLATION_M", (0.0, 0.0, 0.05)):
            camera = world_to_camera((0.0, 0.0, 1.0), IDENTITY_POSE, opencv_conventions=False)
        self.assertAlmostEqual(camera[0], 0.0, places=12)
        self.assertAlmostEqual(camera[1], 0.0, places=12)
        self.assertAlmostEqual(camera[2], 0.95, places=12)


class YawTest(unittest.TestCase):
    """Чистый поворот головы на 90° по yaw."""

    def test_point_along_gaze_axis_stays_on_axis(self) -> None:
        pose = (0.0, 0.0, 0.0, *yaw90_quat())
        gaze_world = transform_point(head_to_world(pose), (0.0, 0.0, 1.0))
        # поворот на 90°: ось взгляда головы смотрит в мир по +x
        self.assertAlmostEqual(gaze_world[0], 1.0, places=12)
        self.assertAlmostEqual(gaze_world[1], 0.0, places=12)
        self.assertAlmostEqual(gaze_world[2], 0.0, places=12)
        camera = world_to_camera((1.0, 0.0, 0.0), pose, t_head_cam=IDENTITY4)
        self.assertAlmostEqual(camera[0], 0.0, places=12)
        self.assertAlmostEqual(camera[1], 0.0, places=12)
        self.assertGreater(camera[2], 0.0)

    def test_point_to_the_side_goes_behind_camera(self) -> None:
        # голова повёрнута на 90° вправо, точка мира по +z теперь слева от неё
        pose = (0.0, 0.0, 0.0, *yaw90_quat())
        camera = world_to_camera((0.0, 0.0, 1.0), pose, t_head_cam=IDENTITY4)
        self.assertLess(camera[2], 0.0)
        self.assertAlmostEqual(camera[0], -1.0, places=12)

    def test_translation_is_applied(self) -> None:
        pose = (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
        camera = world_to_camera((1.0, 2.0, 4.0), pose, t_head_cam=IDENTITY4)
        self.assertAlmostEqual(camera[0], 0.0, places=12)
        self.assertAlmostEqual(camera[1], 0.0, places=12)
        self.assertAlmostEqual(camera[2], 1.0, places=12)
        # точка в стороне от головы: смещение головы учитывается, а не теряется
        side = world_to_camera((2.0, 2.0, 3.0), pose, t_head_cam=IDENTITY4)
        self.assertAlmostEqual(side[0], 1.0, places=12)


class OpenCvConventionTest(unittest.TestCase):
    """Флаг Unity (y вверх) → OpenCV (y вниз) включён в тракт."""

    def test_up_point_lands_above_optical_axis(self) -> None:
        point = (0.0, 1.0, 1.0)
        opencv = world_to_camera(
            point, IDENTITY_POSE, t_head_cam=IDENTITY4, opencv_conventions=True
        )
        unity = world_to_camera(
            point, IDENTITY_POSE, t_head_cam=IDENTITY4, opencv_conventions=False
        )
        self.assertLess(opencv[1], 0.0)  # в кадре «выше центра» — это y < 0
        self.assertGreater(unity[1], 0.0)
        self.assertAlmostEqual(opencv[0], unity[0], places=12)
        self.assertAlmostEqual(opencv[2], unity[2], places=12)

    def test_right_and_forward_axes_unchanged(self) -> None:
        point = (1.0, 0.0, 2.0)
        opencv = world_to_camera(
            point, IDENTITY_POSE, t_head_cam=IDENTITY4, opencv_conventions=True
        )
        unity = world_to_camera(
            point, IDENTITY_POSE, t_head_cam=IDENTITY4, opencv_conventions=False
        )
        self.assertAlmostEqual(opencv[0], unity[0], places=12)
        self.assertAlmostEqual(opencv[2], unity[2], places=12)

    def test_flip_matrix_value(self) -> None:
        expected = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        self.assertEqual(OPENCV_FLIP_Y, expected)


class ConfigTest(unittest.TestCase):
    """T_head_cam живёт в src/config.py и меняется без правки кода камеры."""

    def test_head_cam_matrix_is_built_from_config(self) -> None:
        matrix = head_cam_matrix()
        self.assertAlmostEqual(matrix[3][3], 1.0, places=12)
        for index in range(3):
            self.assertAlmostEqual(matrix[3][index], 0.0, places=12)
        tx, ty, tz = config.HEAD_CAM_TRANSLATION_M
        self.assertAlmostEqual(matrix[0][3], tx, places=12)
        self.assertAlmostEqual(matrix[1][3], ty, places=12)
        self.assertAlmostEqual(matrix[2][3], tz, places=12)
        rotation = tuple(tuple(matrix[i][j] for j in range(3)) for i in range(3))
        yaw, pitch, roll = config.HEAD_CAM_ROTATION_RPY_DEG
        expected = rotation_rpy(yaw, pitch, roll)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(rotation[i][j], expected[i][j], places=12)

    def test_translation_change_without_code_edit(self) -> None:
        # поворот временно нулевой, чтобы проверять именно смещение из конфига
        with mock.patch.object(config, "HEAD_CAM_ROTATION_RPY_DEG", (0.0, 0.0, 0.0)), \
                mock.patch.object(config, "HEAD_CAM_TRANSLATION_M", (0.0, 0.0, 0.20)):
            camera = world_to_camera((0.0, 0.0, 1.0), IDENTITY_POSE, opencv_conventions=False)
        self.assertAlmostEqual(camera[2], 0.80, places=12)
        # без патча — значение из config.py: калибровка 07c, точка перед головой даёт z = 1
        self.assertAlmostEqual(
            world_to_camera((0.0, 0.0, -1.0), IDENTITY_POSE)[2], 1.0, places=12
        )

    def test_rotation_change_without_code_edit(self) -> None:
        # pitch камеры на 90°: ось взгляда головы уходит из поля зрения (z = 0)
        with mock.patch.object(config, "HEAD_CAM_ROTATION_RPY_DEG", (0.0, 90.0, 0.0)), \
                mock.patch.object(config, "HEAD_CAM_TRANSLATION_M", (0.0, 0.0, 0.0)):
            camera = world_to_camera((0.0, 0.0, 1.0), IDENTITY_POSE, opencv_conventions=False)
        self.assertAlmostEqual(camera[0], 0.0, places=12)
        self.assertAlmostEqual(camera[1], 1.0, places=12)
        self.assertAlmostEqual(camera[2], 0.0, places=12)

    def test_explicit_matrix_overrides_config(self) -> None:
        with mock.patch.object(config, "HEAD_CAM_TRANSLATION_M", (0.0, 0.0, 5.0)):
            from_stub = world_to_camera((0.0, 0.0, 1.0), IDENTITY_POSE, t_head_cam=IDENTITY4)
        self.assertAlmostEqual(from_stub[2], 1.0, places=12)

    def test_config_rejects_wrong_values(self) -> None:
        for bad in ((0.0, 0.0), (0.0, 0.0, "z"), (0.0, math.nan, 0.0)):
            with self.subTest(bad=bad):
                with mock.patch.object(config, "HEAD_CAM_TRANSLATION_M", bad):
                    with self.assertRaises(ValueError):
                        head_cam_matrix()


class MatrixTest(unittest.TestCase):
    """Свойства 4×4 и согласованность точки с матрицей."""

    def test_point_matches_matrix(self) -> None:
        pose = (0.3, -0.2, 0.1, 0.1, -0.2, 0.3, 0.92)
        point = (-0.5, 0.4, 1.3)
        matrix = world_to_camera_matrix(pose)
        for camera in (transform_point(matrix, point), world_to_camera(point, pose)):
            self.assertAlmostEqual(camera[0], transform_point(matrix, point)[0], places=12)
            self.assertAlmostEqual(camera[1], transform_point(matrix, point)[1], places=12)
            self.assertAlmostEqual(camera[2], transform_point(matrix, point)[2], places=12)
        # вложенная 4×4, плоский список 16 чисел и кортеж строк — одно и то же
        flat = [value for row in matrix for value in row]
        self.assertEqual(transform_point(flat, point), transform_point(matrix, point))

    def test_rigid_transform_without_opencv_flag(self) -> None:
        pose = (0.3, -0.2, 0.1, 0.1, -0.2, 0.3, 0.92)
        matrix = world_to_camera_matrix(pose, opencv_conventions=False)
        rotation = tuple(tuple(matrix[i][j] for j in range(3)) for i in range(3))
        product = matrix_product(transpose3(rotation), rotation)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(product[i][j], 1.0 if i == j else 0.0, places=12)
        det = (
            rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        self.assertAlmostEqual(det, 1.0, places=12)

    def test_opencv_flag_makes_determinant_negative(self) -> None:
        # Unity (y вверх) → OpenCV (y вниз) — отражение: det = −1, это ожидаемо
        matrix = world_to_camera_matrix(IDENTITY_POSE, opencv_conventions=True)
        rotation = tuple(tuple(matrix[i][j] for j in range(3)) for i in range(3))
        det = (
            rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
            - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
            + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
        )
        self.assertAlmostEqual(det, -1.0, places=12)

    def test_identity_config_identity_pose(self) -> None:
        # конфиг-заглушка (без поворота и смещения) + выключенный переворот y: M = I
        with mock.patch.object(config, "HEAD_CAM_ROTATION_RPY_DEG", (0.0, 0.0, 0.0)), \
                mock.patch.object(config, "HEAD_CAM_TRANSLATION_M", (0.0, 0.0, 0.0)):
            matrix = world_to_camera_matrix(IDENTITY_POSE, opencv_conventions=False)
        self.assertEqual(transform_point(matrix, (0.1, 0.2, 0.3)), (0.1, 0.2, 0.3))


class QuatConventionTest(unittest.TestCase):
    """invert_quat — дискретный вариант конвенции для 07b."""

    def test_invert_quat_transposes_rotation(self) -> None:
        pose = (0.0, 0.0, 0.0, *yaw90_quat())
        forward = head_to_world(pose)
        inverted = head_to_world(pose, invert_quat=True)
        left_rotation = tuple(tuple(forward[i][j] for j in range(3)) for i in range(3))
        left_inverted = tuple(tuple(inverted[i][j] for j in range(3)) for i in range(3))
        self.assertEqual(left_inverted, transpose3(left_rotation))
        # ось взгляда меняет направление на противоположное
        self.assertAlmostEqual(apply3(left_rotation, (0.0, 0.0, 1.0))[0], 1.0, places=12)
        self.assertAlmostEqual(apply3(left_inverted, (0.0, 0.0, 1.0))[0], -1.0, places=12)

    def test_invert_quat_moves_front_to_back(self) -> None:
        pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # для единичного кватерниона не меняет
        point = (0.0, 0.0, 1.0)
        self.assertEqual(
            world_to_camera(point, pose, t_head_cam=IDENTITY4),
            world_to_camera(point, pose, t_head_cam=IDENTITY4, invert_quat=True),
        )
        rotated = (0.0, 0.0, 0.0, *yaw90_quat())
        gaze_world = transform_point(head_to_world(rotated), (0.0, 0.0, 1.0))
        front = world_to_camera(gaze_world, rotated, t_head_cam=IDENTITY4)
        back = world_to_camera(gaze_world, rotated, t_head_cam=IDENTITY4, invert_quat=True)
        self.assertGreater(front[2], 0.0)
        self.assertLess(back[2], 0.0)


class PoseInputTest(unittest.TestCase):
    """Поза принимается и кортежем из 7 чисел, и объектом с head_pos/head_quat."""

    def test_sequence_and_object_agree(self) -> None:
        pose = (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0)
        as_object = SimpleNamespace(head_pos=(0.1, 0.2, 0.3), head_quat=(0.0, 0.0, 0.0, 1.0))
        self.assertEqual(head_to_world(pose), head_to_world(as_object))
        self.assertEqual(
            world_to_camera((0.2, 0.2, 0.3), pose),
            world_to_camera((0.2, 0.2, 0.3), as_object),
        )

    def test_parts_of_sequence(self) -> None:
        position, quaternion = head_pose_parts((1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))
        self.assertEqual(position, (1.0, 2.0, 3.0))
        self.assertEqual(quaternion, (0.0, 0.0, 0.0, 1.0))

    def test_incomplete_object_rejected(self) -> None:
        with self.assertRaises(ValueError):
            head_pose_parts(SimpleNamespace(head_pos=(0.0, 0.0, 0.0)))
        with self.assertRaises(ValueError):
            head_pose_parts(SimpleNamespace(head_quat=(0.0, 0.0, 0.0, 1.0)))


class ValidationTest(unittest.TestCase):
    """Кривые входы — ValueError с понятным сообщением, а не мусор в матрице."""

    def test_wrong_pose_length(self) -> None:
        for pose in ((0.0, 0.0, 0.0), (0.0,) * 6, (0.0,) * 8):
            with self.subTest(length=len(pose)):
                with self.assertRaises(ValueError):
                    head_to_world(pose)

    def test_bad_quaternion(self) -> None:
        for quat in ((0.0, 0.0, 0.0, 0.0), (math.nan, 0.0, 0.0, 1.0), (0.0, math.inf, 0.0, 1.0)):
            with self.subTest(quat=quat):
                with self.assertRaises(ValueError):
                    head_to_world((0.0, 0.0, 0.0, *quat))

    def test_bad_point(self) -> None:
        for point in ((0.0, 0.0), (0.0, 0.0, 0.0, 0.0), (0.0, "y", 0.0), (0.0, math.nan, 0.0)):
            with self.subTest(point=point):
                with self.assertRaises(ValueError):
                    world_to_camera(point, IDENTITY_POSE)

    def test_singular_head_cam_rejected(self) -> None:
        singular = ((0.0,) * 4,) * 4
        with self.assertRaises(ValueError):
            world_to_camera_matrix(IDENTITY_POSE, t_head_cam=singular)

    def test_wrong_head_cam_shape_rejected(self) -> None:
        with self.assertRaises(ValueError):
            world_to_camera_matrix(IDENTITY_POSE, t_head_cam=((1.0, 0.0, 0.0),) * 4)
        with self.assertRaises(ValueError):
            world_to_camera_matrix(IDENTITY_POSE, t_head_cam="matrix")

    def test_quat_rotation_is_orthonormal(self) -> None:
        rotation = quat_to_rotation((0.1, -0.2, 0.3, 0.92))
        product = matrix_product(transpose3(rotation), rotation)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(product[i][j], 1.0 if i == j else 0.0, places=12)


class RotationRpyTest(unittest.TestCase):
    """Порядок поворотов ``R_y(yaw) · R_x(pitch) · R_z(roll)`` зафиксирован."""

    def test_identity(self) -> None:
        identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        self.assertEqual(rotation_rpy(0.0, 0.0, 0.0), identity)

    def test_yaw_turns_forward_to_right(self) -> None:
        rotation = rotation_rpy(90.0, 0.0, 0.0)
        for actual, expected in zip(apply3(rotation, (0.0, 0.0, 1.0)), (1.0, 0.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_pitch_tilts_forward_down(self) -> None:
        rotation = rotation_rpy(0.0, 90.0, 0.0)
        for actual, expected in zip(apply3(rotation, (0.0, 0.0, 1.0)), (0.0, -1.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_roll_applied_first(self) -> None:
        # roll вокруг z: (1,0,0) → (0,1,0); yaw вокруг y это не меняет
        rotation = rotation_rpy(90.0, 0.0, 90.0)
        for actual, expected in zip(apply3(rotation, (1.0, 0.0, 0.0)), (0.0, 1.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_rejects_non_numeric(self) -> None:
        for values in (("a", 0.0, 0.0), (0.0, math.nan, 0.0)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    rotation_rpy(*values)


@unittest.skipIf(DATA is None, NO_DATA)
class RealDataTest(unittest.TestCase):
    """Проверка на реальных позах: композиция согласована, данные не ломают тракт."""

    @classmethod
    def setUpClass(cls) -> None:
        from src.parse_tracking import iter_samples, parse_pose

        cls.samples = []
        for index, sample in enumerate(iter_samples(DATA)):
            if index % 4000:
                continue
            pose = tuple(float(value) for value in parse_pose(sample["Head"]["pose"]))
            cls.samples.append((index, pose, sample))

    def test_gaze_axis_on_real_poses(self) -> None:
        # точка перед головой (0, 0, −1) при калибровке 07c → x ≈ y ≈ 0, z > 0
        for index, pose, _ in self.samples:
            with self.subTest(index=index):
                gaze_world = transform_point(head_to_world(pose), (0.0, 0.0, -1.0))
                camera = world_to_camera(gaze_world, pose)
                self.assertAlmostEqual(camera[0], 0.0, places=9)
                self.assertAlmostEqual(camera[1], 0.0, places=9)
                self.assertGreater(camera[2], 0.0)

    def test_hands_are_finite_and_conventions_differ(self) -> None:
        from src.parse_tracking import extract_sample

        for index, pose, sample in self.samples:
            record = extract_sample(sample)
            matrix = world_to_camera_matrix(pose)
            other = world_to_camera_matrix(pose, invert_quat=True)
            finite = 0
            ahead = {False: 0, True: 0}
            for side in ("left", "right"):
                for name in ("palm", "wrist", "thumb", "index", "middle"):
                    point = tuple(record[f"{side}_{name}_{axis}"] for axis in ("x", "y", "z"))
                    if any(math.isnan(value) for value in point):
                        continue
                    finite += 1
                    a = transform_point(matrix, point)
                    b = transform_point(other, point)
                    self.assertTrue(all(math.isfinite(value) for value in a))
                    self.assertTrue(all(math.isfinite(value) for value in b))
                    ahead[False] += a[2] > 0.0
                    ahead[True] += b[2] > 0.0
            with self.subTest(index=index):
                self.assertGreater(finite, 0)
                # замер 05b: две конвенции дают разные полупространства (вход в 07b)
                self.assertNotEqual(ahead[False], ahead[True])


if __name__ == "__main__":
    unittest.main()
