"""Assert-тесты для src/calib_search.py, задача 07b: конвенция осей и грубый ``T_head_cam``.

Запуск: python -m unittest discover -s tests -t . -v
Тесты на реальном файле трекинга пропускаются, если нет data/trackingData_*.txt или
output/video_meta.json.

Что проверяется (критерии карточки 07b):

* перебор осей — ровно 24 собственные ротации (перестановки со знаками, ``det = +1``),
  без повторов; все 48 (с несобственными) — для понимания, что отброшено;
* метрика эталона: точка внутри своей рамки → 0, в чужой → ``wrong``, мимо → ``miss``,
  за камерой / вне кадра не считается; средняя ошибка считается по попавшим в кадр;
* ранжирование: малое покрытие не даёт выиграть «формально нулевой» ошибкой, при равной
  метрике выбирается минимальная правка ``T_head_cam`` (нулевое смещение и нулевой
  добавочный поворот), а не произвольная точка плато;
* ``Convention.matrix_for`` совпадает с ``camera_model.world_to_camera_matrix`` с тем же
  ``T_head_cam`` и ``opencv_conventions=False`` — подбор не дублирует математику 05b;
* на синтетической сцене с известной конвенцией перебор возвращает именно её;
* на реальных данных (если они есть) победитель — ``+x −y −z``, ``invert_quat=False``,
  нулевое смещение, 39/39 точек в своих рамках, средняя ошибка 0 px;
* ``rpy_from_matrix`` — обратная к ``rotation_rpy`` (включая шарнирный замок) и даёт
  читаемые углы для ``config.py`` (поворот на 180° вокруг ``x`` → ``(0, 180, 0)``).
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import calib_search  # noqa: E402
from src.calib_search import (  # noqa: E402
    ANGLE_RANGE_DEG,
    ANGLE_STEP_DEG,
    COVERAGE_TOLERANCE,
    MAX_MEAN_ERROR_PX,
    OFFSET_RANGE_M,
    OFFSET_STEP_M,
    CalibData,
    Candidate,
    Convention,
    Metrics,
    PointTarget,
    axis_rotations,
    evaluate,
    evaluate_convention,
    format_table,
    load_calib_data,
    offset_grid,
    offset_values,
    rank,
    refine,
    rpy_from_matrix,
    search_discrete,
    sensitivity,
    winner_payload,
)
from src.camera_model import Intrinsics, rotation_rpy, world_to_camera_matrix  # noqa: E402

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
META = ROOT / "output" / "video_meta.json"
NO_DATA = "нет data/trackingData_*.txt — файл данных не коммитится (см. README)"
NO_META = "нет output/video_meta.json — он создаётся задачей 02 (см. README)"

RAW_INTRINSICS = [1079.5, 404.5, 1373.6681390725917, 686.8680648828482]
EYE_W, EYE_H = 1080, 810

#: Победитель, записанный в отчёте 07b: камера смотрит вперёд (``z`` головы — назад).
WINNER_AXES = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))

#: Точно нормированный кватернион поворота на 90° вокруг ``y`` — для сверки с ``camera_model``.
SQRT_HALF = math.sqrt(0.5)


def intrinsics() -> Intrinsics:
    """Per-eye интринсики левого глаза из значений header'а (файл данных не нужен)."""
    return Intrinsics.from_header({"cameraIntrinsics": RAW_INTRINSICS}).per_eye("left")


def project_head(
    matrix: tuple[tuple[float, ...], ...], point: tuple[float, float, float]
) -> tuple[float, float]:
    """Пиксель ``(u, v)`` для точки в системе головы по матрице ``p_cam = A·p + b``."""
    camera = tuple(
        sum(matrix[row][column] * point[column] for column in range(3)) + matrix[row][3]
        for row in range(3)
    )
    eye = intrinsics()
    return (eye.fx * camera[0] / camera[2] + eye.cx, eye.fy * camera[1] / camera[2] + eye.cy)


def point_in_front(
    matrix: tuple[tuple[float, ...], ...], x: float = 0.0, y: float = 0.0
) -> tuple[float, float, float]:
    """Точка системы головы, которая при данной матрице проецируется в ``(cx, cy)``-сдвиг.

    Проще: берём точку на глубине 0.5 м вперёд (``z_head = −0.5``) и сдвигаем её так,
    чтобы в системе камеры получилось ``(X, Y, Z) = (x·Z, y·Z, 0.5)`` — обратный перевод
    через ``Aᵀ``. Так тесты не зависят от конкретной ориентации камеры.
    """
    rotation = tuple(tuple(matrix[row][column] for column in range(3)) for row in range(3))
    translation = tuple(matrix[row][3] for row in range(3))
    target = (x * 0.5, y * 0.5, 0.5)
    # p_head = Aᵀ·(p_cam − b)
    local = tuple(target[row] - translation[row] for row in range(3))
    return tuple(
        sum(rotation[column][row] * local[column] for column in range(3)) for row in range(3)
    )  # type: ignore[return-value]


def make_data(points: list[tuple[tuple[float, float, float], tuple[int, int, int, int],
                                 tuple[tuple[int, int, int, int], ...]]]) -> CalibData:
    """Собрать ``CalibData`` из пар «точка в системе головы → рамки».

    ``frames`` пустые: для метрики и перебора номера кадров не нужны (только для PNG).
    """
    targets = tuple(
        PointTarget(
            frame_index=0,
            side="left",
            joint="palm",
            head=head,
            head_inverted=head,
            own=own,
            others=others,
        )
        for head, own, others in points
    )
    return CalibData(intrinsics=intrinsics(), frames=(), points=targets)


def matrix4(rotation, offset=(0.0, 0.0, 0.0)) -> tuple[tuple[float, ...], ...]:
    """``p_cam = A·(p_head − c)`` в виде 4×4 ``[A | −A·c]``."""
    b = tuple(-sum(rotation[row][column] * offset[column] for column in range(3))
              for row in range(3))
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], b[0]),
        (rotation[1][0], rotation[1][1], rotation[1][2], b[1]),
        (rotation[2][0], rotation[2][1], rotation[2][2], b[2]),
        (0.0, 0.0, 0.0, 1.0),
    )


class AxisRotationsTest(unittest.TestCase):
    """Перебор осей: только собственные ротации, все разные."""

    def test_twenty_four_proper_rotations(self) -> None:
        rotations = axis_rotations(proper=True)
        self.assertEqual(len(rotations), 24)
        self.assertEqual(len({tuple(tuple(row) for row in matrix) for matrix in rotations}), 24)
        for matrix in rotations:
            self.assertAlmostEqual(_det(matrix), 1.0, places=12)
            for row in matrix:
                self.assertEqual(sum(value * value for value in row), 1.0)
                self.assertTrue(all(value in (-1.0, 0.0, 1.0) for value in row))

    def test_all_axis_permutations_are_48(self) -> None:
        self.assertEqual(len(axis_rotations(proper=False)), 48)
        improper = [m for m in axis_rotations(proper=False) if _det(m) < 0]
        self.assertEqual(len(improper), 24)

    def test_winner_is_in_the_search_space(self) -> None:
        self.assertIn(WINNER_AXES, axis_rotations(proper=True))

    def test_axes_text(self) -> None:
        identity = Convention(rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        self.assertEqual(identity.axes, "+x +y +z")
        winner = Convention(rotation=WINNER_AXES)
        self.assertEqual(winner.axes, "+x −y −z")

    def test_own_rotation_rejects_reflections(self) -> None:
        with self.assertRaises(ValueError):
            calib_search.rotation_basis(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)))
        with self.assertRaises(ValueError):
            calib_search.rotation_basis(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 2.0)))


def _det(matrix) -> float:
    """Определитель 3×3 для тестов (независимо от модуля)."""
    (a, b, c), (d, e, f), (g, h, i) = matrix
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


class RpyTest(unittest.TestCase):
    """``rpy_from_matrix`` — обратная к ``rotation_rpy`` в договорённости config.py."""

    def test_round_trip(self) -> None:
        for angles in (
            (0.0, 0.0, 0.0),
            (0.0, 180.0, 0.0),
            (180.0, 0.0, 180.0),
            (30.0, -40.0, 50.0),
            (-120.0, 20.0, 170.0),
            (10.0, 45.0, -60.0),
        ):
            matrix = rotation_rpy(*angles)
            restored = rotation_rpy(*rpy_from_matrix(matrix))
            for row in range(3):
                for column in range(3):
                    self.assertAlmostEqual(restored[row][column], matrix[row][column], places=9,
                                           msg=f"{angles} → {rpy_from_matrix(matrix)}")

    def test_winner_reads_as_pitch_180(self) -> None:
        """Поворот на 180° вокруг x даёт (0, 180, 0), а не эквивалентное (180, 0, 180)."""
        self.assertEqual(rpy_from_matrix(WINNER_AXES), (0.0, 180.0, 0.0))
        self.assertEqual(rpy_from_matrix(rotation_rpy(180.0, 0.0, 180.0)), (0.0, 180.0, 0.0))

    def test_no_negative_zero(self) -> None:
        for value in rpy_from_matrix(WINNER_AXES):
            self.assertGreaterEqual(math.copysign(1.0, value), 0.0, f"−0 в углах: {value!r}")

    def test_gimbal_lock_round_trip(self) -> None:
        matrix = rotation_rpy(30.0, 90.0, 40.0)
        angles = rpy_from_matrix(matrix)
        self.assertEqual(angles[2], 0.0)
        restored = rotation_rpy(*angles)
        for row in range(3):
            for column in range(3):
                self.assertAlmostEqual(restored[row][column], matrix[row][column], places=9)

    def test_broken_matrix_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rpy_from_matrix(((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(ValueError):
            rpy_from_matrix(((1.0, 0.0, math.nan), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


class ConventionTest(unittest.TestCase):
    """Связь гипотезы с матрицами: ``head_cam``, ``matrix_for``, ``camera_translation``."""

    def test_head_cam_is_inverse_of_projection(self) -> None:
        convention = Convention(rotation=WINNER_AXES, camera_offset=(0.01, -0.02, 0.03))
        head_cam = convention.head_cam()
        for row in range(3):
            for column in range(3):
                self.assertAlmostEqual(head_cam[row][column],
                                       WINNER_AXES[column][row], places=12)
        self.assertEqual(tuple(head_cam[row][3] for row in range(3)), (0.01, -0.02, 0.03))
        b = convention.camera_translation
        for row in range(3):
            expected = -sum(WINNER_AXES[row][column] * (0.01, -0.02, 0.03)[column]
                            for column in range(3))
            self.assertAlmostEqual(b[row], expected, places=12)

    def test_matrix_matches_camera_model(self) -> None:
        """Подбор не дублирует 05b: та же матрица получается через ``camera_model``."""
        poses = [
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            ((-0.0852, -0.0275, -0.0059), (0.1626, -0.2861, -0.1165, -0.9371)),
            ((-0.2, 0.1, 0.05), (0.0, SQRT_HALF, 0.0, SQRT_HALF)),
        ]
        for invert in (False, True):
            for offset in ((0.0, 0.0, 0.0), (0.02, -0.01, 0.03)):
                convention = Convention(
                    rotation=WINNER_AXES, invert_quat=invert, camera_offset=offset
                )
                for head_pos, head_quat in poses:
                    expected = world_to_camera_matrix(
                        (head_pos[0], head_pos[1], head_pos[2],
                         head_quat[0], head_quat[1], head_quat[2], head_quat[3]),
                        t_head_cam=convention.head_cam(),
                        opencv_conventions=False,
                        invert_quat=invert,
                    )
                    actual = convention.matrix_for(head_pos, head_quat)
                    for row in range(4):
                        for column in range(4):
                            self.assertAlmostEqual(actual[row][column], expected[row][column],
                                                   places=12)

    def test_name_contains_parameters(self) -> None:
        convention = Convention(rotation=WINNER_AXES, invert_quat=True,
                                camera_offset=(0.02, 0.0, -0.02), perturbation_deg=5.0)
        self.assertIn("+x −y −z", convention.name)
        self.assertIn("invert_quat=True", convention.name)
        self.assertIn("c=(0.02, 0, -0.02)", convention.name)
        self.assertIn("Δ5°", convention.name)


class OffsetGridTest(unittest.TestCase):
    """Сетка смещения из карточки: ±10 см с шагом 2 см."""

    def test_grid_size_and_bounds(self) -> None:
        grid = offset_grid()
        self.assertEqual(len(grid), 11 ** 3)
        self.assertEqual(len(set(grid)), len(grid))
        self.assertIn((0.0, 0.0, 0.0), grid)
        for point in grid:
            for value in point:
                self.assertLessEqual(abs(value), OFFSET_RANGE_M + 1e-12)
                self.assertAlmostEqual(value / OFFSET_STEP_M, round(value / OFFSET_STEP_M),
                                       places=9)

    def test_values_are_clean_floats(self) -> None:
        """0.02, а не 0.020000000000000018: значения попадают в JSON как есть."""
        values = offset_values()
        self.assertEqual(values[0], -OFFSET_RANGE_M)
        self.assertEqual(values[-1], OFFSET_RANGE_M)
        self.assertIn(0.02, values)
        self.assertEqual(len(values), 11)

    def test_broken_grid_rejected(self) -> None:
        with self.assertRaises(ValueError):
            offset_grid(step_m=0.0)
        with self.assertRaises(ValueError):
            offset_grid(limit_m=-0.1)


class EvaluateTest(unittest.TestCase):
    """Метрика эталона 07a на синтетике: своя рамка, чужая, мимо, за камерой, вне кадра."""

    def setUp(self) -> None:
        self.identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        self.matrix = matrix4(self.identity)
        eye = intrinsics()
        self.cx, self.cy = eye.cx, eye.cy

    def _pixel(self, head):
        return project_head(self.matrix, head)

    def test_point_inside_own_box(self) -> None:
        head = point_in_front(self.matrix)
        u, v = self._pixel(head)
        box = (int(u) - 40, int(v) - 40, int(u) + 40, int(v) + 40)
        data = make_data([(head, box, ())])
        metrics = evaluate(data, self.identity)
        self.assertEqual((metrics.good, metrics.wrong, metrics.miss), (1, 0, 0))
        self.assertEqual(metrics.in_frame, 1)
        self.assertEqual(metrics.mean_error_px, 0.0)
        self.assertEqual(metrics.score, 0.0)

    def test_point_in_other_hand_is_wrong(self) -> None:
        head = point_in_front(self.matrix)
        u, v = self._pixel(head)
        other = (int(u) - 20, int(v) - 20, int(u) + 20, int(v) + 20)
        data = make_data([(head, (0, 0, 10, 10), (other,))])
        metrics = evaluate(data, self.identity)
        self.assertEqual(metrics.wrong, 1)
        self.assertEqual(metrics.good, 0)
        self.assertEqual(metrics.violations, 1)
        self.assertGreaterEqual(metrics.score, calib_search.PENALTY_PX)

    def test_point_outside_all_boxes_is_miss(self) -> None:
        head = point_in_front(self.matrix)
        u, v = self._pixel(head)
        box = (int(u) + 200, int(v) + 200, int(u) + 260, int(v) + 260)
        data = make_data([(head, box, ())])
        metrics = evaluate(data, self.identity)
        self.assertEqual(metrics.miss, 1)
        expected = math.hypot(box[0] - u, box[1] - v)
        self.assertAlmostEqual(metrics.mean_error_px, expected, places=6)
        self.assertAlmostEqual(metrics.max_error_px, metrics.mean_error_px, places=6)

    def test_point_behind_camera_not_counted(self) -> None:
        data = make_data([((0.0, 0.0, -0.5), (0, 0, 1080, 810), ())])  # z_cam = −0.5
        metrics = evaluate(data, self.identity)
        self.assertEqual(metrics.in_frame, 0)
        self.assertEqual(metrics.good, 0)
        self.assertTrue(math.isinf(metrics.mean_error_px))
        self.assertEqual(metrics.total, 1)

    def test_point_outside_frame_not_counted(self) -> None:
        data = make_data([((5.0, 0.0, -1.0), (0, 0, 1080, 810), ())])  # u ≫ 1080
        metrics = evaluate(data, self.identity)
        self.assertEqual(metrics.in_frame, 0)
        self.assertTrue(math.isinf(metrics.mean_error_px))

    def test_nan_point_is_skipped(self) -> None:
        data = make_data([
            ((math.nan, 0.0, -0.5), (0, 0, 1080, 810), ()),
            (point_in_front(self.matrix), (0, 0, 1080, 810), ()),
        ])
        metrics = evaluate(data, self.identity)
        self.assertEqual(metrics.in_frame, 1)
        self.assertEqual(metrics.good, 1)

    def test_camera_offset_moves_projection(self) -> None:
        head = point_in_front(self.matrix, x=0.0)
        u, v = self._pixel(head)
        box = (int(u) - 30, int(v) - 30, int(u) + 30, int(v) + 30)
        data = make_data([(head, box, ())])
        shifted = evaluate(data, self.identity, camera_offset=(0.05, 0.0, 0.0))
        self.assertEqual(shifted.good, 0)
        self.assertEqual(shifted.miss, 1)

    def test_broken_rotation_rejected(self) -> None:
        data = make_data([((0.0, 0.0, -0.5), (0, 0, 1080, 810), ())])
        with self.assertRaises(ValueError):
            evaluate(data, WINNER_AXES + ((1.0, 0.0, 0.0),))
        with self.assertRaises(ValueError):
            evaluate(data, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)))
        with self.assertRaises(ValueError):
            evaluate(data, "матрица")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            evaluate("не данные", WINNER_AXES)  # type: ignore[arg-type]


class RankingTest(unittest.TestCase):
    """Ранжирование: сравнимые по покрытию — впереди, внутри — минимальная правка."""

    def _candidate(self, *, offset=(0.0, 0.0, 0.0), in_frame=39, mean=0.0,
                   perturbation=0.0, name="c") -> Candidate:
        convention = Convention(
            rotation=WINNER_AXES, camera_offset=offset, perturbation_deg=perturbation
        )
        metrics = Metrics(good=in_frame, wrong=0, miss=0, in_frame=in_frame, total=45,
                          mean_error_px=mean, max_error_px=mean)
        return Candidate(convention, metrics)

    def test_small_coverage_goes_last(self) -> None:
        full = self._candidate(in_frame=39, mean=0.0, name="full")
        tiny = self._candidate(in_frame=1, mean=0.0, name="tiny")
        ordered = rank([tiny, full])
        self.assertIs(ordered[0], full)
        self.assertEqual(COVERAGE_TOLERANCE, 2)

    def test_minimal_offset_wins_tie(self) -> None:
        zero = self._candidate(offset=(0.0, 0.0, 0.0), name="zero")
        shifted = self._candidate(offset=(0.0, 0.0, 0.02), name="shifted", in_frame=41)
        ordered = rank([shifted, zero])
        self.assertIs(ordered[0], zero)

    def test_minimal_rotation_wins_tie(self) -> None:
        base = self._candidate(perturbation=0.0, name="base", in_frame=39)
        turned = self._candidate(perturbation=5.0, name="turned", in_frame=40)
        ordered = rank([turned, base])
        self.assertIs(ordered[0], base)

    def test_metric_beats_everything(self) -> None:
        bad = self._candidate(mean=120.0, name="bad", in_frame=41)
        good = self._candidate(mean=0.0, name="good", in_frame=39)
        self.assertIs(rank([bad, good])[0], good)

    def test_empty_rank(self) -> None:
        self.assertEqual(rank([]), ())

    def test_duplicates_collapsed(self) -> None:
        """Перебор и уточнение могут вернуть один вариант — в таблице он один."""
        first = self._candidate(name="same")
        second = self._candidate(name="same")
        self.assertEqual(len(rank([first, second])), 1)


class SearchRecoveryTest(unittest.TestCase):
    """Перебор возвращает конвенцию, которой была сгенерирована синтетическая сцена."""

    def _scene(self, offset=(0.0, 0.0, 0.0), box_px: int = 25):
        matrix = matrix4(WINNER_AXES, offset)
        points = []
        for x, y in itertools.product((-0.12, 0.0, 0.12), repeat=2):
            head = point_in_front(matrix, x=x, y=y)
            u, v = project_head(matrix, head)
            box = (int(round(u)) - box_px, int(round(v)) - box_px,
                   int(round(u)) + box_px, int(round(v)) + box_px)
            points.append((head, box, ()))
        return make_data(points)

    def test_recovers_known_convention(self) -> None:
        data = self._scene()
        candidates = search_discrete(data, offset_grid(step_m=0.02, limit_m=0.04))
        winner = candidates[0]
        self.assertEqual(winner.convention.rotation, WINNER_AXES)
        self.assertFalse(winner.convention.invert_quat)
        self.assertEqual(winner.convention.camera_offset, (0.0, 0.0, 0.0))
        self.assertEqual(winner.metrics.mean_error_px, 0.0)
        self.assertEqual(winner.metrics.violations, 0)
        self.assertEqual(winner.metrics.good, 9)

    def test_recovers_offset_within_grid_step(self) -> None:
        data = self._scene(offset=(0.02, 0.0, -0.02), box_px=8)
        candidates = search_discrete(data, offset_grid(step_m=0.02, limit_m=0.06))
        winner = candidates[0]
        self.assertEqual(winner.convention.rotation, WINNER_AXES)
        self.assertEqual(winner.metrics.mean_error_px, 0.0)
        for actual, expected in zip(winner.convention.camera_offset, (0.02, 0.0, -0.02)):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_refine_does_not_invent_rotation(self) -> None:
        """Сетка уточнения не должна «улучшать» метрику за счёт лишнего поворота."""
        data = self._scene()
        candidates = search_discrete(data, offset_grid(step_m=0.02, limit_m=0.04))
        refined, summary = refine(data, candidates[:1], step_deg=5.0, angle_limit_deg=10.0,
                                  step_m=0.02, offset_limit_m=0.04)
        self.assertEqual(refined[0].convention.rotation, WINNER_AXES)
        self.assertEqual(refined[0].convention.camera_offset, (0.0, 0.0, 0.0))
        self.assertEqual(refined[0].convention.perturbation_deg, 0.0)
        self.assertIn("rotation_cells", summary)
        self.assertIn("offset_cells", summary)


class SensitivityTest(unittest.TestCase):
    """Плато чувствительности: при огромной рамке нулевая ошибка у всех параметров."""

    def test_huge_box_means_full_plateau(self) -> None:
        matrix = matrix4(WINNER_AXES)
        head = point_in_front(matrix)
        data = make_data([(head, (0, 0, EYE_W, EYE_H), ())])
        convention = Convention(rotation=WINNER_AXES)
        plateau = sensitivity(data, convention, step_deg=5.0, angle_limit_deg=20.0,
                              step_m=0.02, offset_limit_m=0.10)
        self.assertEqual(set(plateau), {"offset_m", "angle_deg"})
        for axis in "xyz":
            self.assertEqual(plateau["offset_m"][axis], OFFSET_RANGE_M)
        for angle in ("yaw", "pitch", "roll"):
            self.assertEqual(plateau["angle_deg"][angle], ANGLE_RANGE_DEG)

    def test_tight_box_has_empty_plateau(self) -> None:
        matrix = matrix4(WINNER_AXES)
        # Точка вне оптической оси: сдвиг вдоль луча и roll вокруг него её двигают.
        head = point_in_front(matrix, x=0.3, y=0.2)
        u, v = project_head(matrix, head)
        box = (int(u) - 2, int(v) - 2, int(u) + 2, int(v) + 2)
        data = make_data([(head, box, ())])
        plateau = sensitivity(data, Convention(rotation=WINNER_AXES))
        for axis in "xyz":
            self.assertEqual(plateau["offset_m"][axis], 0.0)
        for angle in ("yaw", "pitch", "roll"):
            self.assertEqual(plateau["angle_deg"][angle], 0.0)


class PayloadTest(unittest.TestCase):
    """JSON для 07c: параметры, метрика и признак выполнения критерия."""

    def test_payload_contains_config_values(self) -> None:
        convention = Convention(rotation=WINNER_AXES)
        metrics = Metrics(39, 0, 0, 39, 45, 0.0, 0.0)
        winner = Candidate(convention, metrics)
        payload = winner_payload(winner, {"offset_m": {"x": 0.0}}, [winner], {"changed": 0})
        text = json.dumps(payload, ensure_ascii=False)
        self.assertIn("07b", text)
        body = json.loads(text)
        self.assertEqual(body["winner"]["head_cam_rotation_rpy_deg"], [0.0, 180.0, 0.0])
        self.assertEqual(body["winner"]["head_cam_translation_m"], [0.0, 0.0, 0.0])
        self.assertFalse(body["winner"]["invert_quat"])
        self.assertFalse(body["winner"]["opencv_conventions"])
        self.assertTrue(body["winner"]["criterion_met"])
        self.assertEqual(body["max_mean_error_px"], MAX_MEAN_ERROR_PX)
        self.assertEqual(body["top"][0]["axes"], "+x −y −z")

    def test_table_marks_unsupported_rows(self) -> None:
        full = Candidate(Convention(rotation=WINNER_AXES), Metrics(39, 0, 0, 39, 45, 0.0, 0.0))
        tiny = Candidate(Convention(rotation=WINNER_AXES), Metrics(1, 0, 0, 1, 45, 0.0, 0.0))
        table = format_table([full, tiny])
        self.assertIn("39/45", table)
        self.assertIn("1/45", table)
        self.assertIn("*", table)


class RealDataTest(unittest.TestCase):
    """Проверка на реальных данных (без них тест пропускается)."""

    @classmethod
    def setUpClass(cls) -> None:
        if DATA is None:
            raise unittest.SkipTest(NO_DATA)
        if not META.is_file():
            raise unittest.SkipTest(NO_META)
        # 07b искался без поправки на задержку видеотракта (она появилась в доработке 08a),
        # поэтому здесь фиксируется offset_ns=0 — иначе числа 39/39 и плато поедут
        cls.data = load_calib_data(DATA, META, offset_ns=0)

    def test_frame_set_and_point_count(self) -> None:
        self.assertEqual(len(self.data.frames), 5)
        self.assertEqual(self.data.total, 45)
        self.assertEqual(self.data.intrinsics.cx, 539.5)

    def test_winner_metrics(self) -> None:
        metrics = evaluate(self.data, WINNER_AXES, invert_quat=False, camera_offset=(0, 0, 0))
        self.assertEqual(metrics.good, 39)
        self.assertEqual(metrics.in_frame, 39)
        self.assertEqual(metrics.wrong, 0)
        self.assertEqual(metrics.miss, 0)
        self.assertEqual(metrics.mean_error_px, 0.0)
        self.assertLessEqual(metrics.mean_error_px, MAX_MEAN_ERROR_PX)

    def test_previous_convention_sees_nothing(self) -> None:
        """Старая (05b) конвенция без поворота камеры: кисти оказываются за камерой.

        Поворот ``F = diag(1, −1, 1)`` переворачивал только ``y``, а ``z`` головы (назад)
        оставался «вперёд» камеры — поэтому 05b и показывала 0 % точек перед камерой.
        В метрике 07b это значит «ни один сустав не попал в кадр», а не «попал мимо».
        """
        metrics = evaluate(self.data, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        self.assertEqual(metrics.in_frame, 0)
        self.assertEqual(metrics.violations, 0)

    def test_search_recovers_winner(self) -> None:
        """Полный перебор (~5 с) возвращает ту же конвенцию, что записана в отчёт."""
        candidates = search_discrete(self.data)
        winner = candidates[0]
        self.assertEqual(winner.convention.rotation, WINNER_AXES)
        self.assertFalse(winner.convention.invert_quat)
        self.assertEqual(winner.convention.camera_offset, (0.0, 0.0, 0.0))
        self.assertEqual(winner.metrics.mean_error_px, 0.0)
        self.assertEqual(winner.metrics.violations, 0)

    def test_refine_keeps_minimal_model(self) -> None:
        winner = search_discrete(self.data)[0]
        refined, _ = refine(self.data, [winner])
        self.assertEqual(refined[0].convention.rotation, WINNER_AXES)
        self.assertEqual(refined[0].convention.camera_offset, (0.0, 0.0, 0.0))
        self.assertEqual(refined[0].convention.perturbation_deg, 0.0)

    def test_plateau_matches_the_report(self) -> None:
        """Данные уточняют смещение не лучше шага сетки: x и y — ровно 0, z — ±2 см."""
        plateau = sensitivity(self.data, Convention(rotation=WINNER_AXES))
        self.assertEqual(plateau["offset_m"]["x"], 0.0)
        self.assertEqual(plateau["offset_m"]["y"], 0.0)
        self.assertEqual(plateau["offset_m"]["z"], OFFSET_STEP_M)
        self.assertEqual(plateau["angle_deg"]["yaw"], 0.0)

    def test_matrix_matches_camera_model_on_real_pose(self) -> None:
        from src import sync
        from src.parse_tracking import read_header, timestamp_range

        meta = json.loads(META.read_text(encoding="utf-8"))
        header = read_header(DATA)
        track = sync.load_poses(DATA)
        end = timestamp_range(DATA)[1]
        for frame in self.data.frames:
            moment = sync.frame_time(frame.index, meta, int(header["timeStampNs"]), end)
            pose = track.pose_at(moment.ts_ns)
            convention = Convention(rotation=WINNER_AXES)
            expected = world_to_camera_matrix(
                (pose.head_pos[0], pose.head_pos[1], pose.head_pos[2],
                 pose.head_quat[0], pose.head_quat[1], pose.head_quat[2], pose.head_quat[3]),
                t_head_cam=convention.head_cam(),
                opencv_conventions=False,
                invert_quat=False,
            )
            actual = convention.matrix_for(pose.head_pos, pose.head_quat)
            for row in range(4):
                for column in range(4):
                    self.assertAlmostEqual(actual[row][column], expected[row][column], places=12)


if __name__ == "__main__":
    unittest.main()
