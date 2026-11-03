"""Assert-тесты для задачи 08e: костная разметка (calib_frames) и метрика (calib_bones).

Запуск: python -m unittest discover -s tests -t . -v

Синтетические тесты не зависят от данных: ``evaluate_bones`` принимает проектор снаружи,
поэтому вместо камеры подставляется тождественное отображение «система головы = пиксели».
Тесты на реальных файлах пропускаются, если нет ``data/trackingData_*.txt`` /
``output/video_meta.json`` (не коммитятся).

Что проверяется:

* разметка: 7–9 точек на руку, имена суставов только из ``HAND_JOINTS_FULL``, точки внутри
  кадра, цепочки покрывают все точки, сторона обязательна; кадры 08e согласованы с рамками
  07a (кадр есть, рука не в ``ignore``, все 5 кадров покрыты);
* метрика: точное совпадение проекции с разметкой даёт 0 px, сдвиг — ровно сдвиг,
  поворот даёт излом, точка за камерой не считается, ``min_confidence`` отбрасывает
  отрезки с ``low``-точками;
* геометрия: расстояние до отрезка (внутри/за концом/перпендикуляр), угол между костями
  (0/90/вырожденный отрезок);
* ``sync``: дополнительные суставы (``extra_keys``/``joint_at``) интерполируются по тем же
  правилам, что поза (край, точное попадание, NaN), и не попадают в ``Pose`` (5-точечный
  путь не меняется);
* на реальных данных: 6 кистей / 54 точки / 48 отрезков, и позиции суставов из
  ``load_bone_data`` совпадают с путём ``sync.pose_at_frame``.
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import calib_bones, calib_frames, sync  # noqa: E402
from src.calib_bones import (  # noqa: E402
    EXTRA_KEYS,
    BoneData,
    BoneMetrics,
    BonePoint,
    bone_angle_deg,
    evaluate_bones,
    load_bone_data,
    point_segment_distance,
    sample_segment,
)
from src.calib_frames import (  # noqa: E402
    BONE_CHAINS,
    BONE_FRAMES,
    BONE_KINK_ANGLE_DEG,
    BONE_POINTS_PER_HAND,
    BONE_TOLERANCE_PX,
    CALIB_FRAMES,
    BoneFrame,
    HandMarks,
    JointMark,
    bones_as_json,
    validate_bone_frames,
)
from src.camera_model import Intrinsics  # noqa: E402

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
META = ROOT / "output" / "video_meta.json"
NO_DATA = "нет data/trackingData_*.txt — файл данных не коммитится (см. README)"
NO_META = "нет output/video_meta.json — он создаётся задачей 02 (см. README)"

RAW_INTRINSICS = [1079.5, 404.5, 1373.6681390725917, 686.8680648828482]
MS = 1_000_000


def intrinsics() -> Intrinsics:
    """Per-eye интринсики левого глаза (файл данных не нужен)."""
    return Intrinsics.from_header({"cameraIntrinsics": RAW_INTRINSICS}).per_eye("left")


def marks_points() -> dict[str, tuple[int, int]]:
    """Правдоподобная разметка кисти: 9 точек, все цепочки покрыты."""
    return {
        "wrist": (500, 400),
        "palm": (480, 380),
        "index_metacarpal": (470, 370),
        "index_proximal": (460, 350),
        "index_intermediate": (450, 330),
        "index": (445, 310),
        "middle_proximal": (475, 345),
        "middle_intermediate": (470, 325),
        "middle": (468, 305),
    }


def make_hand(points: dict[str, tuple[int, int]] | None = None,
              low: tuple[str, ...] = ()) -> HandMarks:
    """``HandMarks`` с цепочками 08e из готовых пикселей."""
    source = points if points is not None else marks_points()
    return HandMarks(
        hand="left",
        chains=BONE_CHAINS,
        marks=tuple(
            JointMark(name, source[name], "low" if name in low else "high")
            for name in source
        ),
    )


def make_bone_data(points: dict[str, tuple[int, int]] | None = None,
                   low: tuple[str, ...] = (), index: int = 1800) -> BoneData:
    """``BoneData`` на один кадр/руку: ``head`` = пиксели, проектор — тождество."""
    hand = make_hand(points, low)
    frame = BoneFrame(index=index, hands=(hand,))
    source = hand.points
    return BoneData(
        intrinsics=intrinsics(),
        frames=(frame,),
        points=tuple(
            BonePoint(
                frame_index=index,
                side="left",
                joint=name,
                head=(float(uv[0]), float(uv[1]), 0.0),
                head_inverted=(float(uv[0]), float(uv[1]), 0.0),
            )
            for name, uv in source.items()
        ),
    )


def identity_project(point) -> tuple[float, float]:
    """Проектор тождественный: ``head`` уже в пикселях."""
    return float(point[0]), float(point[1])


def shift_project(dx: float, dy: float):
    """Проектор со сдвигом проекции на ``(dx, dy)`` пикселей."""

    def project(point) -> tuple[float, float]:
        return float(point[0]) + dx, float(point[1]) + dy

    return project


class JointMarkTest(unittest.TestCase):
    """Точка разметки: имя сустава из HAND_JOINTS_FULL, пиксели в кадре, уверенность."""

    def test_valid_mark(self) -> None:
        mark = JointMark("index_proximal", (10, 20))
        self.assertEqual(mark.joint, "index_proximal")
        self.assertEqual(mark.point, (10, 20))
        self.assertEqual(mark.confidence, "high")

    def test_rejects_unknown_joint(self) -> None:
        with self.assertRaises(ValueError):
            JointMark("index_proximal_2", (10, 20))
        with self.assertRaises(ValueError):
            JointMark("", (10, 20))
        with self.assertRaises(ValueError):
            JointMark("Palm", (10, 20))  # type: ignore[arg-type]

    def test_rejects_point_out_of_frame(self) -> None:
        bad = ((-1, 10), (10, -1))
        bad += ((calib_frames.EYE_WIDTH, 10), (10, calib_frames.EYE_HEIGHT))
        for point in bad:
            with self.assertRaises(ValueError, msg=point):
                JointMark("palm", point)

    def test_rejects_bad_point(self) -> None:
        with self.assertRaises(ValueError):
            JointMark("palm", (1.5, 2))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            JointMark("palm", (True, 2))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            JointMark("palm", (1, 2, 3))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            JointMark("palm", [1, 2])  # type: ignore[arg-type]

    def test_rejects_bad_confidence(self) -> None:
        with self.assertRaises(ValueError):
            JointMark("palm", (1, 2), "sure")
        with self.assertRaises(ValueError):
            JointMark("palm", (1, 2), None)  # type: ignore[arg-type]


class HandMarksTest(unittest.TestCase):
    """Кисть: 7–9 точек, сторона обязательна, цепочки покрывают все точки."""

    def test_valid_hand(self) -> None:
        hand = make_hand()
        self.assertEqual(len(hand.marks), 9)
        self.assertEqual(len(hand.bones), 8)
        self.assertEqual(hand.points["palm"], (480, 380))
        self.assertEqual(hand.mark_for("index").joint, "index")

    def test_unknown_side_rejected(self) -> None:
        """Рамка 07a может быть ``unknown``, кости — нет: спроецировать нечего."""
        with self.assertRaises(ValueError):
            HandMarks(hand="unknown", chains=BONE_CHAINS, marks=make_hand().marks)

    def test_point_count_bounds(self) -> None:
        low, high = BONE_POINTS_PER_HAND
        self.assertEqual((low, high), (7, 9))
        only_chain = tuple(name for name in ("wrist", "palm") if name in marks_points())
        del only_chain
        with self.assertRaises(ValueError):
            # 6 точек — меньше нижней границы
            HandMarks(
                hand="left",
                chains=(("palm", "index"),),
                marks=(
                    JointMark("palm", (1, 1)),
                    JointMark("wrist", (2, 2)),
                    JointMark("index", (3, 3)),
                    JointMark("middle", (4, 4)),
                    JointMark("thumb", (5, 5)),
                    JointMark("ring", (6, 6)),
                ),
            )

    def test_duplicate_joint_rejected(self) -> None:
        points = marks_points()
        marks = tuple(
            JointMark(name, uv, "high") for name, uv in points.items()
        ) + (JointMark("palm", (100, 100), "high"),)
        with self.assertRaises(ValueError):
            HandMarks(hand="left", chains=BONE_CHAINS, marks=marks)

    def test_chain_with_unknown_joint_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HandMarks(
                hand="left",
                chains=BONE_CHAINS + (("palm", "thumb"),),
                marks=make_hand().marks,
            )

    def test_chain_shorter_than_two_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HandMarks(hand="left", chains=(("palm",),), marks=make_hand().marks)

    def test_point_outside_chains_rejected(self) -> None:
        """Точка, не входящая ни в одну цепочку, ничего не проверяет — это ошибка разметки."""
        points = dict(marks_points())
        points["thumb"] = (600, 200)
        with self.assertRaises(ValueError):
            HandMarks(hand="left", chains=BONE_CHAINS, marks=make_hand(points).marks)

    def test_as_dict_round_trip(self) -> None:
        hand = make_hand()
        payload = hand.as_dict()
        self.assertEqual(payload["hand"], "left")
        self.assertEqual(payload["chains"][0], list(BONE_CHAINS[0]))
        self.assertEqual(len(payload["marks"]), len(hand.marks))
        self.assertEqual(tuple(payload["marks"][0]["point"]), hand.marks[0].point)


class BoneFrameTest(unittest.TestCase):
    """Кадр костной разметки: номер, хотя бы одна кисть, стороны не повторяются."""

    def test_valid_frame(self) -> None:
        frame = BoneFrame(index=1800, hands=(make_hand(),))
        self.assertEqual(frame.bones, 8)
        self.assertIs(frame.hand_for("left"), frame.hands[0])
        self.assertIsNone(frame.hand_for("right"))

    def test_rejects_bad_index(self) -> None:
        with self.assertRaises(ValueError):
            BoneFrame(index=-1, hands=(make_hand(),))
        with self.assertRaises(ValueError):
            BoneFrame(index=True, hands=(make_hand(),))  # type: ignore[arg-type]

    def test_rejects_empty_hands(self) -> None:
        with self.assertRaises(ValueError):
            BoneFrame(index=0, hands=())

    def test_rejects_duplicate_side(self) -> None:
        with self.assertRaises(ValueError):
            BoneFrame(index=0, hands=(make_hand(), make_hand()))


class ShippedBoneReferenceTest(unittest.TestCase):
    """Разметка 08e, лежащая в репозитории: 6 кистей на 5 кадрах, 48 отрезков."""

    def test_validates_against_07a(self) -> None:
        validate_bone_frames()

    def test_shape(self) -> None:
        self.assertEqual(len(BONE_FRAMES), len(CALIB_FRAMES))
        self.assertEqual([frame.index for frame in BONE_FRAMES],
                         [frame.index for frame in CALIB_FRAMES])
        hands = [hand for frame in BONE_FRAMES for hand in frame.hands]
        self.assertEqual(len(hands), 6)
        self.assertEqual(sum(frame.bones for frame in BONE_FRAMES), 48)
        for hand in hands:
            low, high = BONE_POINTS_PER_HAND
            self.assertTrue(low <= len(hand.marks) <= high, hand.hand)
            self.assertEqual(len(hand.marks), 9)
            self.assertEqual(len(hand.bones), 8)

    def test_all_marks_inside_frame(self) -> None:
        for frame in BONE_FRAMES:
            for hand in frame.hands:
                for mark in hand.marks:
                    self.assertTrue(calib_frames.in_eye_frame(*mark.point), mark)

    def test_hands_are_visible_in_07a(self) -> None:
        """Каждая размеченная рука в 07a существует и не помечена как отсутствующая."""
        for frame in BONE_FRAMES:
            reference = next(f for f in CALIB_FRAMES if f.index == frame.index)
            for hand in frame.hands:
                self.assertNotIn(hand.hand, reference.ignore)
                box = reference.box_for(hand.hand) or reference.box_for("unknown")
                self.assertIsNotNone(box)

    def test_k30_is_low_confidence_and_unknown_box(self) -> None:
        """k=30: рамка 07a — ``unknown``, кости — left с уверенностью low (так и записано)."""
        frame = next(f for f in BONE_FRAMES if f.index == 30)
        hand = frame.hand_for("left")
        self.assertIsNotNone(hand)
        self.assertTrue(all(mark.confidence == "low" for mark in hand.marks
                            if mark.joint not in ("wrist", "palm")))
        reference = next(f for f in CALIB_FRAMES if f.index == 30)
        self.assertIsNotNone(reference.box_for("unknown"))

    def test_json(self) -> None:
        payload = json.loads(bones_as_json())
        self.assertEqual(payload["eye_size"], [calib_frames.EYE_WIDTH, calib_frames.EYE_HEIGHT])
        self.assertEqual(payload["tolerance_px"], BONE_TOLERANCE_PX)
        self.assertEqual(payload["kink_angle_deg"], BONE_KINK_ANGLE_DEG)
        self.assertEqual(payload["points_per_hand"], list(BONE_POINTS_PER_HAND))
        self.assertEqual(len(payload["frames"]), len(BONE_FRAMES))
        self.assertEqual(sum(len(hand["marks"]) for frame in payload["frames"]
                             for hand in frame["hands"]), 54)

    def test_validation_catches_missing_frame(self) -> None:
        with self.assertRaises(ValueError):
            validate_bone_frames(BONE_FRAMES[1:])
        with self.assertRaises(ValueError):
            validate_bone_frames((BoneFrame(index=999, hands=(make_hand(),)),))

    def test_validation_catches_ignored_hand(self) -> None:
        frame = next(f for f in CALIB_FRAMES if f.index == 30)
        self.assertIn("right", frame.ignore)
        bad = BoneFrame(index=30, hands=(
            HandMarks(hand="right", chains=BONE_CHAINS, marks=make_hand().marks),
        ))
        with self.assertRaises(ValueError):
            validate_bone_frames((bad,))


class GeometryTest(unittest.TestCase):
    """Расстояние до отрезка, сэмплирование и угол между костями."""

    def test_distance_inside_segment(self) -> None:
        self.assertEqual(point_segment_distance((5, 5), (0, 0), (10, 0)), 5.0)
        self.assertEqual(point_segment_distance((5, 0), (0, 0), (10, 0)), 0.0)

    def test_distance_beyond_end(self) -> None:
        self.assertAlmostEqual(point_segment_distance((13, 4), (0, 0), (10, 0)), 5.0)
        self.assertAlmostEqual(point_segment_distance((-3, 4), (0, 0), (10, 0)), 5.0)

    def test_degenerate_segment(self) -> None:
        self.assertAlmostEqual(point_segment_distance((3, 4), (0, 0), (0, 0)), 5.0)

    def test_sample_segment(self) -> None:
        points = sample_segment((0.0, 0.0), (10.0, 0.0), 5)
        self.assertEqual(len(points), 5)
        self.assertEqual(points[0], (0.0, 0.0))
        self.assertEqual(points[-1], (10.0, 0.0))
        self.assertAlmostEqual(points[2][0], 5.0)
        with self.assertRaises(ValueError):
            sample_segment((0, 0), (1, 1), 1)
        with self.assertRaises(ValueError):
            sample_segment((0, 0), (1, 1), True)  # type: ignore[arg-type]

    def test_angle(self) -> None:
        self.assertEqual(bone_angle_deg((0, 0), (10, 0), (0, 0), (10, 0)), 0.0)
        self.assertAlmostEqual(bone_angle_deg((0, 0), (10, 0), (0, 0), (0, 10)), 90.0)
        self.assertAlmostEqual(bone_angle_deg((0, 0), (10, 0), (0, 0), (-10, 0)), 0.0)
        self.assertAlmostEqual(bone_angle_deg((0, 0), (10, 0), (0, 0), (10, 10)), 45.0)

    def test_angle_of_degenerate_bone_is_zero(self) -> None:
        self.assertEqual(bone_angle_deg((0, 0), (1, 0), (0, 0), (10, 0)), 0.0)
        self.assertEqual(bone_angle_deg((0, 0), (10, 0), (0, 0), (1, 0)), 0.0)


class EvaluateBonesTest(unittest.TestCase):
    """Метрика: точная проекция → 0, сдвиг → ровно сдвиг, поворот → излом."""

    def test_perfect_projection_scores_zero(self) -> None:
        data = make_bone_data()
        metrics = evaluate_bones(data, identity_project)
        self.assertIsInstance(metrics, BoneMetrics)
        self.assertEqual(metrics.bones, 8)
        self.assertEqual(metrics.in_frame, 8)
        self.assertEqual(metrics.good, 8)
        self.assertEqual(metrics.mean_error_px, 0.0)
        self.assertEqual(metrics.max_error_px, 0.0)
        self.assertEqual(metrics.kinks, 0)
        self.assertEqual(metrics.violations, 0)
        self.assertEqual(metrics.joints, 9)
        self.assertEqual(metrics.hands, 1)

    def test_shift_inside_tolerance_is_good(self) -> None:
        """Сдвиг на 20 px: расстояние до своего отрезка не больше сдвига (часть — вдоль кости)."""
        data = make_bone_data()
        metrics = evaluate_bones(data, shift_project(20.0, 0.0))
        self.assertEqual(metrics.in_frame, 8)
        self.assertEqual(metrics.good, 8)
        self.assertGreater(metrics.mean_error_px, 0.0)
        self.assertLessEqual(metrics.max_error_px, 20.0 + 1e-9)
        self.assertEqual(metrics.kinks, 0)

    def test_shift_beyond_tolerance_is_violation(self) -> None:
        data = make_bone_data()
        metrics = evaluate_bones(data, shift_project(120.0, 0.0))
        self.assertEqual(metrics.in_frame, 8)
        self.assertEqual(metrics.good, 0)
        self.assertEqual(metrics.violations, 8)
        self.assertGreater(metrics.mean_error_px, 100.0)
        self.assertLessEqual(metrics.max_error_px, 120.0 + 1e-9)

    def test_rotation_is_a_kink(self) -> None:
        """Сдвиг не меняет направление, поворот — меняет: считаем изломы отдельно."""
        data = make_bone_data()
        centre = (470.0, 350.0)

        def rotated(point) -> tuple[float, float]:
            dx = float(point[0]) - centre[0]
            dy = float(point[1]) - centre[1]
            return centre[0] - dy, centre[1] + dx  # поворот на 90°

        metrics = evaluate_bones(data, rotated)
        self.assertEqual(metrics.kinks, 8)
        self.assertAlmostEqual(metrics.max_angle_deg, 90.0, places=6)

    def test_behind_camera_not_counted(self) -> None:
        data = make_bone_data()
        metrics = evaluate_bones(data, lambda point: None)
        self.assertEqual(metrics.in_frame, 0)
        self.assertEqual(metrics.good, 0)
        self.assertTrue(math.isinf(metrics.mean_error_px))
        self.assertEqual(metrics.bones, 8)

    def test_min_confidence_filters_low_marks(self) -> None:
        """k=30-подобная разметка: low-точки выкидывают свои отрезки, но не все."""
        data = make_bone_data(low=("index_metacarpal", "index_proximal"))
        full = evaluate_bones(data, identity_project)
        strict = evaluate_bones(data, identity_project, min_confidence="medium")
        self.assertEqual(full.bones, 8)
        # отброшены palm→index_metacarpal, index_metacarpal→index_proximal и
        # index_proximal→index_intermediate (у каждой кости один конец помечен low)
        self.assertEqual(strict.bones, 5)
        self.assertEqual(strict.in_frame, 5)
        self.assertEqual(strict.joints, 7)

    def test_rejects_bad_inputs(self) -> None:
        data = make_bone_data()
        with self.assertRaises(ValueError):
            evaluate_bones("не данные", identity_project)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            evaluate_bones(data, "не проектор")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            evaluate_bones(data, identity_project, tolerance_px=0)
        with self.assertRaises(ValueError):
            evaluate_bones(data, identity_project, kink_angle_deg=-5)
        with self.assertRaises(ValueError):
            evaluate_bones(data, identity_project, min_confidence="sure")

    def test_as_dict_and_format(self) -> None:
        data = make_bone_data()
        metrics = evaluate_bones(data, shift_project(20.0, 0.0))
        payload = metrics.as_dict()
        self.assertEqual(payload["bones"], 8)
        self.assertEqual(payload["in_frame"], 8)
        self.assertEqual(payload["good"], 8)
        lines = metrics.format_lines()
        self.assertIn("8/8", lines[0])
        self.assertIn("изломы", lines[1])

    def test_format_of_empty_projection(self) -> None:
        metrics = evaluate_bones(make_bone_data(), lambda point: None)
        lines = metrics.format_lines()
        self.assertEqual(len(lines), 1)
        self.assertIn("не попало ни одного", lines[0])


class ExtraJointsTrackTest(unittest.TestCase):
    """Дополнительные суставы в ``PoseTrack`` (08e): чтение, интерполяция, ошибки."""

    def _track(self, **kwargs) -> sync.PoseTrack:
        ts = [0, 100 * MS, 200 * MS]
        values = {name: [0.0] * 3 for name in (*sync.POSITION_FIELDS, *sync.QUAT_FIELDS)}
        values["head_x"] = [0.0, 0.1, 0.2]
        values["head_qw"] = [1.0, 1.0, 1.0]
        for axis, base in zip(("x", "y", "z"), (1.0, 2.0, 3.0)):
            values[f"left_index_proximal_{axis}"] = [base, base + 0.1, base + 0.2]
        return sync.PoseTrack(
            ts, values, {"left": [True] * 3, "right": [True] * 3}, **kwargs
        )

    def test_default_has_no_extra_keys(self) -> None:
        track = self._track()
        self.assertEqual(track.extra_keys, ())
        with self.assertRaises(ValueError):
            track.joint_at(0, "left_index_proximal")

    def test_joint_at_interpolates_like_pose(self) -> None:
        track = self._track(extra_keys=("left_index_proximal",))
        self.assertEqual(track.extra_keys, ("left_index_proximal",))
        self.assertEqual(track.joint_at(0, "left_index_proximal"), (1.0, 2.0, 3.0))
        self.assertEqual(track.joint_at(100 * MS, "left_index_proximal")[0], 1.1)
        middle = track.joint_at(50 * MS, "left_index_proximal")
        self.assertAlmostEqual(middle[0], 1.05)
        self.assertAlmostEqual(middle[1], 2.05)
        self.assertAlmostEqual(middle[2], 3.05)

    def test_joint_at_freezes_at_edges(self) -> None:
        track = self._track(extra_keys=("left_index_proximal",))
        self.assertEqual(track.joint_at(-5_000, "left_index_proximal"), (1.0, 2.0, 3.0))
        self.assertEqual(track.joint_at(999 * MS, "left_index_proximal"), (1.2, 2.2, 3.2))

    def test_joint_at_propagates_nan(self) -> None:
        track = self._track(extra_keys=("left_index_proximal",))
        track._values["left_index_proximal_y"] = (2.0, math.nan, 2.2)
        self.assertTrue(math.isnan(track.joint_at(50 * MS, "left_index_proximal")[1]))

    def test_extra_keys_validation(self) -> None:
        with self.assertRaises(ValueError):
            self._track(extra_keys=("left_palm",))  # уже среди 5 точек
        with self.assertRaises(ValueError):
            self._track(extra_keys=("left_index_proximal", "left_index_proximal"))
        with self.assertRaises(ValueError):
            self._track(extra_keys="left_index_proximal")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self._track(extra_keys=("",))
        with self.assertRaises(ValueError):
            sync.PoseTrack(
                [0], {"head_x": [0.0]}, {"left": [True], "right": [True]},
                extra_keys=("left_index_proximal",),
            )

    def test_pose_does_not_include_extras(self) -> None:
        """5-точечный путь не меняется: в Pose только POINT_KEYS."""
        track = self._track(extra_keys=("left_index_proximal",))
        pose = track.pose_at(50 * MS)
        self.assertEqual(set(pose.hands), set(sync.POINT_KEYS))

    def test_pose_time_ns(self) -> None:
        frame = sync.FrameTime(index=0, ts_ns=1_000, t_rel_s=0.0, in_tracking=True)
        self.assertEqual(sync.pose_time_ns(frame, 0), 1_000)
        self.assertEqual(sync.pose_time_ns(frame, -100_000_000), 1_000 - 100_000_000)
        with self.assertRaises(ValueError):
            sync.pose_time_ns(frame, True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            sync.pose_time_ns(frame, 1.5)  # type: ignore[arg-type]


class HeadProjectorTest(unittest.TestCase):
    """Проектор костной метрики: та же математика, что в ``evaluate``."""

    IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def _projector(self, offset=(0.0, 0.0, 0.0)):
        from src import calib_search

        return calib_search.head_projector(intrinsics(), self.IDENTITY, offset)

    def test_centre_of_frame(self) -> None:
        project = self._projector()
        u, v = project((0.0, 0.0, 0.5))
        self.assertAlmostEqual(u, intrinsics().cx)
        self.assertAlmostEqual(v, intrinsics().cy)

    def test_behind_camera_is_none(self) -> None:
        self.assertIsNone(self._projector()((0.0, 0.0, -0.5)))

    def test_outside_frame_is_none(self) -> None:
        self.assertIsNone(self._projector()((5.0, 0.0, 0.5)))

    def test_offset_moves_projection(self) -> None:
        import math

        project = self._projector((0.05, 0.0, 0.0))
        u, _ = project((0.0, 0.0, 0.5))
        shift = abs(intrinsics().fx * 0.05 / 0.5)
        self.assertTrue(math.isclose(intrinsics().cx - shift, u, abs_tol=1e-6)
                        or math.isclose(intrinsics().cx + shift, u, abs_tol=1e-6))


@unittest.skipIf(DATA is None, NO_DATA)
class RealBoneDataTest(unittest.TestCase):
    """Реальные данные: 6 кистей / 54 точки / 48 отрезков и тот же путь времени, что у позы."""

    @classmethod
    def setUpClass(cls) -> None:
        if DATA is None:
            raise unittest.SkipTest(NO_DATA)
        if not META.is_file():
            raise unittest.SkipTest(NO_META)
        cls.data = load_bone_data(DATA, META, offset_ns=0)

    def test_counts(self) -> None:
        self.assertEqual(len(self.data.frames), 5)
        self.assertEqual(self.data.hands, 6)
        self.assertEqual(self.data.total, 54)
        self.assertEqual(self.data.bones, 48)
        self.assertEqual(self.data.intrinsics.cx, 539.5)

    def test_points_are_finite(self) -> None:
        for point in self.data.points:
            self.assertTrue(all(math.isfinite(value) for value in point.head), point)
            self.assertTrue(all(math.isfinite(value) for value in point.head_inverted), point)

    def test_head_frame_matches_pose_path(self) -> None:
        """``load_bone_data`` берёт суставы в позе кадра: сверка с ``sync.pose_at_frame``."""
        from src.camera_model import quat_to_rotation

        track = sync.load_poses(DATA, extra_keys=EXTRA_KEYS)
        meta = json.loads(META.read_text(encoding="utf-8"))
        from src.parse_tracking import read_header, timestamp_range

        header = read_header(DATA)
        end = timestamp_range(DATA)[1]
        for frame in self.data.frames:
            moment = sync.frame_time(frame.index, meta, int(header["timeStampNs"]), end)
            pose = sync.pose_at_frame(track, moment, offset_ns=0)
            t_pose = sync.pose_time_ns(moment, 0)
            rotation = quat_to_rotation(pose.head_quat)
            inverse = tuple(tuple(rotation[j][i] for j in range(3)) for i in range(3))
            for point in self.data.points:
                if point.frame_index != frame.index:
                    continue
                key = f"{point.side}_{point.joint}"
                world = (
                    pose.hands[key]
                    if key in sync.POINT_KEYS
                    else track.joint_at(t_pose, key)
                )
                delta = tuple(world[i] - pose.head_pos[i] for i in range(3))
                expected = tuple(
                    sum(inverse[row][column] * delta[column] for column in range(3))
                    for row in range(3)
                )
                for actual, wanted in zip(point.head, expected):
                    self.assertAlmostEqual(actual, wanted, places=9)

    def test_metric_runs_on_current_config(self) -> None:
        """Метрика на текущей калибровке считается и попадает в разумный диапазон."""
        from src import calib_search

        convention = calib_search.convention_from_config()
        metrics = calib_search.evaluate_bone_convention(self.data, convention)
        self.assertEqual(metrics.bones, 48)
        self.assertEqual(metrics.in_frame, 48)
        self.assertLessEqual(metrics.good, metrics.in_frame)
        self.assertGreaterEqual(metrics.mean_error_px, 0.0)
        self.assertLess(metrics.mean_error_px, 300.0)
        self.assertEqual(len(metrics.rows), 48)


if __name__ == "__main__":
    unittest.main()
