"""Assert-тесты для src/calib_frames.py (задача 07a).

Проверяется внутренняя согласованность эталона (рамки в кадре левого глаза, стороны не
дублируются, номера и время кадров возрастают) и — на реальных данных — то, ради чего он
существует: кадры действительно покрывают разные положения рук, время кадров совпадает с
``sync.frame_time``, а разметка достижима, то есть существует конвенция осей, при которой
проекция кистей ложится в рамки без перепутанных рук.

Тесты на реальных данных пропускаются, если ``data/trackingData_*.txt`` (258 МБ) отсутствует
— файл не коммитится. Синтетические тесты работают всегда.

Запуск: python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calib_frames import (  # noqa: E402
    CALIB_FRAMES,
    EYE_HEIGHT,
    EYE_WIDTH,
    TOLERANCE_PX,
    CalibFrame,
    HandBox,
    as_json,
    in_eye_frame,
    validate_frames,
)

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
VIDEO = next(iter(sorted((ROOT / "data").glob("CameraRecord_*.mp4"))), None)
META = ROOT / "output" / "video_meta.json"
NO_DATA = "нет data/trackingData_*.txt — файл данных не коммитится (см. README)"
NO_VIDEO = "нет data/CameraRecord_*.mp4 — видео не коммитится (см. README)"
NO_META = "нет output/video_meta.json — метаданные видео создаются задачей 02"


def box(bbox: tuple[int, int, int, int] = (10, 10, 100, 100), **kwargs) -> HandBox:
    """Синтетическая рамка кисти (по умолчанию — валидная)."""
    kwargs.setdefault("hand", "right")
    return HandBox(bbox=bbox, **kwargs)


def frame(index: int = 30, **kwargs) -> CalibFrame:
    """Синтетический кадр эталона с одной валидной рамкой."""
    kwargs.setdefault("hands", (box(),))
    return CalibFrame(index=index, t_s=kwargs.pop("t_s", index / 30.0), **kwargs)


class HandBoxTest(unittest.TestCase):
    """Рамка кисти: валидация входа и геометрия."""

    def test_constants(self) -> None:
        self.assertEqual((EYE_WIDTH, EYE_HEIGHT), (1080, 810))
        self.assertEqual(TOLERANCE_PX, 50)

    def test_full_frame_box_is_valid(self) -> None:
        item = box((0, 0, 1080, 810), hand="unknown")
        self.assertEqual(item.center, (540.0, 405.0))
        self.assertTrue(item.contains(0, 0))
        self.assertTrue(item.contains(1080, 810))
        self.assertFalse(item.contains(1081, 405))

    def test_contains_bounds_inclusive(self) -> None:
        item = box((440, 130, 900, 340))
        for u, v, inside in (
            (440, 130, True),
            (900, 340, True),
            (439, 130, False),
            (440, 129, False),
            (901, 200, False),
            (600, 341, False),
        ):
            self.assertIs(item.contains(u, v), inside, f"({u}, {v})")

    def test_rejects_unknown_hand(self) -> None:
        with self.assertRaises(ValueError):
            box(hand="palm")

    def test_rejects_unknown_confidence(self) -> None:
        with self.assertRaises(ValueError):
            box(confidence="sure")

    def test_rejects_non_bool_partial(self) -> None:
        with self.assertRaises(ValueError):
            box(partial=1)  # type: ignore[arg-type]

    def test_rejects_degenerate_bbox(self) -> None:
        with self.assertRaises(ValueError):
            box((100, 200, 100, 300))
        with self.assertRaises(ValueError):
            box((100, 300, 200, 300))

    def test_rejects_bad_bbox_shape(self) -> None:
        with self.assertRaises(ValueError):
            box((1, 2, 3))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            box([1, 2, 3, 4])  # type: ignore[arg-type]

    def test_rejects_non_int_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            box((0.5, 10, 100, 100))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            box((False, 10, 100, 100))  # type: ignore[arg-type]

    def test_rejects_out_of_frame(self) -> None:
        with self.assertRaises(ValueError):
            box((-1, 10, 100, 100))
        with self.assertRaises(ValueError):
            box((0, 0, 1081, 100))
        with self.assertRaises(ValueError):
            box((0, 0, 100, 811))


class CalibFrameTest(unittest.TestCase):
    """Кадр эталона: стороны не дублируются, ``ignore`` не конфликтует с разметкой."""

    def test_box_for(self) -> None:
        item = CalibFrame(
            index=1800,
            t_s=60.0,
            hands=(box((440, 130, 900, 340), hand="right"), box((0, 520, 250, 810), hand="left")),
        )
        self.assertEqual(item.box_for("right").bbox, (440, 130, 900, 340))
        self.assertEqual(item.box_for("left").bbox, (0, 520, 250, 810))
        self.assertIsNone(item.box_for("unknown"))
        self.assertEqual(len(item.labeled_boxes), 2)

    def test_unknown_box_is_not_labeled(self) -> None:
        item = CalibFrame(index=30, t_s=1.0, hands=(box(hand="unknown"),))
        self.assertEqual(item.labeled_boxes, ())

    def test_rejects_empty_hands(self) -> None:
        with self.assertRaises(ValueError):
            CalibFrame(index=30, t_s=1.0, hands=())

    def test_rejects_duplicate_side(self) -> None:
        with self.assertRaises(ValueError):
            CalibFrame(
                index=30,
                t_s=1.0,
                hands=(box(hand="right"), box((0, 0, 10, 10), hand="right")),
            )

    def test_unknown_side_can_repeat(self) -> None:
        CalibFrame(
            index=30,
            t_s=1.0,
            hands=(box(hand="unknown"), box((0, 0, 10, 10), hand="unknown")),
        )

    def test_rejects_ignored_but_labeled_side(self) -> None:
        with self.assertRaises(ValueError):
            CalibFrame(index=30, t_s=1.0, hands=(box(hand="right"),), ignore=("right",))

    def test_rejects_bad_ignore(self) -> None:
        with self.assertRaises(ValueError):
            CalibFrame(index=30, t_s=1.0, hands=(box(),), ignore=("both",))

    def test_rejects_bad_index_or_time(self) -> None:
        with self.assertRaises(ValueError):
            CalibFrame(index=-1, t_s=1.0, hands=(box(),))
        with self.assertRaises(ValueError):
            CalibFrame(index=True, t_s=1.0, hands=(box(),))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CalibFrame(index=30, t_s=-0.1, hands=(box(),))
        with self.assertRaises(ValueError):
            CalibFrame(index=30, t_s=float("nan"), hands=(box(),))


class ValidateSetTest(unittest.TestCase):
    """Проверка набора целиком: возрастание, уникальность, покрытие уверенности."""

    def test_shipped_set_is_valid(self) -> None:
        validate_frames(CALIB_FRAMES)

    def test_shipped_set_shape(self) -> None:
        self.assertEqual([item.index for item in CALIB_FRAMES], [30, 1800, 3600, 5400, 7200])
        # на каждом кадре 1–2 рамки: левая и/или правая кисть
        for item in CALIB_FRAMES:
            self.assertIn(len(item.hands), (1, 2), f"k={item.index}")
        # уверенная разметка есть хотя бы для четырёх кадров из пяти
        high = [item for item in CALIB_FRAMES if any(b.confidence == "high" for b in item.hands)]
        self.assertGreaterEqual(len(high), 4)

    def test_shipped_labels_are_consistent(self) -> None:
        """k=30 — кисть без установленной стороны, на остальных обе стороны размечены."""
        first, *rest = CALIB_FRAMES
        self.assertEqual([b.hand for b in first.hands], ["unknown"])
        self.assertEqual(first.ignore, ("right",))
        for item in rest:
            self.assertEqual({b.hand for b in item.hands}, {"left", "right"})

    def test_rejects_too_few_frames(self) -> None:
        with self.assertRaises(ValueError):
            validate_frames((frame(0), frame(30)))

    def test_rejects_unsorted_indexes(self) -> None:
        with self.assertRaises(ValueError):
            validate_frames((frame(1800), frame(30), frame(3600)))

    def test_rejects_unsorted_times(self) -> None:
        with self.assertRaises(ValueError):
            validate_frames((frame(30, t_s=5.0), frame(60, t_s=1.0), frame(90, t_s=9.0)))

    def test_rejects_duplicate_index(self) -> None:
        with self.assertRaises(ValueError):
            validate_frames((frame(30), frame(60), frame(30)))

    def test_rejects_without_high_confidence(self) -> None:
        frames = tuple(frame(k, hands=(box(confidence="low"),)) for k in (0, 30, 60))
        with self.assertRaises(ValueError):
            validate_frames(frames)


class JsonTest(unittest.TestCase):
    """Эталон для 07b: JSON содержит размер кадра глаза, допуск и все рамки."""

    def test_json_round_trip(self) -> None:
        payload = json.loads(as_json(CALIB_FRAMES))
        self.assertEqual(payload["eye_size"], [EYE_WIDTH, EYE_HEIGHT])
        self.assertEqual(payload["tolerance_px"], TOLERANCE_PX)
        self.assertEqual(len(payload["frames"]), len(CALIB_FRAMES))
        for source, item in zip(CALIB_FRAMES, payload["frames"]):
            self.assertEqual(item["index"], source.index)
            self.assertAlmostEqual(item["t_s"], source.t_s, places=9)
            self.assertEqual(item["ignore"], list(source.ignore))
            self.assertEqual(len(item["hands"]), len(source.hands))
            for hand, source_box in zip(item["hands"], source.hands):
                self.assertEqual(hand["hand"], source_box.hand)
                self.assertEqual(tuple(hand["bbox"]), source_box.bbox)
                self.assertEqual(hand["confidence"], source_box.confidence)
                self.assertEqual(hand["partial"], source_box.partial)

    def test_in_eye_frame(self) -> None:
        self.assertTrue(in_eye_frame(0, 0))
        self.assertTrue(in_eye_frame(1079, 809))
        self.assertFalse(in_eye_frame(1080, 809))
        self.assertFalse(in_eye_frame(0, 810))
        self.assertFalse(in_eye_frame(-1, 0))


class SaveFramesTest(unittest.TestCase):
    """CLI сохраняет 5 PNG полного SBS-кадра в координатах, на которые рассчитана разметка."""

    def test_save_writes_five_frames(self) -> None:
        if VIDEO is None:
            self.skipTest(NO_VIDEO)
        try:
            import cv2  # noqa: PLC0415 — тест без OpenCV не нужен
        except ImportError:
            self.skipTest("cv2 не установлен")
        from src.calib_frames import _save_frames  # noqa: PLC0415 — локальный импорт для skip

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "calib"
            written = _save_frames(VIDEO, out, CALIB_FRAMES, overlay=True)
            self.assertEqual(written, len(CALIB_FRAMES))
            for item in CALIB_FRAMES:
                path = out / f"frame_{item.index:05d}.png"
                self.assertTrue(path.is_file(), path)
                image = cv2.imread(str(path))
                self.assertEqual(image.shape, (EYE_HEIGHT, 2 * EYE_WIDTH, 3))
                # разметка в координатах левого глаза: кадр не обрезан
                marked = out / "annotated" / f"frame_{item.index:05d}_left.png"
                self.assertTrue(marked.is_file(), marked)
                self.assertEqual(cv2.imread(str(marked)).shape, (EYE_HEIGHT, EYE_WIDTH, 3))
            self.assertEqual(len(list(out.glob("*.png"))), len(CALIB_FRAMES))

    def test_missing_video_is_reported(self) -> None:
        from src.calib_frames import _save_frames  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                _save_frames(
                    Path(directory) / "нет_такого.mp4", Path(directory) / "out", CALIB_FRAMES, False
                )


@unittest.skipIf(DATA is None, NO_DATA)
class RealFramesTest(unittest.TestCase):
    """Реальные данные: время кадров, покрытие положений рук и достижимость разметки."""

    @classmethod
    def setUpClass(cls) -> None:
        from src.camera_model import Intrinsics, head_to_world
        from src.parse_tracking import read_header, timestamp_range
        from src.sync import frame_time, load_poses

        if not META.is_file():
            raise unittest.SkipTest(NO_META)
        cls.head_to_world = staticmethod(head_to_world)
        cls.frame_time = staticmethod(frame_time)
        cls.meta = json.loads(META.read_text(encoding="utf-8"))
        cls.header = read_header(DATA)
        cls.header_ns = cls.header["timeStampNs"]
        cls.tracking_end_ns = timestamp_range(DATA)[1]
        cls.track = load_poses(DATA)
        cls.intrinsics = Intrinsics.from_header(cls.header, frame_size=(2160, 810)).per_eye("left")
        cls.poses = {
            item.index: cls.track.pose_at(
                cls.frame_time(item.index, cls.meta, cls.header_ns, cls.tracking_end_ns).ts_ns
            )
            for item in CALIB_FRAMES
        }

    def setUp(self) -> None:
        if not META.is_file():
            self.skipTest(NO_META)

    def test_frame_times_match_sync(self) -> None:
        for item in CALIB_FRAMES:
            moment = self.frame_time(item.index, self.meta, self.header_ns, self.tracking_end_ns)
            self.assertTrue(moment.in_tracking, f"k={item.index}: кадр вне трекинга")
            self.assertAlmostEqual(item.t_s, moment.t_rel_s, places=6, msg=f"k={item.index}")

    def test_hands_are_tracked_on_all_frames(self) -> None:
        for item in CALIB_FRAMES:
            pose = self.poses[item.index]
            self.assertTrue(pose.active["left"], f"k={item.index}: левая рука неактивна")
            self.assertTrue(pose.active["right"], f"k={item.index}: правая рука неактивна")
            for side in ("left", "right"):
                point = pose.hands[f"{side}_palm"]
                self.assertTrue(
                    all(math.isfinite(value) for value in point),
                    f"k={item.index}: NaN в ладони {side}",
                )

    def test_frames_cover_different_hand_positions(self) -> None:
        """≥ 3 различных положения кистей: точки разнесены хотя бы на 10 см."""
        palms = {
            item.index: self.poses[item.index].hands["right_palm"] for item in CALIB_FRAMES
        }
        distinct: list[float] = []
        for index, point in palms.items():
            if all(math.dist(point, kept) > 0.10 for kept in distinct):
                distinct.append(point)
        self.assertGreaterEqual(len(distinct), 3, f"положения правой кисти: {list(palms.values())}")
        # и левая кисть тоже не стоит на месте между крайними кадрами
        left = [self.poses[item.index].hands["left_palm"] for item in CALIB_FRAMES]
        self.assertGreater(max(math.dist(a, b) for a, b in itertools.combinations(left, 2)), 0.05)

    def test_near_and_far_hands_are_covered(self) -> None:
        """Набор покрывает и близкую кисть (крупная в кадре), и далёкую (мелкая)."""
        distances = [
            math.dist(self.poses[item.index].hands[f"{side}_palm"], self.poses[item.index].head_pos)
            for item in CALIB_FRAMES
            for side in ("left", "right")
        ]
        self.assertGreater(max(distances) - min(distances), 0.10)

    def test_ground_truth_is_reachable(self) -> None:
        """Разметка не противоречива: существует конвенция осей, попадающая в рамки.

        Перебираем 24 собственные ротации (перестановки осей со знаками, det = +1) × 2
        конвенции кватерниона — 48 вариантов. Для каждого считаем точки кистей, попавшие
        внутрь кадра левого глаза, и требуем, чтобы в лучшем варианте **каждая** такая
        точка лежала в рамке своей руки (или в рамке ``unknown``), и ни одна не попадала в
        рамку чужой руки. Это проверка эталона, а не финальная калибровка: 07b/07c
        добавят ``T_head_cam`` (смещение + yaw/pitch/roll) и обязаны только улучшить
        результат.
        """
        joins = ("palm", "wrist", "thumb", "index", "middle")
        best: tuple[int, int, int] | None = None
        for invert in (False, True):
            for matrix in _proper_rotations():
                good = wrong = missed = 0
                for item in CALIB_FRAMES:
                    pose = self.poses[item.index]
                    world = self.head_to_world(pose, invert_quat=invert)
                    rotation = (
                        (world[0][0], world[1][0], world[2][0]),
                        (world[0][1], world[1][1], world[2][1]),
                        (world[0][2], world[1][2], world[2][2]),
                    )
                    translation = (world[0][3], world[1][3], world[2][3])
                    for side in ("left", "right"):
                        if side in item.ignore:
                            continue
                        for name in joins:
                            point = pose.hands[f"{side}_{name}"]
                            delta = [point[i] - translation[i] for i in range(3)]
                            head = [
                                sum(rotation[row][column] * delta[column] for column in range(3))
                                for row in range(3)
                            ]
                            camera = [
                                sum(matrix[row][column] * head[column] for column in range(3))
                                for row in range(3)
                            ]
                            if camera[2] <= 0.01:
                                continue
                            u = self.intrinsics.fx * camera[0] / camera[2] + self.intrinsics.cx
                            v = self.intrinsics.fy * camera[1] / camera[2] + self.intrinsics.cy
                            if not in_eye_frame(u, v):
                                continue
                            own = item.box_for(side)
                            if own is not None and own.contains(u, v):
                                good += 1
                            elif any(b.contains(u, v) and b.hand == side for b in item.hands):
                                good += 1
                            elif any(b.contains(u, v) and b.hand == "unknown" for b in item.hands):
                                good += 1
                            else:
                                other = next((b for b in item.hands if b.contains(u, v)), None)
                                if other is None:
                                    missed += 1
                                else:
                                    wrong += 1
                score = (good, -wrong, -missed)
                if best is None or score > best:
                    best = score
        self.assertIsNotNone(best)
        good, neg_wrong, neg_missed = best
        self.assertEqual(neg_wrong, 0, "лучшая конвенция путает руки — разметка противоречива")
        self.assertEqual(neg_missed, 0, "лучшая конвенция не попадает в рамки — разметка смещена")
        self.assertGreaterEqual(good, 35, "лучшая конвенция попадает менее чем в 35 точках")


def _proper_rotations():
    """24 собственные ротации (перестановки осей со знаками, det = +1)."""
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            for row, column in enumerate(permutation):
                matrix[row][column] = signs[row]
            determinant = (
                matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
                - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
                + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
            )
            if determinant > 0:
                yield tuple(tuple(row) for row in matrix)


if __name__ == "__main__":
    unittest.main()
