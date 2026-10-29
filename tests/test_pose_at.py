"""Assert-тесты для src/sync.py (задачи 04b: load_poses/pose_at).

Запуск: python -m unittest discover -s tests -t . -v
Синтетические тесты не зависят от данных; тесты на реальных файлах пропускаются,
если нет data/trackingData_*.txt или output/video_meta.json (не коммитятся).
"""
from __future__ import annotations

import json
import math
import sys
import unittest
from bisect import bisect_right
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.parse_tracking import read_header, timestamp_range  # noqa: E402
from src.sync import (  # noqa: E402
    POINT_KEYS,
    POSITION_FIELDS,
    QUAT_FIELDS,
    FrameTime,
    PoseTrack,
    frame_time,
    iter_frame_poses,
    load_poses,
    pose_at_frame,
    pose_at_frame_split,
    slerp,
    video_time_offset_ns,
)

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
META_PATH = ROOT / "output" / "video_meta.json"
NO_DATA = "нет data/trackingData_*.txt или output/video_meta.json (данные не коммитятся)"

MS = 1_000_000  # наносекунд в миллисекунде


def make_track(
    ts_ms: list[int],
    head_x: list[float] | None = None,
    nan_right_before: int = 0,
    active_right: list[bool] | None = None,
) -> PoseTrack:
    """Синтетический трек: голова движется вдоль x, кисти в нуле (правые — NaN до индекса)."""
    ts = [value * MS for value in ts_ms]
    values = {name: [0.0] * len(ts) for name in (*POSITION_FIELDS, *QUAT_FIELDS)}
    values["head_x"] = list(head_x) if head_x is not None else [float(v) for v in ts_ms]
    values["head_qw"] = [1.0] * len(ts)
    for key in POINT_KEYS:
        if key.startswith("right"):
            for i in range(min(nan_right_before, len(ts))):
                for axis in ("x", "y", "z"):
                    values[f"{key}_{axis}"][i] = math.nan
    if active_right is None:
        active_right = [i >= nan_right_before for i in range(len(ts))]
    return PoseTrack(ts, values, {"left": [True] * len(ts), "right": active_right})


class VideoTimeOffsetTest(unittest.TestCase):
    """Поправка на задержку видеотракта: константа из config и её применение к кадру."""

    def _track(self) -> PoseTrack:
        # голова едет по x со скоростью 1 м/с: 0.0, 0.1, 0.2 м на 0/100/200 мс
        return make_track([0, 100, 200], head_x=[0.0, 0.1, 0.2])

    def test_offset_from_config_in_nanoseconds(self) -> None:
        with mock.patch.object(config, "VIDEO_TIME_OFFSET_MS", -100.0):
            self.assertEqual(video_time_offset_ns(), -100_000_000)
        with mock.patch.object(config, "VIDEO_TIME_OFFSET_MS", 33.5):
            self.assertEqual(video_time_offset_ns(), 33_500_000)
        with mock.patch.object(config, "VIDEO_TIME_OFFSET_MS", 0):
            self.assertEqual(video_time_offset_ns(), 0)

    def test_invalid_offset_is_rejected(self) -> None:
        for bad in (True, "100", None, 1000.1, -1000.5, math.nan, math.inf):
            with self.subTest(value=bad), mock.patch.object(
                config, "VIDEO_TIME_OFFSET_MS", bad
            ), self.assertRaises(ValueError):
                video_time_offset_ns()

    def test_pose_at_frame_explicit_offset(self) -> None:
        track = self._track()
        frame = FrameTime(index=0, ts_ns=100 * MS, t_rel_s=0.1, in_tracking=True)
        self.assertAlmostEqual(pose_at_frame(track, frame, 0).head_pos[0], 0.1)
        self.assertAlmostEqual(pose_at_frame(track, frame, -50 * MS).head_pos[0], 0.05)
        self.assertAlmostEqual(pose_at_frame(track, frame, -100 * MS).head_pos[0], 0.0)
        # за пределами трека поза замораживается — поправка не выдумывает движение
        self.assertAlmostEqual(pose_at_frame(track, frame, -400 * MS).head_pos[0], 0.0)

    def test_pose_at_frame_uses_config_offset_by_default(self) -> None:
        track = self._track()
        frame = FrameTime(index=0, ts_ns=100 * MS, t_rel_s=0.1, in_tracking=True)
        with mock.patch.object(config, "VIDEO_TIME_OFFSET_MS", -50.0):
            self.assertAlmostEqual(pose_at_frame(track, frame).head_pos[0], 0.05)
        with mock.patch.object(config, "VIDEO_TIME_OFFSET_MS", 0.0):
            self.assertAlmostEqual(pose_at_frame(track, frame).head_pos[0], 0.1)

    def test_pose_at_frame_rejects_bad_explicit_offset(self) -> None:
        track = self._track()
        frame = FrameTime(index=0, ts_ns=0, t_rel_s=0.0, in_tracking=True)
        for bad in (True, 1.5, "0", None):
            if bad is None:
                continue  # None = «взять из конфига», это допустимо
            with self.subTest(value=bad), self.assertRaises(ValueError):
                pose_at_frame(track, frame, bad)


class PoseAtSyntheticTest(unittest.TestCase):
    """Линейное движение, границы, дырки NaN — без больших файлов."""

    STEP_MS = 11.122
    TS = [0, 11, 22, 33, 44]  # целые мс — для точных ожиданий

    def test_midpoint_of_linear_motion_is_exact(self) -> None:
        # head_x(t_ms) = t_ms / 1000 (1 м/с) → в середине интервала точное среднее
        track = make_track(self.TS, head_x=[t / 1000.0 for t in self.TS])
        pose = track.pose_at(int(5.5 * MS))
        self.assertAlmostEqual(pose.head_pos[0], 0.0055, places=12)
        self.assertEqual(pose.ts_ns, int(5.5 * MS))
        # ближайший сэмпл (t = 0) даёт ошибку 5.5 мс движения — интерполяция её убирает
        self.assertGreater(abs(0.0055 - 0.0), 0.0)

    def test_interpolation_is_exact_for_uniform_sampling(self) -> None:
        # равномерная выборка линейной функции: интерполяция воспроизводит её точно
        ts_ms = [round(k * 11.122) for k in range(4)]
        track = make_track(ts_ms, head_x=[t / 1000.0 for t in ts_ms])
        for t_ms in (1, 5, 7, 12, 20, 30):
            pose = track.pose_at(round(t_ms * MS))
            self.assertAlmostEqual(pose.head_pos[0], t_ms / 1000.0, places=9)

    def test_exact_sample_hit_returns_sample(self) -> None:
        track = make_track(self.TS, head_x=[0.0, 0.111, 0.222, 0.333, 0.444])
        pose = track.pose_at(2 * 11 * MS)
        self.assertEqual(pose.head_pos[0], 0.222)
        self.assertEqual(pose.ts_ns, 22 * MS)
        self.assertEqual(pose.head_quat, (0.0, 0.0, 0.0, 1.0))

    def test_freeze_before_first_and_after_last(self) -> None:
        track = make_track(self.TS, head_x=[0.0, 0.1, 0.2, 0.3, 0.4])
        before = track.pose_at(-1_000 * MS)
        after = track.pose_at(10_000 * MS)
        self.assertEqual(before.head_pos, track.sample_pose(0).head_pos)
        self.assertEqual(after.head_pos, track.sample_pose(len(track) - 1).head_pos)
        self.assertEqual(before.ts_ns, -1_000 * MS)  # время — запрошенное, не сэмпла
        self.assertEqual(after.ts_ns, 10_000 * MS)

    def test_single_sample_track_freezes_everywhere(self) -> None:
        track = make_track([500], head_x=[1.25])
        for t_ms in (-100, 500, 900):
            pose = track.pose_at(t_ms * MS)
            self.assertEqual(pose.head_pos, (1.25, 0.0, 0.0))
            self.assertEqual(len(pose.hands), 10)

    def test_nan_hole_is_not_bridged(self) -> None:
        # правая рука невалидна до индекса 2: интервал 0–1 и 1–2 даёт NaN целиком
        track = make_track(self.TS, nan_right_before=2)
        for t_ms in (0, 5, 11, 16):
            pose = track.pose_at(t_ms * MS)
            self.assertTrue(
                all(math.isnan(value) for value in pose.hands["right_palm"]),
                f"t={t_ms} мс: дырка правой руки не должна интерполироваться",
            )
        # левая рука в тех же кадрах валидна (нули, а не NaN)
        self.assertEqual(track.pose_at(5 * MS).hands["left_palm"], (0.0, 0.0, 0.0))
        # после валидного сэмпла (2) значение снова конечное
        self.assertEqual(track.pose_at(25 * MS).hands["right_palm"], (0.0, 0.0, 0.0))

    def test_partial_nan_inside_hand(self) -> None:
        # только ось x правого указательного пальца без данных: остальные оси конечны
        ts = [0, 10, 20]
        values = {name: [0.0] * 3 for name in (*POSITION_FIELDS, *QUAT_FIELDS)}
        values["head_qw"] = [1.0] * 3
        values["right_index_x"][0] = math.nan
        track = PoseTrack([t * MS for t in ts], values, {"left": [True] * 3, "right": [True] * 3})
        pose = track.pose_at(5 * MS)
        self.assertTrue(math.isnan(pose.hands["right_index"][0]))
        self.assertEqual(pose.hands["right_index"][1], 0.0)
        self.assertEqual(pose.hands["right_index"][2], 0.0)
        # на валидном участке (11–20 мс) та же ось уже конечна
        self.assertEqual(track.pose_at(15 * MS).hands["right_index"][0], 0.0)

    def test_nearest_index_tie_goes_earlier(self) -> None:
        track = make_track([0, 10, 20])
        self.assertEqual(track.nearest_index(4 * MS), 0)
        self.assertEqual(track.nearest_index(5 * MS), 0)  # ничья → более ранний
        self.assertEqual(track.nearest_index(6 * MS), 1)
        self.assertEqual(track.nearest_index(10 * MS), 1)
        self.assertEqual(track.nearest_index(-50 * MS), 0)
        self.assertEqual(track.nearest_index(50 * MS), 2)

    def test_quaternion_comes_from_nearest_sample(self) -> None:
        # в 04b ориентация не интерполируется: берём кватернион ближайшего сэмпла
        ts = [0, 10, 20]
        values = {name: [0.0] * 3 for name in (*POSITION_FIELDS, *QUAT_FIELDS)}
        values["head_qw"] = [1.0, 0.0, -1.0]
        track = PoseTrack([t * MS for t in ts], values, {"left": [True] * 3, "right": [True] * 3})
        self.assertEqual(track.pose_at(4 * MS).head_quat, (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(track.pose_at(6 * MS).head_quat, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(track.pose_at(16 * MS).head_quat, (0.0, 0.0, 0.0, -1.0))

    def test_active_flag_comes_from_nearest_sample(self) -> None:
        track = make_track(self.TS, nan_right_before=1, active_right=[False, False, True, True, True])
        self.assertFalse(track.pose_at(5 * MS).active["right"])  # ближайший — сэмпл 0
        self.assertFalse(track.pose_at(15 * MS).active["right"])  # ближайший — сэмпл 1
        self.assertTrue(track.pose_at(17 * MS).active["right"])  # ближайший — сэмпл 2

    def test_non_increasing_timestamps_raise(self) -> None:
        for ts in ([0, 10, 10], [0, 20, 10], [5, 0]):
            with self.subTest(ts=ts):
                with self.assertRaises(ValueError):
                    make_track(ts)

    def test_empty_track_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_track([])

    def test_missing_or_short_column_raises(self) -> None:
        track = make_track([0, 10])
        with self.assertRaises(ValueError):
            PoseTrack([0, 10], {"head_x": [0.0, 0.0]}, {"left": [True, True], "right": [True, True]})
        values = {name: [0.0, 0.0] for name in (*POSITION_FIELDS, *QUAT_FIELDS)}
        values["head_x"] = [0.0]  # короче, чем ts
        with self.assertRaises(ValueError):
            PoseTrack([0, 10], values, {"left": [True, True], "right": [True, True]})
        values["head_x"] = [0.0, 0.0]
        with self.assertRaises(ValueError):
            PoseTrack([0, 10], values, {"left": [True, True], "right": [True]})

    def test_bad_time_raises(self) -> None:
        track = make_track(self.TS)
        for bad in (True, 1.0, "0", None):
            with self.subTest(t=bad), self.assertRaises(ValueError):
                track.pose_at(bad)  # type: ignore[arg-type]
                track.nearest_index(bad)  # type: ignore[arg-type]

    def test_to_flat_matches_csv_columns(self) -> None:
        from src.parse_tracking import CSV_COLUMNS

        track = make_track(self.TS, head_x=[t / 1000.0 for t in self.TS], nan_right_before=1)
        flat = track.pose_at(5 * MS).to_flat()
        self.assertEqual(set(flat), set(CSV_COLUMNS) - {"t_ns", "t_rel_s"})
        self.assertEqual(flat["left_active"], 1)
        self.assertEqual(flat["right_active"], 0)
        self.assertTrue(math.isnan(flat["right_index_x"]))
        self.assertAlmostEqual(flat["head_x"], 0.005, places=12)


class PoseAtFrameSplitTest(unittest.TestCase):
    """08c: раздельные сдвиги времени для головы и кистей (инструмент проверки каналов)."""

    def _track(self) -> PoseTrack:
        # голова: 0, 0.1, 0.2 м по x (1 м/с); правая ладонь: 0, −0.2, −0.4 м (−2 м/с)
        ts = [0, 100 * MS, 200 * MS]
        values = {name: [0.0, 0.0, 0.0] for name in (*POSITION_FIELDS, *QUAT_FIELDS)}
        values["head_x"] = [0.0, 0.1, 0.2]
        values["head_qw"] = [1.0, 1.0, 1.0]
        for key in POINT_KEYS:
            values[f"{key}_x"] = [0.0, -0.2, -0.4] if key == "right_palm" else [0.0, 0.0, 0.0]
        return PoseTrack(ts, values, {"left": [True] * 3, "right": [True] * 3})

    def _frame(self) -> FrameTime:
        return FrameTime(index=0, ts_ns=100 * MS, t_rel_s=0.1, in_tracking=True)

    def test_equal_offsets_match_single_offset(self) -> None:
        track = self._track()
        frame = self._frame()
        for offset in (0, -50 * MS, 100 * MS, -400 * MS):
            with self.subTest(offset=offset):
                split = pose_at_frame_split(track, frame, offset, offset)
                plain = pose_at_frame(track, frame, offset)
                self.assertEqual(split.head_pos, plain.head_pos)
                self.assertEqual(split.head_quat, plain.head_quat)
                self.assertEqual(split.hands, plain.hands)
                self.assertEqual(split.active, plain.active)

    def test_channels_take_their_own_time(self) -> None:
        track = self._track()
        frame = self._frame()
        pose = pose_at_frame_split(track, frame, 0, 0)
        self.assertAlmostEqual(pose.head_pos[0], 0.1)  # ровно t = 100 мс
        self.assertAlmostEqual(pose.hands["right_palm"][0], -0.2)
        mixed = pose_at_frame_split(track, frame, -100 * MS, 0)
        self.assertAlmostEqual(mixed.head_pos[0], 0.0)
        self.assertAlmostEqual(mixed.hands["right_palm"][0], -0.2)
        self.assertEqual(mixed.ts_ns, frame.ts_ns)  # поза отнесена ко времени кадра
        swapped = pose_at_frame_split(track, frame, 0, -100 * MS)
        self.assertAlmostEqual(swapped.head_pos[0], 0.1)
        self.assertAlmostEqual(swapped.hands["right_palm"][0], 0.0)

    def test_split_interpolates_each_channel(self) -> None:
        track = self._track()
        frame = FrameTime(index=1, ts_ns=150 * MS, t_rel_s=0.15, in_tracking=True)
        pose = pose_at_frame_split(track, frame, -75 * MS, -25 * MS)
        self.assertAlmostEqual(pose.head_pos[0], 0.075)  # 0.1 м + 0.1 м/с · (−25 мс)
        self.assertAlmostEqual(pose.hands["right_palm"][0], -0.25)  # −0.2 м + (−2 м/с)·25 мс

    def test_invalid_offsets_raise(self) -> None:
        track = self._track()
        frame = self._frame()
        for bad in (True, 1.5, "0", None):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                pose_at_frame_split(track, frame, bad, 0)
            with self.subTest(value=bad), self.assertRaises(ValueError):
                pose_at_frame_split(track, frame, 0, bad)


@unittest.skipIf(DATA is None or not META_PATH.is_file(), NO_DATA)
class PoseAtRealDataTest(unittest.TestCase):
    """Все 7279 кадров и 21 787 сэмплов (один разбор файла на класс)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        cls.header_ns = read_header(DATA)["timeStampNs"]
        cls.first_ns, cls.last_ns = timestamp_range(DATA)
        cls.track = load_poses(DATA)

    def test_all_frames_resolve_without_exceptions(self) -> None:
        frames = list(iter_frame_poses(self.track, self.meta, self.header_ns, self.last_ns))
        self.assertEqual(len(frames), 7279)
        self.assertEqual(len(self.track), 21_787)
        for frame, pose in frames:
            self.assertEqual(pose.ts_ns, frame.ts_ns)
            self.assertEqual(len(pose.hands), 10)
            self.assertAlmostEqual(math.hypot(*pose.head_quat), 1.0, places=6)

    def test_time_error_and_tracking_gaps(self) -> None:
        # Критерий 04b: |t − ближайший сэмпл| ≤ половины шага (5.6 мс). На регулярной сетке
        # это так; 24 кадра (0.33 %) попадают в паузы трекинга до 67.4 мс — там nearest
        # далеко по определению, и как раз нужна интерполяция.
        over = 0
        max_dt = 0
        for k in range(7279):
            frame = frame_time(k, self.meta, self.header_ns, self.last_ns)
            if not frame.in_tracking or frame.ts_ns < self.track.ts[0]:
                continue
            dt = abs(frame.ts_ns - self.track.ts[self.track.nearest_index(frame.ts_ns)])
            max_dt = max(max_dt, dt)
            over += dt > 5_600_000
        self.assertEqual(over, 24)
        self.assertAlmostEqual(max_dt / 1e6, 32.673547, places=5)
        gaps = [b - a for a, b in zip(self.track.ts, self.track.ts[1:])]
        self.assertEqual(max(gaps), 67_403_008)

    def test_linear_interpolation_matches_neighbour_lerp(self) -> None:
        # Реальные данные: в кадре k = 707 поза = w·p_i + (1−w)·p_{i+1} по каждой оси.
        k = 707  # w ≈ 0.49, т.е. кадр почти в середине интервала между сэмплами
        ts = frame_time(k, self.meta, self.header_ns, self.last_ns).ts_ns
        i = bisect_right(self.track.ts, ts) - 1
        w = (ts - self.track.ts[i]) / (self.track.ts[i + 1] - self.track.ts[i])
        self.assertTrue(0.0 < w < 1.0)
        pose = self.track.pose_at(ts)
        a = self.track.sample_pose(i)
        b = self.track.sample_pose(i + 1)
        for axis_index in range(3):
            expected = a.head_pos[axis_index] + (b.head_pos[axis_index] - a.head_pos[axis_index]) * w
            self.assertAlmostEqual(pose.head_pos[axis_index], expected, places=12)
        # ориентация — SLERP от тех же двух сэмплов с тем же весом (04c); позиции — lerp
        self.assertEqual(pose.head_quat, slerp(a.head_quat, b.head_quat, w))

    def test_first_and_last_samples_are_frozen(self) -> None:
        first = self.track.pose_at(self.header_ns)
        last = self.track.pose_at(self.last_ns + 100 * MS)
        self.assertEqual(first.head_pos, self.track.sample_pose(0).head_pos)
        self.assertEqual(last.head_pos, self.track.sample_pose(len(self.track) - 1).head_pos)
        # k = 0: первый кадр раньше первого сэмпла (первый сэмпл +18.4 мс от header)
        self.assertGreater(self.track.ts[0], self.header_ns)

    def test_tail_frames_are_frozen_last_sample(self) -> None:
        frames = list(iter_frame_poses(self.track, self.meta, self.header_ns, self.last_ns))
        tail = [pose for frame, pose in frames[-3:]]
        last_sample = self.track.sample_pose(len(self.track) - 1)
        for pose in tail:
            self.assertEqual(pose.head_pos, last_sample.head_pos)

    def test_right_hand_invalid_at_start_and_at_end(self) -> None:
        # 135 невалидных сэмплов — это 66 в начале (до 0.83 с) и 69 в конце (с 241.80 с),
        # а не только «в начале», как считалось в заметках к 03c.
        early = self.track.pose_at(self.header_ns + 500 * MS)
        self.assertTrue(all(math.isnan(v) for v in early.hands["right_palm"]))
        self.assertFalse(early.active["right"])
        middle = self.track.pose_at(self.header_ns + 2000 * MS)
        self.assertFalse(math.isnan(middle.hands["right_palm"][0]))
        self.assertTrue(middle.active["right"])
        tail = self.track.pose_at(self.last_ns - 200 * MS)
        self.assertTrue(all(math.isnan(v) for v in tail.hands["right_palm"]))
        self.assertFalse(tail.active["right"])


if __name__ == "__main__":
    unittest.main()
