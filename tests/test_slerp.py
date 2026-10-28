"""Assert-тесты для src/sync.py (задача 04c: slerp и ориентация головы в pose_at).

Запуск: python -m unittest discover -s tests -t . -v
Синтетические тесты не зависят от данных; тесты на реальных файлах пропускаются,
если нет data/trackingData_*.txt или output/video_meta.json (не коммитятся).
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import unittest
from bisect import bisect_right
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parse_tracking import read_header, timestamp_range  # noqa: E402
from src.sync import (  # noqa: E402
    POSITION_FIELDS,
    QUAT_FIELDS,
    SLERP_EPS,
    PoseTrack,
    frame_time,
    iter_frame_poses,
    load_poses,
    slerp,
)

DATA = next(iter(sorted((ROOT / "data").glob("trackingData_*.txt"))), None)
META_PATH = ROOT / "output" / "video_meta.json"
NO_DATA = "нет data/trackingData_*.txt или output/video_meta.json (данные не коммитятся)"

MS = 1_000_000  # наносекунд в миллисекунде
IDENTITY = (0.0, 0.0, 0.0, 1.0)


def quat_angle(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Угол поворота между кватернионами, рад (0…π); устойчив к малым углам."""
    norm = math.hypot(*a) * math.hypot(*b)
    dot = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / norm))
    return 2.0 * math.atan2(math.sqrt(max(0.0, 1.0 - dot * dot)), abs(dot))


def quat_z(degrees: float) -> tuple[float, float, float, float]:
    """Поворот на ``degrees`` вокруг оси z (порядок x, y, z, w)."""
    half = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def random_unit_quat(rng: random.Random) -> tuple[float, float, float, float]:
    """Равномерный случайный поворот: 4 гауссианы, нормированные на 1."""
    values = [rng.gauss(0.0, 1.0) for _ in range(4)]
    norm = math.hypot(*values)
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def make_quat_track(
    ts_ms: list[int],
    quats: list[tuple[float, float, float, float]],
    head_x: list[float] | None = None,
) -> PoseTrack:
    """Синтетический трек: задаём только кватернионы головы (и, если нужно, head_x)."""
    ts = [value * MS for value in ts_ms]
    values = {name: [0.0] * len(ts) for name in (*POSITION_FIELDS, *QUAT_FIELDS)}
    for index, (qx, qy, qz, qw) in enumerate(quats):
        values["head_qx"][index] = qx
        values["head_qy"][index] = qy
        values["head_qz"][index] = qz
        values["head_qw"][index] = qw
    if head_x is not None:
        values["head_x"] = list(head_x)
    return PoseTrack(ts, values, {"left": [True] * len(ts), "right": [True] * len(ts)})


class SlerpContractTest(unittest.TestCase):
    """Контракт slerp: концы, 90° на половине, 1000 пар, double cover, вырожденные."""

    def test_endpoints_are_exact_including_double_cover(self) -> None:
        turned = quat_z(120.0)
        cases = (
            (IDENTITY, turned),  # dot = cos 60° = 0.5
            (IDENTITY, tuple(-value for value in turned)),  # dot < 0 — двойное покрытие
            (turned, IDENTITY),
        )
        negative_cases = 0
        for q0, q1 in cases:
            dot = sum(a * b for a, b in zip(q0, q1))
            negative_cases += dot < 0.0
            for u, expected in ((0.0, q0), (1.0, q1)):
                with self.subTest(q0=q0, q1=q1, u=u):
                    result = slerp(q0, q1, u)
                    self.assertEqual(len(result), 4)
                    for got, want in zip(result, expected):
                        # при dot < 0 внутри идём к −q1, но u = 1 отдаёт исходный q1
                        self.assertAlmostEqual(got, want, delta=1e-12)
        self.assertEqual(negative_cases, 1)  # пара с dot < 0 действительно проверена

    def test_quarter_turn_midpoint_is_45_degrees(self) -> None:
        q0 = IDENTITY
        q1 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))  # поворот на 90°
        result = slerp(q0, q1, 0.5)
        self.assertAlmostEqual(quat_angle(q0, result), math.pi / 4, delta=1e-9)
        self.assertAlmostEqual(quat_angle(result, q1), math.pi / 4, delta=1e-9)
        # ожидаемый поворот на 45°: z = sin(22.5°), w = cos(22.5°)
        self.assertAlmostEqual(result[2], math.sin(math.pi / 8), delta=1e-12)
        self.assertAlmostEqual(result[3], math.cos(math.pi / 8), delta=1e-12)

    def test_1000_random_pairs_unit_norm_and_shortest_arc(self) -> None:
        rng = random.Random(20261028)
        negative_dot = 0
        for _ in range(1000):
            q0 = random_unit_quat(rng)
            q1 = random_unit_quat(rng)
            u = rng.random()
            negative_dot += sum(a * b for a, b in zip(q0, q1)) < 0.0
            result = slerp(q0, q1, u)
            self.assertLessEqual(abs(math.hypot(*result) - 1.0), 1e-6)
            # на кратчайшей дуге угол(q0, res) + угол(res, q1) = угол(q0, q1);
            # на длинной дуге сумма была бы 2π − угол(q0, q1)
            total = quat_angle(q0, q1)
            split = quat_angle(q0, result) + quat_angle(result, q1)
            self.assertAlmostEqual(split, total, delta=1e-9)
        # ~половина пар дала dot < 0: двойное покрытие проверено на 500+ случаях
        self.assertGreater(negative_dot, 400)

    def test_opposite_quaternions_give_zero_rotation(self) -> None:
        q0 = quat_z(30.0)
        q1 = tuple(-value for value in q0)  # q1 = −q0: тот же поворот
        result = slerp(q0, q1, 0.5)
        # угол 0°, а не 180°: пошли по короткой дуге, а не через −q0
        self.assertAlmostEqual(quat_angle(q0, result), 0.0, delta=1e-9)
        self.assertAlmostEqual(abs(sum(a * b for a, b in zip(q0, result))), 1.0, delta=1e-12)
        end = slerp(q0, q1, 1.0)
        for got, want in zip(end, q1):
            self.assertAlmostEqual(got, want, delta=1e-12)  # u = 1 → исходный q1

    def test_equal_and_near_equal_quats(self) -> None:
        q = quat_z(37.0)
        for u in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(u=u):
                for got, want in zip(slerp(q, q, u), q):
                    self.assertAlmostEqual(got, want, delta=1e-15)
        # почти равные (угол 1e-6 рад): 1 − dot = 1.25e-13 < SLERP_EPS → ветка nlerp,
        # формула через sin на таком угле делила бы на ~0
        tiny = (math.sin(0.5e-6), 0.0, 0.0, math.cos(0.5e-6))
        self.assertLess(1.0 - sum(a * b for a, b in zip(IDENTITY, tiny)), SLERP_EPS)
        mixed = tuple(0.5 * a + 0.5 * b for a, b in zip(IDENTITY, tiny))
        norm = math.hypot(*mixed)
        for got, want in zip(slerp(IDENTITY, tiny, 0.5), (value / norm for value in mixed)):
            self.assertAlmostEqual(got, want, delta=1e-15)

    def test_non_unit_input_is_normalised(self) -> None:
        # ||q| − 1| = 1 > 1e-6 → нормируем вход, а не падаем; выход единичный
        result = slerp((0.0, 0.0, 0.0, 2.0), (0.0, 0.0, 0.0, 3.0), 0.5)
        self.assertAlmostEqual(math.hypot(*result), 1.0, delta=1e-12)
        self.assertAlmostEqual(result[3], 1.0, delta=1e-12)
        for u in (0.0, 1.0):
            with self.subTest(u=u):
                self.assertAlmostEqual(math.hypot(*slerp((0.0, 0.0, 0.0, 2.0), IDENTITY, u)),
                                       1.0, delta=1e-12)

    def test_invalid_inputs_raise(self) -> None:
        for u in (-0.1, 1.1, float("nan"), float("inf"), True, "0.5"):
            with self.subTest(u=u), self.assertRaises(ValueError):
                slerp(IDENTITY, quat_z(90.0), u)  # type: ignore[arg-type]
        bad_quats = (
            (0.0, 0.0, 0.0),  # не 4 компоненты
            (0.0, 0.0, 0.0, 0.0),  # нулевая норма
            (math.nan, 0.0, 0.0, 1.0),  # не конечное число
            (0.0, 0.0, 0.0, math.inf),
            ("a", 0.0, 0.0, 1.0),  # не число
        )
        for bad in bad_quats:
            with self.subTest(q=bad), self.assertRaises(ValueError):
                slerp(bad, IDENTITY, 0.5)  # type: ignore[arg-type]
            with self.subTest(q=bad), self.assertRaises(ValueError):
                slerp(IDENTITY, bad, 0.5)  # type: ignore[arg-type]


class PoseAtSlerpTest(unittest.TestCase):
    """Интеграция 04c: pose_at интерполирует ориентацию, позиции остались линейными."""

    def test_midpoint_is_45_degrees(self) -> None:
        track = make_quat_track([0, 100], [IDENTITY, quat_z(90.0)], head_x=[0.0, 1.0])
        pose = track.pose_at(50 * MS)
        self.assertAlmostEqual(quat_angle(IDENTITY, pose.head_quat), math.pi / 4, delta=1e-9)
        # в 04b здесь был бы ближайший сэмпл (ничья → более ранний, 0°) — SLERP включён
        near = track.sample_pose(track.nearest_index(50 * MS), 50 * MS)
        self.assertAlmostEqual(quat_angle(IDENTITY, near.head_quat), 0.0, delta=1e-12)
        # позиции по-прежнему линейны (регресс 04b)
        self.assertAlmostEqual(pose.head_pos[0], 0.5, places=12)
        self.assertAlmostEqual(pose.head_pos[1], 0.0, places=15)

    def test_pose_at_matches_slerp_of_neighbour_samples(self) -> None:
        q1 = quat_z(90.0)
        track = make_quat_track([0, 100], [IDENTITY, q1])
        for w in (0.1, 0.25, 0.5, 0.9):
            with self.subTest(w=w):
                pose = track.pose_at(round(w * 100 * MS))
                for got, want in zip(pose.head_quat, slerp(IDENTITY, q1, w)):
                    self.assertAlmostEqual(got, want, delta=1e-12)

    def test_exact_samples_and_freeze_unchanged(self) -> None:
        q1 = quat_z(90.0)
        track = make_quat_track([0, 100], [IDENTITY, q1])
        self.assertEqual(track.pose_at(0).head_quat, IDENTITY)  # точное попадание в сэмпл
        self.assertEqual(track.pose_at(100 * MS).head_quat, q1)
        self.assertEqual(track.pose_at(-500 * MS).head_quat, IDENTITY)  # заморозка до
        self.assertEqual(track.pose_at(10_000 * MS).head_quat, q1)  # заморозка после

    def test_orientation_is_interpolated_not_taken_from_nearest(self) -> None:
        # t = 74 мс: nearest — второй сэмпл (90°), SLERP — 66.6°; разница ~23°,
        # то есть это не выбор ближайшего, а действительно интерполяция
        track = make_quat_track([0, 100], [IDENTITY, quat_z(90.0)])
        pose = track.pose_at(74 * MS)
        self.assertAlmostEqual(quat_angle(IDENTITY, pose.head_quat),
                               math.radians(66.6), delta=1e-9)
        self.assertGreater(quat_angle(pose.head_quat, quat_z(90.0)), math.radians(20.0))

    def test_broken_quat_falls_back_to_nearest(self) -> None:
        # нулевой кватернион интерполировать нечем: кадр не падает, берём nearest
        track = make_quat_track([0, 100], [IDENTITY, (0.0, 0.0, 0.0, 0.0)])
        self.assertEqual(track.pose_at(25 * MS).head_quat, IDENTITY)
        self.assertEqual(track.pose_at(75 * MS).head_quat, (0.0, 0.0, 0.0, 0.0))


@unittest.skipIf(DATA is None or not META_PATH.is_file(), NO_DATA)
class SlerpRealDataTest(unittest.TestCase):
    """Ориентация 04c на реальных данных: 7 279 кадров, 21 787 сэмплов (один разбор)."""

    DEBUG_FRAMES = (30, 1800, 3600, 5400, 7200)

    @classmethod
    def setUpClass(cls) -> None:
        cls.meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        cls.header_ns = read_header(DATA)["timeStampNs"]
        cls.first_ns, cls.last_ns = timestamp_range(DATA)
        cls.track = load_poses(DATA)
        cls.quats = [cls.track.sample_pose(index).head_quat for index in range(len(cls.track))]
        cls.frame_deltas = cls._frame_deltas()

    @classmethod
    def _frame_deltas(cls) -> list[tuple[int, float, float]]:
        """(k, дельта nearest°, дельта SLERP°) для каждой пары соседних кадров с данными."""
        rows: list[tuple[int, float, float]] = []
        prev: tuple[tuple[float, ...], tuple[float, ...]] | None = None
        for k in range(cls.meta["frame_count"]):
            frame = frame_time(k, cls.meta, cls.header_ns, cls.last_ns)
            if not frame.in_tracking or frame.ts_ns < cls.track.ts[0]:
                prev = None
                continue
            pose = cls.track.pose_at(frame.ts_ns)
            near = cls.track.sample_pose(cls.track.nearest_index(frame.ts_ns), frame.ts_ns)
            if prev is not None:
                rows.append(
                    (
                        k,
                        math.degrees(quat_angle(prev[0], near.head_quat)),
                        math.degrees(quat_angle(prev[1], pose.head_quat)),
                    )
                )
            prev = (near.head_quat, pose.head_quat)
        return rows

    def test_all_frames_have_finite_unit_quats(self) -> None:
        frames = list(iter_frame_poses(self.track, self.meta, self.header_ns, self.last_ns))
        self.assertEqual(len(frames), 7279)
        for frame, pose in frames:
            self.assertEqual(len(pose.head_quat), 4)
            for value in pose.head_quat:
                self.assertTrue(math.isfinite(value), f"k={frame.index}: не конечное число")
            self.assertLessEqual(abs(math.hypot(*pose.head_quat) - 1.0), 1e-6)
        # на входе норма тоже единичная: max ||q| − 1| = 1.23e-07
        worst = max(abs(math.hypot(*quat) - 1.0) for quat in self.quats)
        self.assertLessEqual(worst, 1e-6)
        self.assertLessEqual(worst, 2e-7)

    def test_consecutive_samples_never_need_double_cover(self) -> None:
        # 21 786 пар соседних сэмплов: ни одной с dot < 0, min dot = 0.999242 —
        # голова в этой записи почти не разворачивается скачком; сам double cover
        # покрыт синтетикой (SlerpContractTest)
        dots = [
            sum(a * b for a, b in zip(self.quats[i], self.quats[i + 1]))
            for i in range(len(self.quats) - 1)
        ]
        self.assertEqual(len(dots), 21_786)
        self.assertEqual(sum(1 for dot in dots if dot < 0.0), 0)
        self.assertGreater(min(dots), 0.99)

    def test_debug_frames_nearest_vs_slerp(self) -> None:
        rows = []
        for k in self.DEBUG_FRAMES:
            frame = frame_time(k, self.meta, self.header_ns, self.last_ns)
            self.assertTrue(frame.in_tracking)
            pose = self.track.pose_at(frame.ts_ns)
            near = self.track.sample_pose(self.track.nearest_index(frame.ts_ns), frame.ts_ns)
            rows.append((k, math.degrees(quat_angle(near.head_quat, pose.head_quat))))
        # Измерено: 0.0399°, 0.0083°, 0.0139°, 0.0203°, 0.0104° — голова в этих кадрах
        # поворачивается на 0.16…0.26° за сэмпл, поэтому nearest и SLERP близки;
        # различие всё же ненулевое (nearest отстаёт/опережает до половины шага).
        for k, difference in rows:
            with self.subTest(k=k):
                self.assertLess(difference, 0.05)
        self.assertGreater(rows[0][1], 0.0)
        # Покадровая дельта на этих кадрах — не скачок: max 0.755° < 1°
        steps = {k: slerp_step for k, _, slerp_step in self.frame_deltas}
        for k in self.DEBUG_FRAMES:
            with self.subTest(k=k):
                self.assertLess(steps[k], 1.0)

    def test_pose_at_uses_slerp_between_real_samples(self) -> None:
        k = 707  # w ≈ 0.49: кадр почти в середине интервала между сэмплами
        ts = frame_time(k, self.meta, self.header_ns, self.last_ns).ts_ns
        i = bisect_right(self.track.ts, ts) - 1
        w = (ts - self.track.ts[i]) / (self.track.ts[i + 1] - self.track.ts[i])
        expected = slerp(
            self.track.sample_pose(i).head_quat,
            self.track.sample_pose(i + 1).head_quat,
            w,
        )
        pose = self.track.pose_at(ts)
        for got, want in zip(pose.head_quat, expected):
            self.assertAlmostEqual(got, want, delta=1e-12)
        self.assertNotEqual(pose.head_quat, self.track.sample_pose(i).head_quat)

    def test_slerp_has_no_bigger_jumps_than_nearest(self) -> None:
        nearest = [row[1] for row in self.frame_deltas]
        slerp_deg = [row[2] for row in self.frame_deltas]
        self.assertEqual(len(nearest), 7274)
        # max: 6.916° (nearest) → 5.330° (SLERP), т.е. меньше на 23 %
        self.assertAlmostEqual(max(nearest), 6.9162, delta=1e-3)
        self.assertAlmostEqual(max(slerp_deg), 5.3296, delta=1e-3)
        self.assertLess(max(slerp_deg), max(nearest))
        # кадров с дельтой > 1°: 331 (nearest) → 325 (SLERP)
        over_nearest = sum(1 for value in nearest if value > 1.0)
        over_slerp = sum(1 for value in slerp_deg if value > 1.0)
        self.assertEqual(over_nearest, 331)
        self.assertEqual(over_slerp, 325)
        self.assertLess(over_slerp, over_nearest)
        # p95 совпадает до тысячных (0.975°), std практически тот же (0.3208 vs 0.3210)
        def p95(values: list[float]) -> float:
            ordered = sorted(values)
            return ordered[math.ceil(0.95 * len(ordered)) - 1]

        self.assertAlmostEqual(p95(nearest), 0.975245, delta=1e-3)
        self.assertAlmostEqual(p95(slerp_deg), 0.975491, delta=1e-3)
        self.assertLess(abs(statistics.pstdev(nearest) - statistics.pstdev(slerp_deg)), 1e-3)


if __name__ == "__main__":
    unittest.main()
