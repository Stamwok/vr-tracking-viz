"""sync.py — время кадра видео в шкале трекинга (04a) и поза на этот момент (04b).

``frame_time(k)`` ставит номеру кадра видео момент времени, в котором нужно брать позу
трекинга. Обе шкалы — Unix-наносекунды: ``header.timeStampNs`` из файла трекинга
описывает первый кадр видео, поэтому t = header_ns + k · step, где step берётся из
``output/video_meta.json``:

    step = duration / frame_count = 242.661 / 7279 = 33.3371 мс

Шаг считается целочисленно и с округлением, чтобы не терять наносекунды на больших
числах: offset_ns = round(k · duration_ns / frame_count).

Почему не 1/fps: cv2 отдаёт 29.9976 fps, это дало бы 33.3360 мс и расхождение 8.7 мс
к концу записи. Формула «duration / frame_count» зафиксирована в карточке 04a,
поэтому следуем ей, а расхождение с 1/fps отмечено в docs/NOTES.md.

Граница трекинга: последний сэмпл — не ``header + tracking_duration_s``. В
``video_meta.json`` поле ``tracking_duration_s`` = 242.542 с — это span first→last,
а относительно header последний сэмпл находится на 242.5607 с. Поэтому конец
трекинга передаётся явно (``parse_tracking.timestamp_range``), а ``in_tracking``
у кадров за ним — False («нет данных», последние ~50 мс записи).

Задача 04b: ``load_poses()`` загружает сэмплы колонками (позиции и флаги), ``pose_at(t)``
интерполирует позиции между соседними сэмплами; ориентация берётся от ближайшего сэмпла
(SLERP — задача 04c). За пределами диапазона сэмплов поза замораживается, NaN (нет данных)
через дырку не интерполируется.

Запуск:
    python src/sync.py output/video_meta.json data/trackingData_20260505_165740.txt
    python src/sync.py output/video_meta.json data/trackingData_20260505_165740.txt --poses
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from array import array
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parse_tracking import (  # noqa: E402
    HAND_JOINTS,
    extract_sample,
    iter_samples,
    read_header,
    timestamp_range,
)

#: Плоские ключи 10 точек кистей: "left_palm" … "right_middle" (по 5 на руку).
POINT_KEYS = tuple(f"{side}_{name}" for side in ("left", "right") for name, _ in HAND_JOINTS)

#: Колонки позиций, которые интерполируются: голова (3) + 10 точек (30).
POSITION_FIELDS = (
    "head_x",
    "head_y",
    "head_z",
    *(f"{key}_{axis}" for key in POINT_KEYS for axis in ("x", "y", "z")),
)

#: Колонки кватерниона головы (в 04b берутся от ближайшего сэмпла, SLERP — в 04c).
QUAT_FIELDS = ("head_qx", "head_qy", "head_qz", "head_qw")


@dataclass(frozen=True)
class FrameTime:
    """Момент кадра в шкале трекинга.

    ``ts_ns`` — Unix ns (его сравниваем с ``timeStampNs`` сэмплов), ``t_rel_s`` —
    то же время относительно header (для логов и графиков), ``in_tracking`` —
    покрыт ли кадр сэмплами трекинга (для хвоста видео — False).
    """

    index: int
    ts_ns: int
    t_rel_s: float
    in_tracking: bool


def _positive_int(value: object, what: str) -> int:
    """Целое > 0 (bool не считается числом: isinstance(True, int) — True)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what}: ожидалось целое число, получено {value!r}")
    if value <= 0:
        raise ValueError(f"{what}: ожидалось > 0, получено {value}")
    return value


def _int_ns(value: object, what: str) -> int:
    """Целое число наносекунд (bool отвергаем по той же причине)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what}: ожидалось целое число (Unix ns), получено {value!r}")
    return value


def duration_ns(meta: dict) -> int:
    """Длительность видео из ``video_meta.json`` в наносекундах."""
    duration = meta.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError(f"video_meta.duration: ожидалось число секунд, получено {duration!r}")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"video_meta.duration: ожидалось > 0, получено {duration!r}")
    value = round(float(duration) * 1e9)
    if value <= 0:
        raise ValueError(f"video_meta.duration: слишком мала для наносекунд: {duration!r}")
    return value


def frame_time(k: int, meta: dict, header_ns: int, tracking_end_ns: int) -> FrameTime:
    """Время кадра ``k`` (0 ≤ k < frame_count) в шкале трекинга.

    ``meta`` — разобранный ``output/video_meta.json`` (нужны ``frame_count`` и
    ``duration``), ``header_ns`` — ``header.timeStampNs``, ``tracking_end_ns`` —
    ``timeStampNs`` последнего сэмпла трекинга. Выход за границы кадров — ValueError,
    а не тихое расширение диапазона: молчаливый k = frame_count дал бы кадр,
    которого в видео нет.
    """
    frame_count = _positive_int(meta.get("frame_count"), "video_meta.frame_count")
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError(f"k: ожидался целочисленный номер кадра, получено {k!r}")
    if not 0 <= k < frame_count:
        raise ValueError(f"k={k} вне диапазона кадров 0…{frame_count - 1}")

    header_ns = _int_ns(header_ns, "header_ns")
    tracking_end_ns = _int_ns(tracking_end_ns, "tracking_end_ns")
    total_ns = duration_ns(meta)
    offset_ns = (k * total_ns + frame_count // 2) // frame_count
    ts_ns = header_ns + offset_ns
    return FrameTime(index=k, ts_ns=ts_ns, t_rel_s=offset_ns / 1e9, in_tracking=ts_ns <= tracking_end_ns)


def iter_frames(meta: dict, header_ns: int, tracking_end_ns: int) -> Iterator[FrameTime]:
    """Все кадры видео по порядку (k = 0 … frame_count − 1)."""
    frame_count = _positive_int(meta.get("frame_count"), "video_meta.frame_count")
    for k in range(frame_count):
        yield frame_time(k, meta, header_ns, tracking_end_ns)


@dataclass(frozen=True)
class Pose:
    """Поза в конкретный момент: голова + 10 точек кистей (NaN = нет данных).

    ``ts_ns`` — момент, к которому отнесена поза (время запроса), а не время сэмпла:
    между сэмплами поза интерполирована, а на краях диапазона — заморожена.
    ``hands`` индексируется ключами :data:`POINT_KEYS`; ``active`` — флаг ``isActive``
    ближайшего сэмпла. Рисовать всё равно нужно по NaN: у руки может быть активен лишь
    часть суставов, и ``active`` этого не различает.
    """

    ts_ns: int
    head_pos: tuple[float, float, float]
    head_quat: tuple[float, float, float, float]
    hands: dict[str, tuple[float, float, float]]
    active: dict[str, bool]

    def to_flat(self) -> dict[str, float | int]:
        """Плоская запись в порядке колонок tracking.csv (пригодится рендеру и графикам)."""
        flat: dict[str, float | int] = {
            "head_x": self.head_pos[0],
            "head_y": self.head_pos[1],
            "head_z": self.head_pos[2],
            "head_qx": self.head_quat[0],
            "head_qy": self.head_quat[1],
            "head_qz": self.head_quat[2],
            "head_qw": self.head_quat[3],
        }
        for key, (x, y, z) in self.hands.items():
            flat[f"{key}_x"], flat[f"{key}_y"], flat[f"{key}_z"] = x, y, z
        for side in ("left", "right"):
            flat[f"{side}_active"] = 1 if self.active[side] else 0
        return flat


class PoseTrack:
    """Все сэмплы трекинга в памяти как колонки — случайный доступ по времени.

    Храним только нужное: ``ts`` (Unix ns, строго возрастает), 33 позиции и 2 флага.
    21 787 сэмплов занимают ≈ 6 МБ, поэтому файл читается один раз (``load_poses``).
    """

    def __init__(
        self,
        ts: Sequence[int],
        values: Mapping[str, Sequence[float]],
        active: Mapping[str, Sequence[bool]],
    ) -> None:
        self.ts = tuple(int(t) for t in ts)
        if not self.ts:
            raise ValueError("PoseTrack: нет сэмплов (нужен хотя бы один)")
        for a, b in zip(self.ts, self.ts[1:]):
            if b <= a:
                raise ValueError(f"PoseTrack: время сэмплов не возрастает ({a} → {b})")
        for name in (*POSITION_FIELDS, *QUAT_FIELDS):
            column = values.get(name) if values else None
            if column is None:
                raise ValueError(f"PoseTrack: нет колонки {name!r}")
            if len(column) != len(self.ts):
                raise ValueError(
                    f"PoseTrack: колонка {name!r} длиной {len(column)}, а сэмплов {len(self.ts)}"
                )
        for side in ("left", "right"):
            column = active.get(side) if active else None
            if column is None:
                raise ValueError(f"PoseTrack: нет флага активности {side!r}")
            if len(column) != len(self.ts):
                raise ValueError(
                    f"PoseTrack: флаг {side!r} длиной {len(column)}, а сэмплов {len(self.ts)}"
                )
        self._values = {name: tuple(values[name]) for name in (*POSITION_FIELDS, *QUAT_FIELDS)}
        self._active = {side: tuple(bool(v) for v in active[side]) for side in ("left", "right")}

    def __len__(self) -> int:
        return len(self.ts)

    def nearest_index(self, t_ns: int) -> int:
        """Индекс сэмпла, ближайшего по времени к ``t_ns`` (ничья → более ранний).

        За пределами диапазона — крайний сэмпл (поза замораживается, см. ``pose_at``).
        """
        t = _int_ns(t_ns, "t_ns")
        if t <= self.ts[0]:
            return 0
        if t >= self.ts[-1]:
            return len(self.ts) - 1
        i = bisect_right(self.ts, t) - 1
        if self.ts[i] == t:
            return i
        return i if (t - self.ts[i]) <= (self.ts[i + 1] - t) else i + 1

    def sample_pose(self, index: int, ts_ns: int | None = None) -> Pose:
        """Поза ровно из сэмпла ``index`` (без интерполяции), время по умолчанию — его."""
        if ts_ns is None:
            ts_ns = self.ts[index]
        values = self._values
        head = (values["head_x"][index], values["head_y"][index], values["head_z"][index])
        quat = tuple(values[name][index] for name in QUAT_FIELDS)
        hands = {
            key: (
                values[f"{key}_x"][index],
                values[f"{key}_y"][index],
                values[f"{key}_z"][index],
            )
            for key in POINT_KEYS
        }
        active = {side: self._active[side][index] for side in ("left", "right")}
        return Pose(
            ts_ns=ts_ns,
            head_pos=head,
            head_quat=(quat[0], quat[1], quat[2], quat[3]),
            hands=hands,
            active=active,
        )

    def _lerp(self, field: str, i: int, j: int, w: float) -> float:
        """Линейная интерполяция колонки между сэмплами; NaN с любой стороны → NaN."""
        a = self._values[field][i]
        b = self._values[field][j]
        if math.isnan(a) or math.isnan(b):
            return math.nan
        return a + (b - a) * w

    def pose_at(self, t_ns: int) -> Pose:
        """Поза на момент ``t_ns`` (Unix ns).

        Правила (карточка 04b):

        * ``t`` до первого или после последнего сэмпла — заморозка крайнего значения
          (так закрываются 3 хвостовых кадра видео, где трекинга уже нет);
        * попадание ровно в сэмпл — этот сэмпл, без float-арифметики;
        * между сэмплами i и i+1 — линейная интерполяция 33 позиций (3 головы + 30 кистей);
        * NaN — «нет данных»: интервал с NaN на любом конце целиком даёт NaN, через
          дырку (например, первые 0.8 с неактивной правой руки) не мостим;
        * кватернион головы и флаги активности — от ближайшего сэмпла (SLERP — 04c).
        """
        t = _int_ns(t_ns, "t_ns")
        if t <= self.ts[0]:
            return self.sample_pose(0, t)
        if t >= self.ts[-1]:
            return self.sample_pose(len(self.ts) - 1, t)
        i = bisect_right(self.ts, t) - 1
        if self.ts[i] == t:
            return self.sample_pose(i, t)
        j = i + 1
        w = (t - self.ts[i]) / (self.ts[j] - self.ts[i])
        nearest = self.nearest_index(t)
        values = {name: self._lerp(name, i, j, w) for name in POSITION_FIELDS}
        head = (values["head_x"], values["head_y"], values["head_z"])
        quat = tuple(self._values[name][nearest] for name in QUAT_FIELDS)
        hands = {
            key: (values[f"{key}_x"], values[f"{key}_y"], values[f"{key}_z"]) for key in POINT_KEYS
        }
        active = {side: self._active[side][nearest] for side in ("left", "right")}
        return Pose(
            ts_ns=t,
            head_pos=head,
            head_quat=(quat[0], quat[1], quat[2], quat[3]),
            hands=hands,
            active=active,
        )


def load_poses(tracking_path: str | Path) -> PoseTrack:
    """Читает файл трекинга потоково и складывает сэмплы колонками в :class:`PoseTrack`.

    Разбор — ``parse_tracking.extract_sample`` (поза головы + 5 точек на руку, NaN для
    невалидных). Ошибка формата позы поднимается с номером строки через ``iter_samples``,
    но без поля сэмпла: при погоне за конкретным суставом включите ``where`` в отладке.
    """
    ts = array("q")
    values: dict[str, array] = {name: array("d") for name in (*POSITION_FIELDS, *QUAT_FIELDS)}
    active: dict[str, array] = {"left": array("b"), "right": array("b")}
    for sample in iter_samples(tracking_path):
        record = extract_sample(sample)
        ts.append(int(record["timeStampNs"]))
        for name in values:
            values[name].append(float(record[name]))
        for side in active:
            active[side].append(int(record[f"{side}_active"]))
    return PoseTrack(ts, values, active)


def iter_frame_poses(
    track: PoseTrack, meta: dict, header_ns: int, tracking_end_ns: int
) -> Iterator[tuple[FrameTime, Pose]]:
    """Пары «время кадра → поза» для всего видео (k = 0 … frame_count − 1)."""
    for frame in iter_frames(meta, header_ns, tracking_end_ns):
        yield frame, track.pose_at(frame.ts_ns)


def _load_meta(path: Path) -> dict:
    """Читает video_meta.json; не-объект — ValueError."""
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: ожидался JSON-объект video_meta, получено {type(meta).__name__}")
    return meta


def _poses_report(track: PoseTrack, meta: dict, header_ns: int, tracking_end_ns: int) -> dict:
    """Статистика 04b: покрытие кадров, точность по времени/позиции, дырки правой руки.

    «Ошибка интерполяции» меряется двумя числами: смещение по времени до ближайшего
    сэмпла (на регулярной сетке ≤ половины шага) и расхождение интерполированной позиции
    с ближайшим сэмплом (в метрах). Кадры, попавшие в паузы трекинга (шаг сэмплов
    больше номинального), считаются отдельно: там nearest неизбежно далеко, а линейная
    интерполяция как раз и нужна, чтобы поза не «прыгала».
    """
    stats = {
        "frames": 0,
        "max_dt_ns": 0,
        "frames_over_half_step": 0,
        "max_gap_ns": 0,
        "max_dp_m": 0.0,
        "max_dp_where": "",
        "right_no_data": 0,
        "right_partial": 0,
        "before_first_sample": 0,
        "in_tracking": 0,
    }
    half_step_ns = 5_600_000  # половина медианного шага 11.122 мс (карточка 04b: ≤ 5.6 мс)
    stats["max_gap_ns"] = max((b - a for a, b in zip(track.ts, track.ts[1:])), default=0)
    for frame, pose in iter_frame_poses(track, meta, header_ns, tracking_end_ns):
        stats["frames"] += 1
        t = frame.ts_ns
        if t < track.ts[0]:
            # первый кадр раньше первого сэмпла (+18.4 мс) — это заморозка, не пауза
            stats["before_first_sample"] += 1
            continue
        if not frame.in_tracking:
            continue
        stats["in_tracking"] += 1
        nearest = track.nearest_index(t)
        dt_ns = abs(t - track.ts[nearest])
        stats["max_dt_ns"] = max(stats["max_dt_ns"], dt_ns)
        if dt_ns > half_step_ns:
            stats["frames_over_half_step"] += 1
        near = track.sample_pose(nearest, t)
        for key, xyz in pose.hands.items():
            for axis_index, value in enumerate(xyz):
                other = near.hands[key][axis_index]
                if math.isnan(value) or math.isnan(other):
                    continue
                delta = abs(value - other)
                if delta > stats["max_dp_m"]:
                    stats["max_dp_m"] = delta
                    stats["max_dp_where"] = f"{key}.{('x', 'y', 'z')[axis_index]}"
        right = [value for key in POINT_KEYS if key.startswith("right") for value in pose.hands[key]]
        nan_count = sum(math.isnan(value) for value in right)
        if nan_count == len(right):
            stats["right_no_data"] += 1
        elif nan_count:
            stats["right_partial"] += 1
    return stats


def main() -> None:
    """CLI: граница трекинга, кадры без данных; с --poses — покрытие поз и точность 04b."""
    parser = argparse.ArgumentParser(
        description="Время кадра видео в шкале трекинга и поза на этот момент (задачи 04a/04b)."
    )
    parser.add_argument("meta", type=Path, help="output/video_meta.json")
    parser.add_argument("tracking", type=Path, help="data/trackingData_*.txt")
    parser.add_argument(
        "--poses",
        action="store_true",
        help="загрузить все сэмплы и проверить pose_at на всех кадрах (медленно: 258 МБ)",
    )
    args = parser.parse_args()

    for path in (args.meta, args.tracking):
        if not path.is_file():
            raise SystemExit(f"нет файла: {path}")

    meta = _load_meta(args.meta)
    header = read_header(args.tracking)
    header_ns = header["timeStampNs"]
    first_ns, last_ns = timestamp_range(args.tracking)

    frame_count = _positive_int(meta.get("frame_count"), "video_meta.frame_count")
    total_ns = duration_ns(meta)
    frames = list(iter_frames(meta, header_ns, last_ns))
    outside = [frame.index for frame in frames if not frame.in_tracking]

    print(f"{args.meta}: кадров {frame_count}, длительность {total_ns / 1e9:.3f} с, "
          f"шаг {total_ns / frame_count / 1e6:.4f} мс")
    print(f"  header (первый кадр): {header_ns}")
    print(f"  трекинг: первый сэмпл {first_ns - header_ns:+d} нс, последний {last_ns - header_ns:+d} нс "
          f"({(last_ns - header_ns) / 1e9:.6f} с) от header")
    if outside:
        print(f"  кадров в трекинге: {len(frames) - len(outside)}, без данных: {len(outside)} "
              f"(k = {outside[0]}…{outside[-1]})")
    else:
        print(f"  кадров в трекинге: {len(frames)}, без данных: 0")

    samples = sorted({0, frame_count // 2, *range(max(0, frame_count - 3), frame_count)})
    for k in samples:
        frame = frames[k]
        flag = "ok" if frame.in_tracking else "НЕТ ДАННЫХ"
        print(f"  k={frame.index:<5} t={frame.t_rel_s:+10.6f} с  ts_ns={frame.ts_ns}  {flag}")

    if not args.poses:
        print("  (позы: добавьте --poses, чтобы проверить pose_at на всех кадрах)")
        return

    track = load_poses(args.tracking)
    stats = _poses_report(track, meta, header_ns, last_ns)
    share = stats["frames_over_half_step"] / stats["in_tracking"] * 100
    print()
    print(f"  позы: сэмплов {len(track)}, кадров {stats['frames']} — без исключений")
    print(f"  кадров с данными {stats['in_tracking']}: |t − ближайший сэмпл| ≤ 5.6 мс у "
          f"{100 - share:.2f} %, max {stats['max_dt_ns'] / 1e6:.3f} мс")
    print(f"  вне половины шага: {stats['frames_over_half_step']} кадров ({share:.2f} %) — "
          f"попали в паузы трекинга (макс. пауза {stats['max_gap_ns'] / 1e6:.1f} мс)")
    print(f"  интерполяция vs nearest: max Δ позиции = {stats['max_dp_m'] * 100:.2f} см "
          f"({stats['max_dp_where']})")
    print(f"  правая рука: кадров без данных {stats['right_no_data']}, частично {stats['right_partial']}")
    first_pose = track.pose_at(header_ns - 1_000_000_000)
    last_pose = track.pose_at(last_ns + 1_000_000_000)
    sample0 = track.sample_pose(0)
    sample_last = track.sample_pose(len(track) - 1)
    print(f"  заморозка до первого сэмпла: {'совпадает' if first_pose.head_pos == sample0.head_pos else 'РАСХОЖДЕНИЕ'}")
    print(f"  заморозка после последнего:  {'совпадает' if last_pose.head_pos == sample_last.head_pos else 'РАСХОЖДЕНИЕ'}")
    pose0 = track.pose_at(frames[0].ts_ns)
    left_palm = pose0.hands["left_palm"]
    right_palm = pose0.hands["right_palm"]
    print(f"  k=0: голова ({pose0.head_pos[0]:.3f}, {pose0.head_pos[1]:.3f}, {pose0.head_pos[2]:.3f}), "
          f"левая ладонь ({left_palm[0]:.3f}, {left_palm[1]:.3f}, {left_palm[2]:.3f}), "
          f"правая ладонь {'нет данных' if math.isnan(right_palm[0]) else 'есть'}")
    last_frame = frames[-1]
    print(f"  k={last_frame.index}: t = {last_frame.t_rel_s:.6f} с, последний сэмпл "
          f"{track.ts[-1] - header_ns:+d} нс → заморозка на {(last_frame.ts_ns - track.ts[-1]) / 1e6:.2f} мс")


if __name__ == "__main__":
    main()
