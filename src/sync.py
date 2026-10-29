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
интерполирует позиции между соседними сэмплами; ориентация берётся от ближайшего сэмпла.
Задача 04c: ориентация головы интерполируется через :func:`slerp` (двойное покрытие,
кратчайшая дуга, при малом угле — нормализованный lerp). За пределами диапазона сэмплов
поза замораживается, NaN (нет данных) через дырку не интерполируется.

Задержка видеотракта (доработка 08a): кадр показывает сцену раньше своей метки, поэтому поза
берётся не в ``frame.ts_ns``, а в ``frame.ts_ns + VIDEO_TIME_OFFSET_MS`` — см.
:func:`video_time_offset_ns` и :func:`pose_at_frame`. Рендер и проверка калибровки зовут именно
``pose_at_frame``, чтобы жить в одной шкале времени.

Доработка 08c: :func:`pose_at_frame_split` берёт голову и кисти со своими сдвигами времени —
это инструмент проверки гипотезы «каналы запаздывают по-разному». По итогам проверки гипотеза
не подтвердилась (разница оценок меньше точности одиночной константы), поэтому рендер по-прежнему
зовёт ``pose_at_frame`` с одной поправкой.

Запуск:
    python src/sync.py output/video_meta.json data/trackingData_20260505_165740.txt
    python src/sync.py output/video_meta.json data/trackingData_20260505_165740.txt --poses
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
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

#: Колонки кватерниона головы (x, y, z, w) — интерполируются через :func:`slerp`.
QUAT_FIELDS = ("head_qx", "head_qy", "head_qz", "head_qw")

#: Оси позиции сустава в колонках: ``left_index_proximal_x`` и т.д.
AXES = ("x", "y", "z")


def _extra_keys(keys: Sequence[str]) -> tuple[str, ...]:
    """Проверить список дополнительных ключей суставов (08e): имена ``"left_palm"`` и т.п.

    Ключи из :data:`POINT_KEYS` запрещены: их колонки уже есть, дублировать их в ``values``
    значило бы завести два источника одной величины.
    """
    if isinstance(keys, str):
        raise ValueError("extra_keys: ожидался список ключей, получена строка")
    result: list[str] = []
    for key in keys:
        if not isinstance(key, str) or not key:
            raise ValueError(f"extra_keys: ожидалось непустое имя ключа, получено {key!r}")
        if key in POINT_KEYS:
            raise ValueError(f"extra_keys: {key!r} уже среди 5 точек рендера (POINT_KEYS)")
        if key in result:
            raise ValueError(f"extra_keys: ключ {key!r} повторяется")
        result.append(key)
    return tuple(result)


def _extra_fields(keys: Sequence[str]) -> tuple[str, ...]:
    """Имена колонок дополнительных суставов: ``<key>_x/_y/_z``."""
    return tuple(f"{key}_{axis}" for key in keys for axis in AXES)

#: Порог «малого угла» в SLERP: при ``1 − |dot| < SLERP_EPS`` sin(theta) в формуле
#: деления уже сопоставим с погрешностью float, поэтому берём нормализованный lerp.
#: 1 − dot = 2·sin²(theta/2) ≈ theta²/2, т.е. eps = 1e-9 соответствует theta ≈ 4.5e-5 рад
#: (0.0026°): расхождение nlerp и SLERP там ~theta³ (≈ 1e-13 рад), а запас до float-шума
#: (dot near 1 разрешается шагом ~1e-16) — семь порядков.
SLERP_EPS = 1e-9

#: Допуск нормы на входе SLERP: при ``||q| − 1| > QUAT_NORM_TOL`` кватернион
#: нормализуется (а не отвергается) — данные приходят из парсера с погрешностью парсинга.
QUAT_NORM_TOL = 1e-6


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


#: Допустимый модуль поправки времени кадра, мс. Больше — почти наверняка опечатка в конфиге,
#: а не измеренная задержка: 1 с сдвигал бы позу на 30 кадров.
MAX_VIDEO_TIME_OFFSET_MS = 1000.0


def video_time_offset_ns() -> int:
    """Поправка времени кадра из ``src/config.py``, наносекунды (может быть отрицательной).

    Отрицательная поправка значит «поза из прошлого»: кадр снят раньше своей метки. Значение
    проверяется: конечное число, ``|value| ≤ MAX_VIDEO_TIME_OFFSET_MS``, не ``bool``. Модуль
    ``config`` импортируется внутри функции, чтобы ``sync`` оставался импортируемым без конфига
    (так же поступает ``camera_model``: числа читаются на каждом вызове, подмена в тестах работает).
    """
    from src import config  # noqa: PLC0415 — локальный импорт, чтобы не тянуть конфиг при импорте

    value = getattr(config, "VIDEO_TIME_OFFSET_MS", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config.VIDEO_TIME_OFFSET_MS: ожидалось число мс, получено {value!r}")
    if not math.isfinite(value) or abs(value) > MAX_VIDEO_TIME_OFFSET_MS:
        raise ValueError(
            f"config.VIDEO_TIME_OFFSET_MS: ожидалось |value| ≤ {MAX_VIDEO_TIME_OFFSET_MS:g} мс, "
            f"получено {value!r}"
        )
    return int(round(float(value) * 1e6))


def pose_time_ns(frame: FrameTime, offset_ns: int | None = None) -> int:
    """Момент, в котором берётся поза кадра: ``frame.ts_ns + offset_ns`` (задача 08e).

    ``offset_ns=None`` — поправка из ``config.VIDEO_TIME_OFFSET_MS`` (``video_time_offset_ns``).
    Вынесено из :func:`pose_at_frame`, чтобы костная разметка 08e брала дополнительные
    суставы ровно в тот же момент времени, что и поза головы, — иначе две метрики одной
    проверки жили бы в разных шкалах.
    """
    if offset_ns is None:
        offset_ns = video_time_offset_ns()
    if isinstance(offset_ns, bool) or not isinstance(offset_ns, int):
        raise ValueError(f"offset_ns: ожидалось целое число, получено {offset_ns!r}")
    return int(frame.ts_ns) + offset_ns


def pose_at_frame(track: PoseTrack, frame: FrameTime, offset_ns: int | None = None) -> Pose:
    """Поза на кадр с поправкой на задержку видеотракта: ``t = frame.ts_ns + offset_ns``.

    ``offset_ns=None`` — :func:`video_time_offset_ns` (константа ``src/config.py``); явное
    значение нужно диагностике (скан сдвига) и тестам. Сама ``frame_time`` остаётся «чистой»
    шкалой видео: поправка — отдельное решение, которое видно в одном месте.
    """
    return track.pose_at(pose_time_ns(frame, offset_ns))


def pose_at_frame_split(
    track: PoseTrack, frame: FrameTime, head_offset_ns: int, hand_offset_ns: int
) -> Pose:
    """Поза кадра, где **голова и кисти** взяты со своими сдвигами времени (08c, диагностика).

    Смысл: если каналы head и hand запаздывают по-разному, единая поправка времени не может
    посадить на кисть одновременно позу камеры и точки кистей. Функция собирает такую позу
    из двух запросов: поза головы (она же поза камеры) — из ``t + head_offset_ns``, кисти —
    из ``t + hand_offset_ns``. ``Pose.ts_ns`` при этом равен времени кадра: поза относится
    к моменту видео, а не к одному из двух сдвинутых моментов.

    Зачем в ``sync``, а не в диагностике: это та же операция «время → поза», что и
    :func:`pose_at_frame`; раздельные сдвиги каналов — гипотеза о синхронизации, а не о
    картинке. В рендере функция **не используется**: 08c показала, что данные не разделяют
    каналы (разница оценок меньше точности одиночной константы), поэтому у рендера осталась
    одна поправка ``config.VIDEO_TIME_OFFSET_MS`` — см. ``docs/NOTES.md``, 2026-10-29.

    При равных сдвигах возвращается ровно та же поза, что у
    :func:`pose_at_frame` (проверяется тестом), поэтому сравнение «одна константа против
    двух» не вносит расхождения в интерполяцию.
    """
    head_offset_ns = _int_ns(head_offset_ns, "head_offset_ns")
    hand_offset_ns = _int_ns(hand_offset_ns, "hand_offset_ns")
    if head_offset_ns == hand_offset_ns:
        return track.pose_at(int(frame.ts_ns) + head_offset_ns)
    head_pose = track.pose_at(int(frame.ts_ns) + head_offset_ns)
    hand_pose = track.pose_at(int(frame.ts_ns) + hand_offset_ns)
    return Pose(
        ts_ns=int(frame.ts_ns),
        head_pos=head_pose.head_pos,
        head_quat=head_pose.head_quat,
        hands=hand_pose.hands,
        active=hand_pose.active,
    )


def _normalized(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Делит четвёрку на её норму (норма > 0 гарантирована вызывающим)."""
    norm = math.hypot(*q)
    return q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm


def _unit_quat(q: object, what: str) -> tuple[float, float, float, float]:
    """Проверяет кватернион и при необходимости нормирует его.

    Требования: ровно 4 конечных числа, норма > 0. Отклонение нормы в пределах
    :data:`QUAT_NORM_TOL` не трогаем (возвращаем как есть), больше — делим на норму:
    данные из парсера уже близки к единичной норме, но контракт :func:`slerp` обещает
    единичный результат, и падать из-за отклонения 1e-4 смысла нет.
    """
    if isinstance(q, (str, bytes)):
        raise ValueError(f"{what}: ожидалась последовательность из 4 чисел, получено {q!r}")
    try:
        values = tuple(float(value) for value in q)
    except (TypeError, ValueError):
        raise ValueError(f"{what}: ожидалась последовательность из 4 чисел, получено {q!r}") from None
    if len(values) != 4:
        raise ValueError(f"{what}: ожидалось 4 компоненты (x, y, z, w), получено {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{what}: компоненты должны быть конечными числами, получено {values!r}")
    norm = math.hypot(*values)
    if norm == 0.0:
        raise ValueError(f"{what}: нулевой кватернион (норма 0), поворот не определён")
    if abs(norm - 1.0) > QUAT_NORM_TOL:
        values = tuple(value / norm for value in values)
    return values[0], values[1], values[2], values[3]


def slerp(
    q0: Sequence[float], q1: Sequence[float], u: float
) -> tuple[float, float, float, float]:
    """Сферическая интерполяция кватернионов по кратчайшей дуге (задача 04c).

    Кватернионы — последовательности 4 float в порядке ``x, y, z, w``; ``u ∈ [0, 1]``,
    иначе ``ValueError``. Вход валидируется: конечные числа, норма > 0; при
    ``||q| − 1| > 1e-6`` кватернион нормируется (не падаем), поэтому выход всегда
    единичный: ``||q| − 1| ≤ 1e-6``.

    Контракт концов буквальный: ``u = 0`` возвращает (нормированный) ``q0``, ``u = 1`` —
    ровно (нормированный) ``q1``, в том числе при ``dot < 0``, когда внутри дуга идёт
    к ``−q1``. Кватернионы ``q`` и ``−q`` задают один поворот, поэтому двойное покрытие
    ``dot < 0`` разворачивается в ``−q1`` (далее работаем с ``|dot|``) — иначе
    интерполяция пошла бы по длинной дуге (> 180°).

    Малый угол: при ``1 − |dot| < SLERP_EPS`` (≈ 4.5e-5 рад, см. константу) формула
    ``sin`` делила бы на ~0, поэтому берём нормализованный lerp (nlerp) — на таком угле
    он отличается от SLERP на ~theta³, т.е. меньше 1e-13 рад. Знаменатель nlerp не
    обращается в 0: после разворота ``dot ≥ 0``, значит ``|(1−u)·q0 + u·q1|² ≥ 1/2``.
    """
    if isinstance(u, bool) or not isinstance(u, (int, float)):
        raise ValueError(f"u: ожидалось число в [0, 1], получено {u!r}")
    u = float(u)
    if not math.isfinite(u) or not 0.0 <= u <= 1.0:
        raise ValueError(f"u: ожидалось 0 ≤ u ≤ 1, получено {u!r}")

    a = _unit_quat(q0, "q0")
    b = _unit_quat(q1, "q1")
    if u == 0.0:
        return a
    if u == 1.0:
        return b

    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:  # двойное покрытие: q и −q — один поворот, идём по короткой дуге
        b = (-b[0], -b[1], -b[2], -b[3])
        dot = -dot
    dot = min(dot, 1.0)  # float-шум может дать 1 + 1e-16

    if 1.0 - dot < SLERP_EPS:
        mixed = tuple((1.0 - u) * x + u * y for x, y in zip(a, b))
        return _normalized(mixed)

    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    s0 = math.sin((1.0 - u) * theta) / sin_theta
    s1 = math.sin(u * theta) / sin_theta
    return _normalized(
        (
            s0 * a[0] + s1 * b[0],
            s0 * a[1] + s1 * b[1],
            s0 * a[2] + s1 * b[2],
            s0 * a[3] + s1 * b[3],
        )
    )


def _is_usable_quat(q: Sequence[float]) -> bool:
    """Кватернион можно интерполировать: компоненты конечны, норма > 0."""
    return all(math.isfinite(value) for value in q) and math.hypot(*q) > 0.0


def _quat_angle_rad(a: Sequence[float], b: Sequence[float]) -> float:
    """Угол поворота между двумя кватернионами, рад (0…π).

    Берём ``dot = a·b / (|a|·|b|)`` (для поворотов ``|dot|`` из-за двойного покрытия) и
    ``angle = 2·atan2(sqrt(1 − dot²), |dot|)``: при малых углах atan2 не теряет точность,
    в отличие от ``2·acos(dot)``, где ``dot → 1``. Деление на нормы обязательно: кватернионы
    из данных не строго единичные (max ||q| − 1| ≈ 1.2e-7), и без него dot выходит за 1.
    """
    norm_a = math.hypot(*a)
    norm_b = math.hypot(*b)
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("угол между кватернионами: нулевой кватернион")
    dot = sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)
    dot = max(-1.0, min(1.0, dot))
    vec_norm = math.sqrt(max(0.0, 1.0 - dot * dot))
    return 2.0 * math.atan2(vec_norm, abs(dot))


def _percentile(values: Sequence[float], q: float) -> float:
    """Перцентиль ``q`` (0…1) ближайшим рангом: ceil(q·n)-й элемент по возрастанию."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


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

    Храним только нужное: ``ts`` (Unix ns, строго возрастает), 33 позиции, кватернион
    головы (4 колонки) и 2 флага. 21 787 сэмплов занимают ≈ 6 МБ, поэтому файл читается
    один раз (``load_poses``).

    ``extra_keys`` (задача 08e) — дополнительные суставы, которых нет среди 5 точек рендера
    (``index_proximal`` и т.п.): их позиции лежат в тех же колонках, что и остальные, но в
    :class:`Pose` не попадают — рендер и CSV остаются 5-точечными (их расширение — 08f).
    Читаются они через :meth:`joint_at` с той же интерполяцией, что и поза.
    """

    def __init__(
        self,
        ts: Sequence[int],
        values: Mapping[str, Sequence[float]],
        active: Mapping[str, Sequence[bool]],
        extra_keys: Sequence[str] = (),
    ) -> None:
        self.ts = tuple(int(t) for t in ts)
        if not self.ts:
            raise ValueError("PoseTrack: нет сэмплов (нужен хотя бы один)")
        for a, b in zip(self.ts, self.ts[1:]):
            if b <= a:
                raise ValueError(f"PoseTrack: время сэмплов не возрастает ({a} → {b})")
        extra = _extra_keys(extra_keys)
        for name in (*POSITION_FIELDS, *QUAT_FIELDS, *_extra_fields(extra)):
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
        self.extra_keys = extra
        self._values = {
            name: tuple(values[name])
            for name in (*POSITION_FIELDS, *QUAT_FIELDS, *_extra_fields(extra))
        }
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

    def _sample_quat(self, index: int) -> tuple[float, float, float, float]:
        """Кватернион головы сэмпла ``index`` в порядке ``x, y, z, w``."""
        values = self._values
        return (
            values["head_qx"][index],
            values["head_qy"][index],
            values["head_qz"][index],
            values["head_qw"][index],
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

        Правила (карточки 04b/04c):

        * ``t`` до первого или после последнего сэмпла — заморозка крайнего значения
          (так закрываются 3 хвостовых кадра видео, где трекинга уже нет);
        * попадание ровно в сэмпл — этот сэмпл, без float-арифметики;
        * между сэмплами i и i+1 — линейная интерполяция 33 позиций (3 головы + 30 кистей)
          и SLERP кватерниона головы (04c);
        * NaN — «нет данных»: интервал с NaN на любом конце целиком даёт NaN, через
          дырку (например, первые 0.8 с неактивной правой руки) не мостим;
        * флаги активности — от ближайшего сэмпла;
        * если кватернион сэмпла невалиден (не конечный или нулевой), ориентация берётся
          от ближайшего сэмпла: интерполировать нечего, а ронять весь кадр из-за одного
          битого сэмпла хуже. В реальных данных таких сэмплов нет (max ||q| − 1| ≈ 1.2e-7).
        """
        t = _int_ns(t_ns, "t_ns")
        i, j, w = self._bracket(t)
        if j is None:
            return self.sample_pose(i, t)
        nearest = self.nearest_index(t)
        values = {name: self._lerp(name, i, j, w) for name in POSITION_FIELDS}
        head = (values["head_x"], values["head_y"], values["head_z"])
        q_i = self._sample_quat(i)
        q_j = self._sample_quat(j)
        if _is_usable_quat(q_i) and _is_usable_quat(q_j):
            quat = slerp(q_i, q_j, w)
        else:
            quat = self._sample_quat(nearest)
        hands = {
            key: (values[f"{key}_x"], values[f"{key}_y"], values[f"{key}_z"]) for key in POINT_KEYS
        }
        active = {side: self._active[side][nearest] for side in ("left", "right")}
        return Pose(
            ts_ns=t,
            head_pos=head,
            head_quat=quat,
            hands=hands,
            active=active,
        )

    def _bracket(self, t_ns: int) -> tuple[int, int | None, float]:
        """Соседние сэмплы и вес интерполяции на момент ``t_ns``: ``(i, j, w)``.

        ``j = None`` значит «интерполировать нечего»: момент попал ровно в сэмпл ``i`` или
        вышел за диапазон (край замораживается) — те же три случая, что разбирает
        :meth:`pose_at`. Вынесено отдельно, чтобы :meth:`joint_at` считал дополнительные
        суставы по ровно тем же правилам, что и позу.
        """
        t = _int_ns(t_ns, "t_ns")
        if t <= self.ts[0]:
            return 0, None, 0.0
        if t >= self.ts[-1]:
            return len(self.ts) - 1, None, 0.0
        i = bisect_right(self.ts, t) - 1
        if self.ts[i] == t:
            return i, None, 0.0
        j = i + 1
        return i, j, (t - self.ts[i]) / (self.ts[j] - self.ts[i])

    def joint_at(self, t_ns: int, key: str) -> tuple[float, float, float]:
        """Позиция дополнительного сустава ``key`` на момент ``t_ns`` (задача 08e).

        ``key`` — ключ из ``extra_keys`` (``"left_index_proximal"`` и т.п.). Правила те же,
        что у :meth:`pose_at`: край замораживается, точное попадание в сэмпл берётся без
        float-арифметики, NaN на любом конце интервала даёт NaN (через «нет данных» не мостим).
        Значения совпадают с :meth:`pose_at` для тех же суставов, если они входят в позу.
        """
        if key not in self.extra_keys:
            raise ValueError(
                f"PoseTrack.joint_at: {key!r} нет среди extra_keys {self.extra_keys}"
            )
        i, j, w = self._bracket(t_ns)
        if j is None:
            return tuple(  # type: ignore[return-value]
                self._values[f"{key}_{axis}"][i] for axis in AXES
            )
        return tuple(  # type: ignore[return-value]
            self._lerp(f"{key}_{axis}", i, j, w) for axis in AXES
        )


def load_poses(tracking_path: str | Path, extra_keys: Sequence[str] = ()) -> PoseTrack:
    """Читает файл трекинга потоково и складывает сэмплы колонками в :class:`PoseTrack`.

    Разбор — ``parse_tracking.extract_sample`` (поза головы + 5 точек на руку, NaN для
    невалидных). Ошибка формата позы поднимается с номером строки через ``iter_samples``,
    но без поля сэмпла: при погоне за конкретным суставом включите ``where`` в отладке.

    ``extra_keys`` (08e) — дополнительные суставы, которых нет среди 5 точек рендера; они
    читаются теми же колонками и доступны через :meth:`PoseTrack.joint_at`, но в
    :class:`Pose` не попадают. Неизвестное имя сустава — ``ValueError`` на первом сэмпле
    (а не ``KeyError`` из середины разбора).
    """
    extra = _extra_keys(extra_keys)
    fields = (*POSITION_FIELDS, *QUAT_FIELDS, *_extra_fields(extra))
    ts = array("q")
    values: dict[str, array] = {name: array("d") for name in fields}
    active: dict[str, array] = {"left": array("b"), "right": array("b")}
    checked = False
    for sample in iter_samples(tracking_path):
        record = extract_sample(sample)
        if not checked:
            missing = [name for name in _extra_fields(extra) if name not in record]
            if missing:
                raise ValueError(
                    f"load_poses: в данных нет колонок {missing} (extra_keys={extra!r})"
                )
            checked = True
        ts.append(int(record["timeStampNs"]))
        for name in values:
            values[name].append(float(record[name]))
        for side in active:
            active[side].append(int(record[f"{side}_active"]))
    return PoseTrack(ts, values, active, extra_keys=extra)


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
    """Статистика 04b/04c: покрытие кадров, точность по времени/позиции, дырки правой руки,
    гладкость ориентации.

    «Ошибка интерполяции» меряется двумя числами: смещение по времени до ближайшего
    сэмпла (на регулярной сетке ≤ половины шага) и расхождение интерполированной позиции
    с ближайшим сэмплом (в метрах). Кадры, попавшие в паузы трекинга (шаг сэмплов
    больше номинального), считаются отдельно: там nearest неизбежно далеко, а линейная
    интерполяция как раз и нужна, чтобы поза не «прыгала».

    Ориентация (04c): покадровая угловая дельта между соседними кадрами (шаг 33.34 мс) —
    отдельно для nearest и для SLERP: max, p95, std и число дельт > 1°. Плюс контроль
    входа: максимальное отклонение нормы кватерниона от 1 и число соседних пар с dot < 0
    (двойное покрытие: если бы такие пары были, SLERP обязан разворачивать q1).
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
        "quat_norm_max_dev": 0.0,
        "quat_dot_negative": 0,
        "orientation_frames": 0,
    }
    half_step_ns = 5_600_000  # половина медианного шага 11.122 мс (карточка 04b: ≤ 5.6 мс)
    stats["max_gap_ns"] = max((b - a for a, b in zip(track.ts, track.ts[1:])), default=0)
    prev_quat: tuple[float, float, float, float] | None = None
    for index in range(len(track)):
        quat = track._sample_quat(index)
        stats["quat_norm_max_dev"] = max(
            stats["quat_norm_max_dev"], abs(math.hypot(*quat) - 1.0)
        )
        if prev_quat is not None and sum(a * b for a, b in zip(prev_quat, quat)) < 0.0:
            stats["quat_dot_negative"] += 1
        prev_quat = quat

    nearest_deltas_deg: list[float] = []
    slerp_deltas_deg: list[float] = []
    prev_quats: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    for frame, pose in iter_frame_poses(track, meta, header_ns, tracking_end_ns):
        stats["frames"] += 1
        t = frame.ts_ns
        if t < track.ts[0]:
            # первый кадр раньше первого сэмпла (+18.4 мс) — это заморозка, не пауза
            stats["before_first_sample"] += 1
            prev_quats = None
            continue
        if not frame.in_tracking:
            prev_quats = None
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
        if prev_quats is not None:
            nearest_deltas_deg.append(math.degrees(_quat_angle_rad(prev_quats[0], near.head_quat)))
            slerp_deltas_deg.append(math.degrees(_quat_angle_rad(prev_quats[1], pose.head_quat)))
        prev_quats = (near.head_quat, pose.head_quat)
        right = [value for key in POINT_KEYS if key.startswith("right") for value in pose.hands[key]]
        nan_count = sum(math.isnan(value) for value in right)
        if nan_count == len(right):
            stats["right_no_data"] += 1
        elif nan_count:
            stats["right_partial"] += 1

    stats["orientation_frames"] = len(slerp_deltas_deg)
    for prefix, deltas in (("nearest", nearest_deltas_deg), ("slerp", slerp_deltas_deg)):
        stats[f"{prefix}_deg_max"] = max(deltas, default=0.0)
        stats[f"{prefix}_deg_p95"] = _percentile(deltas, 0.95)
        stats[f"{prefix}_deg_std"] = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
        stats[f"{prefix}_deg_over_1"] = sum(1 for delta in deltas if delta > 1.0)
    return stats


def main() -> None:
    """CLI: граница трекинга, кадры без данных; с --poses — покрытие поз, точность 04b
    и гладкость ориентации 04c (nearest vs SLERP)."""
    parser = argparse.ArgumentParser(
        description="Время кадра видео в шкале трекинга и поза на этот момент (04a–04c)."
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
    ori_frames = stats["orientation_frames"]
    share_ori = stats["nearest_deg_over_1"] / ori_frames * 100 if ori_frames else 0.0
    print(f"  ориентация: пар кадров {ori_frames} (шаг 33.34 мс), угловая дельта за кадр:")
    print(f"    nearest: max {stats['nearest_deg_max']:.3f}°, p95 {stats['nearest_deg_p95']:.3f}°, "
          f"std {stats['nearest_deg_std']:.3f}°, > 1° — {stats['nearest_deg_over_1']} кадров "
          f"({share_ori:.2f} %)")
    print(f"    SLERP:   max {stats['slerp_deg_max']:.3f}°, p95 {stats['slerp_deg_p95']:.3f}°, "
          f"std {stats['slerp_deg_std']:.3f}°, > 1° — {stats['slerp_deg_over_1']} кадров "
          f"({stats['slerp_deg_over_1'] / ori_frames * 100 if ori_frames else 0.0:.2f} %)")
    print(f"  кватернион головы: max ||q| − 1| = {stats['quat_norm_max_dev']:.2e}, "
          f"соседних пар с dot < 0: {stats['quat_dot_negative']}")
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
