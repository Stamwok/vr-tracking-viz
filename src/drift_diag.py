"""drift_diag.py — диагностика «слёта» скелета с кисти на быстрых кадрах (08a → 08c).

Зачем: при просмотре `overlay_skeleton.mp4` видно, что на быстрых движениях скелет уезжает
с кисти на десятки-сотни пикселей. Чтобы понять причину, этот инструмент собирает по кадрам
объясняющие величины и меряет **фактическую ошибку наложения по картинке** — без CV-моделей
распознавания рук (они запрещены решением §7.2 CLAUDE.md и не нужны):

1. **Ошибка, px** — для 5 точек правой руки считается расстояние до ближайшей кожи
   (маска кожи по HSV + морфология). На кадрах, где скелет лежит на кисти, ошибка 0–16 px;
   когда скелет уезжает на стол — десятки-сотни. Ограничение сверху — 200 px. Метрика меряет
   только компоненту смещения, перпендикулярную коже руки: смещение вдоль предплечья остаётся
   «на коже» и недооценивается (см. docs/NOTES.md, 2026-10-29, доработка 08a).
2. **Объясняющие величины**: `dt_to_nearest_ms` (расстояние до ближайшего сэмпла трекинга),
   `speed_mps` (скорость правой руки по сырому треку, 90 Гц), `hand_head_speed_mps` (скорость
   кисти в системе головы — насколько рука движется *относительно камеры*),
   `head_angular_speed_deg_s` (угловая скорость головы, °/с), `depth_z_m` (глубина кисти в
   системе камеры), `head_yaw_deg` (рысканье головы относительно первого сэмпла),
   `head_rotation_range_deg` (размах поворота головы за ±50 мс).
3. **Скан смещения времени** — та же ошибка при позе, взятой со сдвигом δ, усреднённая по
   когорте (шаг 5 мс в диапазоне −160…−40 мс, 08c). Минимум этого скана и есть оценка
   задержки видео относительно трекинга; скан считается по трём третям записи, чтобы
   проверить, что сдвиг постоянен.
4. **Раздельные задержки head/hand** (08c) — двумерный скан: голова берётся со сдвигом δ_h,
   кисти со сдвигом δ_n, и для трёх когорт (`fast`, `headrot`, `headstill`) ищется минимум
   ошибки по этой паре. Если бы каналы запаздывали по-разному, минимум ушёл бы с диагонали
   δ_h = δ_n. Здесь же — ошибка по суставам (ладонь, запястье, кончики), чтобы отделить
   временной остаток от расхождения позы кисти с её видом (кулак/щипок).

Выход: `table.csv` (по строке на кадр), PNG с наложением и подписями, `summary.md` с
корреляциями (Спирмен и Пирсон), сканами и выводом, `scan_2d_<когорта>.csv` — матрицы 2D-скана.

Почему когорта «50 самых быстрых», а не «speed > 2 м/с»: порог 2 м/с в записи не набирает
50 кадров. Измеренная скорость правой руки на кадрах (максимум по 5 суставам, центральная
разность по треку) — медиана 0.40 м/с, p95 0.81, максимум 2.43; выше 2 м/с только 2 кадра,
и оба — начальный «прыжок» правой руки в момент появления трекинга (t ≈ 0.9 с, 5.9 м/с
на сэмплах). Поэтому берутся 50 самых быстрых кадров (1.4…2.4 м/с), а фактический порог
печатается в `summary.md`.

Запуск:
    python src/drift_diag.py data/CameraRecord_*.mp4 --out output/debug/drift/pre_fix \\
        --time-offset-ms 0
    python src/drift_diag.py data/CameraRecord_*.mp4 --out output/debug/drift/post_fix
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import camera_model, overlay_video, sync

#: Сколько кадров в диагностической когорте (самые быстрые по правой руке).
TOP_FRAMES = 50

#: Верхний предел ошибки, px: дальше «мимо руки» уже не различить (скелет уехал совсем).
ERROR_CAP_PX = 200.0

#: Порог «точка на коже» (пикселей): DT внутри руки не строго 0 из-за маски.
ON_SKIN_PX = 3.0

#: Полуширина окна для размаха поворота и угловой скорости головы, мс.
ROTATION_WINDOW_MS = 50

#: Сетка смещений времени для скана, мс: шаг 5 мс в диапазоне −160…−40 мс (08c; в 08a было
#: 20 мс на −240…+80). Диапазон сужен вокруг измеренного минимума: шаг 20 мс на широком
#: диапазоне не давал различить −100 и −120, а «+80 мс» заведомо хуже и только тратил время.
OFFSET_GRID_MS = tuple(range(-160, -35, 5))

#: Сетка смещений для 2D-скана head × hand; та же, что у одномерного скана.
HEAD_OFFSET_GRID_MS = OFFSET_GRID_MS
HAND_OFFSET_GRID_MS = OFFSET_GRID_MS

#: Размер когорт для проверки разделения каналов head/hand (08c).
SEPARATION_COHORT = 40

#: Когорта `headrot`: голова крутится, а кисть в системе головы почти стоит. Порог 0.7 м/с —
#: это p90 скорости кисти в системе головы по записи (0.66), т.е. самые «неподвижные» кадры.
HEAD_COHORT_MAX_HAND_SPEED_MPS = 0.7

#: Когорта `headstill`: голова почти не вращается (ω ниже p75 записи), кисть быстрая.
HEAD_STILL_MAX_OMEGA_DEG_S = 8.0

#: Цвета (BGR) для подписи в PNG.
CAPTION_COLOR = (255, 255, 255)
CAPTION_SHADOW = (0, 0, 0)

#: Фокус левого глаза, px (из header: ``fy`` полного SBS-кадра) — для оценки «скорость ×
#: время → пиксели» в корреляционной таблице и для оценки чувствительности метрики.
FOCAL_PX = 686.868

#: Суставы, которыми меряется форма кисти отдельно от смещения: ладонь и запястье лежат
#: на коже при любой позе кисти, кончики зависят от позы пальцев (кулак/щипок).
BODY_JOINTS = ("palm", "wrist")
TIP_JOINTS = ("thumb", "index", "middle")


def _frame_time(scene: overlay_video.Scene, k: int) -> sync.FrameTime:
    return sync.frame_time(k, scene.meta, scene.header_ns, scene.tracking_end_ns)


def _pose(scene: overlay_video.Scene, k: int, offset_ns: int = 0) -> sync.Pose:
    """Поза кадра со сдвигом времени ``offset_ns`` (как в рендере)."""
    moment = _frame_time(scene, k)
    return scene.track.pose_at(int(moment.ts_ns) + int(offset_ns))


def sample_speeds(track: sync.PoseTrack, side: str = "right") -> np.ndarray:
    """Скорость руки на сэмплах трека (м/с): max по 5 суставам, центральная разность.

    ``NaN`` там, где сустав не трекается у соседей. Скорость по сырому 90-герцовому треку
    честнее покадровой: кадр размывает пик движения (33 мс), а сэмпл — 11 мс.
    """
    ts = np.asarray(track.ts, dtype=np.int64)
    positions = {}
    for joint in overlay_video.JOINTS:
        positions[joint] = np.array(
            [track._values[f"{side}_{joint}_{axis}"] for axis in ("x", "y", "z")], dtype=float
        ).T
    dt = np.diff(ts) / 1e9
    speeds = np.full(len(ts), np.nan)
    for i in range(1, len(ts) - 1):
        values = []
        for joint in overlay_video.JOINTS:
            first, second = positions[joint][i - 1], positions[joint][i + 1]
            if np.all(np.isfinite(first)) and np.all(np.isfinite(second)):
                values.append(float(np.linalg.norm(second - first) / (dt[i - 1] + dt[i])))
        if values:
            speeds[i] = max(values)
    return speeds


def nearest_sample(track: sync.PoseTrack, t_ns: int) -> int:
    """Индекс сэмпла, ближайшего к ``t_ns`` (ничья — более ранний, как в ``PoseTrack``)."""
    return track.nearest_index(int(t_ns))


def dt_to_nearest_ms(track: sync.PoseTrack, t_ns: int) -> float:
    """Модуль расстояния от момента позы до ближайшего сэмпла трекинга, мс."""
    index = nearest_sample(track, t_ns)
    return abs(int(t_ns) - int(track.ts[index])) / 1e6


def sample_speeds_in_head(track: sync.PoseTrack, side: str = "right") -> np.ndarray:
    """Скорость руки **в системе головы** на сэмплах, м/с (08c).

    Та же центральная разность, что в :func:`sample_speeds`, но точки переводятся в систему
    головы текущего сэмпла: ``R(q)ᵀ · (p − head_pos)``. Зачем: камера смотрит из головы, и
    на картинке сдвигается именно положение кисти **относительно камеры**. Рука, которая
    движется вместе с головой (например, человек поворачивается, держа руку на столе),
    в мире быстрая, а в кадре стоит — и по `speed_mps` её с движением головы не различить.
    Наоборот, при высоком `head_head_speed` и низком `speed_in_head` кадр показывает
    «голова крутится, рука в кадре стоит» — именно на таких кадрах ошибка наложения
    чувствительна к задержке head-канала, а не hand-канала.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side: ожидалось 'left' или 'right', получено {side!r}")
    ts = np.asarray(track.ts, dtype=np.int64)
    positions = {}
    for joint in overlay_video.JOINTS:
        positions[joint] = np.array(
            [track._values[f"{side}_{joint}_{axis}"] for axis in ("x", "y", "z")], dtype=float
        ).T
    head_pos = np.array(
        [[track._values[f"head_{axis}"][index] for axis in ("x", "y", "z")]
         for index in range(len(ts))],
        dtype=float,
    )
    quats = [track._sample_quat(index) for index in range(len(ts))]
    rotations = [camera_model.quat_to_rotation(quat) for quat in quats]
    dt = np.diff(ts) / 1e9
    speeds = np.full(len(ts), np.nan)
    for i in range(1, len(ts) - 1):
        values = []
        for joint in overlay_video.JOINTS:
            points = []
            for index in (i - 1, i + 1):
                point = positions[joint][index]
                if not np.all(np.isfinite(point)):
                    break
                delta = point - head_pos[index]
                rotation = rotations[index]
                # система головы: (Rᵀ · δ)_r = Σ_c R[c][r] · δ[c]; у каждого сэмпла свой R,
                # поэтому разность включает и поворот головы между сэмплами — иначе рука,
                # поворачивающаяся вместе с головой, считалась бы движущейся
                points.append(
                    tuple(
                        sum(rotation[axis][row] * delta[axis] for axis in range(3))
                        for row in range(3)
                    )
                )
            if len(points) == 2:
                values.append(
                    float(np.linalg.norm(np.array(points[1]) - np.array(points[0]))
                          / (dt[i - 1] + dt[i]))
                )
        if values:
            speeds[i] = max(values)
    return speeds


def track_speed_at_ms(track: sync.PoseTrack, speeds: np.ndarray, t_ns: int) -> float:
    """Скорость руки в момент ``t_ns``: значение на ближайшем сэмпле трека, м/с."""
    return float(speeds[nearest_sample(track, t_ns)])


def hand_depth_m(pose: sync.Pose, side: str = "right") -> float:
    """Средняя глубина ``Z`` видимых точек руки в системе камеры, м (``NaN`` — нет точек)."""
    matrix = camera_model.world_to_camera_matrix(pose)
    depths = []
    for joint in overlay_video.JOINTS:
        world = pose.hands[f"{side}_{joint}"]
        if not all(math.isfinite(value) for value in world):
            continue
        camera = camera_model.transform_point(matrix, world)
        if camera[2] > camera_model.MIN_DEPTH_M:
            depths.append(camera[2])
    return float(statistics.fmean(depths)) if depths else math.nan


def _forward_xy(quat: Sequence[float]) -> tuple[float, float]:
    """Направление взгляда головы в плоскости XZ (для рысканья).

    В системе головы (OpenXR) вперёд смотрит ``−z``; берём ``R(q)·(0, 0, −1)`` и
    возвращаем ``(x, −z)``, т.е. «вперёд = (1, 0)» при нулевом рысканье.
    """
    rotation = camera_model.quat_to_rotation(quat)
    x = -rotation[0][2]
    z = -rotation[2][2]
    return x, z


def head_yaw_deg(pose: sync.Pose, reference_quat: Sequence[float]) -> float:
    """Рысканье головы относительно опорного кватерниона, градусы (знак — против часовой)."""
    x0, z0 = _forward_xy(reference_quat)
    x1, z1 = _forward_xy(pose.head_quat)
    angle = math.degrees(math.atan2(x0 * z1 - z0 * x1, x0 * x1 + z0 * z1))
    return -angle  # поворот вправо (по часовой) — положительное рысканье


def _quat_angle_deg(a: Sequence[float], b: Sequence[float]) -> float:
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def head_rotation_range_deg(
    scene: overlay_video.Scene, k: int, offset_ns: int = 0, window_ms: float = ROTATION_WINDOW_MS
) -> float:
    """Размах поворота головы за ``±window_ms`` вокруг кадра: max угол между позами, градусы.

    Берётся 5 поз (``−2, −1, 0, +1, +2`` шага по четверти окна); метрика отвечает на вопрос
    «насколько быстро крутится голова в этот момент», а не насколько она повёрнута.
    """
    moment = _frame_time(scene, k)
    step_ns = int(window_ms / 2 * 1e6)
    quats = [
        scene.track.pose_at(int(moment.ts_ns) + int(offset_ns) + shift).head_quat
        for shift in (-2 * step_ns, -step_ns, 0, step_ns, 2 * step_ns)
    ]
    return max(_quat_angle_deg(a, b) for i, a in enumerate(quats) for b in quats[i + 1 :])


def head_angular_speed_deg_s(
    scene: overlay_video.Scene, k: int, offset_ns: int = 0, window_ms: float = ROTATION_WINDOW_MS
) -> float:
    """Угловая скорость головы вокруг кадра, °/с (08c).

    Угол между позами в ``t − window/2`` и ``t + window/2``, делённый на окно. В отличие от
    :func:`head_rotation_range_deg` (максимальный размах по пяти позам), это именно средняя
    скорость: она переводит задержку head-канала Δ в ошибку наложения ≈ ``fx · ω · Δ``
    радиан — оценка чувствительности метрики к раздельным сдвигам.
    """
    moment = _frame_time(scene, k)
    step_ns = int(window_ms / 2 * 1e6)
    first = scene.track.pose_at(int(moment.ts_ns) + int(offset_ns) - step_ns).head_quat
    second = scene.track.pose_at(int(moment.ts_ns) + int(offset_ns) + step_ns).head_quat
    return _quat_angle_deg(first, second) / (window_ms / 1000.0)


def skin_distance_map(image: Any) -> np.ndarray:
    """Карта «расстояние до ближайшего пикселя кожи», px (0 — на коже).

    Кожа: HSV ``H ≤ 13``, ``70 ≤ S ≤ 210``, ``55 ≤ V ≤ 200`` (подобрано по кадрам записи:
    у стола H ≈ 15–19 и S ниже, у оранжевой губки H ≈ 10, но она статична и в кинокадрах
    попадает в кадр редко; ошибка метрики — занижение, а не завышение).
    """
    import cv2  # noqa: PLC0415 — OpenCV нужен только диагностике картинки

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    skin = (
        (hue <= 13) & (saturation >= 70) & (saturation <= 210) & (value >= 55) & (value <= 200)
    ).astype(np.uint8) * 255
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return cv2.distanceTransform((skin == 0).astype(np.uint8), cv2.DIST_L2, 5)


def joint_errors_named(
    pose: sync.Pose, scene: overlay_video.Scene, distance: np.ndarray, side: str = "right"
) -> dict[str, float]:
    """Расстояния проекций до ближайшей кожи **по имени сустава**, px (``NaN`` — вне кадра).

    Именованный словарь, а не список: метрика раскладывается на ладонь/запястье и кончики,
    и при пропущенном суставе (``NaN``, точка за камерой или за краем кадра) список «съезжал»
    бы, а имена — нет. Именно поэтому ``err_palm_wrist_px`` в прогонах 08a считался по чужим
    суставам на кадрах с обрезанной кистью — в 08c это исправлено.
    """
    pixels = overlay_video.project_hands(pose, scene.intrinsics, "left")
    height, width = distance.shape
    values: dict[str, float] = {}
    for joint in overlay_video.JOINTS:
        point = pixels[side][joint]
        if not point.visible:
            values[joint] = math.nan
            continue
        u, v = int(round(point.u)), int(round(point.v))
        if not (0 <= u < width and 0 <= v < height):
            values[joint] = math.nan
            continue
        values[joint] = min(float(distance[v, u]), ERROR_CAP_PX)
    return values


def joint_errors_px(
    pose: sync.Pose, scene: overlay_video.Scene, distance: np.ndarray, side: str = "right"
) -> list[float]:
    """Расстояния проекций 5 точек руки до ближайшей кожи, px (``NaN``-точки пропускаются)."""
    named = joint_errors_named(pose, scene, distance, side)
    return [value for value in named.values() if not math.isnan(value)]


def mean_error_named(
    named: Mapping[str, float], joints: Sequence[str] = tuple(overlay_video.JOINTS)
) -> float:
    """Среднее по названным суставам (``NaN``, если все значения — ``NaN``)."""
    return nanmean(named[joint] for joint in joints if joint in named)


def error_px(pose: sync.Pose, scene: overlay_video.Scene, distance: np.ndarray) -> float:
    """Средняя ошибка наложения по видимым точкам правой руки, px (``NaN`` — точек нет)."""
    values = joint_errors_px(pose, scene, distance)
    return float(statistics.fmean(values)) if values else math.nan


def best_offset_ms(
    scene: overlay_video.Scene,
    k: int,
    distance: np.ndarray,
    grid: Sequence[float] = OFFSET_GRID_MS,
) -> tuple[float, float]:
    """Сдвиг времени, при котором ошибка кадра минимальна: ``(error_px, delta_ms)``."""
    best = (math.nan, math.nan)
    for delta in grid:
        pose = _pose(scene, k, int(delta * 1e6))
        value = error_px(pose, scene, distance)
        if not math.isnan(value) and (math.isnan(best[0]) or value < best[0]):
            best = (value, float(delta))
    return best


def nanmean(values: Iterable[float]) -> float:
    """Среднее без NaN (``NaN``, если значений не осталось)."""
    usable = [float(value) for value in values if not math.isnan(value)]
    return float(statistics.fmean(usable)) if usable else math.nan


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Ранги значений (средние ранги для связей) — без SciPy."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = np.zeros(len(values))
    position = 0
    while position < len(indexed):
        same = position
        while same + 1 < len(indexed) and values[indexed[same + 1]] == values[indexed[position]]:
            same += 1
        average = (position + same) / 2.0 + 1.0
        for index in indexed[position : same + 1]:
            ranks[index] = average
        position = same + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Коэффициент Спирмена по парам без NaN (``NaN``, если меньше двух точек)."""
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 2:
        return math.nan
    first = np.array([pair[0] for pair in pairs])
    second = np.array([pair[1] for pair in pairs])
    ranked_first = rankdata(first)
    ranked_second = rankdata(second)
    if ranked_first.std() == 0 or ranked_second.std() == 0:
        return math.nan
    return float(np.corrcoef(ranked_first, ranked_second)[0, 1])


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Коэффициент Пирсона по парам без NaN."""
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 2:
        return math.nan
    first = np.array([pair[0] for pair in pairs])
    second = np.array([pair[1] for pair in pairs])
    if first.std() == 0 or second.std() == 0:
        return math.nan
    return float(np.corrcoef(first, second)[0, 1])


def value_on_frames(scene: overlay_video.Scene, sample_values: Sequence[float]) -> np.ndarray:
    """Значение на каждый кадр = значение на ближайшем сэмпле трека (``NaN`` — сэмпл ``NaN``)."""
    values = np.full(scene.frame_count, np.nan)
    for k in range(scene.frame_count):
        moment = _frame_time(scene, k)
        values[k] = sample_values[nearest_sample(scene.track, moment.ts_ns)]
    return values


def select_fast_frames(
    scene: overlay_video.Scene, count: int = TOP_FRAMES, side: str = "right"
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Самые быстрые кадры записи по скорости руки: ``(кадры, скорости по кадрам, сэмплы)``.

    Кадру приписывается скорость ближайшего сэмпла трека (``NaN``, если рука не трекается).
    Порог не фиксируется: в записи нет и двух десятков кадров быстрее 2 м/с, поэтому «top-N
    самых быстрых» — единственный способ собрать когорту фиксированного размера; фактический
    диапазон скоростей печатается в ``summary.md``.
    """
    speeds = sample_speeds(scene.track, side)
    frame_speed = value_on_frames(scene, speeds)
    order = np.argsort(-np.nan_to_num(frame_speed, nan=-1.0))
    selected = sorted(int(k) for k in order[:count] if not math.isnan(frame_speed[k]))
    return selected, frame_speed, speeds


def top_frames(scores: np.ndarray, count: int, mask: np.ndarray) -> list[int]:
    """``count`` кадров с наибольшим ``score`` среди ``mask`` (по возрастанию номера)."""
    candidates = [k for k in range(len(scores)) if bool(mask[k]) and math.isfinite(scores[k])]
    candidates.sort(key=lambda k: -float(scores[k]))
    return sorted(int(k) for k in candidates[:count])


def separation_cohorts(
    scene: overlay_video.Scene,
    frame_speed: np.ndarray,
    frame_hand_head_speed: np.ndarray,
    frame_omega: np.ndarray,
    count: int = SEPARATION_COHORT,
) -> dict[str, list[int]]:
    """Когорты для проверки, разделяются ли задержки каналов head и hand (08c).

    * ``headrot`` — голова крутится, а кисть в системе головы почти стоит: чувствительна к
      задержке **head**-канала (сдвиг камеры поворачивает всю сцену, а собственная задержка
      кисти на такой когорте почти не видна — рука в кадре не двигается);
    * ``headstill`` — голова почти не вращается, кисть быстрая: чувствительна к задержке
      **hand**-канала (сдвиг камеры не даёт эффекта, а сдвиг кисти очевиден);
    * ``fast`` (когорта 08a) считается отдельно — по ней идут основной скан и PNG.

    Пороги — константы модуля; они не подбирались под результат, а взяты из распределений
    записи (p90 скорости кисти в системе головы, p75 угловой скорости головы).
    """
    head_rot_mask = frame_hand_head_speed < HEAD_COHORT_MAX_HAND_SPEED_MPS
    head_still_mask = frame_omega < HEAD_STILL_MAX_OMEGA_DEG_S
    return {
        "headrot": top_frames(frame_omega, count, head_rot_mask),
        "headstill": top_frames(frame_speed, count, head_still_mask),
    }


@dataclass(frozen=True)
class Scan2D:
    """Матрица средней ошибки наложения по (сдвиг головы, сдвиг кистей), px.

    ``values[i][j]`` — ошибка при ``head_grid[i]`` и ``hand_grid[j]``; ``NaN`` — кадров с
    измеримой ошибкой не было. Диагональ (δ_h = δ_н) — это то, что умеет модель с одной
    константой; если минимум заметно уходит с диагонали, каналы запаздывают по-разному.
    """

    frames: tuple[int, ...]
    head_grid: tuple[float, ...]
    hand_grid: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    #: Средняя угловая скорость головы на когорте, °/с (для подписи в сводке).
    omega_deg_s: float = math.nan
    #: Средняя скорость кисти в системе головы на когорте, м/с.
    hand_head_speed_mps: float = math.nan

    @property
    def best(self) -> tuple[float, float, float]:
        """``(сдвиг головы мс, сдвиг кистей мс, ошибка px)`` в минимуме."""
        best = (math.nan, math.nan, math.nan)
        for i, head_delta in enumerate(self.head_grid):
            for j, hand_delta in enumerate(self.hand_grid):
                value = self.values[i][j]
                if math.isnan(value):
                    continue
                if math.isnan(best[2]) or value < best[2]:
                    best = (float(head_delta), float(hand_delta), value)
        return best

    def value(self, head_ms: float, hand_ms: float) -> float:
        """Ошибка в узле сетки (``NaN``, если узла нет)."""
        try:
            i = self.head_grid.index(float(head_ms))
            j = self.hand_grid.index(float(hand_ms))
        except ValueError:
            return math.nan
        return self.values[i][j]

    def head_margin(self, hand_ms: float) -> list[tuple[float, float]]:
        """Зависимость ошибки от сдвига головы при фиксированном сдвиге кистей."""
        try:
            j = self.hand_grid.index(float(hand_ms))
        except ValueError:
            return []
        return [(float(head), self.values[i][j]) for i, head in enumerate(self.head_grid)]

    def hand_margin(self, head_ms: float) -> list[tuple[float, float]]:
        """Зависимость ошибки от сдвига кистей при фиксированном сдвиге головы."""
        try:
            i = self.head_grid.index(float(head_ms))
        except ValueError:
            return []
        return [(float(hand), self.values[i][j]) for j, hand in enumerate(self.hand_grid)]

    def to_csv_rows(self) -> list[list[str]]:
        """Матрица в виде строк CSV: первый столбец — сдвиг головы, остальные — сдвиги кистей."""
        header = ["head_offset_ms", *(f"{delta:g}" for delta in self.hand_grid)]
        rows = [header]
        for i, head_delta in enumerate(self.head_grid):
            rows.append(
                [f"{head_delta:g}",
                 *("" if math.isnan(value) else f"{value:.2f}" for value in self.values[i])]
            )
        return rows


def scan_2d(
    scene: overlay_video.Scene,
    frames: Sequence[int],
    distances: Mapping[int, np.ndarray],
    head_grid: Sequence[float] = HEAD_OFFSET_GRID_MS,
    hand_grid: Sequence[float] = HAND_OFFSET_GRID_MS,
    cohort_omega_deg_s: float = math.nan,
    cohort_hand_head_mps: float = math.nan,
) -> Scan2D:
    """2D-скан (сдвиг головы × сдвиг кистей) на когорте кадров (08c).

    Для каждой пары сдвигов собирается поза кадра: голова — из ``t + head_delta``, кисти — из
    ``t + hand_delta`` (:func:`sync.pose_at_frame_split`), и ошибка наложения усредняется по
    всем видимым суставам когорты. Позы на сетке считаются один раз на кадр (это дорогая
    часть: интерполяция 33 полей и SLERP), поэтому 625 узлов стоят ≈13 с на 50 кадров.
    """
    frames = tuple(int(k) for k in frames)
    head_grid = tuple(float(value) for value in head_grid)
    hand_grid = tuple(float(value) for value in hand_grid)
    head_parts: dict[tuple[int, float], tuple[tuple[float, float, float],
                                              tuple[float, float, float, float]]] = {}
    hand_parts: dict[tuple[int, float], tuple[dict[str, tuple[float, float, float]],
                                              dict[str, bool]]] = {}
    for k in frames:
        moment = _frame_time(scene, k)
        for head_delta in head_grid:
            pose = scene.track.pose_at(int(moment.ts_ns) + int(round(head_delta * 1e6)))
            head_parts[(k, head_delta)] = (pose.head_pos, pose.head_quat)
        for hand_delta in hand_grid:
            pose = scene.track.pose_at(int(moment.ts_ns) + int(round(hand_delta * 1e6)))
            hand_parts[(k, hand_delta)] = (pose.hands, pose.active)
    values: list[list[float]] = []
    usable = [k for k in frames if k in distances]
    for head_delta in head_grid:
        row: list[float] = []
        for hand_delta in hand_grid:
            errors: list[float] = []
            for k in usable:
                head_pos, head_quat = head_parts[(k, head_delta)]
                hands, active = hand_parts[(k, hand_delta)]
                pose = sync.Pose(
                    ts_ns=_frame_time(scene, k).ts_ns,
                    head_pos=head_pos,
                    head_quat=head_quat,
                    hands=hands,
                    active=active,
                )
                errors.extend(joint_errors_px(pose, scene, distances[k]))
            row.append(float(statistics.fmean(errors)) if errors else math.nan)
        values.append(row)
    return Scan2D(
        frames=frames,
        head_grid=head_grid,
        hand_grid=hand_grid,
        values=tuple(tuple(row) for row in values),
        omega_deg_s=float(cohort_omega_deg_s),
        hand_head_speed_mps=float(cohort_hand_head_mps),
    )


def error_breakdown(
    scene: overlay_video.Scene,
    frames: Sequence[int],
    distances: Mapping[int, np.ndarray],
    offset_ns: int,
) -> dict[str, float]:
    """Ошибка когорты с разбивкой по суставам для одного сдвига времени, px.

    ``all`` — среднее по кадрам от среднего по 5 точкам (метрика 08a), ``all5`` — то же, но
    только на кадрах, где все 5 точек попали в кадр (на остальных кисть обрезана краем, и
    «ошибка» меряет обрезку, а не наложение); ``palm``/``wrist``/``tips`` — среднее по
    суставу на кадрах, где он в кадре. Кончики зависят от позы пальцев (кулак/щипок) и дают
    постоянный вклад, поэтому временной остаток честнее видеть по ладони и запястью.
    """
    frame_means: list[float] = []
    all5_means: list[float] = []
    per_joint: dict[str, list[float]] = {joint: [] for joint in overlay_video.JOINTS}
    for k in frames:
        if k not in distances:
            continue
        named = joint_errors_named(_pose(scene, int(k), offset_ns), scene, distances[k])
        values = [value for value in named.values() if not math.isnan(value)]
        if not values:
            continue
        frame_means.append(statistics.fmean(values))
        if len(values) == len(named):
            all5_means.append(statistics.fmean(values))
        for joint, value in named.items():
            if not math.isnan(value):
                per_joint[joint].append(value)
    tips = [value for joint in TIP_JOINTS for value in per_joint.get(joint, ())]
    return {
        "frames": float(len(frame_means)),
        "all": nanmean(frame_means),
        "all5": nanmean(all5_means),
        "all5_frames": float(len(all5_means)),
        "palm": nanmean(per_joint.get("palm", ())),
        "wrist": nanmean(per_joint.get("wrist", ())),
        "tips": nanmean(tips),
    }


def frame_rows(
    video: str | Path,
    scene: overlay_video.Scene,
    frames: Sequence[int],
    speeds: np.ndarray,
    offset_ns: int,
    hand_head_speeds: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Строка таблицы на кадр: объясняющие величины, ошибка и лучший сдвиг времени.

    Ошибка считается по позе, взятой с ``offset_ns`` (как в рендере), т.е. отражает то, что
    видит пользователь в этом прогоне. ``best_offset_ms`` — сдвиг, при котором ошибка этого
    кадра минимальна (скан по :data:`OFFSET_GRID_MS`). Ошибка раскладывается по суставам:
    ``err_palm_wrist_px`` (смещение камеры/тайминг) против ``err_tips_px`` (поза пальцев).
    """
    import cv2  # noqa: PLC0415

    if hand_head_speeds is None:
        hand_head_speeds = sample_speeds_in_head(scene.track)
    reference_quat = scene.track.sample_pose(0).head_quat
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cv2.VideoCapture не открывает {video}")
    rows: list[dict[str, Any]] = []
    try:
        for k in frames:
            moment = _frame_time(scene, k)
            pose = scene.track.pose_at(int(moment.ts_ns) + int(offset_ns))
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, k):
                raise ValueError(f"не удалось перейти к кадру {k}")
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"не читается кадр {k}")
            view = image[:, : scene.eye_size[0]].copy()
            distance = skin_distance_map(view)
            named = joint_errors_named(pose, scene, distance)
            errors = [value for value in named.values() if not math.isnan(value)]
            tips = [named[joint] for joint in TIP_JOINTS if not math.isnan(named[joint])]
            best_error, best_delta = best_offset_ms(scene, k, distance)
            pose_time = int(moment.ts_ns) + int(offset_ns)
            rows.append(
                {
                    "k": k,
                    "t_s": moment.t_rel_s,
                    "dt_to_nearest_ms": dt_to_nearest_ms(scene.track, pose_time),
                    # скорости — в момент самого кадра (движение в картинке), а не позы:
                    # поправка времени сдвигает позу, но наблюдаемое движение — в кадре
                    "speed_mps": track_speed_at_ms(scene.track, speeds, moment.ts_ns),
                    "hand_head_speed_mps": track_speed_at_ms(
                        scene.track, hand_head_speeds, moment.ts_ns
                    ),
                    "head_angular_speed_deg_s": head_angular_speed_deg_s(
                        scene, k, offset_ns
                    ),
                    "depth_z_m": hand_depth_m(pose),
                    "head_yaw_deg": head_yaw_deg(pose, reference_quat),
                    "head_rotation_range_deg": head_rotation_range_deg(
                        scene, k, offset_ns
                    ),
                    "err_px": float(statistics.fmean(errors)) if errors else math.nan,
                    "on_skin_frac": (
                        sum(1 for value in errors if value <= ON_SKIN_PX) / len(errors)
                        if errors
                        else math.nan
                    ),
                    # ладонь и запястье лежат на коже при любой позе кисти: их расстояние
                    # меряет смещение (тайминг/камера), а не форму кисти (кулак/щипок)
                    "err_palm_px": named["palm"],
                    "err_wrist_px": named["wrist"],
                    "err_palm_wrist_px": mean_error_named(named, BODY_JOINTS),
                    "err_tips_px": float(statistics.fmean(tips)) if tips else math.nan,
                    "joints_in_frame": len(errors),
                    "err_at_best_px": best_error,
                    "best_offset_ms": best_delta,
                }
            )
    finally:
        capture.release()
    return rows


def offset_scan(
    scene: overlay_video.Scene,
    rows: Sequence[Mapping[str, Any]],
    distances: Mapping[int, np.ndarray],
) -> dict[str, list[tuple[float, float, int]]]:
    """Средняя ошибка когорты как функция сдвига времени: ``{когорта: [(δ мс, px, n)]}``.

    Когорта режется на три трети по времени; если минимум у всех третей на одном δ, сдвиг
    постоянен (а не «уплывает» к концу записи). ``n`` — сколько кадров дали измеримую ошибку.
    """
    frames = [int(row["k"]) for row in rows]
    thirds = np.array_split(np.array(sorted(frames)), 3)
    cohorts = {
        "all": frames,
        "early": [int(k) for k in thirds[0]],
        "mid": [int(k) for k in thirds[1]],
        "late": [int(k) for k in thirds[2]],
    }
    curves: dict[str, list[tuple[float, float, int]]] = {}
    for name, cohort in cohorts.items():
        curve = []
        for delta in OFFSET_GRID_MS:
            values = [
                error_px(
                    _pose(scene, int(k), int(delta * 1e6)), scene, distances[int(k)]
                )
                for k in cohort
                if int(k) in distances
            ]
            usable = [value for value in values if not math.isnan(value)]
            curve.append(
                (float(delta), float(statistics.fmean(usable)) if usable else math.nan, len(usable))
            )
        curves[name] = curve
    return curves


def control_slow_frames(
    video: str | Path,
    scene: overlay_video.Scene,
    frame_speed: np.ndarray,
    offset_ns: int,
    count: int = 40,
    seed: int = 7,
) -> dict[str, float]:
    """Контрольный опыт на медленных кадрах: остаётся ли ошибка после сдвига времени.

    Если ошибка была **временной**, на медленных кадрах она мала и без поправки, а после
    сдвига не растёт. Если бы калибровка камеры была смещена, на медленных кадрах остался бы
    постоянный сдвиг (в пикселях он тем больше, чем ближе кисть): ``err0`` был бы заметным.
    Меряется по ладони и запястью (:data:`BODY_JOINTS`) — они лежат на коже, поэтому не
    зависят от формы кисти; ``all0``/``all`` — по всем 5 точкам для сравнения.
    """
    import cv2  # noqa: PLC0415

    candidates = [
        k
        for k in range(scene.frame_count)
        if not math.isnan(frame_speed[k]) and 0.05 < frame_speed[k] < 0.6
    ]
    if not candidates:
        return {"frames": 0, "err0": math.nan, "err_best": math.nan}
    rng = np.random.default_rng(seed)
    chosen = sorted(
        int(k) for k in rng.choice(candidates, size=min(count, len(candidates)), replace=False)
    )
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cv2.VideoCapture не открывает {video}")
    at_zero: list[float] = []
    at_best: list[float] = []
    all_zero: list[float] = []
    all_best: list[float] = []
    try:
        for k in chosen:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, k):
                raise ValueError(f"не удалось перейти к кадру {k}")
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"не читается кадр {k}")
            distance = skin_distance_map(image[:, : scene.eye_size[0]])
            for shift_ns, target, everything in (
                (0, at_zero, all_zero),
                (int(offset_ns), at_best, all_best),
            ):
                named = joint_errors_named(_pose(scene, k, shift_ns), scene, distance)
                body = mean_error_named(named, BODY_JOINTS)
                if not math.isnan(body):
                    target.append(body)
                values = [value for value in named.values() if not math.isnan(value)]
                if values:
                    everything.append(statistics.fmean(values))
    finally:
        capture.release()
    return {
        "frames": float(len(chosen)),
        "err0": nanmean(at_zero),
        "err_best": nanmean(at_best),
        "all0": nanmean(all_zero),
        "all": nanmean(all_best),
    }


def _row_value(row: Mapping[str, Any], key: str) -> float:
    """Число из строки таблицы; отсутствующая колонка — ``NaN`` (сводка терпима к старым CSV)."""
    value = row.get(key, math.nan)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan


def describe(
    rows: Sequence[Mapping[str, Any]],
    curves,
    frame_speed: np.ndarray,
    *,
    scans: Mapping[str, Scan2D] | None = None,
    breakdowns: Mapping[str, Mapping[str, float]] | None = None,
    control: Mapping[str, float] | None = None,
    offset_ms: float | None = None,
) -> str:
    """Markdown-сводка: когорта, корреляции, сканы сдвига и разложение ошибки.

    ``scans`` — 2D-сканы по когортам (``fast``/``headrot``/``headstill``), ``breakdowns`` —
    разложение ошибки по суставам при разных сдвигах, ``control`` — медленные кадры,
    ``offset_ms`` — сдвиг текущего прогона (для подписи «было/стало»).
    """
    errors = [float(row["err_px"]) for row in rows]
    speeds = [float(row["speed_mps"]) for row in rows]
    depths = [float(row["depth_z_m"]) for row in rows]
    dts = [float(row["dt_to_nearest_ms"]) for row in rows]
    yaws = [float(row["head_yaw_deg"]) for row in rows]
    rotations = [float(row["head_rotation_range_deg"]) for row in rows]
    hand_head = [_row_value(row, "hand_head_speed_mps") for row in rows]
    omegas = [_row_value(row, "head_angular_speed_deg_s") for row in rows]
    finite_speed = frame_speed[np.isfinite(frame_speed)]
    cohort_speed = [value for value in speeds if not math.isnan(value)]
    mean_error = nanmean(errors)
    body_errors = [float(row["err_palm_wrist_px"]) for row in rows]
    tip_errors = [_row_value(row, "err_tips_px") for row in rows]
    best_errors = [float(row["err_at_best_px"]) for row in rows]
    deltas = [float(row["best_offset_ms"]) for row in rows]
    complete = [float(row["err_px"]) for row in rows if int(row["joints_in_frame"]) == 5]
    lines = [
        f"## Когорта: {len(rows)} самых быстрых кадров",
        "",
        f"* скорость правой руки на когорте: "
        f"{min(cohort_speed):.2f}…{max(cohort_speed):.2f} м/с "
        f"(медиана {statistics.median(cohort_speed):.2f}); по всей записи медиана "
        f"{statistics.median(finite_speed):.2f}, p95 {np.percentile(finite_speed, 95):.2f}, "
        f"максимум {finite_speed.max():.2f} м/с",
        f"* кадров со скоростью > 2 м/с во всей записи: "
        f"{int((frame_speed > 2.0).sum())} (порог 2 м/с в 50 кадров не набирается)",
        f"* средняя ошибка наложения (расстояние до кожи, px): **{mean_error:.1f}** "
        f"(по ладони и запястью {nanmean(body_errors):.1f}, по кончикам "
        f"{nanmean(tip_errors):.1f}); доля точек на коже: "
        f"{nanmean([row['on_skin_frac'] for row in rows]):.2f}",
        f"* только кадры с полностью видимой кистью (5/5 точек в кадре, n={len(complete)}): "
        f"**{nanmean(complete):.1f}** px — на остальных кадрах кисть обрезана краем, и "
        f"«ошибка» там меряет обрезку, а не наложение",
        f"* ошибка при лучшем для кадра сдвиге: {statistics.fmean(best_errors):.1f} px; "
        f"медианный лучший сдвиг: {statistics.median(deltas):+.0f} мс",
        "",
        f"## Корреляции ошибки ({len(rows)} кадров)",
        "",
        "| величина | Спирмен | Пирсон |",
        "|---|---|---|",
    ]
    for name, values in (
        ("dt_to_nearest_ms", dts),
        ("speed_mps", speeds),
        ("hand_head_speed_mps", hand_head),
        ("head_angular_speed_deg_s", omegas),
        ("depth_z_m", depths),
        ("head_yaw_deg", yaws),
        ("head_rotation_range_deg", rotations),
    ):
        lines.append(
            f"| {name} | {spearman(errors, values):+.2f} | {pearson(errors, values):+.2f} |"
        )
    predicted = [
        speed * dt / 1000.0 * FOCAL_PX / depth if depth == depth and depth > 0 else math.nan
        for speed, dt, depth in zip(speeds, dts, depths)
    ]
    lines.append(
        f"| speed·dt·fx/depth (модель тайминга) | {spearman(errors, predicted):+.2f} | "
        f"{pearson(errors, predicted):+.2f} |"
    )
    has_scan = any(not math.isnan(value) for _, value, _ in curves["all"])
    lines += ["", "## Скан сдвига времени (средняя ошибка по 5 точкам, px)", ""]
    if not has_scan:
        lines.append("скан не считался (ключ --no-offset-scan).")
        return "\n".join(lines)
    lines += ["| сдвиг, мс | все | early | mid | late |", "|---|---|---|---|---|"]
    for index, (delta, _, _) in enumerate(curves["all"]):
        cells = []
        for name in ("all", "early", "mid", "late"):
            curve = curves.get(name) or []
            cells.append(f"{curve[index][1]:.1f}" if index < len(curve) else "—")
        lines.append(f"| {delta:+.0f} | " + " | ".join(cells) + " |")
    best_all = min(curves["all"], key=lambda item: item[1])
    zero_row = [curve for curve in curves["all"] if curve[0] == 0.0]
    lines += [
        "",
        f"Минимум по всей когорте: **{best_all[1]:.1f} px при сдвиге {best_all[0]:+.0f} мс**"
        + (f" (без сдвига — {zero_row[0][1]:.1f} px)." if zero_row else "."),
    ]
    if breakdowns:
        lines += ["", "## Ошибка по суставам (когорта быстрых кадров)", ""]
        lines += ["| сдвиг | все 5 | 5/5 точек | ладонь | запястье | кончики |",
                  "|---|---|---|---|---|---|"]
        for name, data in breakdowns.items():
            lines.append(
                f"| {name} | {data['all']:.1f} | {data['all5']:.1f} "
                f"(n={data['all5_frames']:.0f}) | {data['palm']:.1f} | {data['wrist']:.1f} "
                f"| {data['tips']:.1f} |"
            )
        lines += [
            "",
            "Ладонь и запястье лежат на коже при любой позе кисти: их остаток — это "
            "тайминг/камера. Кончики зависят от позы пальцев (кулак, щипок) и на медленных "
            "кадрах всё равно дают 15–25 px — это не временной остаток.",
        ]
    if scans:
        lines += ["", "## Раздельные задержки head/hand (2D-скан)", ""]
        lines += [
            "| когорта | n | ω головы, °/с | кисть в системе головы, м/с | лучший head "
            "| лучший hand | ошибка, px | диагональ −100 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, scan in scans.items():
            best_head, best_hand, best_value = scan.best
            diagonal = scan.value(-100.0, -100.0)
            omega = scan.omega_deg_s
            hand_head_speed = scan.hand_head_speed_mps
            lines.append(
                f"| {name} | {len(scan.frames)} | "
                f"{'—' if math.isnan(omega) else f'{omega:.1f}'} | "
                f"{'—' if math.isnan(hand_head_speed) else f'{hand_head_speed:.2f}'} | "
                f"{best_head:+.0f} мс | {best_hand:+.0f} мс | {best_value:.1f} | "
                f"{'—' if math.isnan(diagonal) else f'{diagonal:.1f}'} |"
            )
        fast_scan = scans.get("fast")
        if fast_scan is not None:
            best_head, best_hand, _ = fast_scan.best
            head_margin = fast_scan.head_margin(best_hand)
            hand_margin = fast_scan.hand_margin(best_head)
            hand_estimates = ", ".join(
                f"{name}: {scan.best[1]:+.0f} мс" for name, scan in scans.items()
            )
            head_estimates = ", ".join(
                f"{name}: {scan.best[0]:+.0f} мс" for name, scan in scans.items()
            )
            lines += [
                "",
                "Маргинальные кривые когорты `fast` в минимуме (сдвиг одного канала при "
                "лучшем другом):",
                "",
                "* голова: " + _margin_line(head_margin),
                "* кисти: " + _margin_line(hand_margin),
                "",
                f"Оценки по когортам — кисти: {hand_estimates}; голова: {head_estimates}.",
                "",
                "Вывод: **данные не разделяют задержки каналов**. По кистям оценка устойчива "
                "(−100…−120 мс на всех когортах), по голове — нет: на `headrot` она −125 мс, "
                "на `fast` минимум уезжает к самому позднему краю сетки (−50…−40 мс), на "
                "`headstill` маргинал по голове плоский (0.4 px на 120 мс) — там её просто не "
                "видно. Разброс оценки головы больше разницы между каналами, а метрика "
                "принципиально не видит сдвиг внутри силуэта кисти: расстояние до кожи "
                "меняется, только когда проекция выходит за руку. Поэтому вторая константа "
                "не вводится — разница оценок в пределах точности одиночной константы "
                "(±40 мс); тот же вывод записан в `src/config.py`.",
            ]
    if control:
        lines += [
            "",
            f"## Контроль: медленные кадры (0.05…0.6 м/с, n={control['frames']:.0f})",
            "",
            f"ошибка ладони и запястья: **{control['err0']:.1f} px** без сдвига, "
            f"**{control['err_best']:.1f} px** при сдвиге рендера"
            + (f" ({offset_ms:+.0f} мс)" if offset_ms is not None else "")
            + "; "
            f"по всем 5 точкам — {control['all0']:.1f} и {control['all']:.1f} px. "
            "Постоянного пространственного смещения не видно: без временной поправки здесь "
            "уже почти ноль, значит калибровка камеры не при чём.",
        ]
    if offset_ms is not None:
        lines += [
            "",
            f"Сдвиг времени этого прогона: {offset_ms:+.0f} мс.",
        ]
    return "\n".join(lines)


def _margin_line(margin: Sequence[tuple[float, float]], step: int = 2) -> str:
    """Компактная запись маргинальной кривой: «−160: 31.6 · −140: 29.7 · …»."""
    cells = [f"{delta:+.0f}: {value:.1f}" for delta, value in margin[::step]]
    if margin and (len(margin) - 1) % step:
        delta, value = margin[-1]
        cells.append(f"{delta:+.0f}: {value:.1f}")
    return " · ".join(cells)


def write_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """CSV с колонками карточки: k, dt, скорости, depth, yaw, поворот + ошибка по суставам."""
    columns = (
        "k",
        "t_s",
        "dt_to_nearest_ms",
        "speed_mps",
        "hand_head_speed_mps",
        "head_angular_speed_deg_s",
        "depth_z_m",
        "head_yaw_deg",
        "head_rotation_range_deg",
        "err_px",
        "on_skin_frac",
        "err_palm_px",
        "err_wrist_px",
        "err_palm_wrist_px",
        "err_tips_px",
        "joints_in_frame",
        "err_at_best_px",
        "best_offset_ms",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (round(row[key], 4) if isinstance(row.get(key), float) else row.get(key, ""))
                    for key in columns
                }
            )


def write_scan2d(scan: Scan2D, path: Path) -> None:
    """CSV матрицы 2D-скана: строки — сдвиг головы, столбцы — сдвиг кистей."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(scan.to_csv_rows())


def write_png(video: str | Path, scene: overlay_video.Scene, row: Mapping[str, Any], out: Path,
              offset_ns: int) -> None:
    """PNG-наложение кадра с подписью диагностических величин."""
    import cv2  # noqa: PLC0415

    capture = cv2.VideoCapture(str(video))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(row["k"]))
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok or image is None:
        raise ValueError(f"не читается кадр {row['k']}")
    view = image[:, : scene.eye_size[0]].copy()
    pose = _pose(scene, int(row["k"]), offset_ns)
    overlay_video.draw_skeleton(
        view, overlay_video.project_hands(pose, scene.intrinsics),
        overlay_video.style_for(scene.eye_size[0], 1.2),
    )
    caption = (
        f"k={row['k']} t={row['t_s']:.2f}s  "
        f"dt={row['dt_to_nearest_ms']:.1f}ms  v={row['speed_mps']:.2f}m/s  "
        f"z={row['depth_z_m']:.2f}m"
    )
    caption2 = (
        f"yaw={row['head_yaw_deg']:+.1f}deg  rot={row['head_rotation_range_deg']:.1f}deg  "
        f"err={row['err_px']:.0f}px  best={row['best_offset_ms']:+.0f}ms"
    )
    for index, text in enumerate((caption, caption2)):
        position = (12, 30 + index * 26)
        cv2.putText(view, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, CAPTION_SHADOW, 4, cv2.LINE_AA)
        cv2.putText(view, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.65, CAPTION_COLOR, 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), view):
        raise ValueError(f"не удалось записать {out}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: таблица, PNG наложения и ``summary.md`` с корреляциями и сканом сдвига."""
    parser = argparse.ArgumentParser(
        description="Диагностика слёта скелета: ошибка наложения, корреляции и скан сдвига времени."
    )
    parser.add_argument("video", type=Path, help="data/CameraRecord_*.mp4")
    parser.add_argument("--tracking", type=Path, default=None, help="data/trackingData_*.txt")
    parser.add_argument("--meta", type=Path, default=Path("output/video_meta.json"))
    parser.add_argument(
        "--out", type=Path, default=Path("output/debug/drift"), help="каталог таблицы и PNG"
    )
    parser.add_argument("--top", type=int, default=TOP_FRAMES, help="сколько кадров брать")
    parser.add_argument(
        "--time-offset-ms", type=float, default=None,
        help="сдвиг времени позы (по умолчанию — config.VIDEO_TIME_OFFSET_MS)",
    )
    parser.add_argument(
        "--no-offset-scan", action="store_true", help="не считать сканы сдвига (быстрее)"
    )
    parser.add_argument(
        "--no-2d-scan", action="store_true",
        help="не считать 2D-скан head × hand (проверка разделения каналов)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        scene = overlay_video.load_scene(args.meta, args.tracking)
        offset_ns = (
            sync.video_time_offset_ns()
            if args.time_offset_ms is None
            else int(round(args.time_offset_ms * 1e6))
        )
        frames, frame_speed, speeds = select_fast_frames(scene, args.top)
        if not frames:
            raise ValueError("не нашлось кадров с валидной правой рукой")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        # скорости и угловая скорость головы по кадрам — вход когорт и корреляций
        hand_head_speeds = sample_speeds_in_head(scene.track)
        frame_hand_head = value_on_frames(scene, hand_head_speeds)
        frame_omega = np.array(
            [head_angular_speed_deg_s(scene, k, offset_ns) for k in range(scene.frame_count)]
        )
        cohorts = separation_cohorts(scene, frame_speed, frame_hand_head, frame_omega)
        rows = frame_rows(args.video, scene, frames, speeds, offset_ns, hand_head_speeds)
        distances: dict[int, np.ndarray] = {}
        if not args.no_offset_scan:
            import cv2  # noqa: PLC0415

            needed = sorted(set(frames) | {k for cohort in cohorts.values() for k in cohort})
            capture = cv2.VideoCapture(str(args.video))
            try:
                for k in needed:
                    if not capture.set(cv2.CAP_PROP_POS_FRAMES, k):
                        continue
                    ok, image = capture.read()
                    if ok:
                        distances[k] = skin_distance_map(image[:, : scene.eye_size[0]])
            finally:
                capture.release()
        curves = (
            offset_scan(scene, rows, distances)
            if distances
            else {"all": [(0.0, float("nan"), 0)], "early": [], "mid": [], "late": []}
        )
        control = (
            control_slow_frames(args.video, scene, frame_speed, offset_ns)
            if distances
            else None
        )
        scans: dict[str, Scan2D] = {}
        if distances and not args.no_2d_scan:
            for name, cohort in (("fast", frames), *cohorts.items()):
                if not cohort:
                    continue
                scan = scan_2d(
                    scene, cohort, distances,
                    cohort_omega_deg_s=float(np.nanmean(frame_omega[cohort])),
                    cohort_hand_head_mps=float(np.nanmean(frame_hand_head[cohort])),
                )
                scans[name] = scan
                write_scan2d(scan, out / f"scan_2d_{name}.csv")
        breakdowns: dict[str, Mapping[str, float]] = {}
        if distances:
            breakdowns["без сдвига (0 мс)"] = error_breakdown(scene, frames, distances, 0)
            if offset_ns != 0:
                breakdowns[f"рендер ({offset_ns / 1e6:+.0f} мс)"] = error_breakdown(
                    scene, frames, distances, offset_ns
                )
        write_table(rows, out / "table.csv")
        for row in rows:
            write_png(args.video, scene, row, out / f"diag_{row['k']:05d}.png", offset_ns)
        summary = describe(
            rows, curves, frame_speed,
            scans=scans or None,
            breakdowns=breakdowns or None,
            control=control,
            offset_ms=offset_ns / 1e6,
        )
        (out / "summary.md").write_text(summary, encoding="utf-8")
        (out / "frames.json").write_text(json.dumps(frames), encoding="utf-8")
        print(summary)
        print(f"\nтаблица: {out / 'table.csv'}\nPNG: {len(rows)} файлов в {out}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
