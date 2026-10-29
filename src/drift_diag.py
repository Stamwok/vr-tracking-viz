"""drift_diag.py — диагностика «слёта» скелета с кисти на быстрых кадрах (доработка 08a).

Зачем: при просмотре `overlay_skeleton.mp4` видно, что на быстрых движениях скелет уезжает
с кисти на десятки-сотни пикселей. Чтобы понять причину, этот инструмент собирает по кадрам
объясняющие величины и меряет **фактическую ошибку наложения по картинке** — без CV-моделей
распознавания рук (они запрещены решением §7.2 CLAUDE.md и не нужны):

1. **Ошибка, px** — для 5 точек правой руки считается расстояние до ближайшего пикселя кожи
   (маска кожи по HSV + морфология). На кадрах, где скелет лежит на кисти, ошибка 0–16 px;
   когда скелет уезжает на стол — десятки-сотни. Ограничение сверху — 200 px. Метрика меряет
   только компоненту смещения, перпендикулярную коже руки: смещение вдоль предплечья остаётся
   «на коже» и недооценивается (см. docs/NOTES.md, 2026-10-29, доработка 08a).
2. **Объясняющие величины**: `dt_to_nearest_ms` (расстояние до ближайшего сэмпла трекинга),
   `speed_mps` (скорость правой руки по сырому треку, 90 Гц), `depth_z_m` (глубина кисти в
   системе камеры), `head_yaw_deg` (рысканье головы относительно первого сэмпла),
   `head_rotation_range_deg` (размах поворота головы за ±50 мс).
3. **Скан смещения времени** — та же ошибка при позе, взятой со сдвигом δ, усреднённая по
   когорте. Минимум этого скана и есть оценка задержки видео относительно трекинга; скан
   считается по трём третям записи, чтобы проверить, что сдвиг постоянен.

Выход: `table.csv` (по строке на кадр), PNG с наложением и подписями, `summary.md` с
корреляциями (Спирмен и Пирсон) и выводом.

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
from bisect import bisect_left
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

#: Полуширина окна для размаха поворота головы, мс.
ROTATION_WINDOW_MS = 50

#: Сетка смещений времени для скана, мс (группами по 20 мс).
OFFSET_GRID_MS = tuple(range(-240, 81, 20))

#: Цвета (BGR) для подписи в PNG.
CAPTION_COLOR = (255, 255, 255)
CAPTION_SHADOW = (0, 0, 0)

#: Фокус левого глаза, px (из header: ``fy`` полного SBS-кадра) — для оценки «скорость ×
#: время → пиксели» в корреляционной таблице.
FOCAL_PX = 686.868


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


def joint_errors_px(
    pose: sync.Pose, scene: overlay_video.Scene, distance: np.ndarray, side: str = "right"
) -> list[float]:
    """Расстояния проекций 5 точек руки до ближайшей кожи, px (``NaN`` — вне кадра)."""
    pixels = overlay_video.project_hands(pose, scene.intrinsics, "left")
    height, width = distance.shape
    values: list[float] = []
    for joint in overlay_video.JOINTS:
        point = pixels[side][joint]
        if not point.visible:
            continue
        u, v = int(round(point.u)), int(round(point.v))
        if not (0 <= u < width and 0 <= v < height):
            continue
        values.append(min(float(distance[v, u]), ERROR_CAP_PX))
    return values


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
    frame_speed = np.full(scene.frame_count, np.nan)
    for k in range(scene.frame_count):
        moment = _frame_time(scene, k)
        frame_speed[k] = speeds[nearest_sample(scene.track, moment.ts_ns)]
    order = np.argsort(-np.nan_to_num(frame_speed, nan=-1.0))
    selected = sorted(int(k) for k in order[:count] if not math.isnan(frame_speed[k]))
    return selected, frame_speed, speeds


def frame_rows(
    video: str | Path,
    scene: overlay_video.Scene,
    frames: Sequence[int],
    speeds: np.ndarray,
    offset_ns: int,
) -> list[dict[str, Any]]:
    """Строка таблицы на кадр: объясняющие величины, ошибка и лучший сдвиг времени.

    Ошибка считается по позе, взятой с ``offset_ns`` (как в рендере), т.е. отражает то, что
    видит пользователь в этом прогоне. ``best_offset_ms`` — сдвиг, при котором ошибка этого
    кадра минимальна (скан по :data:`OFFSET_GRID_MS`).
    """
    import cv2  # noqa: PLC0415

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
            errors = joint_errors_px(pose, scene, distance)
            body_errors = [value for value, name in zip(errors, overlay_video.JOINTS)
                           if name in ("palm", "wrist")]
            best_error, best_delta = best_offset_ms(scene, k, distance)
            pose_time = int(moment.ts_ns) + int(offset_ns)
            rows.append(
                {
                    "k": k,
                    "t_s": moment.t_rel_s,
                    "dt_to_nearest_ms": dt_to_nearest_ms(scene.track, pose_time),
                    # скорость — в момент самого кадра (движение в картинке), а не позы:
                    # поправка времени сдвигает позу, но наблюдаемое движение — в кадре
                    "speed_mps": track_speed_at_ms(scene.track, speeds, moment.ts_ns),
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
                    # уже, чем ошибка по 5 точкам: ладонь и запястье лежат на коже, поэтому
                    # их расстояние меряет смещение, а не форму кисти (кулак/открытая ладонь)
                    "err_palm_wrist_px": nanmean(body_errors),
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
    speeds: np.ndarray,
    best_delta_ms: float,
    count: int = 40,
    seed: int = 7,
) -> dict[str, float]:
    """Контрольный опыт на медленных кадрах: остаётся ли ошибка после сдвига времени.

    Если ошибка была **временной**, на медленных кадрах она мала и без поправки, а после
    сдвига не растёт. Если бы калибровка камеры была смещена, на медленных кадрах остался бы
    постоянный сдвиг (в пикселях он тем больше, чем ближе кисть): ``err0`` был бы заметным.
    Меряется по ладони и запястью — они лежат на коже, поэтому не зависят от формы кисти.
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
    try:
        for k in chosen:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, k):
                raise ValueError(f"не удалось перейти к кадру {k}")
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"не читается кадр {k}")
            distance = skin_distance_map(image[:, : scene.eye_size[0]])
            for offset_ns, target in ((0, at_zero), (int(best_delta_ms * 1e6), at_best)):
                errors = joint_errors_px(_pose(scene, k, offset_ns), scene, distance)
                body = [
                    value
                    for value, name in zip(errors, overlay_video.JOINTS)
                    if name in ("palm", "wrist")
                ]
                target.extend(body)
    finally:
        capture.release()
    return {"frames": len(chosen), "err0": nanmean(at_zero), "err_best": nanmean(at_best)}


def describe(rows: Sequence[Mapping[str, Any]], curves, frame_speed: np.ndarray) -> str:
    """Markdown-сводка: когорта, ошибка, корреляции и скан сдвига времени."""
    errors = [float(row["err_px"]) for row in rows]
    speeds = [float(row["speed_mps"]) for row in rows]
    depths = [float(row["depth_z_m"]) for row in rows]
    dts = [float(row["dt_to_nearest_ms"]) for row in rows]
    yaws = [float(row["head_yaw_deg"]) for row in rows]
    rotations = [float(row["head_rotation_range_deg"]) for row in rows]
    finite_speed = frame_speed[np.isfinite(frame_speed)]
    cohort_speed = [value for value in speeds if not math.isnan(value)]
    mean_error = nanmean(errors)
    body_errors = [float(row["err_palm_wrist_px"]) for row in rows]
    best_errors = [float(row["err_at_best_px"]) for row in rows]
    deltas = [float(row["best_offset_ms"]) for row in rows]
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
        f"(по ладони и запястью {nanmean(body_errors):.1f}); "
        f"доля точек на коже: {nanmean([row['on_skin_frac'] for row in rows]):.2f}",
        f"* ошибка при лучшем для кадра сдвиге: {statistics.fmean(best_errors):.1f} px; "
        f"медианный лучший сдвиг: {statistics.median(deltas):+.0f} мс",
        "",
        "## Корреляции ошибки (50 кадров)",
        "",
        "| величина | Спирмен | Пирсон |",
        "|---|---|---|",
    ]
    for name, values in (
        ("dt_to_nearest_ms", dts),
        ("speed_mps", speeds),
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
    lines += ["", "## Скан сдвига времени (средняя ошибка, px)", ""]
    header = "| когорта | " + " | ".join(f"{delta:+.0f} мс" for delta, _, _ in curves["all"]) + " |"
    lines += [header, "|" + "---|" * (len(curves["all"]) + 1)]
    for name, curve in curves.items():
        lines.append(
            f"| {name} | " + " | ".join(f"{value:.1f}" for _, value, _ in curve) + " |"
        )
    best_all = min(curves["all"], key=lambda item: item[1])
    lines += [
        "",
        f"Минимум по всей когорте: **{best_all[1]:.1f} px при сдвиге {best_all[0]:+.0f} мс** "
        f"(без сдвига — {curves['all'][[d for d, _, _ in curves['all']].index(0.0)][1]:.1f} px).",
    ]
    return "\n".join(lines)


def write_table(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """CSV с колонками карточки: k, dt, speed, depth, yaw, rotation + ошибка."""
    columns = (
        "k",
        "t_s",
        "dt_to_nearest_ms",
        "speed_mps",
        "depth_z_m",
        "head_yaw_deg",
        "head_rotation_range_deg",
        "err_px",
        "on_skin_frac",
        "err_palm_wrist_px",
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
                    key: (round(value, 4) if isinstance(value, float) else value)
                    for key, value in row.items()
                }
            )


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
        "--no-offset-scan", action="store_true", help="не считать скан сдвига (быстрее)"
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
        rows = frame_rows(args.video, scene, frames, speeds, offset_ns)
        distances = {}
        if not args.no_offset_scan:
            import cv2  # noqa: PLC0415

            capture = cv2.VideoCapture(str(args.video))
            try:
                for k in frames:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, k)
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
        best_all = min(curves["all"], key=lambda item: item[1]) if curves["all"] else (0.0, 0.0, 0)
        control = (
            control_slow_frames(args.video, scene, frame_speed, speeds, best_all[0])
            if curves["all"]
            else {"frames": 0, "err0": math.nan, "err_best": math.nan}
        )
        write_table(rows, out / "table.csv")
        for row in rows:
            write_png(args.video, scene, row, out / f"diag_{row['k']:05d}.png", offset_ns)
        summary = describe(rows, curves, frame_speed)
        summary += (
            f"\n\n## Контроль: медленные кадры (0.05…0.6 м/с, n={control['frames']})\n\n"
            f"ошибка ладони и запястья: **{control['err0']:.1f} px** без сдвига, "
            f"**{control['err_best']:.1f} px** при лучшем сдвиге {best_all[0]:+.0f} мс. "
            "Постоянного пространственного смещения не видно: без временной поправки здесь уже "
            "почти ноль, значит калибровка камеры не при чём.\n"
            f"\nСдвиг времени в этом прогоне: {offset_ns / 1e6:+.0f} мс "
            f"({'config.VIDEO_TIME_OFFSET_MS' if args.time_offset_ms is None else 'задан вручную'}).\n"
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
