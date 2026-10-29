"""overlay_video.py — покадровый рендер скелета кистей поверх видео камеры шлема (задача 08a).

Что делает кадр: берёт левую половину SBS-кадра (левый глаз, 1080×810), для номера кадра
``k`` считает время (:func:`sync.frame_time`, 04a) и позу на этот момент
(:meth:`sync.PoseTrack.pose_at`: линейная интерполяция позиций + SLERP ориентации, 04b/04c),
переводит 10 точек кистей в координаты камеры (:func:`camera_model.world_to_camera_matrix`
с константами из :mod:`src.config`) и в пиксели (:func:`camera_model.project` по per-eye
интринсикам, 05a–05c), затем рисует по 5 точек на руку и линии «ладонь → запястье /
большой / указательный / средний». Левая и правая руки — разные цвета.

Границы задачи: след (trail), HUD и маркер головы — задача 08b. Поза головы = поза камеры
(камера в голове, ``T_head_cam`` из ``config.py``), поэтому «маркер головы» имеет смысл
только как элемент HUD, а не как точка сцены.

Почему на входе сырой файл трекинга (JSON Lines), а не ``tracking.csv``: для синхронизации
нужен ``header.timeStampNs`` — метка первого кадра видео (04a). В ``tracking.csv`` её нет
(там только ``t_ns`` сэмплов), поэтому по CSV нельзя восстановить время кадра точно.
CSV остаётся входом графиков (09), а рендер читает ``data/trackingData_*.txt`` потоково
(:func:`sync.load_poses`, ≈6 МБ в памяти, ≈3 с на 21 787 сэмплов).

Задержка видеотракта. Поза кадра берётся не в ``frame_time(k)``, а со сдвигом
``config.VIDEO_TIME_OFFSET_MS`` (:func:`sync.pose_at_frame`): кадр показывает сцену раньше
своей метки, и без поправки скелет уезжает с кисти на быстрых движениях (диагностика —
``src/drift_diag.py``, 29.10). Сдвиг применяется и в проверке калибровки (``calib_search``),
поэтому метрика 07a описывает именно то, что рисует рендер. ``--time-offset-ms`` перекрывает
константу для сравнений.

Кодеки. Пишем H.264 (``avc1``), при недоступности — MPEG-4 part 2 (``mp4v``): в сборке
OpenCV без OpenH264 первый вариант просто не откроется. Фактический файл после рендера
проверяется :func:`probe_output` (открывается заново, число кадров и размер сверяются).

Запуск (весь клип, 7279 кадров, ≈242.66 с):
    python src/overlay_video.py data/CameraRecord_20260505_165740.mp4 \\
        --tracking data/trackingData_20260505_165740.txt --out output/overlay_skeleton.mp4

Короткий фрагмент и PNG-проверка на 5 отладочных кадрах 07a:
    python src/overlay_video.py data/CameraRecord_*.mp4 --start 1790 --end 1810 --scale 1.5
    python src/overlay_video.py data/CameraRecord_*.mp4 --stills output/debug/overlay
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import camera_model, sync
from src.parse_tracking import HAND_JOINTS, read_header, timestamp_range

#: Имена 5 суставов на руку в порядке отрисовки (те же, что в ``tracking.csv``).
JOINTS = tuple(name for name, _ in HAND_JOINTS)

#: Кости скелета: ладонь → запястье и три «пальцевые» точки (ТЗ: 3–5 точек на руку).
HAND_CHAINS = (
    ("palm", "wrist"),
    ("palm", "thumb"),
    ("palm", "index"),
    ("palm", "middle"),
)

#: Ширина кадра глаза, к которой отнесены радиус и толщина по умолчанию.
REFERENCE_WIDTH = 1080

#: Радиус точки и толщина линии при ``scale = 1`` на кадре 1080×810, px.
POINT_RADIUS_PX = 4.0
LINE_THICKNESS_PX = 2.0

#: Цвета рук (BGR): левая — голубая, правая — красная. Те же цвета, что в разметке
#: эталона 07a (``calib_frames``/``calib_search``), поэтому PNG наложения и разметка
#: читаются одинаково; между собой и с фоном сцены цвета различимы.
COLORS: Mapping[str, tuple[int, int, int]] = {"left": (255, 140, 0), "right": (0, 0, 255)}

#: Тёмная обводка точки и её толщина: цвет не теряется на светлых предметах (губки, бумага).
OUTLINE_COLOR = (0, 0, 0)
OUTLINE_THICKNESS = 1

#: Кодеки для :class:`cv2.VideoWriter` в порядке предпочтения: H.264, затем MPEG-4 part 2.
FOURCC_CANDIDATES = ("avc1", "mp4v")

#: Предел координаты при отрисовке: cv2 ждёт int32, а проекция «видимой» точки у края кадра
#: может уйти на тысячи пикселей. Обрезку отрезка по краю кадра cv2 делает сам, поэтому
#: координата не отбрасывается, а только ограничивается по модулю.
COORD_LIMIT = 1_000_000

#: Кадры 07a по умолчанию для режима ``--stills`` (совпадают с ``camera_model.DEBUG_FRAMES``).
DEBUG_FRAMES = (30, 1800, 3600, 5400, 7200)


@dataclass(frozen=True)
class Style:
    """Толщины отрисовки в пикселях уже для конкретного кадра (масштаб применён)."""

    point_radius: int
    line_thickness: int

    def __post_init__(self) -> None:
        for name in ("point_radius", "line_thickness"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Style.{name}: ожидалось целое число, получено {value!r}")
            if value < 1:
                raise ValueError(f"Style.{name}: должно быть ≥ 1, получено {value!r}")


def style_for(width: int, scale: float = 1.0) -> Style:
    """Стиль для кадра шириной ``width``: радиус/толщина растут вместе с кадром.

    ``scale`` — пользовательский множитель (CLI ``--scale``): ``1.0`` — значения по
    умолчанию для 1080×810, ``2.0`` — вдвое толще. Итог округляется и не может быть
    меньше 1 px, иначе на уменьшенном кадре точки исчезают.
    """
    if isinstance(width, bool) or not isinstance(width, int):
        raise ValueError(f"width: ожидалось целое число, получено {width!r}")
    if width <= 0:
        raise ValueError(f"width: должно быть > 0, получено {width!r}")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(f"scale: ожидалось число, получено {scale!r}")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale: должно быть конечным и > 0, получено {scale!r}")
    factor = width / REFERENCE_WIDTH * float(scale)
    return Style(
        point_radius=max(1, int(round(POINT_RADIUS_PX * factor))),
        line_thickness=max(1, int(round(LINE_THICKNESS_PX * factor))),
    )


def frame_range(meta: Mapping[str, Any], start: int = 0, end: int | None = None) -> tuple[int, int]:
    """Диапазон кадров для рендера: ``(start, end)`` включительно, проверенный по метаданным.

    ``end=None`` — до последнего кадра видео. Границы — номера кадров 0…frame_count−1;
    ``start > end``, выход за пределы и нецелые значения — ``ValueError`` (bool тоже).
    """
    count = meta.get("frame_count") if isinstance(meta, Mapping) else None
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"meta.frame_count: ожидалось целое > 0, получено {count!r}")
    for name, value in (("start", start), ("end", end)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{name}: ожидался номер кадра (int), получено {value!r}")
    last = count - 1 if end is None else end
    if not 0 <= start <= last <= count - 1:
        raise ValueError(
            f"кадры {start}…{last}: ожидалось 0 ≤ start ≤ end ≤ {count - 1} "
            f"(в видео {count} кадров)"
        )
    return start, last


def parse_frames(text: str) -> tuple[int, ...]:
    """Разобрать список кадров ``"30, 1800, 3600"`` в кортеж int (для ``--frames``)."""
    if not isinstance(text, str):
        raise ValueError(f"frames: ожидалась строка, получено {text!r}")
    frames = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            frames.append(int(chunk))
        except ValueError:
            raise ValueError(f"frames: {chunk!r} — не номер кадра") from None
    if not frames:
        raise ValueError("frames: пустой список")
    if len(set(frames)) != len(frames):
        raise ValueError(f"frames: повторяющиеся кадры в {text!r}")
    return tuple(frames)


@dataclass(frozen=True)
class Scene:
    """Всё, что нужно рендеру: метаданные видео, синхронизация и позы трекинга.

    Собирается один раз (:func:`load_scene`): загрузка трекинга занимает ≈3 с, а вызов
    :func:`render_video`/`render_stills` её уже не повторяет.
    """

    meta: dict
    header_ns: int
    tracking_end_ns: int
    track: sync.PoseTrack
    intrinsics: camera_model.Intrinsics
    #: Поправка времени кадра (нс): поза берётся из ``frame_time(k).ts_ns + time_offset_ns``.
    #: Значение — :func:`sync.video_time_offset_ns` (``config.VIDEO_TIME_OFFSET_MS``).
    time_offset_ns: int

    @property
    def frame_count(self) -> int:
        """Сколько кадров в видео (из ``video_meta.json``)."""
        return int(self.meta["frame_count"])

    @property
    def duration_s(self) -> float:
        """Длительность видео в секундах (из ``video_meta.json``)."""
        return float(self.meta["duration"])

    @property
    def fps(self) -> float:
        """Кадров в секунду для записи: ``frame_count / duration`` (29.9976…).

        Не номинальные 30: при записи 30 fps контейнер дал бы 7279/30 = 242.63 с вместо
        242.661 с, то есть видео «сжалось» бы на 30 мс.
        """
        return self.frame_count / self.duration_s

    @property
    def eye_size(self) -> tuple[int, int]:
        """Размер одного глаза SBS-пары (``1080, 810``) — размер выходного кадра."""
        return self.intrinsics.eye_w, self.intrinsics.eye_h


def load_scene(
    meta_path: str | Path = "output/video_meta.json",
    tracking_path: str | Path | None = None,
    *,
    data_dir: str | Path = "data",
) -> Scene:
    """Прочитать метаданные видео и трекинг, проверить согласованность.

    ``tracking_path=None`` — единственный ``data/trackingData_*.txt`` (иначе ``ValueError``:
    молча выбрать один из нескольких файлов значило бы отрендерить не ту запись).
    Размер кадра из ``video_meta.json`` передаётся в :meth:`Intrinsics.from_header`, поэтому
    расхождение интринсиков с видео (не тот файл, другая раскладка) падает сразу.
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path}: ожидался JSON-объект video_meta")
    frame_count = meta.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError(f"{meta_path}: frame_count должен быть целым > 0")
    duration = meta.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError(f"{meta_path}: duration должен быть числом > 0")

    tracking = Path(tracking_path) if tracking_path is not None else find_tracking(data_dir)
    if not tracking.is_file():
        raise FileNotFoundError(f"нет файла трекинга: {tracking}")

    frame_size = None
    if isinstance(meta.get("width"), int) and isinstance(meta.get("height"), int):
        frame_size = (meta["width"], meta["height"])
    header = read_header(tracking)
    intrinsics = camera_model.Intrinsics.from_header(header, frame_size)
    if frame_size is not None and (intrinsics.full_w, intrinsics.full_h) != frame_size:
        raise ValueError("интринсики не описывают полный кадр видео")
    return Scene(
        meta=meta,
        header_ns=int(header["timeStampNs"]),
        tracking_end_ns=int(timestamp_range(tracking)[1]),
        track=sync.load_poses(tracking),
        intrinsics=intrinsics,
        time_offset_ns=sync.video_time_offset_ns(),
    )


def find_tracking(data_dir: str | Path = "data") -> Path:
    """Единственный файл ``trackingData_*.txt`` в ``data_dir`` (иначе понятная ошибка)."""
    candidates = sorted(Path(data_dir).glob("trackingData_*.txt"))
    if not candidates:
        raise ValueError(f"в {data_dir}/ нет trackingData_*.txt — укажите --tracking")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"в {data_dir}/ несколько файлов трекинга ({names}) — укажите --tracking")
    return candidates[0]


def project_hands(
    pose: sync.Pose,
    intrinsics: camera_model.Intrinsics,
    side: str = "left",
) -> dict[str, dict[str, camera_model.ProjectedPoint]]:
    """10 точек кистей кадра в пикселях глаза: ``side → joint → ProjectedPoint``.

    Матрица ``мир → камера`` считается один раз на кадр (поза головы + константы
    ``src/config.py``), дальше каждая точка переводится той же матрицей. Точка за камерой,
    ближе 1 см или с ``NaN`` («нет данных» у неактивной руки) даёт ``visible = False`` —
    рендер такую точку и линии к ней пропускает.
    """
    eye = camera_model.eye_intrinsics(intrinsics, side)
    matrix = camera_model.world_to_camera_matrix(pose)
    result: dict[str, dict[str, camera_model.ProjectedPoint]] = {}
    for hand in ("left", "right"):
        pixels: dict[str, camera_model.ProjectedPoint] = {}
        for joint in JOINTS:
            world = pose.hands[f"{hand}_{joint}"]
            if not all(math.isfinite(value) for value in world):
                pixels[joint] = camera_model.ProjectedPoint(math.nan, math.nan, False)
                continue
            pixels[joint] = camera_model.project(
                camera_model.transform_point(matrix, world), eye
            )
        result[hand] = pixels
    return result


def _coord(value: float) -> int:
    """Координата для cv2: целое, ограниченное :data:`COORD_LIMIT` (защита от int32)."""
    if not math.isfinite(value):
        raise ValueError(f"координата: нефинитное значение {value!r}")
    return max(-COORD_LIMIT, min(COORD_LIMIT, int(round(value))))


def draw_skeleton(
    image: Any,
    pixels: Mapping[str, Mapping[str, camera_model.ProjectedPoint]],
    style: Style,
    *,
    colors: Mapping[str, tuple[int, int, int]] = COLORS,
) -> int:
    """Нарисовать скелет на кадре (in place) и вернуть число видимых точек.

    Порядок: сначала линии (под точками), потом точки. Пара, в которой хотя бы одна точка
    невидима, линией не соединяется — иначе отрезок «прибил» бы руку к точке за камерой.
    Точки, вышедшие за край кадра, всё равно рисуются: cv2 обрежет круг/отрезок по границе,
    и видимая часть кисти останется на месте.
    """
    import cv2  # noqa: PLC0415 — локальный импорт: чистые функции модуля работают без OpenCV

    if not isinstance(style, Style):
        raise ValueError(f"style: ожидался Style, получено {style!r}")
    if image is None or getattr(image, "ndim", 0) != 3:
        raise ValueError("image: ожидался BGR-кадр (H×W×3)")
    drawn = 0
    for side in ("left", "right"):
        color = colors[side]
        joints = pixels.get(side)
        if joints is None:
            raise ValueError(f"pixels: нет руки {side!r}")
        for start, end in HAND_CHAINS:
            first, second = joints[start], joints[end]
            if not (first.visible and second.visible):
                continue
            cv2.line(
                image,
                (_coord(first.u), _coord(first.v)),
                (_coord(second.u), _coord(second.v)),
                color,
                style.line_thickness,
                cv2.LINE_AA,
            )
        for joint in JOINTS:
            point = joints[joint]
            if not point.visible:
                continue
            center = (_coord(point.u), _coord(point.v))
            cv2.circle(image, center, style.point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(
                image, center, style.point_radius, OUTLINE_COLOR, OUTLINE_THICKNESS, cv2.LINE_AA
            )
            drawn += 1
    return drawn


def _eye_view(image: Any, scene: Scene) -> Any:
    """Левая половина SBS-кадра (левый глаз) — копия под рисование.

    Размер проверяется до рисования: если видео не то (или не SBS), координаты левого
    глаза недействительны, и молча нарисовать «съехавший» скелет хуже, чем упасть.
    """
    height, width = image.shape[:2]
    expected = (scene.intrinsics.full_w, scene.intrinsics.full_h)
    if (width, height) != expected:
        raise ValueError(
            f"кадр {width}×{height}, ожидался полный SBS-кадр {expected[0]}×{expected[1]}"
        )
    eye_width, eye_height = scene.eye_size
    return image[:eye_height, :eye_width].copy()


def _draw_pose(image: Any, pose: sync.Pose, scene: Scene, style: Style) -> int:
    """Спроецировать кисти кадра и нарисовать их на уже обрезанном кадре глаза."""
    return draw_skeleton(image, project_hands(pose, scene.intrinsics), style)


def _codec_name(path: Path) -> str:
    """Кодек из заголовка mp4 (по четырёхбуквенному тегу в первых мегабайтах)."""
    tags = {
        b"avc1": "H.264 (avc1)",
        b"avc3": "H.264 (avc3)",
        b"hvc1": "H.265 (hvc1)",
        b"hev1": "H.265 (hev1)",
        b"mp4v": "MPEG-4 part 2 (mp4v)",
        b"vp09": "VP9",
        b"av01": "AV1",
    }
    with path.open("rb") as handle:
        head = handle.read(4 * 1024 * 1024)
    for tag, name in tags.items():
        if tag in head:
            return name
    return "unknown"


def probe_output(
    path: str | Path, expected_frames: int, expected_size: tuple[int, int]
) -> dict[str, Any]:
    """Прочитать записанный файл заново: число кадров, размер, кодек.

    Это не формальность: ``VideoWriter.isOpened()`` у части сборок OpenCV возвращает True,
    а файл получается пустым или с обрезанным числом кадров. Ошибка здесь означает, что
    рендер надо повторять, а не публиковать.
    """
    import cv2  # noqa: PLC0415

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"файл не записан (или пуст): {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"записанный файл не открывается: {path}")
    try:
        frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        size = (
            int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
        ok, image = cap.read()
    finally:
        cap.release()
    if not ok or image is None:
        raise ValueError(f"из {path} не читается ни одного кадра")
    if size != expected_size:
        raise ValueError(f"размер {size}, ожидался {expected_size}")
    if frames != expected_frames:
        raise ValueError(f"кадров {frames}, ожидалось {expected_frames}")
    return {"frame_count": frames, "size": size, "codec": _codec_name(path)}


@dataclass(frozen=True)
class RenderStats:
    """Итог рендера: что записано, чем, за сколько и сколько точек нарисовано."""

    out: Path
    first_frame: int
    last_frame: int
    size: tuple[int, int]
    fps: float
    codec: str
    elapsed_s: float
    bytes: int
    joints_drawn: int
    frames_after_tracking: int
    time_offset_ms: float = 0.0

    @property
    def frames(self) -> int:
        """Сколько кадров записано."""
        return self.last_frame - self.first_frame + 1

    @property
    def duration_s(self) -> float:
        """Длительность записанного файла по метаданным записи."""
        return self.frames / self.fps if self.fps else 0.0

    @property
    def frames_per_s(self) -> float:
        """Скорость рендера, кадр/с."""
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0


def _open_writer(
    out: Path, fps: float, size: tuple[int, int], candidates: Sequence[str], log: Callable[[str], None]
) -> tuple[Any, str]:
    """Первый открывшийся ``VideoWriter`` из списка кодеков (H.264 → mp4v)."""
    import cv2  # noqa: PLC0415

    if not candidates:
        raise ValueError("fourcc_candidates: пустой список")
    for fourcc in candidates:
        writer = cv2.VideoWriter(
            str(out), cv2.VideoWriter_fourcc(*fourcc), fps, size
        )
        if writer.isOpened():
            if fourcc != candidates[0]:
                log(f"  кодек {candidates[0]!r} недоступен, пишу {fourcc!r}")
            return writer, fourcc
        writer.release()
    raise ValueError(f"ни один кодек не открылся: {', '.join(candidates)}")


def render_video(
    video: str | Path,
    out: str | Path,
    scene: Scene,
    *,
    start: int = 0,
    end: int | None = None,
    scale: float = 1.0,
    fourcc_candidates: Sequence[str] = FOURCC_CANDIDATES,
    progress_every: int = 300,
    log: Callable[[str], None] = print,
) -> RenderStats:
    """Отрендерить диапазон кадров ``start…end`` в ``out`` и вернуть :class:`RenderStats`.

    Пишется только левый глаз (1080×810). ``progress_every`` — как часто печатать строку
    прогресса (0 — молча); итоговая строка печатается всегда. Скорость рендера в статистике
    — фактическое время (от открытия видео до закрытия файла), а не оценка.
    """
    import cv2  # noqa: PLC0415

    if isinstance(progress_every, bool) or not isinstance(progress_every, int):
        raise ValueError(f"progress_every: ожидалось целое число, получено {progress_every!r}")
    if progress_every < 0:
        raise ValueError(f"progress_every: должно быть ≥ 0, получено {progress_every!r}")
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"нет видео: {video}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    first, last = frame_range(scene.meta, start, end)
    size = scene.eye_size
    style = style_for(size[0], scale)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cv2.VideoCapture не открывает {video}")
    writer = None
    try:
        if first:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
                raise ValueError(f"не удалось перейти к кадру {first}")
        writer, _fourcc = _open_writer(out, scene.fps, size, fourcc_candidates, log)
        total = last - first + 1
        started = time.perf_counter()
        joints = 0
        after_tracking = 0
        completed = False
        try:
            for k in range(first, last + 1):
                ok, image = capture.read()
                if not ok or image is None:
                    raise ValueError(f"видео кончилось на кадре {k} (ожидались по {last})")
                view = _eye_view(image, scene)
                moment = sync.frame_time(k, scene.meta, scene.header_ns, scene.tracking_end_ns)
                pose = sync.pose_at_frame(scene.track, moment, scene.time_offset_ns)
                joints += _draw_pose(view, pose, scene, style)
                if not moment.in_tracking:
                    after_tracking += 1
                writer.write(view)
                done = k - first + 1
                if progress_every and (done % progress_every == 0 or done == total):
                    elapsed = time.perf_counter() - started
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (total - done) / rate if rate else 0.0
                    log(
                        f"  кадр {k:5d}/{last} ({100.0 * done / total:5.1f} %) — "
                        f"{elapsed:6.1f} с, {rate:5.1f} кадр/с, осталось ~{eta:5.1f} с"
                    )
            completed = True
        finally:
            writer.release()
            writer = None
            elapsed = time.perf_counter() - started
            if not completed:
                out.unlink(missing_ok=True)
    finally:
        if writer is not None:
            writer.release()
        capture.release()

    probe = probe_output(out, total, size)
    return RenderStats(
        out=out,
        first_frame=first,
        last_frame=last,
        size=probe["size"],
        fps=scene.fps,
        codec=probe["codec"],
        elapsed_s=elapsed,
        bytes=out.stat().st_size,
        joints_drawn=joints,
        frames_after_tracking=after_tracking,
        time_offset_ms=scene.time_offset_ns / 1e6,
    )


def render_stills(
    video: str | Path,
    scene: Scene,
    out_dir: str | Path,
    *,
    frames: Iterable[int] = DEBUG_FRAMES,
    scale: float = 1.0,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Сохранить PNG-кадры с наложением (по умолчанию 5 отладочных кадров 07a).

    Это визуальная проверка наложения перед полным рендером: PNG 1080×810, по одному на
    кадр, ``frame_{k:05d}.png`` в ``out_dir``. Артефакт регенерируемый (``output/`` в
    ``.gitignore``), поэтому коммитится не он, а выводы о проверке.
    """
    import cv2  # noqa: PLC0415

    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"нет видео: {video}")
    indices = sorted(set(frames))
    style = style_for(scene.eye_size[0], scale)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cv2.VideoCapture не открывает {video}")
    written: list[Path] = []
    try:
        for k in indices:
            frame_range(scene.meta, k, k)  # проверка номера кадра
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, k):
                raise ValueError(f"не удалось перейти к кадру {k}")
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"не читается кадр {k}")
            view = _eye_view(image, scene)
            moment = sync.frame_time(k, scene.meta, scene.header_ns, scene.tracking_end_ns)
            pose = sync.pose_at_frame(scene.track, moment, scene.time_offset_ns)
            drawn = _draw_pose(view, pose, scene, style)
            if not moment.in_tracking:
                log(f"  кадр {k:5d}: трекинга уже нет — поза заморожена (по 04a)")
            path = out / f"frame_{k:05d}.png"
            if not cv2.imwrite(str(path), view):
                raise ValueError(f"не удалось записать {path}")
            written.append(path)
            log(f"  кадр {k:5d} (t = {moment.t_rel_s:9.3f} с): точек {drawn}/10 → {path}")
    finally:
        capture.release()
    return written


def _format_stats(stats: RenderStats) -> str:
    """Итог рендера одной таблицей — эти числа идут в README/заметки."""
    megabytes = stats.bytes / (1024 * 1024)
    lines = [
        f"записано: {stats.out}",
        f"  кадры: {stats.first_frame}…{stats.last_frame} ({stats.frames}), "
        f"{stats.size[0]}×{stats.size[1]}, {stats.fps:.4f} fps → {stats.duration_s:.3f} с",
        f"  кодек: {stats.codec}; файл {megabytes:.1f} МиБ; "
        f"поправка времени {stats.time_offset_ms:+.0f} мс",
        f"  рендер: {stats.elapsed_s:.1f} с, {stats.frames_per_s:.1f} кадр/с "
        f"(видимых суставов {stats.joints_drawn})",
        f"  кадров вне трекинга (поза заморожена): {stats.frames_after_tracking}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: полный рендер, фрагмент (``--start/--end``) или PNG-проверка (``--stills``)."""
    parser = argparse.ArgumentParser(
        description="Рендер видео с наложением скелета кистей (задача 08a)."
    )
    parser.add_argument("video", type=Path, help="data/CameraRecord_*.mp4 (полный SBS-кадр)")
    parser.add_argument(
        "--tracking",
        type=Path,
        default=None,
        help="data/trackingData_*.txt (по умолчанию — единственный в data/)",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("output/video_meta.json"),
        help="video_meta.json (по умолчанию output/video_meta.json)",
    )
    parser.add_argument("--out", type=Path, default=Path("output/overlay_skeleton.mp4"),
                        help="куда писать видео (по умолчанию output/overlay_skeleton.mp4)")
    parser.add_argument("--start", type=int, default=0, help="первый кадр (0-based, по умолчанию 0)")
    parser.add_argument("--end", type=int, default=None, help="последний кадр включительно")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="множитель радиуса точек и толщины линий (по умолчанию 1.0)")
    parser.add_argument("--fourcc", default=",".join(FOURCC_CANDIDATES),
                        help=f"кодеки через запятую (по умолчанию {','.join(FOURCC_CANDIDATES)})")
    parser.add_argument("--progress-every", type=int, default=300,
                        help="строка прогресса каждые N кадров (0 — не печатать)")
    parser.add_argument("--time-offset-ms", type=float, default=None,
                        help="сдвиг времени позы (по умолчанию — config.VIDEO_TIME_OFFSET_MS)")
    parser.add_argument("--stills", type=Path, default=None,
                        help="режим PNG: сохранить кадры --frames в этот каталог и выйти")
    parser.add_argument("--frames", default=",".join(str(k) for k in DEBUG_FRAMES),
                        help="кадры для --stills (по умолчанию 5 кадров 07a)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        scene = load_scene(args.meta, args.tracking)
        if args.time_offset_ms is not None:
            scene = replace(
                scene,
                time_offset_ns=int(round(args.time_offset_ms * 1e6)),
            )
        print(
            f"видео: {scene.frame_count} кадров, {scene.duration_s:.3f} с, "
            f"{scene.fps:.4f} fps; SBS {scene.intrinsics.full_w}×{scene.intrinsics.full_h} "
            f"→ глаз {scene.eye_size[0]}×{scene.eye_size[1]}"
        )
        print(
            f"трекинг: {len(scene.track)} сэмплов; конец трекинга "
            f"+{(scene.tracking_end_ns - scene.header_ns) / 1e9:.3f} с от header; "
            f"поправка времени {scene.time_offset_ns / 1e6:+.0f} мс"
        )
        if args.stills is not None:
            paths = render_stills(
                args.video, scene, args.stills, frames=parse_frames(args.frames), scale=args.scale
            )
            print(f"PNG с наложением: {len(paths)} файлов в {args.stills}")
            return 0
        fourcc = tuple(part.strip() for part in args.fourcc.split(",") if part.strip())
        stats = render_video(
            args.video,
            args.out,
            scene,
            start=args.start,
            end=args.end,
            scale=args.scale,
            fourcc_candidates=fourcc,
            progress_every=args.progress_every,
        )
    except (OSError, ValueError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1
    print(_format_stats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
