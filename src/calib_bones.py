"""calib_bones.py — метрика на костях: проекция скелета против разметки 08e.

Задача 07a дала эталон «точка внутри прямоугольника кисти»: он различает конвенции осей
(у неверной кисти уезжают на сотни пикселей), но внутри рамки 370×460 px ничего не говорит
о форме — 5 точек могут лежать в рамке, а «скелет» из 26 джойнтов при этом расходиться с
видимыми пальцами. Задача 08e добавляет второй, более тонкий эталон:

* **разметка** (``src/calib_frames.py``, ``BONE_FRAMES``): 7–9 пиксельных точек на руку —
  запястье, ладонь, пястная кость, фаланги указательного и среднего пальцев;
* **метрика** (этот модуль): проекция отрезка между соседними размеченными суставами
  сравнивается с размеченным отрезком (расстояние от точек проекции до эталонного отрезка),
  плюс счётчик «изломов» — костей, направление которых ушло от эталона больше чем на
  :data:`src.calib_frames.BONE_KINK_ANGLE_DEG`.

Что именно считается (одна функция :func:`evaluate_bones`):

1. Каждый размеченный сустав проецируется ровно тем же путём, что и 5 точек в
   ``calib_search.evaluate`` (поза головы + конвенция осей + интринсики левого глаза).
   Проектор передаётся снаружи (``project``), поэтому модуль не зависит от ``calib_search``
   и не тянет за собой перебор конвенций — тесты подставляют свой проектор.
2. Для каждой кости (соседняя пара в цепочке) проекция отрезка сэмплируется в
   :data:`BONE_SAMPLES` точках, для каждой берётся расстояние до **размеченного** отрезка
   этой же кости (0 — если попала внутрь, т.е. в пределы отрезка). Среднее по кости —
   «ошибка кости», максимум — худшая точка кости.
3. Кость «в допуске», если её худшая точка уложилась в ``tolerance_px``. Излом — если угол
   между проекцией кости и разметкой больше ``kink_angle_deg`` (направление, а не
   положение: кость можно сдвинуть вдоль себя, и расстояние это почти не заметит).

Почему это дополнение, а не замена: прямоугольник 07a остаётся критерием 07b/07c
(``--verify-config`` печатает 40/40) — он не зависит от формы кисти. Костная метрика
строже и потому чувствительнее к разметке (ручная точность ±50 px, короткие фаланги
30–40 px), поэтому её вердикт справочный: числа печатаются, но код возврата определяет
старая метрика. Так 08e не меняет поведение 5-точечного пути.

Запуск (через калибровку, вместе со старой метрикой):
    python src/calib_search.py data/trackingData_*.txt --verify-config
    python src/calib_search.py data/trackingData_*.txt --verify-config --bones-table
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import calib_frames, camera_model, sync  # noqa: E402
from src.parse_tracking import read_header, timestamp_range  # noqa: E402

#: Имена суставов, которых нет среди 5 точек рендера (08d) и которые нужны костям.
#: Берутся из ``HAND_JOINTS_FULL``: ``index_proximal`` — основная фаланга указательного,
#: ``middle_intermediate`` — средняя фаланга среднего и т.д. ``index`` и ``middle``
#: (кончики) уже входят в 5-точечный набор, поэтому читаются из позы.
EXTRA_JOINT_NAMES: tuple[str, ...] = (
    "index_metacarpal",
    "index_proximal",
    "index_intermediate",
    "middle_proximal",
    "middle_intermediate",
)

#: Суставы, которые уже есть в ``sync.Pose.hands`` (5-точечный набор): их позиция на момент
#: кадра берётся из позы, а не из дополнительных колонок.
POSE_JOINT_NAMES: tuple[str, ...] = ("palm", "wrist", "index", "middle")

#: Полные плоские ключи дополнительных суставов: ``left_index_proximal`` и т.д.
EXTRA_KEYS: tuple[str, ...] = tuple(
    f"{side}_{joint}" for side in calib_frames.BONE_HANDS for joint in EXTRA_JOINT_NAMES
)

#: Допуск костной метрики, px: та же ручная точность разметки, что у рамок 07a.
BONE_TOLERANCE_PX = calib_frames.BONE_TOLERANCE_PX

#: Порог излома, градусы (см. ``calib_frames.BONE_KINK_ANGLE_DEG``).
BONE_KINK_ANGLE_DEG = calib_frames.BONE_KINK_ANGLE_DEG

#: Сколько точек проекции кости сравнивать с размеченным отрезком (включая концы).
BONE_SAMPLES = 5

#: Короче этой длины отрезок считается вырожденным: направление по нему не считается.
BONE_MIN_LENGTH_PX = 8.0

#: Уровни уверенности разметки для фильтра ``min_confidence``: чем больше, тем строже.
CONFIDENCE_LEVELS = {"low": 0, "medium": 1, "high": 2}

#: Путь по умолчанию к метаданным видео (нужны ``frame_count`` и ``duration``).
DEFAULT_META = Path("output/video_meta.json")

#: Проектор «точка в системе головы → пиксель левого глаза (или None)».
Projector = Callable[[Sequence[float]], "tuple[float, float] | None"]


@dataclass(frozen=True)
class BonePoint:
    """Размеченный сустав на кадре: имя, сторона и позиция в системе головы.

    ``head`` — позиция для ``invert_quat=False``, ``head_inverted`` — для
    ``invert_quat=True`` (обе считаются один раз при загрузке, как ``PointTarget``
    в ``calib_search``): конвенция кватерниона выбирается уже на этапе проекции.
    """

    frame_index: int
    side: str
    joint: str
    head: tuple[float, float, float]
    head_inverted: tuple[float, float, float]


@dataclass(frozen=True)
class BoneData:
    """Данные костной метрики: интринсики левого глаза, разметка и позиции суставов."""

    intrinsics: camera_model.Intrinsics
    frames: tuple[calib_frames.BoneFrame, ...]
    points: tuple[BonePoint, ...]

    @property
    def total(self) -> int:
        """Сколько суставов размечено (всех кадров и рук)."""
        return len(self.points)

    @property
    def hands(self) -> int:
        """Сколько кистей размечено."""
        return sum(len(frame.hands) for frame in self.frames)

    @property
    def bones(self) -> int:
        """Сколько отрезков размечено."""
        return sum(frame.bones for frame in self.frames)


@dataclass(frozen=True)
class BoneRow:
    """Результат по одной кости: ошибка проекции и поворот относительно разметки."""

    frame_index: int
    side: str
    first: str
    second: str
    mean_px: float
    max_px: float
    angle_deg: float

    def as_dict(self) -> dict:
        """Плоское представление для JSON."""
        return {
            "frame": self.frame_index,
            "side": self.side,
            "bone": [self.first, self.second],
            "mean_px": round(self.mean_px, 3),
            "max_px": round(self.max_px, 3),
            "angle_deg": round(self.angle_deg, 3),
        }


@dataclass(frozen=True)
class BoneMetrics:
    """Сводка костной метрики по всем размеченным кистям.

    ``bones``/``in_frame`` — сколько костей размечено и сколько из них целиком попало в кадр
    (оба конца перед камерой и внутри кадра); ``good`` — из попавших уложились в допуск
    ``tolerance_px``. ``mean_error_px``/``max_error_px`` — по всем сэмплам всех попавших
    костей. ``kinks``/``max_angle_deg`` — по направлению (см. ``BONE_KINK_ANGLE_DEG``).
    """

    joints: int
    hands: int
    bones: int
    in_frame: int
    good: int
    mean_error_px: float
    max_error_px: float
    kinks: int
    max_angle_deg: float
    rows: tuple[BoneRow, ...]

    @property
    def violations(self) -> int:
        """Кости в кадре, которые не уложились в допуск."""
        return self.in_frame - self.good

    def as_dict(self) -> dict:
        """Плоское представление для JSON (без построчных данных)."""
        return {
            "joints": self.joints,
            "hands": self.hands,
            "bones": self.bones,
            "in_frame": self.in_frame,
            "good": self.good,
            "mean_error_px": round(self.mean_error_px, 3),
            "max_error_px": round(self.max_error_px, 3),
            "kinks": self.kinks,
            "max_angle_deg": round(self.max_angle_deg, 3),
        }

    def format_lines(
        self,
        tolerance_px: float = BONE_TOLERANCE_PX,
        kink_angle_deg: float = BONE_KINK_ANGLE_DEG,
    ) -> list[str]:
        """Строки отчёта для CLI (без отступа: его добавляет вызывающий)."""
        if self.in_frame == 0:
            return [
                f"кости (08e): размечено {self.bones} отрезков, но в кадр не попало ни одного — "
                "проверить форму нечем"
            ]
        lines = [
            f"кости (08e): {self.in_frame}/{self.bones} отрезков в кадре "
            f"({self.hands} кистей, {self.joints} точек); в допуске ±{tolerance_px:g} px "
            f"{self.good}/{self.in_frame}, средняя ошибка {self.mean_error_px:.1f} px, "
            f"максимальная {self.max_error_px:.1f} px"
        ]
        lines.append(
            f"  изломы (направление кости ушло от разметки > {kink_angle_deg:g}°): "
            f"{self.kinks}/{self.in_frame}, максимальный поворот {self.max_angle_deg:.1f}°"
        )
        return lines


def _joint_world(
    pose: sync.Pose, track: sync.PoseTrack, t_ns: int, side: str, joint: str
) -> tuple[float, float, float]:
    """Позиция сустава на момент кадра: из позы (5 точек) или из дополнительных колонок."""
    if joint in POSE_JOINT_NAMES:
        return pose.hands[f"{side}_{joint}"]
    return track.joint_at(t_ns, f"{side}_{joint}")


def load_bone_data(
    tracking_path: str | Path,
    meta_path: str | Path = DEFAULT_META,
    frames: Sequence[calib_frames.BoneFrame] = calib_frames.BONE_FRAMES,
    offset_ns: int | None = None,
    track: sync.PoseTrack | None = None,
) -> BoneData:
    """Прочитать позиции размеченных суставов на моменты кадров 08e (система головы).

    Время кадра — ``sync.frame_time``, поза — ``sync.pose_at_frame`` (интерполяция, SLERP,
    поправка ``config.VIDEO_TIME_OFFSET_MS``) — тот же путь, что у рендера и у метрики рамок.
    Дополнительные 26-джойнтные суставы читаются ``PoseTrack.joint_at`` на **тот же** момент
    (``sync.pose_time_ns``), поэтому обе метрики живут в одной шкале времени.

    ``track`` — уже загруженный трек с ``extra_keys`` (чтобы не читать 258 МБ дважды);
    ``None`` — загрузить самому (один проход по файлу, ≈4 с).
    """
    if track is None:
        track = sync.load_poses(tracking_path, extra_keys=EXTRA_KEYS)
    else:
        missing = [key for key in EXTRA_KEYS if key not in track.extra_keys]
        if missing:
            raise ValueError(
                f"load_bone_data: в переданном track нет колонок {missing[:3]}… "
                f"(нужен sync.load_poses(..., extra_keys=EXTRA_KEYS))"
            )
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    header = read_header(tracking_path)
    header_ns = int(header["timeStampNs"])
    tracking_end_ns = timestamp_range(tracking_path)[1]
    intrinsics = camera_model.Intrinsics.from_header(header, frame_size=(2160, 810))
    points: list[BonePoint] = []
    for frame in frames:
        moment = sync.frame_time(frame.index, meta, header_ns, tracking_end_ns)
        t_pose = sync.pose_time_ns(moment, offset_ns)
        pose = track.pose_at(t_pose)
        rotation = camera_model.quat_to_rotation(pose.head_quat)
        inverse = _transpose3(rotation)
        for hand in frame.hands:
            for mark in hand.marks:
                world = _joint_world(pose, track, t_pose, hand.hand, mark.joint)
                if not all(math.isfinite(value) for value in world):
                    continue
                delta = (
                    world[0] - pose.head_pos[0],
                    world[1] - pose.head_pos[1],
                    world[2] - pose.head_pos[2],
                )
                points.append(
                    BonePoint(
                        frame_index=frame.index,
                        side=hand.hand,
                        joint=mark.joint,
                        head=_matvec3(inverse, delta),
                        head_inverted=_matvec3(rotation, delta),
                    )
                )
    if not points:
        raise ValueError("кости: ни одного размеченного сустава не нашлось в данных (все NaN?)")
    return BoneData(
        intrinsics=intrinsics.per_eye("left"), frames=tuple(frames), points=tuple(points)
    )


def _matvec3(
    matrix: camera_model.Matrix3, point: Sequence[float]
) -> tuple[float, float, float]:
    """``matrix · point`` для 3×3."""
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    )


def _transpose3(matrix: Sequence[Sequence[float]]) -> camera_model.Matrix3:
    """Транспонированная 3×3 (для поворота — обратная)."""
    return tuple(  # type: ignore[return-value]
        tuple(matrix[j][i] for j in range(3)) for i in range(3)
    )


def point_segment_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    """Расстояние от точки до отрезка ``start→end``: 0, если точка лежит на нём."""
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    px, py = float(point[0]), float(point[1])
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length2
    t = min(1.0, max(0.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def sample_segment(
    start: Sequence[float], end: Sequence[float], count: int = BONE_SAMPLES
) -> list[tuple[float, float]]:
    """``count`` точек вдоль отрезка, включая оба конца (для сравнения проекции с эталоном)."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError(f"sample_segment.count: ожидалось целое ≥ 2, получено {count!r}")
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    return [
        (ax + (bx - ax) * i / (count - 1), ay + (by - ay) * i / (count - 1))
        for i in range(count)
    ]


def bone_angle_deg(
    first: Sequence[float], second: Sequence[float],
    reference_first: Sequence[float], reference_second: Sequence[float],
) -> float:
    """Угол между направлениями двух отрезков, градусы: 0…90 (у кости нет начала и конца).

    Вырожденный отрезок (короче :data:`BONE_MIN_LENGTH_PX`) даёт 0.0: направление по нему
    не определено, и штрафовать за это нечего.
    """
    vx = float(second[0]) - float(first[0])
    vy = float(second[1]) - float(first[1])
    wx = float(reference_second[0]) - float(reference_first[0])
    wy = float(reference_second[1]) - float(reference_first[1])
    norm_v = math.hypot(vx, vy)
    norm_w = math.hypot(wx, wy)
    if norm_v < BONE_MIN_LENGTH_PX or norm_w < BONE_MIN_LENGTH_PX:
        return 0.0
    cosine = (vx * wx + vy * wy) / (norm_v * norm_w)
    cosine = min(1.0, max(-1.0, cosine))
    angle = math.degrees(math.acos(cosine))
    return min(angle, 180.0 - angle)


def evaluate_bones(
    data: BoneData,
    project: Projector,
    invert_quat: bool = False,
    tolerance_px: float = BONE_TOLERANCE_PX,
    kink_angle_deg: float = BONE_KINK_ANGLE_DEG,
    samples: int = BONE_SAMPLES,
    min_confidence: str = "low",
) -> BoneMetrics:
    """Костная метрика: проекция отрезков скелета против размеченных отрезков 08e.

    ``project`` — проектор «точка в системе головы → пиксель левого глаза» (``None``, если
    точка за камерой или вне кадра); его собирает ``calib_search`` из конвенции и интринсиков,
    тесты подставляют свой. ``invert_quat`` выбирает из :class:`BonePoint` ту же конвенцию
    кватерниона, что и метрика рамок.

    ``min_confidence`` отбрасывает отрезки, у которых хотя бы один конец размечен хуже
    указанного уровня: ``"low"`` — считать всё, ``"medium"`` — только точки, где фаланги
    действительно видны (это «строгая» выборка: на k = 30 разметка угадана по буграм и в неё
    не входит). Так одна и та же функция даёт и полную, и надёжную метрику.
    """
    if not isinstance(data, BoneData):
        raise ValueError(f"data: ожидался BoneData, получено {data!r}")
    if not callable(project):
        raise ValueError(f"project: ожидалась функция, получено {project!r}")
    tolerance_px = _positive_float(tolerance_px, "tolerance_px")
    kink_angle_deg = _positive_float(kink_angle_deg, "kink_angle_deg")
    if min_confidence not in CONFIDENCE_LEVELS:
        raise ValueError(
            f"min_confidence: ожидалось одно из {tuple(CONFIDENCE_LEVELS)}, "
            f"получено {min_confidence!r}"
        )
    threshold = CONFIDENCE_LEVELS[min_confidence]

    projected: dict[tuple[int, str, str], tuple[float, float]] = {}
    for point in data.points:
        head = point.head_inverted if invert_quat else point.head
        pixel = project(head)
        if pixel is not None:
            projected[(point.frame_index, point.side, point.joint)] = (
                float(pixel[0]),
                float(pixel[1]),
            )

    errors: list[float] = []
    rows: list[BoneRow] = []
    considered = 0
    kept_hands: set[tuple[int, str]] = set()
    kept_joints: set[tuple[int, str, str]] = set()
    for frame in data.frames:
        for hand in frame.hands:
            marks = hand.points
            for first, second in hand.bones:
                if min(
                    CONFIDENCE_LEVELS[hand.mark_for(first).confidence],
                    CONFIDENCE_LEVELS[hand.mark_for(second).confidence],
                ) < threshold:
                    continue
                considered += 1
                kept_hands.add((frame.index, hand.hand))
                kept_joints.update(
                    {(frame.index, hand.hand, first), (frame.index, hand.hand, second)}
                )
                start = projected.get((frame.index, hand.hand, first))
                end = projected.get((frame.index, hand.hand, second))
                if start is None or end is None:
                    continue
                distances = [
                    point_segment_distance(sample, marks[first], marks[second])
                    for sample in sample_segment(start, end, samples)
                ]
                mean_error = sum(distances) / len(distances)
                max_error = max(distances)
                errors.extend(distances)
                rows.append(
                    BoneRow(
                        frame_index=frame.index,
                        side=hand.hand,
                        first=first,
                        second=second,
                        mean_px=mean_error,
                        max_px=max_error,
                        angle_deg=bone_angle_deg(start, end, marks[first], marks[second]),
                    )
                )
    if not rows:
        return BoneMetrics(
            joints=len(kept_joints),
            hands=len(kept_hands),
            bones=considered,
            in_frame=0,
            good=0,
            mean_error_px=math.inf,
            max_error_px=math.inf,
            kinks=0,
            max_angle_deg=0.0,
            rows=(),
        )
    return BoneMetrics(
        joints=len(kept_joints),
        hands=len(kept_hands),
        bones=considered,
        in_frame=len(rows),
        good=sum(1 for row in rows if row.max_px <= tolerance_px),
        mean_error_px=sum(errors) / len(errors),
        max_error_px=max(errors),
        kinks=sum(1 for row in rows if row.angle_deg > kink_angle_deg),
        max_angle_deg=max(row.angle_deg for row in rows),
        rows=tuple(rows),
    )


def _positive_float(value: object, what: str) -> float:
    """Конечное число > 0 (bool отвергаем: ``isinstance(True, int)`` — True)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what}: ожидалось число, получено {value!r}")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{what}: ожидалось конечное число > 0, получено {value!r}")
    return value


def format_table(metrics: BoneMetrics) -> str:
    """Построчная таблица костей (для ``--bones-table``): кадр, рука, кость, px, угол."""
    if not metrics.rows:
        return "кости: ни один отрезок не попал в кадр"
    lines = [
        "| k | рука | кость | средняя, px | макс, px | поворот, ° |",
        "|---|---|---|---|---|---|",
    ]
    for row in metrics.rows:
        lines.append(
            f"| {row.frame_index} | {row.side} | {row.first}→{row.second} | "
            f"{row.mean_px:.1f} | {row.max_px:.1f} | {row.angle_deg:.1f} |"
        )
    return "\n".join(lines)
