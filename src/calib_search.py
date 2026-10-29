"""calib_search.py — подбор конвенции осей и грубого ``T_head_cam`` (задача 07b).

Задача 07a дала эталон: 5 кадров и прямоугольники кистей в пикселях левого глаза
(``src/calib_frames.py``). Задача 05b дала формулу ``p_cam = A · p_head + b``, но её
параметры (оси и смещение камеры) не были известны: экстринсики из header'а с позой
головы не согласуются (расхождение ~36°, см. ``docs/NOTES.md`` за 27.10).

Этот модуль подбирает параметры по эталону, а не «на глаз»:

1. **Дискретный перебор** — 24 собственные ротации осей (перестановки ``x, y, z`` со
   знаками, ``det = +1``) × 2 конвенции кватерниона (``invert_quat``) = 48 вариантов.
   Для каждого — сетка смещения камеры в пределах ±10 см с шагом 2 см (11³ = 1331
   положений). Несобственные ротации (``det = −1``) в перебор не входят: они дают
   зеркальное отражение сцены, а не поворот камеры.
2. **Уточнение top-3** — грубая сетка ``yaw/pitch/roll`` ±20° с шагом 5° вокруг
   найденной ориентации и такая же сетка смещения ±10 см с шагом 2 см (координатный
   спуск: повороты → смещение → повороты).
3. **Метрика** (эталон 07a): для каждого сустава ``palm, wrist, thumb, index, middle``,
   попавшего в кадр левого глаза, считается расстояние до прямоугольника **своей**
   руки (0 — внутри). Среднее по всем таким точкам и есть «средняя ошибка, px»
   из критерия карточки (≤ 60 px). Дополнительно считаются попадания в прямоугольник
   чужой руки (``wrong``) и промахи мимо всех прямоугольников (``miss``) — это
   штрафы, которые не дают «улучшить» метрику, выбросив точки из кадра.
4. **Плато** — после победы проверяется, насколько метрика чувствительна к параметрам:
   перебираются одномерные сдвиги по каждой оси/углу и печатается, до какого значения
   ошибка остаётся нулевой. Это честная граница точности: разметка ±50 px при
   расстоянии ~0.45 м соответствует ≈ ±3.3 см смещения камеры, поэтому меньшее
   смещение данные не различают, и в плато выбирается минимальная правка модели
   (нулевое смещение), а не произвольная точка.

Почему конвенция осей — это ровно ``A = diag(1, −1, −1)``: в системе головы (OpenXR)
``x`` вправо, ``y`` вверх, ``z`` назад, а камера смотрит вперёд и её изображение имеет
``y`` вниз — значит ``x_cam = x_head``, ``y_cam = −y_head``, ``z_cam = −z_head``.
Прежний флаг ``opencv_conventions`` из 05b переворачивал только ``y``, поэтому ``z``
оставался «назад» и кисти оказывались за камерой (0 % точек впереди, см. 05b); полная
ориентация камеры — поворот на 180° вокруг ``x`` головы, и он записывается в
``T_head_cam``, а флаг ``opencv_conventions`` при этом выключается: переворот ``y`` —
часть калиброванной ориентации камеры, а не отдельная поправка.

Параметры победителя сохраняются в JSON (``output/debug/calib/calibration.json``) —
это вход задачи 07c, которая переносит их в ``src/config.py``. Числа в ``config.py``
здесь не меняются: подбор только печатает/сохраняет результат. Обратная проверка —
режим ``--verify-config`` (07c): он собирает конвенцию из ``src/config.py``
(:func:`convention_from_config`), прогоняет её по эталону 07a и сверяет метрику с 07b —
то есть отвечает на вопрос «рендер по текущему конфигу воспроизводится?».

Запуск:
    python src/calib_search.py data/trackingData_*.txt
    python src/calib_search.py data/trackingData_*.txt --video data/CameraRecord_*.mp4
    python src/calib_search.py data/trackingData_*.txt --json > output/debug/calib/calibration.json
    python src/calib_search.py data/trackingData_*.txt --verify-config
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import calib_frames, camera_model, sync  # noqa: E402
from src.parse_tracking import read_header, timestamp_range  # noqa: E402

#: Суставы, которые сравниваются с разметкой (5 точек на руку, как в 03c/07a).
JOINTS = ("palm", "wrist", "thumb", "index", "middle")

#: Штраф за точку в чужом прямоугольнике или мимо всех, px: больше любой реальной
#: ошибки попадания (кадр 1080 px), поэтому «улучшить» метрику ценой чужих рук нельзя.
PENALTY_PX = 200.0

#: Критерий готовности карточки 07b: средняя ошибка победителя, px.
MAX_MEAN_ERROR_PX = 60.0

#: Точность ручной разметки 07a (±50 px) — она же граница разрешения по смещению.
TOLERANCE_PX = calib_frames.TOLERANCE_PX

#: Грубая сетка смещения камеры: диапазон (± м) и шаг (м) из карточки 07b.
OFFSET_RANGE_M = 0.10
OFFSET_STEP_M = 0.02

#: Грубая сетка уточнения поворота: диапазон (± °) и шаг (°) из карточки 07b.
ANGLE_RANGE_DEG = 20.0
ANGLE_STEP_DEG = 5.0

#: Сколько вариантов печатать в таблице и сколько картинок сохранять.
TOP_TABLE = 5
TOP_OVERLAYS = 3

#: Куда сохраняются PNG и JSON победителя (``output/`` в ``.gitignore``).
DEFAULT_OUT = Path("output/debug/calib")

#: Допуск покрытия, точек. Метрика считается только по суставам, попавшим в кадр, поэтому
#: конвенция, у которой в кадре 1–5 точек вместо 39, «улучшить» ошибку может, но проверить
#: её нечем. Кандидаты, у которых в кадре меньше, чем у лучшего по перебору минус этот
#: допуск, считаются несравнимыми и уходят в конец таблицы (2 точки — суставы у самой
#: границы кадра, которые при сдвиге на шаг сетки уходят за край и возвращаются).
COVERAGE_TOLERANCE = 2


@dataclass(frozen=True)
class Metrics:
    """Результат проверки одной конвенции по эталону 07a.

    ``good`` — точки внутри прямоугольника своей руки (или ``unknown``, если сторона
    на кадре не установлена), ``wrong`` — внутри прямоугольника **чужой** руки,
    ``miss`` — внутри кадра, но мимо всех прямоугольников, ``in_frame`` — сколько
    суставов вообще попало в кадр левого глаза. ``mean_error_px`` — среднее расстояние
    до своей области по попавшим в кадр точкам (0 у всех точек внутри), т.е. главная
    метрика карточки; ``max_error_px`` — худшая такая точка.
    """

    good: int
    wrong: int
    miss: int
    in_frame: int
    total: int
    mean_error_px: float
    max_error_px: float

    @property
    def violations(self) -> int:
        """Точки в чужой руке или мимо всех прямоугольников — метрика это наказывает."""
        return self.wrong + self.miss

    @property
    def score(self) -> float:
        """Скаляр для сравнения: средняя ошибка плюс штраф за каждую точку-нарушение."""
        return self.mean_error_px + PENALTY_PX * self.violations

    def as_dict(self) -> dict:
        """Плоское представление для JSON."""
        return {
            "good": self.good,
            "wrong": self.wrong,
            "miss": self.miss,
            "in_frame": self.in_frame,
            "total": self.total,
            "mean_error_px": round(self.mean_error_px, 3),
            "max_error_px": round(self.max_error_px, 3),
            "score": round(self.score, 3),
        }


#: Матрица 3×3 как три строки (совместимо с ``camera_model.Matrix3``).
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True)
class Convention:
    """Гипотеза о камере: оси (``A``), конвенция кватерниона и положение камеры.

    ``rotation`` — ``A``, поворот из системы головы в систему камеры (``x`` вправо,
    ``y`` вниз, ``z`` вперёд). ``camera_offset`` — положение камеры в системе головы
    (``x`` вправо, ``y`` вверх, ``z`` назад — как в данных); тогда
    ``p_cam = A · (p_head − camera_offset)``.
    ``perturbation_deg`` — насколько ориентация отличается от дискретного варианта
    (0 у вариантов из перебора 24 ротаций, иначе — максимальный сдвиг yaw/pitch/roll).
    """

    rotation: Matrix3
    invert_quat: bool = False
    camera_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    perturbation_deg: float = 0.0

    @property
    def axes(self) -> str:
        """Компактная запись осей: ``"+x −y −z"`` (камера = знак·ось головы)."""
        letters = "xyz"
        parts = []
        for row in self.rotation:
            index = max(range(3), key=lambda i: abs(row[i]))
            sign = "+" if row[index] > 0 else "−"
            parts.append(f"{sign}{letters[index]}")
        return " ".join(parts)

    @property
    def name(self) -> str:
        """Имя варианта для таблиц и JSON."""
        x, y, z = self.camera_offset
        flip = "invert_quat=True" if self.invert_quat else "invert_quat=False"
        text = f"{self.axes}, {flip}, c=({x:g}, {y:g}, {z:g})"
        if self.perturbation_deg:
            text += f", Δ{self.perturbation_deg:g}°"
        return text

    @property
    def camera_translation(self) -> tuple[float, float, float]:
        """``b`` из ``p_cam = A·p_head + b``: смещение камеры в её собственных осях."""
        b = _matvec3(self.rotation, self.camera_offset)
        return (-b[0], -b[1], -b[2])

    def head_cam(self) -> camera_model.Matrix4:
        """``T_head_cam`` — поза камеры в системе головы (``cam`` → ``head``), 4×4.

        Обратная матрица к ``[A | b]``: ``p_head = Aᵀ · (p_cam − b)``, т.е.
        ``C = Aᵀ`` и ``c = camera_offset``. Именно в таком виде параметры попадают в
        ``src/config.py`` (07c): ``HEAD_CAM_ROTATION_RPY_DEG`` — углы ``C``,
        ``HEAD_CAM_TRANSLATION_M`` — ``c``.
        """
        rotation = _transpose3(self.rotation)
        return _homogeneous(rotation, self.camera_offset)

    def matrix_for(
        self, head_pos: Sequence[float], head_quat: Sequence[float]
    ) -> camera_model.Matrix4:
        """``T_cam_world`` кадра: 4×4, переводящая точку мира в координаты камеры.

        Та же величина, что ``camera_model.world_to_camera_matrix(pose, t_head_cam, ...,
        opencv_conventions=False, invert_quat=...)`` — тест сверяет их напрямую.
        """
        world_rotation = camera_model.quat_to_rotation(head_quat)
        if self.invert_quat:
            world_rotation = _transpose3(world_rotation)
        # p_head = Rᵀ · (p_world − t); p_cam = A · p_head + b
        rotation = _matmul3(self.rotation, _transpose3(world_rotation))
        b = self.camera_translation
        shifted = _matvec3(rotation, head_pos)
        translation = (b[0] - shifted[0], b[1] - shifted[1], b[2] - shifted[2])
        return (
            (rotation[0][0], rotation[0][1], rotation[0][2], translation[0]),
            (rotation[1][0], rotation[1][1], rotation[1][2], translation[1]),
            (rotation[2][0], rotation[2][1], rotation[2][2], translation[2]),
            (0.0, 0.0, 0.0, 1.0),
        )


@dataclass(frozen=True)
class Candidate:
    """Конвенция вместе с её метрикой на эталоне (строка таблицы)."""

    convention: Convention
    metrics: Metrics

    @property
    def key(self) -> tuple:
        """Ключ сортировки — от важного к второстепенному.

        1. средняя ошибка (метрика карточки, округлена, чтобы 0.0 не дрожал от float);
        2. число точек-нарушений (чужая рука / мимо всех прямоугольников);
        3. минимальная правка модели: наибольшее смещение камеры по оси — в плато
           нулевой ошибки выбирается «камера в начале координат головы», потому что
           данные меньшее смещение не различают и выдумывать его незачем; затем
           меньший добавленный поворот (сетка уточнения не должна выигрывать «на глаз»
           только за счёт лишней точки у края кадра);
        4. больше точек в кадре (покрытие: меньше точек «уехало» за кадр);
        5. имя — детерминированность.

        Сравнимость кандидатов по покрытию проверяет :func:`rank`: здесь покрытие —
        уже практически ничья, поэтому ключ можно использовать и для поиска внутри
        одной конвенции (:func:`refine`).
        """
        return (
            round(self.metrics.mean_error_px, 6),
            self.metrics.violations,
            round(max(abs(value) for value in self.convention.camera_offset), 6),
            self.convention.perturbation_deg,
            -self.metrics.in_frame,
            self.convention.name,
        )

    def as_dict(self) -> dict:
        """Строка таблицы для JSON."""
        return {
            "convention": self.convention.name,
            "axes": self.convention.axes,
            "invert_quat": self.convention.invert_quat,
            "camera_offset_m": list(self.convention.camera_offset),
            "perturbation_deg": self.convention.perturbation_deg,
            "metrics": self.metrics.as_dict(),
        }


def rank(
    candidates: Iterable[Candidate], tolerance: int = COVERAGE_TOLERANCE
) -> tuple[Candidate, ...]:
    """Отсортировать кандидатов и увести несравнимые по покрытию в конец.

    Метрика (расстояние до рамки) определена только на попавших в кадр суставах, поэтому
    вариант, у которого в кадре 5 точек вместо 39, формально может иметь нулевую ошибку.
    Такие строки не выбрасываются (они показывают, что перебор честный), но сортируются
    после сравнимых: сначала идут кандидаты с покрытием не меньше
    ``best_in_frame − tolerance``, и только потом остальные. Внутри группы — по
    :attr:`Candidate.key`.
    """
    items = list(candidates)
    if not items:
        return ()
    best_coverage = max(item.metrics.in_frame for item in items)
    threshold = best_coverage - max(0, tolerance)
    items.sort(key=lambda item: (item.metrics.in_frame < threshold, item.key))
    unique: list[Candidate] = []
    seen: set[tuple] = set()
    for item in items:
        marker = (
            item.convention.rotation,
            item.convention.invert_quat,
            item.convention.camera_offset,
            item.convention.perturbation_deg,
        )
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(item)
    return tuple(unique)


@dataclass(frozen=True)
class PointTarget:
    """Сустав на кадре: система головы в двух конвенциях и разметка для сравнения.

    ``head`` — позиция в системе головы для ``invert_quat=False``, ``head_inverted`` —
    для ``invert_quat=True`` (обе считаются один раз при загрузке). ``own`` — bbox
    своей руки (или ``unknown``), ``others`` — bbox чужих рук на этом кадре (для
    подсчёта перепутанных рук).
    """

    frame_index: int
    side: str
    joint: str
    head: tuple[float, float, float]
    head_inverted: tuple[float, float, float]
    own: tuple[int, int, int, int]
    others: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class CalibData:
    """Данные калибровки: интринсики левого глаза, суставы кадров и сами кадры.

    ``frames`` — записи ``CalibFrame`` (для наложения нужно знать номер и время),
    ``points`` — плоский список :class:`PointTarget` по всем кадрам.
    """

    intrinsics: camera_model.Intrinsics
    frames: tuple[calib_frames.CalibFrame, ...]
    points: tuple[PointTarget, ...]

    @property
    def total(self) -> int:
        """Сколько суставов участвует в метрике (кадры из ``ignore`` не входят)."""
        return len(self.points)


def _matvec3(matrix: Matrix3, point: Sequence[float]) -> tuple[float, float, float]:
    """``matrix · point`` для 3×3."""
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
    )


def _matmul3(a: Matrix3, b: Matrix3) -> Matrix3:
    """Произведение 3×3: ``a · b``."""
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def _transpose3(matrix: Matrix3) -> Matrix3:
    """Транспонированная 3×3 (для поворота — обратная)."""
    return tuple(  # type: ignore[return-value]
        tuple(matrix[j][i] for j in range(3)) for i in range(3)
    )


def _homogeneous(rotation: Matrix3, translation: Sequence[float]) -> camera_model.Matrix4:
    """4×4 из поворота и сдвига: ``[R | t; 0 0 0 1]``."""
    tx, ty, tz = translation
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], tx),
        (rotation[1][0], rotation[1][1], rotation[1][2], ty),
        (rotation[2][0], rotation[2][1], rotation[2][2], tz),
        (0.0, 0.0, 0.0, 1.0),
    )


def axis_rotations(proper: bool = True) -> tuple[Matrix3, ...]:
    """Перестановки осей со знаками: 24 собственные (``det = +1``) или все 48.

    Собственная ротация нужна потому, что несобственная (``det = −1``) зеркалит
    изображение: сцена отразилась бы по горизонтали, и «попадание на кисть» достигалось
    бы обманом, а не ориентацией камеры. Порядок детерминирован (перестановки, затем
    знаки), поэтому результат воспроизводим.
    """
    matrices: list[Matrix3] = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            rows = [[0.0, 0.0, 0.0] for _ in range(3)]
            for row, column in enumerate(permutation):
                rows[row][column] = signs[row]
            matrix = tuple(tuple(row) for row in rows)  # type: ignore[arg-type]
            determinant = _determinant3(matrix)  # type: ignore[arg-type]
            if proper and determinant <= 0:
                continue
            matrices.append(matrix)  # type: ignore[arg-type]
    return tuple(matrices)


def _determinant3(matrix: Matrix3) -> float:
    """Определитель 3×3 по правилу треугольников."""
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def offset_grid(
    step_m: float = OFFSET_STEP_M, limit_m: float = OFFSET_RANGE_M
) -> tuple[tuple[float, float, float], ...]:
    """Сетка смещения камеры: ``(x, y, z)`` с шагом ``step_m`` в пределах ±``limit_m``.

    Значения округляются до шести знаков, чтобы ``0.02`` не превращался в
    ``0.020000000000000018`` и попадал в JSON как есть.
    """
    if step_m <= 0:
        raise ValueError(f"step_m: должно быть > 0, получено {step_m}")
    if limit_m < 0:
        raise ValueError(f"limit_m: должно быть ≥ 0, получено {limit_m}")
    count = int(round(limit_m / step_m))
    values = [round(index * step_m, 6) for index in range(-count, count + 1)]
    return tuple(itertools.product(values, values, values))


def offset_values(step_m: float = OFFSET_STEP_M, limit_m: float = OFFSET_RANGE_M) -> tuple[float, ...]:
    """Одномерный набор значений смещения (для одномерного анализа чувствительности)."""
    count = int(round(limit_m / step_m))
    return tuple(round(index * step_m, 6) for index in range(-count, count + 1))


def rpy_from_matrix(rotation: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """Углы ``(yaw, pitch, roll)`` для ``R = R_y(yaw)·R_x(pitch)·R_z(roll)``.

    Обратная операция к :func:`camera_model.rotation_rpy` в договорённости ``config.py``.
    У разложения есть две ветви (``pitch`` и ``180° − pitch``); выбирается та, у которой
    меньше ``|yaw| + |roll|``, чтобы в ``config.py`` попадали читаемые углы: например,
    поворот на 180° вокруг ``x`` даёт ``(0, 180, 0)``, а не ``(180, 0, 180)`` — это один
    и тот же поворот. При ``|pitch| ≈ 90°`` (шарнирный замок) ``roll`` полагается 0 и
    ``yaw`` берётся из оставшихся элементов — этого достаточно, чтобы матрица
    восстановилась.
    """
    if isinstance(rotation, (str, bytes)):
        raise ValueError(f"rotation: ожидалась матрица 3×3, получено {rotation!r}")
    rows = [tuple(float(value) for value in row) for row in rotation]  # type: ignore[arg-type]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(f"rotation: ожидалась матрица 3×3, получено {len(rows)} строк")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("rotation: нефинитный элемент матрицы")
    m = rows
    # m12 = −sin(pitch); при |m12| ≈ 1 — шарнирный замок.
    if abs(abs(m[1][2]) - 1.0) < 1e-9:
        pitch = math.degrees(math.copysign(math.pi / 2.0, -m[1][2]))
        # cp = 0: yaw и roll входят только разностью/суммой; берём roll = 0.
        yaw = math.degrees(math.atan2(-m[2][0], m[0][0]))
        return (0.0 if abs(round(yaw, 9)) < 1e-9 else round(yaw, 9), round(pitch, 9), 0.0)
    first = math.asin(max(-1.0, min(1.0, -m[1][2])))
    branches = [
        (
            math.atan2(m[0][2], m[2][2]) if math.cos(first) != 0 else 0.0,
            first,
            math.atan2(m[1][0], m[1][1]),
        ),
        (
            math.atan2(-m[0][2], -m[2][2]),
            math.pi - first,
            math.atan2(-m[1][0], -m[1][1]),
        ),
    ]
    best = min(
        ((math.degrees(y), math.degrees(p), math.degrees(r)) for y, p, r in branches),
        key=lambda angles: (abs(angles[0]) + abs(angles[2]), abs(angles[1])),
    )
    return tuple(0.0 if abs(round(value, 9)) < 1e-9 else round(value, 9) for value in best)


def rotation_basis(rotation: Sequence[Sequence[float]]) -> Matrix3:
    """Проверить, что это собственная ротация, и вернуть её в виде :data:`Matrix3`.

    ``ValueError`` для не-3×3, нефинитных значений, ``det ≤ 0`` и неортогональной
    матрицы: подбору нужны только точные перестановки осей со знаками, а не любая
    матрица (иначе «победитель» мог бы оказаться растяжением).
    """
    if isinstance(rotation, (str, bytes)):
        raise ValueError(f"rotation: ожидалась матрица 3×3, получено {rotation!r}")
    rows = [tuple(float(value) for value in row) for row in rotation]  # type: ignore[arg-type]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(f"rotation: ожидалась матрица 3×3, получено {len(rows)} строк")
    if not all(math.isfinite(value) for row in rows for value in row):
        raise ValueError("rotation: нефинитный элемент матрицы")
    matrix: Matrix3 = tuple(rows)  # type: ignore[assignment]
    determinant = _determinant3(matrix)
    if abs(determinant - 1.0) > 1e-9:
        raise ValueError(f"rotation: определитель {determinant:g}, ожидалась собственная ротация")
    product = _matmul3(matrix, _transpose3(matrix))
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            if abs(product[i][j] - expected) > 1e-9:
                raise ValueError("rotation: матрица не ортогональна")
    return matrix


def load_calib_data(
    tracking_path: str | Path,
    meta_path: str | Path = "output/video_meta.json",
    frames: Sequence[calib_frames.CalibFrame] = calib_frames.CALIB_FRAMES,
    offset_ns: int | None = None,
) -> CalibData:
    """Прочитать позы 5 кадров эталона и перевести суставы в систему головы.

    Время кадра берётся из ``sync.frame_time`` (та же шкала, что у видео), поза — из
    ``sync.pose_at_frame`` (интерполяция + SLERP + поправка на задержку видеотракта
    ``config.VIDEO_TIME_OFFSET_MS``) — ровно так же, как в рендере 08a. Для каждого сустава
    считаются обе конвенции кватерниона сразу, чтобы перебор не пересчитывал их 48 раз.

    ``offset_ns=None`` — поправка из ``config`` (то, что видит пользователь); явное значение
    нужно тестам, которые фиксируют числа калибровки 07b: она искалась без поправки
    (``offset_ns=0``), и её метрика 39/39 к ним и относится.
    """
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    header = read_header(tracking_path)
    header_ns = int(header["timeStampNs"])
    tracking_end_ns = timestamp_range(tracking_path)[1]
    intrinsics = camera_model.Intrinsics.from_header(header, frame_size=(2160, 810))
    track = sync.load_poses(tracking_path)

    points: list[PointTarget] = []
    for item in frames:
        moment = sync.frame_time(item.index, meta, header_ns, tracking_end_ns)
        pose = sync.pose_at_frame(track, moment, offset_ns)
        rotation = camera_model.quat_to_rotation(pose.head_quat)
        inverse = _transpose3(rotation)
        for side in ("left", "right"):
            if side in item.ignore:
                continue
            own_box = item.box_for(side) or item.box_for("unknown")
            if own_box is None:
                raise ValueError(
                    f"эталон k={item.index}: у стороны {side!r} нет прямоугольника "
                    "(и нет unknown) — нечего сравнивать"
                )
            others = tuple(
                box.bbox for box in item.hands if box.hand not in (side, "unknown")
            )
            for joint in JOINTS:
                world = pose.hands[f"{side}_{joint}"]
                if not all(math.isfinite(value) for value in world):
                    continue
                delta = (world[0] - pose.head_pos[0], world[1] - pose.head_pos[1],
                         world[2] - pose.head_pos[2])
                points.append(
                    PointTarget(
                        frame_index=item.index,
                        side=side,
                        joint=joint,
                        head=_matvec3(inverse, delta),
                        head_inverted=_matvec3(rotation, delta),
                        own=own_box.bbox,
                        others=others,
                    )
                )
    if not points:
        raise ValueError("эталон: ни одного сустава для сравнения (все NaN?)")
    return CalibData(
        intrinsics=intrinsics.per_eye("left"),
        frames=tuple(frames),
        points=tuple(points),
    )


def _distance_to_box(u: float, v: float, bbox: Sequence[int]) -> float:
    """Расстояние от пикселя до прямоугольника: 0 внутри (границы включительно)."""
    x0, y0, x1, y1 = bbox
    dx = 0.0 if x0 <= u <= x1 else min(abs(u - x0), abs(u - x1))
    dy = 0.0 if y0 <= v <= y1 else min(abs(v - y0), abs(v - y1))
    return math.hypot(dx, dy)


def evaluate(
    data: CalibData,
    rotation: Sequence[Sequence[float]],
    invert_quat: bool = False,
    camera_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> Metrics:
    """Проверить конвенцию по эталону 07a и вернуть :class:`Metrics`.

    Оси задаются матрицей ``rotation`` (голова → камера), ``camera_offset`` — положение
    камеры в системе головы. Точки за камерой (``Z ≤ 1 см``) и вышедшие за кадр левого
    глаза не считаются: эталон 07a размечен только там, где кисть видна, а часть
    суставов законно уходит за край (частично обрезанные руки).
    """
    if not isinstance(data, CalibData):
        raise ValueError(f"data: ожидался CalibData, получено {data!r}")
    matrix = rotation_basis(rotation)
    offset = (float(camera_offset[0]), float(camera_offset[1]), float(camera_offset[2]))
    b = _matvec3(matrix, offset)
    b = (-b[0], -b[1], -b[2])
    intr = data.intrinsics
    eye_w, eye_h = calib_frames.EYE_WIDTH, calib_frames.EYE_HEIGHT

    good = wrong = miss = in_frame = 0
    errors: list[float] = []
    for point in data.points:
        head = point.head_inverted if invert_quat else point.head
        x, y, z = _matvec3(matrix, head)
        x, y, z = x + b[0], y + b[1], z + b[2]
        if z <= camera_model.MIN_DEPTH_M:
            continue
        u = intr.fx * x / z + intr.cx
        v = intr.fy * y / z + intr.cy
        if not (0.0 <= u < eye_w and 0.0 <= v < eye_h):
            continue
        in_frame += 1
        distance = _distance_to_box(u, v, point.own)
        if distance == 0.0:
            good += 1
            errors.append(0.0)
            continue
        if any(_distance_to_box(u, v, box) == 0.0 for box in point.others):
            wrong += 1
        else:
            miss += 1
        errors.append(distance)
    mean_error = sum(errors) / len(errors) if errors else math.inf
    max_error = max(errors) if errors else math.inf
    return Metrics(
        good=good,
        wrong=wrong,
        miss=miss,
        in_frame=in_frame,
        total=data.total,
        mean_error_px=mean_error,
        max_error_px=max_error,
    )


def evaluate_convention(data: CalibData, convention: Convention) -> Metrics:
    """То же, что :func:`evaluate`, но по объекту :class:`Convention`."""
    return evaluate(
        data,
        convention.rotation,
        invert_quat=convention.invert_quat,
        camera_offset=convention.camera_offset,
    )


def convention_from_config() -> Convention:
    """Конвенция, собранная из ``src/config.py`` (07c), — проверка калибровки без перебора.

    Обратная операция к :meth:`Convention.head_cam`: ``rotation`` — это ``A = Rᵀ``
    поворота ``T_head_cam`` (``cam → head``), ``camera_offset`` — сдвиг из конфига,
    ``invert_quat`` — флаг оттуда же. Значения берутся через
    :func:`camera_model.check_config`, поэтому противоречивый конфиг (запись осей не
    совпадает с матрицей, ``det ≠ +1``, флаг не ``bool``) падает здесь, а не в метрике.
    :func:`evaluate_convention` на этой конвенции обязан воспроизвести метрику 07b:
    39 из 39 суставов в рамках своих рук, 0 чужих, 0 промахов, 0.0 px.
    """
    summary = camera_model.check_config()
    matrix = summary["head_cam_matrix"]
    rotation: Matrix3 = tuple(  # type: ignore[assignment]
        tuple(matrix[i][j] for j in range(3)) for i in range(3)
    )
    return Convention(
        rotation=_transpose3(rotation),
        invert_quat=summary["invert_quat"],
        camera_offset=summary["head_cam_translation_m"],  # type: ignore[arg-type]
    )


def search_discrete(
    data: CalibData,
    offsets: Iterable[tuple[float, float, float]] = offset_grid(),
) -> tuple[Candidate, ...]:
    """Перебор 48 конвенций осей по сетке смещения: для каждой — лучший вариант.

    Возвращает всех кандидатов, отсортированных по :attr:`Candidate.key`, — в таблицу
    top-5 попадают первые. Для каждой конвенции берётся только лучший сдвиг, поэтому
    в таблице одна строка на конвенцию, а не на 1331 положение камеры.
    """
    grid = tuple(offsets)
    if not grid:
        raise ValueError("offsets: пустая сетка смещения")
    best: list[Candidate] = []
    for rotation in axis_rotations(proper=True):
        for invert in (False, True):
            winner: Candidate | None = None
            for offset in grid:
                convention = Convention(
                    rotation=rotation, invert_quat=invert, camera_offset=offset
                )
                candidate = Candidate(convention, evaluate_convention(data, convention))
                if winner is None or candidate.key < winner.key:
                    winner = candidate
            if winner is not None:
                best.append(winner)
    return rank(best)


def rotation_grid(
    base_rpy: Sequence[float],
    step_deg: float = ANGLE_STEP_DEG,
    limit_deg: float = ANGLE_RANGE_DEG,
) -> tuple[tuple[float, float, float], ...]:
    """Сетка абсолютных ``(yaw, pitch, roll)`` вокруг ``base_rpy`` с шагом ``step_deg``."""
    count = int(round(limit_deg / step_deg))
    yaw, pitch, roll = (float(value) for value in base_rpy)
    return tuple(
        (yaw + dy, pitch + dp, roll + dr)
        for dy, dp, dr in itertools.product(
            [index * step_deg for index in range(-count, count + 1)], repeat=3
        )
    )


def refine(
    data: CalibData,
    candidates: Sequence[Candidate],
    step_deg: float = ANGLE_STEP_DEG,
    angle_limit_deg: float = ANGLE_RANGE_DEG,
    step_m: float = OFFSET_STEP_M,
    offset_limit_m: float = OFFSET_RANGE_M,
) -> tuple[tuple[Candidate, ...], dict]:
    """Уточнить top-3 грубой сеткой поворотов и смещения (координатный спуск).

    Порядок как в карточке: сначала повороты ``yaw/pitch/roll`` ±20° с шагом 5° при
    найденном смещении, затем смещение ±10 см с шагом 2 см при лучшем повороте, затем
    ещё один проход по поворотам. Возвращает лучших кандидатов (по одному на входной) и
    сводку: сколько угловых и сдвиговых проверок выполнено и у скольких вариантов
    уточнение изменило результат.
    """
    offsets = offset_grid(step_m, offset_limit_m)
    refined: list[Candidate] = []
    summary: dict = {"rotation_cells": 0, "offset_cells": 0, "changed": 0}
    def try_convention(convention: Convention, best: Candidate) -> Candidate:
        """Построить кандидата, посчитать метрику и оставить лучшего по ``key``."""
        metrics = evaluate_convention(data, convention)
        candidate = Candidate(convention, metrics)
        return candidate if candidate.key < best.key else best

    for base in candidates:
        best = base
        # 1) повороты вокруг дискретной ориентации: C = R_y·R_x·R_z, A = Cᵀ
        base_rpy = rpy_from_matrix(_transpose3(base.convention.rotation))
        angles = rotation_grid(base_rpy, step_deg, angle_limit_deg)
        summary["rotation_cells"] += len(angles)
        for yaw, pitch, roll in angles:
            best = try_convention(
                Convention(
                    rotation=_transpose3(camera_model.rotation_rpy(yaw, pitch, roll)),
                    invert_quat=base.convention.invert_quat,
                    camera_offset=base.convention.camera_offset,
                    perturbation_deg=max(abs(yaw - base_rpy[0]), abs(pitch - base_rpy[1]),
                                         abs(roll - base_rpy[2])),
                ),
                best,
            )
        # 2) смещение камеры при лучшем повороте
        summary["offset_cells"] += len(offsets)
        for offset in offsets:
            best = try_convention(
                Convention(
                    rotation=best.convention.rotation,
                    invert_quat=best.convention.invert_quat,
                    camera_offset=offset,
                    perturbation_deg=best.convention.perturbation_deg,
                ),
                best,
            )
        # 3) ещё один проход по поворотам при лучшем смещении
        base_rpy = rpy_from_matrix(_transpose3(best.convention.rotation))
        for yaw, pitch, roll in angles:
            best = try_convention(
                Convention(
                    rotation=_transpose3(camera_model.rotation_rpy(yaw, pitch, roll)),
                    invert_quat=best.convention.invert_quat,
                    camera_offset=best.convention.camera_offset,
                    perturbation_deg=max(abs(yaw - base_rpy[0]), abs(pitch - base_rpy[1]),
                                         abs(roll - base_rpy[2])),
                ),
                best,
            )
        summary["changed"] += int(best.key != base.key)
        refined.append(best)
    return tuple(refined), summary


def sensitivity(
    data: CalibData,
    convention: Convention,
    step_deg: float = ANGLE_STEP_DEG,
    angle_limit_deg: float = ANGLE_RANGE_DEG,
    step_m: float = OFFSET_STEP_M,
    offset_limit_m: float = OFFSET_RANGE_M,
) -> dict:
    """Насколько метрика чувствительна к каждому параметру победителя (границы плато).

    Для каждой оси смещения и каждого угла перебираются одномерные сдвиги (остальные
    параметры фиксированы) и считается максимальный сдвиг, при котором ошибка ещё
    нулевая. Это и есть разрешение эталона: внутри плато любые параметры одинаково
    хороши, поэтому 07c фиксирует минимальную правку, а не «середину плато».
    """
    base_rpy = rpy_from_matrix(_transpose3(convention.rotation))

    def zero(offset: Sequence[float], angles: Sequence[float]) -> bool:
        body = camera_model.rotation_rpy(*angles)
        metrics = evaluate(
            data,
            _transpose3(body),
            invert_quat=convention.invert_quat,
            camera_offset=offset,
        )
        return metrics.mean_error_px == 0.0 and metrics.violations == 0

    angles = list(base_rpy)
    offset = list(convention.camera_offset)
    report: dict = {"offset_m": {}, "angle_deg": {}}
    for axis, name in enumerate("xyz"):
        limit = 0.0
        for delta in offset_values(step_m, offset_limit_m):
            if delta == 0.0:
                continue
            probe = list(offset)
            probe[axis] += delta
            if zero(probe, angles):
                limit = max(limit, abs(delta))
        report["offset_m"][name] = round(limit, 6)
    for index, name in enumerate(("yaw", "pitch", "roll")):
        limit = 0.0
        for delta in [value * step_deg for value in range(-int(angle_limit_deg / step_deg),
                                                         int(angle_limit_deg / step_deg) + 1)]:
            if delta == 0.0:
                continue
            probe = list(angles)
            probe[index] += delta
            if zero(offset, probe):
                limit = max(limit, abs(delta))
        report["angle_deg"][name] = round(limit, 6)
    return report


def winner_payload(
    winner: Candidate, plateau: dict, top: Sequence[Candidate], summary: dict
) -> dict:
    """JSON для задачи 07c: параметры победителя, метрика, плато и top-5."""
    convention = winner.convention
    body = _transpose3(convention.rotation)
    yaw, pitch, roll = rpy_from_matrix(body)
    return {
        "task": "07b",
        "source": "src/calib_search.py",
        "frames": [
            {"index": frame.index, "t_s": frame.t_s} for frame in calib_frames.CALIB_FRAMES
        ],
        "tolerance_px": TOLERANCE_PX,
        "max_mean_error_px": MAX_MEAN_ERROR_PX,
        "winner": {
            "convention": convention.name,
            "axes_head_to_camera": convention.axes,
            "head_to_camera_rotation": [list(row) for row in convention.rotation],
            "invert_quat": convention.invert_quat,
            "opencv_conventions": False,
            "head_cam_rotation_rpy_deg": [yaw, pitch, roll],
            "head_cam_translation_m": list(convention.camera_offset),
            "head_cam_matrix": [list(row) for row in convention.head_cam()],
            "camera_frame_translation": list(convention.camera_translation),
            "metrics": winner.metrics.as_dict(),
            "criterion_met": winner.metrics.mean_error_px <= MAX_MEAN_ERROR_PX
            and winner.metrics.violations == 0,
        },
        "plateau": plateau,
        "refinement": summary,
        "top": [candidate.as_dict() for candidate in top],
    }


def format_table(candidates: Sequence[Candidate]) -> str:
    """Таблица top-N: конвенция, смещение, покрытие, нарушения и метрика."""
    lines = [
        f"{'#':>2}  {'оси (камера от головы)':<10}{'invert_quat':>12}{'c, м':>22}"
        f"{'в кадре':>9}{'внутри':>8}{'чужая':>7}{'мимо':>6}{'ошибка, px':>12}",
        "-" * 90,
    ]
    best_coverage = max((candidate.metrics.in_frame for candidate in candidates), default=0)
    for number, candidate in enumerate(candidates, start=1):
        m = candidate.metrics
        c = candidate.convention.camera_offset
        text = f"({c[0]:g}, {c[1]:g}, {c[2]:g})"
        mark = " " if m.in_frame >= best_coverage - COVERAGE_TOLERANCE else "*"
        lines.append(
            f"{number:>2}{mark} {candidate.convention.axes:<10}"
            f"{str(candidate.convention.invert_quat):>12}{text:>22}"
            f"{f'{m.in_frame}/{m.total}':>9}{m.good:>8}{m.wrong:>7}{m.miss:>6}"
            f"{m.mean_error_px:>12.1f}"
        )
    if any(
        candidate.metrics.in_frame < best_coverage - COVERAGE_TOLERANCE
        for candidate in candidates
    ):
        lines.append(
            f"* меньше {COVERAGE_TOLERANCE} точек от лучшего покрытия: в кадре слишком мало "
            "суставов — ошибка сравнима формально, но не по существу"
        )
    return "\n".join(lines)


def _overlay_tile(
    frame_index: int,
    pose: sync.Pose,
    convention: Convention,
    intrinsics: camera_model.Intrinsics,
    image: Any,
    boxes: Sequence[calib_frames.HandBox],
) -> Any:
    """Наложить проекцию кистей на левую половину кадра — одна плитка контактного листа.

    Рисуются 5 суставов каждой руки (крестики + подписи), линии между ними и
    прямоугольники эталона 07a: так на картинке сразу видно и попадание в разметку, и
    саму позу скелета. Плитка уменьшается вдвое, чтобы 5 кадров вставали в лист.
    """
    import cv2  # noqa: PLC0415 — локальный импорт: без OpenCV модуль подбора работает

    width = calib_frames.EYE_WIDTH
    height = calib_frames.EYE_HEIGHT
    view = image[:, :width].copy()
    for box in boxes:
        color = {"left": (255, 128, 0), "right": (0, 0, 255)}.get(box.hand, (0, 200, 0))
        cv2.rectangle(view, (box.bbox[0], box.bbox[1]), (box.bbox[2], box.bbox[3]), color, 1)
    matrix = convention.matrix_for(pose.head_pos, pose.head_quat)
    colors = {"left": (255, 128, 0), "right": (0, 0, 255)}
    for side in ("left", "right"):
        pixels = []
        for joint in JOINTS:
            world = pose.hands[f"{side}_{joint}"]
            if not all(math.isfinite(value) for value in world):
                continue
            camera = (
                sum(matrix[0][axis] * world[axis] for axis in range(3)) + matrix[0][3],
                sum(matrix[1][axis] * world[axis] for axis in range(3)) + matrix[1][3],
                sum(matrix[2][axis] * world[axis] for axis in range(3)) + matrix[2][3],
            )
            if camera[2] <= camera_model.MIN_DEPTH_M:
                continue
            u = intrinsics.fx * camera[0] / camera[2] + intrinsics.cx
            v = intrinsics.fy * camera[1] / camera[2] + intrinsics.cy
            pixels.append((u, v))
            if 0 <= u < width and 0 <= v < height:
                cv2.drawMarker(
                    view, (int(round(u)), int(round(v))), colors[side], cv2.MARKER_CROSS, 14, 2
                )
                cv2.putText(
                    view, joint[:2], (int(round(u)) + 8, int(round(v)) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[side], 1, cv2.LINE_AA,
                )
        for start_point, end_point in zip(pixels, pixels[1:]):
            cv2.line(
                view,
                (int(round(start_point[0])), int(round(start_point[1]))),
                (int(round(end_point[0])), int(round(end_point[1]))),
                colors[side], 1, cv2.LINE_AA,
            )
    caption = f"k={frame_index}  {convention.axes}  c={convention.camera_offset}"
    cv2.putText(view, caption, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    tile = cv2.resize(view, (width // 2, height // 2))
    cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (60, 60, 60), 1)
    return tile


def save_overlay_sheets(
    video: str | Path,
    tracking: str | Path,
    data: CalibData,
    candidates: Sequence[Candidate],
    out_dir: str | Path = DEFAULT_OUT,
    meta_path: str | Path = "output/video_meta.json",
    offset_ns: int | None = None,
) -> list[Path]:
    """Сохранить контактный лист (5 кадров) с наложением для top-3 — ``top1..3.png``.

    OpenCV и видео нужны только здесь: подбор, метрика и JSON работают без них.
    Листы — регенерируемый артефакт в ``output/`` (в git не идут). ``offset_ns`` — поправка
    времени кадра (``None`` → ``config.VIDEO_TIME_OFFSET_MS``), та же, что в рендере 08a.
    """
    import cv2  # noqa: PLC0415

    video = Path(video)
    if not video.is_file():
        raise SystemExit(f"нет файла: {video}")
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    header = read_header(tracking)
    header_ns = int(header["timeStampNs"])
    tracking_end_ns = timestamp_range(tracking)[1]
    track = sync.load_poses(tracking)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"не открывается видео: {video}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        for number, candidate in enumerate(candidates[:TOP_OVERLAYS], start=1):
            tiles = []
            for frame in data.frames:
                moment = sync.frame_time(frame.index, meta, header_ns, tracking_end_ns)
                pose = sync.pose_at_frame(track, moment, offset_ns)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame.index)
                ok, image = capture.read()
                if not ok:
                    raise SystemExit(f"не читается кадр k={frame.index}")
                tiles.append(
                    _overlay_tile(
                        frame.index, pose, candidate.convention, data.intrinsics, image, frame.hands
                    )
                )
            while len(tiles) % 3:
                tiles.append(tiles[0] * 0)
            rows = [cv2.hconcat(tiles[index:index + 3]) for index in range(0, len(tiles), 3)]
            path = out / f"top{number}.png"
            cv2.imwrite(str(path), cv2.vconcat(rows))
            written.append(path)
    finally:
        capture.release()
    return written


def _verify_config(data: CalibData) -> int:
    """Режим ``--verify-config`` (07c): метрика конвенции из ``src/config.py`` на эталоне 07a.

    Никакого перебора: параметры берутся из конфига, поэтому код проверки — ровно тот
    путь, которым пойдёт рендер (08a). Ненулевая ошибка или чужие руки означают, что
    конфиг «расползся» относительно калибровки 07b, и это возвращает код 1.
    """
    try:
        summary = camera_model.check_config()
        convention = convention_from_config()
    except (ValueError, TypeError) as exc:
        print(f"ошибка конфига: {exc}", file=sys.stderr)
        return 1
    print("\nкалибровка из src/config.py (07c):")
    print(f"  AXIS_CONVENTION = {summary['axis_convention']}, "
          f"INVERT_QUAT = {summary['invert_quat']}, Y_DOWN_FLIP = {summary['y_down_flip']}")
    print(f"  HEAD_CAM_ROTATION_RPY_DEG = {summary['head_cam_rotation_rpy_deg']}, "
          f"HEAD_CAM_TRANSLATION_M = {summary['head_cam_translation_m']}")
    print(f"  поправка времени кадра (config.VIDEO_TIME_OFFSET_MS): "
          f"{sync.video_time_offset_ns() / 1e6:+.0f} мс")
    print("  проверка конфига: запись осей совпадает с матрицей поворота, det = +1")
    metrics = evaluate_convention(data, convention)
    print(f"  метрика на эталоне: {metrics.good}/{metrics.in_frame} суставов в кадре — в рамках "
          f"своих рук; чужих {metrics.wrong}, мимо {metrics.miss}, "
          f"средняя ошибка {metrics.mean_error_px:.1f} px "
          f"(всего {metrics.total}, {metrics.total - metrics.in_frame} законно вне кадра)")
    criterion = metrics.mean_error_px <= MAX_MEAN_ERROR_PX and metrics.violations == 0
    print(f"  критерий 07b (≤ {MAX_MEAN_ERROR_PX:g} px и без чужих рук): "
          f"{'выполнен' if criterion else 'НЕ выполнен'}")
    return 0 if criterion else 1


def main(argv: Iterable[str] | None = None) -> int:
    """CLI: перебор 48 конвенций, уточнение top-3, метрика, PNG и JSON для 07c.

    ``--verify-config`` вместо перебора проверяет уже зафиксированную в ``src/config.py``
    калибровку (07c) на эталоне 07a: именно этот путь пойдёт рендер.
    """
    parser = argparse.ArgumentParser(
        description="Подбор конвенции осей и грубого T_head_cam по 5 кадрам с разметкой (07b)."
    )
    parser.add_argument("tracking", type=Path, help="файл трекинга, например data/trackingData_*.txt")
    parser.add_argument(
        "--meta", type=Path, default=Path("output/video_meta.json"),
        help="video_meta.json с duration/frame_count (по умолчанию output/video_meta.json)",
    )
    parser.add_argument(
        "--video", type=Path, default=None,
        help="видео для PNG с наложением (data/CameraRecord_*.mp4); без него PNG не пишутся",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, help="куда писать PNG и JSON (output/debug/calib)",
    )
    parser.add_argument("--top", type=int, default=TOP_TABLE, help="сколько строк таблицы печатать")
    parser.add_argument(
        "--no-refine", action="store_true", help="пропустить уточнение top-3 (быстрее)"
    )
    parser.add_argument(
        "--verify-config", action="store_true",
        help="не искать, а проверить калибровку src/config.py на эталоне 07a (07c)",
    )
    parser.add_argument("--json", action="store_true", help="напечатать JSON победителя в stdout")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        data = load_calib_data(args.tracking, args.meta)
    except (OSError, ValueError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1

    print(f"эталон 07a: {len(data.frames)} кадров, {data.total} суставов "
          f"(левый глаз {calib_frames.EYE_WIDTH}×{calib_frames.EYE_HEIGHT}, "
          f"допуск разметки ±{TOLERANCE_PX} px)")

    if args.verify_config:
        return _verify_config(data)
    print("перебор: 24 собственные ротации осей × 2 конвенции кватерниона × "
          f"{len(offset_grid())} положений камеры")
    candidates = search_discrete(data)
    print("\n" + format_table(candidates[:args.top]))
    print("\nперебор конвенций без уточнения: лучший вариант и есть окно выше")

    if not args.no_refine:
        refined, summary = refine(data, candidates[:TOP_OVERLAYS])
        print(f"\nуточнение top-{TOP_OVERLAYS}: повороты ±{ANGLE_RANGE_DEG:g}° шаг "
              f"{ANGLE_STEP_DEG:g}°, смещение ±{OFFSET_RANGE_M:g} м шаг {OFFSET_STEP_M:g} м"
              f" ({summary['rotation_cells']} угловых и {summary['offset_cells']} "
              f"сдвиговых проб на вариант); изменили результат: {summary['changed']}"
              f" из {len(refined)}")
        print(format_table(refined))
        pool = list(candidates)
        pool.extend(refined)
        candidates = rank(pool)
    else:
        summary = {"rotation_cells": 0, "offset_cells": 0, "changed": 0}

    winner = candidates[0]
    plateau = sensitivity(data, winner.convention)
    print("\nчувствительность победителя (до какого сдвига ошибка остаётся нулевой):")
    print("  смещение камеры, м: " + ", ".join(
        f"{axis} ≤ {value:g}" for axis, value in plateau["offset_m"].items()))
    print("  поворот камеры, °: " + ", ".join(
        f"{name} ≤ {value:g}" for name, value in plateau["angle_deg"].items()))
    print("  → внутри этого плато эталон не различает параметры; берётся минимальная "
          "правка (нулевое смещение и нулевой добавочный поворот)")

    print("\nпобедитель:")
    print(f"  оси (камера от головы): {winner.convention.axes}")
    print(f"  invert_quat: {winner.convention.invert_quat}, "
          f"opencv_conventions: False (переворот y входит в ориентацию камеры)")
    print(f"  смещение камеры (система головы, x вправо / y вверх / z назад): "
          f"{winner.convention.camera_offset}")
    yaw, pitch, roll = rpy_from_matrix(_transpose3(winner.convention.rotation))
    print(f"  для src/config.py: HEAD_CAM_ROTATION_RPY_DEG = ({yaw:g}, {pitch:g}, {roll:g}), "
          f"HEAD_CAM_TRANSLATION_M = {winner.convention.camera_offset}")
    metrics = winner.metrics
    criterion = metrics.mean_error_px <= MAX_MEAN_ERROR_PX and metrics.violations == 0
    print(f"  метрика: {metrics.good}/{metrics.in_frame} точек в своих рамках, "
          f"чужих {metrics.wrong}, мимо {metrics.miss}, "
          f"средняя ошибка {metrics.mean_error_px:.1f} px")
    print(f"  критерий карточки (≤ {MAX_MEAN_ERROR_PX:g} px и без перепутанных рук): "
          f"{'выполнен' if criterion else 'НЕ выполнен'}")

    payload = winner_payload(winner, plateau, candidates[:args.top], summary)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "calibration.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  параметры для 07c: {json_path}")
    if args.video is not None:
        paths = save_overlay_sheets(
            args.video, args.tracking, data, candidates[:TOP_OVERLAYS], args.out, args.meta
        )
        for path in paths:
            print(f"  наложение: {path}")
    else:
        print("  PNG с наложением: не записаны (укажите --video data/CameraRecord_*.mp4)")
    if args.json:
        print()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if criterion else 1


if __name__ == "__main__":
    raise SystemExit(main())
