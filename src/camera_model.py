"""camera_model.py — модель камеры: интринсики (05a), world→camera (05b), project (05c).

Задача 05a: ``Intrinsics`` — разобранная строка ``cameraIntrinsics`` из header'а файла
трекинга. В файле она записана как ``[cx, cy, fx, fy]`` полного SBS-кадра 2160×810:

    1079.5 = 2160/2 − 0.5,  404.5 = 810/2 − 0.5   →  это cx, cy (half-pixel конвенция);
    1373.668 ≈ 2 × 686.868 (с точностью 0.068 px) →  fx посчитан от ПОЛНОЙ ширины SBS.

В header нет width/height, поэтому размер кадра выводится из центра: ``2·cx + 1`` и
``2·cy + 1`` обязаны быть целыми — это же отсекает порядок ``[fx, fy, cx, cy]``
(там ``2·cx + 1 = 2748.336`` — не целое). Вывод и уточнения — в ``docs/intrinsics.md``.

Для одного глаза (1080×810) пиксели квадратные: ``fx = fy = 686.868``, ``cx = 539.5``
(правый глаз: ``539.5 + 1080 = 1619.5``), ``cy = 404.5``; FOV ≈ 76.35° × 61.05°,
диагональ ≈ 89.00°. Разница ``fx_full/2`` и ``fy`` — 0.034 px (0.005 %), на проекцию
не влияет; выбран ``fx_eye = fy``, чтобы пиксели были строго квадратными.

Задача 05b: поза головы на кадр (позиция + кватернион ``x, y, z, w``) → координаты
камеры. Экстринсики из header'а не используются (их поворот расходится с ``Head.pose``
на ~36°), поэтому камера считается жёстко закреплённой в голове:

    p_cam = F · T_head_cam⁻¹ · T_head_world⁻¹ · p_world

* ``T_head_world`` — поза головы (:func:`head_to_world`: поворот из кватерниона, сдвиг —
  позиция головы); ``⁻¹`` переводит точку из мира в систему головы.
* ``T_head_cam`` — поза камеры в системе головы (:func:`head_cam_matrix`, константа из
  ``src/config.py``; заглушка 05b — без поворота, 5 см вперёд, финал — 07c).
* ``F = diag(1, −1, 1)`` — переход Unity (``y`` вверх) → OpenCV (``y`` вниз) по флагу
  ``opencv_conventions``: камера смотрит по ``+z``, ``x`` вправо, ``y`` вниз, как ждут
  интринсики из 05a и ``cv2.projectPoints``. Нужен отдельным флагом, потому что это
  конвенция осей экрана, а не свойство «камера в голове».

Конвенция кватерниона — отдельный флаг ``invert_quat`` (``False`` по умолчанию — поза
понимается как поворот головы в мире, как в спецификации формата). Замер на данных:
при ``invert_quat=False`` **0 %** из 8 690 точек кистей оказываются перед головой, при
``True`` — **92.7 %**; значит в записи кватернион ведёт себя как обратный поворот
(Unity-подобная левосторонняя система). Окончательный выбор — за 07b, где вариант
подбирается по наложению на кадры, а не по этой грубой статистике.

В карточке 05b композиция записана как ``T_cam_world(t) = T_head_world(t) @ T_head_cam``:
это та же формула, если ``T_head_cam`` — поза камеры в голове (переводит координаты
камеры в координаты головы), а для координат камеры берётся обратная композиция.

Задача 05c (``project``) сюда не входит.

Запуск:
    python src/camera_model.py data/trackingData_20260505_165740.txt
    python src/camera_model.py data/trackingData_20260505_165740.txt --world-to-camera
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.parse_tracking import (  # noqa: E402
    QUAT_NORM_TOL,
    count_samples,
    extract_sample,
    iter_samples,
    parse_pose,
    read_header,
)

#: Допуск проверки центра кадра и целочисленности выведенного размера (px).
SIZE_TOL = 1e-6


@dataclass(frozen=True)
class Intrinsics:
    """Пинхол-интринсики кадра ``full_w × full_h`` (для одного глаза — ``eye_w × eye_h``).

    Поля — ровно ``fx, fy, cx, cy, full_w, full_h, eye_w, eye_h``; размеры — целые
    пиксели, фокусные и центр — конечные положительные числа. Для полного SBS-кадра
    ``full_w = 2·cx + 1 = 2160``, поэтому ``eye_w == full_w // 2`` задаёт ширину одного
    глаза. В per-eye модели ``full_w/full_h`` — размер изображения глаза, а ``eye_w``
    по тому же инварианту равна половине этой ширины (в FOV и проекции не участвует).

    Полный SBS-кадр: ``fx=1373.668``, ``fy=686.868``, ``cx=1079.5``, ``cy=404.5``,
    ``full_w/full_h=2160/810``, ``eye_w/eye_h=1080/810``.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    full_w: int
    full_h: int
    eye_w: int
    eye_h: int

    def __post_init__(self) -> None:
        for name in ("fx", "fy", "cx", "cy"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name}: ожидалось число, получено {value!r}")
            if not math.isfinite(value):
                raise ValueError(f"{name}: нефинитное значение {value!r}")
            if value <= 0:
                raise ValueError(f"{name}: должно быть > 0, получено {value!r}")
        for name in ("full_w", "full_h", "eye_w", "eye_h"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name}: ожидалось целое число, получено {value!r}")
            if value <= 0:
                raise ValueError(f"{name}: должно быть > 0, получено {value!r}")
        if self.full_w % 2 != 0:
            raise ValueError(f"full_w: для SBS должно быть чётным, получено {self.full_w}")
        if self.eye_w != self.full_w // 2:
            raise ValueError(
                f"eye_w: ожидалось full_w // 2 = {self.full_w // 2}, получено {self.eye_w}"
            )

    @classmethod
    def from_header(cls, header: dict, frame_size: tuple[int, int] | None = None) -> "Intrinsics":
        """Разобрать ``header["cameraIntrinsics"]`` как ``[cx, cy, fx, fy]`` (файловый порядок).

        ``frame_size=None`` — размер кадра выводится из центра по half-pixel конвенции
        OpenCV: ``full_w = 2·cx + 1``, ``full_h = 2·cy + 1``. Оба значения обязаны быть
        целыми (допуск :data:`SIZE_TOL`); иначе ``ValueError`` — это ловит перепутанный
        порядок полей, например ``[fx, fy, cx, cy]`` (там ``2·cx + 1 = 2748.336``).

        ``frame_size=(w, h)`` — размер известен из видео: тогда центр обязан совпадать с
        ``(w/2 − 0.5, h/2 − 0.5)`` с точностью :data:`SIZE_TOL`, иначе ``ValueError``.

        Допущения: модель пинхол (дисторсии в header нет), число глаз — два (SBS),
        ``fx/fy`` — полного кадра, ``eye_w = full_w // 2``.
        """
        if not isinstance(header, dict) or "cameraIntrinsics" not in header:
            raise ValueError("header: нет поля cameraIntrinsics")
        raw = header["cameraIntrinsics"]
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise ValueError(f"cameraIntrinsics: ожидалось 4 числа [cx, cy, fx, fy], получено {raw!r}")
        numbers = []
        for value in raw:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"cameraIntrinsics: не число: {value!r}")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"cameraIntrinsics: нефинитное значение {value!r}")
            if number <= 0:
                raise ValueError(f"cameraIntrinsics: значения должны быть > 0, получено {value!r}")
            numbers.append(number)
        cx, cy, fx, fy = numbers

        if frame_size is None:
            dims = []
            for name, coord, value in (("full_w", "cx", cx), ("full_h", "cy", cy)):
                derived = 2.0 * value + 1.0
                nearest = round(derived)
                if abs(derived - nearest) > SIZE_TOL:
                    raise ValueError(
                        f"{name}: 2·{coord} + 1 = {derived!r} — не целое; "
                        "порядок полей, вероятно, не [cx, cy, fx, fy]"
                    )
                dims.append(int(nearest))
            full_w, full_h = dims
        else:
            if not isinstance(frame_size, (list, tuple)) or len(frame_size) != 2:
                raise ValueError(f"frame_size: ожидалось (width, height), получено {frame_size!r}")
            full_w, full_h = frame_size
            for name, value in (("frame_size.width", full_w), ("frame_size.height", full_h)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{name}: ожидалось целое число, получено {value!r}")
                if value <= 0:
                    raise ValueError(f"{name}: должно быть > 0, получено {value!r}")
            for name, coord, expected in (
                ("cx", cx, full_w / 2.0 - 0.5),
                ("cy", cy, full_h / 2.0 - 0.5),
            ):
                if abs(coord - expected) > SIZE_TOL:
                    raise ValueError(
                        f"{name}: {coord!r} не совпадает с центром кадра {full_w}×{full_h} "
                        f"(ожидалось {expected!r}, half-pixel конвенция)"
                    )

        return cls(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            full_w=full_w,
            full_h=full_h,
            eye_w=full_w // 2,
            eye_h=full_h,
        )

    def per_eye(self, side: str) -> "Intrinsics":
        """Интринсики одного глаза SBS-пары: ``"left"`` или ``"right"`` (иначе ``ValueError``).

        Для глаза берётся ``fx = fy = fy`` полного кадра (пиксели квадратные), центр
        ``cx = 540 − 0.5 = 539.5`` у левого и ``539.5 + 1080 = 1619.5`` у правого — обе
        координаты в полном SBS-кадре. После обрезки правой половины (``frame[:, 1080:]``)
        нужно вычесть 1080 (``cx = 539.5``), иначе проекция уедет на ширину глаза.
        ``cy = 404.5``, размер изображения глаза — 1080×810.
        """
        if side not in ("left", "right"):
            raise ValueError(f"side: ожидалось 'left' или 'right', получено {side!r}")
        eye_cx = self.eye_w / 2.0 - 0.5
        if side == "right":
            eye_cx += self.eye_w
        return Intrinsics(
            fx=self.fy,
            fy=self.fy,
            cx=eye_cx,
            cy=self.eye_h / 2.0 - 0.5,
            full_w=self.eye_w,
            full_h=self.eye_h,
            eye_w=self.eye_w // 2,
            eye_h=self.eye_h,
        )

    def fov_deg(self) -> tuple[float, float, float]:
        """Поле зрения ``(h, v, diag)`` в градусах: ``2·atan(half/f)`` по ``full_w/full_h``."""
        tan_h = (self.full_w / 2.0) / self.fx
        tan_v = (self.full_h / 2.0) / self.fy
        return (
            2.0 * math.degrees(math.atan(tan_h)),
            2.0 * math.degrees(math.atan(tan_v)),
            2.0 * math.degrees(math.atan(math.hypot(tan_h, tan_v))),
        )


# --- 05b: поза головы → координаты камеры ---------------------------------------------------

#: Матрица 3×3 как три строки-кортежа (строка — коэффициенты при x, y, z).
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

#: Однородная матрица 4×4 как четыре строки-кортежа; точка — столбец ``(x, y, z, 1)``.
Matrix4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]

#: Единичная 4×4 — «камера совпадает с головой» без поворота и сдвига.
IDENTITY4: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

#: Переход Unity → OpenCV по осям: ``x`` вправо и ``z`` вперёд совпадают, ``y`` меняет знак.
#: Применяется к координатам камеры, когда ``opencv_conventions=True`` (по умолчанию):
#: в Unity (и в позе головы из данных) ``y`` смотрит вверх, в OpenCV-изображении — вниз.
OPENCV_FLIP_Y: Matrix4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

#: Допуск вырожденности при обращении 4×4: |det| меньше — матрица считается необратимой.
SINGULAR_TOL = 1e-12


def _number(value: object, what: str) -> float:
    """Конечное число из входных данных (``bool`` числом не считается)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what}: ожидалось число, получено {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what}: нефинитное значение {value!r}")
    return number


def _vector(values: object, length: int, what: str) -> tuple[float, ...]:
    """Ровно ``length`` конечных чисел из последовательности (строка/байты — ошибка)."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{what}: ожидалась последовательность чисел, получено {values!r}")
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError(
            f"{what}: ожидалась последовательность из {length} чисел, получено {values!r}"
        ) from None
    if len(items) != length:
        raise ValueError(f"{what}: ожидалось {length} чисел, получено {len(items)}")
    return tuple(_number(item, f"{what}[{index}]") for index, item in enumerate(items))


def _matrix4(matrix: object, what: str) -> Matrix4:
    """4×4 из вложенной последовательности 4×4 (или плоского списка 16 чисел)."""
    if isinstance(matrix, (str, bytes)):
        raise ValueError(f"{what}: ожидалась матрица 4×4, получено {matrix!r}")
    try:
        rows = list(matrix)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError(f"{what}: ожидалась матрица 4×4, получено {matrix!r}") from None
    if len(rows) == 16 and all(isinstance(row, (int, float)) for row in rows):
        flat = _vector(rows, 16, what)
        return tuple(  # type: ignore[return-value]
            tuple(flat[index : index + 4]) for index in range(0, 16, 4)
        )
    if len(rows) != 4:
        raise ValueError(f"{what}: ожидалось 4 строки, получено {len(rows)}")
    return tuple(  # type: ignore[return-value]
        _vector(row, 4, f"{what}[{index}]") for index, row in enumerate(rows)
    )


def _matmul3(a: Matrix3, b: Matrix3) -> Matrix3:
    """Произведение 3×3: ``a · b`` (применяется к столбцу как ``a·(b·p)``)."""
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )


def _transpose3(matrix: Matrix3) -> Matrix3:
    """Транспонированная 3×3 — обратная для поворота (``Rᵀ = R⁻¹``)."""
    return tuple(  # type: ignore[return-value]
        tuple(matrix[j][i] for j in range(3)) for i in range(3)
    )


def _matmul4(a: Matrix4, b: Matrix4) -> Matrix4:
    """Произведение 4×4: ``a · b``."""
    return tuple(  # type: ignore[return-value]
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)) for i in range(4)
    )


def _homogeneous(rotation: Matrix3, translation: Sequence[float]) -> Matrix4:
    """4×4 из поворота 3×3 и сдвига: ``[R | t; 0 0 0 1]``."""
    tx, ty, tz = translation
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], tx),
        (rotation[1][0], rotation[1][1], rotation[1][2], ty),
        (rotation[2][0], rotation[2][1], rotation[2][2], tz),
        (0.0, 0.0, 0.0, 1.0),
    )


def _minor3(matrix: Matrix4, row: int, column: int) -> Matrix3:
    """Минор 3×3: матрица без строки ``row`` и столбца ``column``."""
    keep_rows = [i for i in range(4) if i != row]
    keep_columns = [j for j in range(4) if j != column]
    return tuple(  # type: ignore[return-value]
        tuple(matrix[i][j] for j in keep_columns) for i in keep_rows
    )


def _det3(matrix: Matrix3) -> float:
    """Определитель 3×3 по правилу треугольников."""
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _cofactor4(matrix: Matrix4, row: int, column: int) -> float:
    """Алгебраическое дополнение элемента ``(row, column)``: знак + минор 3×3."""
    minor = _det3(_minor3(matrix, row, column))
    return -minor if (row + column) % 2 else minor


def _inverse4(matrix: Matrix4, what: str = "матрица") -> Matrix4:
    """Обратная 4×4 через алгебраические дополнения.

    ``ValueError`` при ``|det| < SINGULAR_TOL``.

    Общая обратная, а не «транспонировать поворот»: в 07b через ``t_head_cam`` могут
    прийти матрицы перестановки/знаков осей (в том числе с det = −1), и они должны
    работать без особых случаев. Для жёстких матриц результат тот же.
    """
    det = sum(matrix[0][j] * _cofactor4(matrix, 0, j) for j in range(4))
    if abs(det) < SINGULAR_TOL:
        raise ValueError(f"{what}: матрица необратима (det = {det:g})")
    return tuple(  # type: ignore[return-value]
        tuple(_cofactor4(matrix, j, i) / det for j in range(4)) for i in range(4)
    )


def _matvec4(matrix: Matrix4, point: Sequence[float]) -> tuple[float, float, float]:
    """``matrix · (x, y, z, 1)`` → ``(x, y, z)`` однородных координат (``w`` не делим)."""
    x, y, z = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
    )


def quat_to_rotation(quat: Sequence[float]) -> Matrix3:
    """Кватернион ``x, y, z, w`` → матрица поворота 3×3 (стандартная формула, правая система).

    Активное вращение: ``R`` переводит вектор из системы головы в мировую, ``Rᵀ`` — обратно
    (для поворота ``Rᵀ = R⁻¹``). На данных этой записи прочтение ``head → world`` через ``R``
    не сходится — см. :func:`head_to_world` и флаг ``invert_quat``. При
    ``||q| − 1| > QUAT_NORM_TOL`` кватернион нормируется (в данных отклонение ≤ 1.5e-7, но
    после SLERP и подгонок запас полезен); нулевой и нефинитный кватернион — ``ValueError``.
    """
    x, y, z, w = _vector(quat, 4, "quaternion")
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("quaternion: нулевой кватернион (норма 0)")
    if abs(norm - 1.0) > QUAT_NORM_TOL:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def rotation_rpy(yaw_deg: float, pitch_deg: float, roll_deg: float) -> Matrix3:
    """Поворот ``R_y(yaw) · R_x(pitch) · R_z(roll)`` — порядок из ``config.py``.

    ``yaw`` — вокруг ``y`` (вверх), ``pitch`` — вокруг ``x`` (вправо), ``roll`` — вокруг
    ``z`` (вперёд). Повороты внутренние (roll применяется первым к вектору).
    """
    yaw = math.radians(_number(yaw_deg, "yaw_deg"))
    pitch = math.radians(_number(pitch_deg, "pitch_deg"))
    roll = math.radians(_number(roll_deg, "roll_deg"))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    around_y: Matrix3 = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    around_x: Matrix3 = ((1.0, 0.0, 0.0), (0.0, cp, -sp), (0.0, sp, cp))
    around_z: Matrix3 = ((cr, -sr, 0.0), (sr, cr, 0.0), (0.0, 0.0, 1.0))
    return _matmul3(_matmul3(around_y, around_x), around_z)


def head_pose_parts(head_pose: object) -> tuple[tuple[float, float, float], tuple[float, ...]]:
    """Разобрать позу головы на ``(позиция, кватернион)``.

    Принимает либо последовательность из 7 чисел ``x, y, z, qx, qy, qz, qw`` (как в
    ``parse_pose``), либо объект с атрибутами ``head_pos`` и ``head_quat`` (``sync.Pose``) —
    импортировать ``sync`` ради типа не нужно.
    """
    position = getattr(head_pose, "head_pos", None)
    quaternion = getattr(head_pose, "head_quat", None)
    if position is not None or quaternion is not None:
        if position is None or quaternion is None:
            raise ValueError(
                "head_pose: у объекта должны быть и head_pos, и head_quat "
                f"(есть {position!r} и {quaternion!r})"
            )
        return _vector(position, 3, "head_pose.head_pos"), _vector(
            quaternion, 4, "head_pose.head_quat"
        )
    values = _vector(head_pose, 7, "head_pose")
    return (values[0], values[1], values[2]), values[3:]


def head_to_world(head_pose: object, *, invert_quat: bool = False) -> Matrix4:
    """``T_head_world`` — поза головы: ``head`` → ``world`` (поворот из кватерниона, сдвиг).

    ``invert_quat=True`` — кватернион трактуется как обратный поворот (берётся ``Rᵀ``).
    Это дискретный вариант конвенции, который перебирается в 07b: в данных поза приходит
    из Unity (левосторонняя система), и замер на 8 690 точках кистей показывает, что при
    ``invert_quat=False`` **0 %** точек оказываются перед головой (``z < 0``), а при
    ``invert_quat=True`` — **92.7 %**. Остаточная ориентация осей всё равно не сходится с
    кадром (07b), поэтому по умолчанию оставлена «спецификационная» конвенция, а флаг —
    вход для калибровки, а не скрытая подгонка.
    """
    position, quaternion = head_pose_parts(head_pose)
    rotation = quat_to_rotation(quaternion)
    if invert_quat:
        rotation = _transpose3(rotation)
    return _homogeneous(rotation, position)


def head_cam_matrix() -> Matrix4:
    """``T_head_cam`` из ``src/config.py`` — поза камеры в системе головы (``cam`` → ``head``).

    Читает ``config.HEAD_CAM_ROTATION_RPY_DEG`` и ``config.HEAD_CAM_TRANSLATION_M`` при
    каждом вызове, поэтому калибровка (07b/07c) меняет только ``config.py``. Значения
    проверяются: ровно три конечных числа в каждом кортеже, иначе ``ValueError``.
    """
    yaw, pitch, roll = _vector(
        config.HEAD_CAM_ROTATION_RPY_DEG, 3, "config.HEAD_CAM_ROTATION_RPY_DEG"
    )
    translation = _vector(config.HEAD_CAM_TRANSLATION_M, 3, "config.HEAD_CAM_TRANSLATION_M")
    return _homogeneous(rotation_rpy(yaw, pitch, roll), translation)


def world_to_camera_matrix(
    head_pose: object,
    t_head_cam: Matrix4 | Sequence[Sequence[float]] | None = None,
    *,
    opencv_conventions: bool = True,
    invert_quat: bool = False,
) -> Matrix4:
    """``T_cam_world`` для кадра: 4×4, переводящая точку мира в координаты камеры.

    ``M = F · T_head_cam⁻¹ · T_head_world⁻¹`` (см. шапку модуля). ``t_head_cam=None``
    берёт константу из :func:`head_cam_matrix` (``src/config.py``); явное значение —
    для подбора 07b: матрица понимается так же, как поза камеры в голове (``cam`` →
    ``head``), принимается и вложенная 4×4, и плоский список 16 чисел. ``M`` считается
    один раз на кадр, дальше 10 точек кистей переводятся :func:`transform_point`.

    ``opencv_conventions=False`` — оставить оси как в Unity (``y`` вверх), без
    переворота ``y``: этим флагом проверяется сама конвенция осей камеры.
    ``invert_quat`` — дискретный вариант конвенции кватерниона (см. :func:`head_to_world`).
    """
    head_cam = head_cam_matrix() if t_head_cam is None else _matrix4(t_head_cam, "t_head_cam")
    cam_head = _inverse4(head_cam, "t_head_cam")
    head_world = head_to_world(head_pose, invert_quat=invert_quat)
    world_head = _inverse4(head_world, "head_pose")
    matrix = _matmul4(cam_head, world_head)
    if opencv_conventions:
        matrix = _matmul4(OPENCV_FLIP_Y, matrix)
    return matrix


def transform_point(
    matrix: Sequence[Sequence[float]], point: Sequence[float]
) -> tuple[float, float, float]:
    """Применить однородную 4×4 к точке ``(x, y, z)`` (``w = 1``), вернуть ``(x, y, z)``."""
    return _matvec4(_matrix4(matrix, "matrix"), _vector(point, 3, "point"))


def world_to_camera(
    p_world: Sequence[float],
    head_pose: object,
    t_head_cam: Matrix4 | Sequence[Sequence[float]] | None = None,
    *,
    opencv_conventions: bool = True,
    invert_quat: bool = False,
) -> tuple[float, float, float]:
    """Точка мира → координаты камеры кадра: ``(x, y, z)``, ``z`` — глубина вдоль взгляда.

    ``p_world`` — метры в мировой системе (как в ``tracking.csv``), ``head_pose`` — поза
    головы на кадр (7 чисел или ``sync.Pose``), ``t_head_cam`` — поза камеры в голове
    (``None`` → константа из ``src/config.py``). При ``opencv_conventions=True`` (по
    умолчанию) ``x`` вправо, ``y`` вниз, ``z`` вперёд — как ждут интринсики 05a.

    Точка на оси взгляда головы ``(0, 0, 1)`` при заглушке ``config.py`` даёт
    ``(0, 0, 0.95)``: ``x ≈ y ≈ 0``, ``z > 0`` (камера в 5 см впереди центра головы).
    """
    matrix = world_to_camera_matrix(
        head_pose, t_head_cam, opencv_conventions=opencv_conventions, invert_quat=invert_quat
    )
    return transform_point(matrix, p_world)


def _sample_pose(sample: dict, where: str | None = None) -> tuple[float, ...]:
    """Поза головы 7 числами из сэмпла файла трекинга (``Head.pose``), для CLI-проверки."""
    return tuple(float(value) for value in parse_pose(sample["Head"]["pose"], where))


def _display(value: float) -> float:
    """Мелкие значения (−5e-4…5e-4) печатать как 0.000, а не как «-0.000»."""
    return 0.0 if abs(value) < 5e-4 else value


def _head_cam_report(tracking: Path, samples: int) -> int:
    """Печать ``T_head_cam`` из config.py и проверка перевода мира в камеру (задача 05b).

    На равномерно выбранных сэмплах печатает позу головы, точку взгляда в координатах
    камеры (обязана дать ``x ≈ y ≈ 0``, ``z > 0`` при любой позе) и сколько точек кистей
    оказалось впереди камеры (``z > 0``). Колонка «кисти впереди» показана для обеих
    конвенций кватерниона (``invert_quat`` False/True): расхождение — это ровно тот
    открытый вопрос, который закрывает 07b, и данные для него видны сразу.
    """
    header_ns = int(read_header(tracking)["timeStampNs"])
    total = count_samples(tracking)
    if total == 0:
        raise ValueError(f"{tracking}: нет сэмплов")
    count = max(1, min(samples, total))
    if count > 1:
        indices = sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})
    else:
        indices = [0]

    print(tracking)
    print("  T_head_cam из src/config.py (поза камеры в системе головы, cam → head):")
    for row in head_cam_matrix():
        print("    [" + " ".join(f"{value:>9.4f}" for value in row) + " ]")
    print(
        f"  HEAD_CAM_ROTATION_RPY_DEG = {config.HEAD_CAM_ROTATION_RPY_DEG}, "
        f"HEAD_CAM_TRANSLATION_M = {config.HEAD_CAM_TRANSLATION_M}"
    )
    print(
        "  проверка на позах: точка взгляда (0, 0, 1) головы и 10 точек кистей → координаты камеры"
    )
    print(
        f"  {'#':>6}{'t, с':>9}  {'голова (x, y, z), м':<28}{'взгляд → p_cam':<24}"
        "кисти впереди (z > 0): invert_quat False / True"
    )
    for index, sample in enumerate(iter_samples(tracking)):
        if index not in indices:
            continue
        pose = _sample_pose(sample, f"строка {index + 2}")
        record = extract_sample(sample)
        t_rel = (int(sample["timeStampNs"]) - header_ns) / 1e9
        matrix = world_to_camera_matrix(pose)
        other = world_to_camera_matrix(pose, invert_quat=True)
        gaze_world = transform_point(head_to_world(pose), (0.0, 0.0, 1.0))
        gaze = transform_point(matrix, gaze_world)
        ahead = {False: 0, True: 0}
        finite = 0
        for side in ("left", "right"):
            for name in ("palm", "wrist", "thumb", "index", "middle"):
                point = tuple(record[f"{side}_{name}_{axis}"] for axis in ("x", "y", "z"))
                if any(math.isnan(value) for value in point):
                    continue
                finite += 1
                ahead[False] += transform_point(matrix, point)[2] > 0.0
                ahead[True] += transform_point(other, point)[2] > 0.0
        missing = 10 - finite
        hands = f"{ahead[False]}/{finite} ({missing} NaN) / {ahead[True]}/{finite}"
        head = f"({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f})"
        gaze_text = "(" + ", ".join(f"{_display(value):.3f}" for value in gaze) + ")"
        print(f"  {index:>6}{t_rel:>9.3f}  {head:<28}{gaze_text:<24}{hands:>26}")
    print("  допущение: камера жёстко закреплена в голове (T_head_cam — константа, 07c).")
    print("  кисти впереди: доля точек с z > 0 — проверка конвенции осей; финал — 07b/07c.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI: интринсики SBS-кадра (05a) или перевод мира в координаты камеры (05b)."""
    parser = argparse.ArgumentParser(
        description="Интринсики из header (05a) и перевод мира в координаты камеры (05b)."
    )
    parser.add_argument(
        "tracking", type=Path, help="файл трекинга, например data/trackingData_20260505_165740.txt"
    )
    parser.add_argument(
        "--world-to-camera",
        action="store_true",
        help="показать T_head_cam из config.py и проверить перевод на реальных позах (05b)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="сколько поз проверить в режиме --world-to-camera (по умолчанию 5)",
    )
    args = parser.parse_args(argv)

    if args.world_to_camera:
        try:
            return _head_cam_report(args.tracking, args.samples)
        except (OSError, ValueError) as exc:
            print(f"ошибка: {exc}", file=sys.stderr)
            return 1

    try:
        header = read_header(args.tracking)
        intrinsics = Intrinsics.from_header(header)
    except (OSError, ValueError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1

    print(args.tracking)
    print(f"  cameraIntrinsics (файловый порядок): {header['cameraIntrinsics']}")
    print("  интерпретация: [cx, cy, fx, fy] полного SBS-кадра")
    print(
        f"  полный кадр: {intrinsics.full_w}×{intrinsics.full_h} "
        f"(2·cx + 1 = {2 * intrinsics.cx + 1:g}, 2·cy + 1 = {2 * intrinsics.cy + 1:g})"
    )
    print(
        f"    fx = {intrinsics.fx:.6f} px, fy = {intrinsics.fy:.6f} px, "
        f"cx = {intrinsics.cx:g}, cy = {intrinsics.cy:g}"
    )
    fov_h, fov_v, fov_d = intrinsics.fov_deg()
    print(f"    FOV: h = {fov_h:.2f}°, v = {fov_v:.2f}°, диагональ = {fov_d:.2f}°")
    print("  per-eye (пиксели квадратные, fx = fy; шов SBS на x = 1080):")
    print(
        f"    {'глаз':<7}{'fx, px':>12}{'fy, px':>12}{'cx, px':>12}{'cy, px':>11}"
        f"{'кадр':>11}{'hFOV':>9}{'vFOV':>9}{'диаг.':>9}"
    )
    for side, title in (("left", "левый"), ("right", "правый")):
        eye = intrinsics.per_eye(side)
        eye_h, eye_v, eye_d = eye.fov_deg()
        print(
            f"    {title:<7}{eye.fx:>12.6f}{eye.fy:>12.6f}{eye.cx:>12.1f}{eye.cy:>11.1f}"
            f"{f'{eye.full_w}×{eye.full_h}':>11}{eye_h:>8.2f}°{eye_v:>8.2f}°{eye_d:>8.2f}°"
        )
    print("  допущения: модель пинхол, дисторсии в header нет; правая половина — для стерео.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
