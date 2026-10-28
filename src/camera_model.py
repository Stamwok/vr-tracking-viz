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

Задачи 05b (``world_to_camera``) и 05c (``project``) сюда не входят.

Запуск:
    python src/camera_model.py data/trackingData_20260505_165740.txt
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parse_tracking import read_header  # noqa: E402

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
        ``cx = 540 − 0.5 = 539.5`` у левого и ``539.5 + 1080 = 1619.5`` у правого (координата
        в полном SBS-кадре; после обрезки правой половины без сдвига остаётся 1619.5),
        ``cy = 404.5``. Размер изображения глаза — 1080×810.
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


def main(argv: list[str] | None = None) -> int:
    """CLI: сырые интринсики из header, интерпретация ``[cx, cy, fx, fy]`` и per-eye таблица."""
    parser = argparse.ArgumentParser(
        description="Разобрать cameraIntrinsics из header и показать per-eye модель SBS-кадра."
    )
    parser.add_argument(
        "tracking", type=Path, help="файл трекинга, например data/trackingData_20260505_165740.txt"
    )
    args = parser.parse_args(argv)

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
