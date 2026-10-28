"""parse_tracking.py — стриминговый ридер файла трекинга (JSON Lines).

Задача 03a: читаем 258 МБ построчно, отдаём header и итератор сэмплов.
Задача 03b: ``parse_pose()`` — строка "x,y,z,qx,qy,qz,qw" с десятичной запятой → 7 float
(проверка |q| ≈ 1.0). Дальше: 03c — Head + Hand (5 точек на руку, флаг isActive),
03d — output/tracking.csv и статистика.

Формат файла (21 788 строк = header + 21 787 сэмплов):
  * строка 1 — header: ``timeStampNs`` (int), ``cameraIntrinsics`` и ``cameraExtrinsics``
    записаны JSON-строками; в ``cameraExtrinsics`` две матрицы 4×4 (16 чисел плоским
    массивом, row-major) склеены через ``"|"``,
  * строки 2… — по одному JSON-объекту на сэмпл (≈11.8 КБ), пустые строки пропускаем.

Десятичная запятая в pose совпадает с разделителем полей: ``"-0,08517717,..."`` после
``split(",")`` даёт два токена на число. ``parse_pose()`` склеивает пары.

Запуск:
    python src/parse_tracking.py data/trackingData_20260505_165740.txt
"""
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

Matrix4 = list[list[float]]

#: Допуск проверки нормы кватерниона: в данных отклонения ≤ 1.5e-7, так что 1e-6 с запасом.
QUAT_NORM_TOL = 1e-6

_WHOLE_TOKEN = re.compile(r"[+-]?\d+")


def _next_nonempty_line(fh) -> str:
    """Первая непустая строка файла (header); ValueError, если файл пуст."""
    for line in fh:
        stripped = line.strip()
        if stripped:
            return stripped
    raise ValueError("файл пуст: нет header")


def _parse_intrinsics(value: Any) -> list[float]:
    """cameraIntrinsics: JSON-строка (или список) из 4 чисел [cx, cy, fx, fy]."""
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, list) or len(data) != 4:
        raise ValueError(f"cameraIntrinsics: ожидался массив из 4 чисел, получено {data!r}")
    return [float(v) for v in data]


def _parse_matrix(value: Any, name: str) -> Matrix4:
    """Матрица 4×4: плоский массив из 16 чисел (row-major) или вложенный список 4×4."""
    data = json.loads(value) if isinstance(value, str) else value
    if isinstance(data, list) and len(data) == 16 and all(isinstance(v, (int, float)) for v in data):
        data = [data[i : i + 4] for i in range(0, 16, 4)]
    if (
        not isinstance(data, list)
        or len(data) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in data)
    ):
        raise ValueError(f"{name}: ожидалась матрица 4x4, получено {data!r}")
    matrix = [[float(v) for v in row] for row in data]
    if not all(math.isfinite(v) for row in matrix for v in row):
        raise ValueError(f"{name}: нефинитные значения в матрице")
    return matrix


def _parse_extrinsics(value: Any) -> list[Matrix4]:
    """cameraExtrinsics: две матрицы 4×4 (левый/правый глаз), в файле склеены через "|"."""
    if isinstance(value, str):
        parts = value.split("|")
        if len(parts) != 2:
            raise ValueError(f"cameraExtrinsics: ожидалось 2 матрицы через '|', получено {len(parts)}")
    elif isinstance(value, list) and len(value) == 2:
        parts = value
    else:
        raise ValueError(f"cameraExtrinsics: неожиданный формат: {value!r}")
    return [_parse_matrix(p, f"cameraExtrinsics[{i}]") for i, p in enumerate(parts)]


def read_header(path: str | Path) -> dict:
    """Читает и разбирает header (первую непустую строку файла).

    Возвращает dict: ``timeStampNs`` (int, Unix ns первого кадра), ``cameraIntrinsics``
    (4 float, порядок [cx, cy, fx, fy] полного SBS-кадра) и ``cameraExtrinsics``
    (2 матрицы 4×4 — левый и правый глаз).
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = json.loads(_next_nonempty_line(fh))
    if not isinstance(raw, dict):
        raise ValueError("header: ожидался JSON-объект")

    missing = [key for key in ("timeStampNs", "cameraIntrinsics", "cameraExtrinsics") if key not in raw]
    if missing:
        raise ValueError(f"header: нет полей {missing}")

    timestamp = raw["timeStampNs"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise ValueError(f"header: timeStampNs должен быть int, получено {type(timestamp).__name__}")

    return {
        "notice": raw.get("notice"),
        "timeStampNs": timestamp,
        "cameraIntrinsics": _parse_intrinsics(raw["cameraIntrinsics"]),
        "cameraExtrinsics": _parse_extrinsics(raw["cameraExtrinsics"]),
    }


def iter_samples(path: str | Path) -> Iterator[dict]:
    """Генератор сэмплов: построчный json.loads после header, пустые строки пропускаются.

    Файл читается потоково — в памяти живёт одна строка, поэтому объём файла не важен.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        header_skipped = False
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if not header_skipped:
                header_skipped = True
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"строка {line_no}: некорректный JSON ({exc})") from exc


def count_samples(path: str | Path) -> int:
    """Число сэмплов в файле (без header), без загрузки файла в память."""
    return sum(1 for _ in iter_samples(path))


def parse_pose(s: str, where: str | None = None) -> list[float]:
    """Строка позы с десятичной запятой → список float.

    Формат: ``"x,y,z,qx,qy,qz,qw"``, где числа записаны с десятичной запятой, которая
    совпадает с разделителем полей: ``"-0,08517717,-0,02745685,..."`` → 14 токенов.
    Штатное правило (чётное число токенов): каждое число — пара ``<целая часть>,<дробная>``,
    склеиваем ``float(t[i] + "." + t[i+1])`` с шагом 2.

    Нечётное число токенов — стиль ``Body.va`` (``"0,0,0"``): числа записаны без дробной
    части, поэтому токены берутся как есть, но только если все они целые; иначе формат
    неоднозначен и бросается ValueError.

    Для 7 чисел (поза) проверяется ``|q| ≈ 1.0`` с допуском :data:`QUAT_NORM_TOL`;
    в сообщении об ошибке — ``where`` (например, «строка 1234, Head.pose»), чтобы было
    видно, какой сэмпл сломан.
    """
    tokens = [token.strip() for token in s.split(",")]
    if not s or s.isspace() or len(tokens) == 1:
        raise ValueError(f"parse_pose{_ctx(where)}: пустая строка позы: {s!r}")

    if len(tokens) % 2 == 0:
        try:
            values = [float(tokens[i] + "." + tokens[i + 1]) for i in range(0, len(tokens), 2)]
        except ValueError as exc:
            raise ValueError(f"parse_pose{_ctx(where)}: не число в {s!r}") from exc
    elif all(_WHOLE_TOKEN.fullmatch(token) for token in tokens):
        values = [float(token) for token in tokens]
    else:
        raise ValueError(
            f"parse_pose{_ctx(where)}: нечётное число токенов ({len(tokens)}) и не все токены "
            f"целые — формат неоднозначен: {s!r}"
        )

    if len(values) == 7:
        norm = math.sqrt(sum(q * q for q in values[3:]))
        if abs(norm - 1.0) > QUAT_NORM_TOL:
            raise ValueError(
                f"parse_pose{_ctx(where)}: |q| = {norm:.9f}, ожидалось 1.0 ± {QUAT_NORM_TOL:g} "
                f"(q = {values[3:]})"
            )
    return values


def _ctx(where: str | None) -> str:
    """Суффикс с контекстом для сообщений об ошибках."""
    return f" ({where})" if where else ""


def main() -> None:
    """CLI: печатает header и сводку сэмплов."""
    parser = argparse.ArgumentParser(description="Прочитать header и посчитать сэмплы файла трекинга.")
    parser.add_argument("tracking", type=Path, help="путь к txt, например data/trackingData_*.txt")
    args = parser.parse_args()

    if not args.tracking.is_file():
        raise SystemExit(f"нет файла: {args.tracking}")

    header = read_header(args.tracking)
    ts = header["timeStampNs"]
    when = datetime.fromtimestamp(ts / 1e9, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    print(args.tracking)
    print(f"  header: timeStampNs={ts} ({when})")
    print(f"  cameraIntrinsics (cx, cy, fx, fy) = {header['cameraIntrinsics']}")
    print(f"  cameraExtrinsics: {len(header['cameraExtrinsics'])} матрицы 4x4")
    for i, matrix in enumerate(header["cameraExtrinsics"]):
        translation = matrix[0][3], matrix[1][3], matrix[2][3]
        print(f"    [{i}] 4-й столбец (t) = ({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f})")

    count = 0
    first_ts = last_ts = None
    for count, sample in enumerate(iter_samples(args.tracking), start=1):
        sample_ts = sample.get("timeStampNs")
        if sample_ts is None:
            continue
        if first_ts is None:
            first_ts = sample_ts
        last_ts = sample_ts
    print(f"  сэмплов: {count}")
    if first_ts is not None and last_ts is not None:
        print(f"  t = {(first_ts - ts) / 1e9:.3f} … {(last_ts - ts) / 1e9:.3f} с от header")

    first_sample = next(iter_samples(args.tracking), None)
    if first_sample is not None and "Head" in first_sample:
        pose = parse_pose(first_sample["Head"]["pose"], where="первый сэмпл, Head.pose")
        norm = math.sqrt(sum(q * q for q in pose[3:]))
        print(f"  первый Head.pose = {[round(v, 6) for v in pose]}, |q| = {norm:.7f}")


if __name__ == "__main__":
    main()
