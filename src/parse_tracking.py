"""parse_tracking.py — стриминговый ридер файла трекинга (JSON Lines).

Задача 03a: читаем 258 МБ построчно, отдаём header и итератор сэмплов.
Задача 03b: ``parse_pose()`` — строка "x,y,z,qx,qy,qz,qw" с десятичной запятой → 7 float
(проверка |q| ≈ 1.0). Задача 03c: ``extract_sample()`` — плоская запись: 3+4 головы,
точки рук, невалидные точки → NaN. Задача 03d: ``write_tracking_csv()`` —
output/tracking.csv + сводная статистика. Задача 08d: в данные и CSV попадают все
26 джойнтов OpenXR на руку (``HAND_JOINTS_FULL``), рендер по-прежнему использует
5-точечный набор ``HAND_JOINTS``.

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
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

Matrix4 = list[list[float]]

#: Допуск проверки нормы кватерниона: в данных отклонения ≤ 1.5e-7, так что 1e-6 с запасом.
QUAT_NORM_TOL = 1e-6

_WHOLE_TOKEN = re.compile(r"[+-]?\d+")

#: Все 26 джойнтов OpenXR на руку: (имя, индекс в ``HandJointLocations``).
#:
#: Индексы OpenXR: 0 palm, 1 wrist, 2–5 thumb, 6–10 index, 11–15 middle, 16–20 ring,
#: 21–25 little. Кончик пальца сохраняет короткое имя (``thumb``/``index``/...),
#: потому что эти имена уже используются рендером и калибровкой; промежуточные
#: фаланги получают полные имена (``*_metacarpal/proximal/intermediate/distal``).
HAND_JOINTS_FULL = (
    ("palm", 0),
    ("wrist", 1),
    ("thumb_metacarpal", 2),
    ("thumb_proximal", 3),
    ("thumb_distal", 4),
    ("thumb", 5),
    ("index_metacarpal", 6),
    ("index_proximal", 7),
    ("index_intermediate", 8),
    ("index_distal", 9),
    ("index", 10),
    ("middle_metacarpal", 11),
    ("middle_proximal", 12),
    ("middle_intermediate", 13),
    ("middle_distal", 14),
    ("middle", 15),
    ("ring_metacarpal", 16),
    ("ring_proximal", 17),
    ("ring_intermediate", 18),
    ("ring_distal", 19),
    ("ring", 20),
    ("little_metacarpal", 21),
    ("little_proximal", 22),
    ("little_intermediate", 23),
    ("little_distal", 24),
    ("little", 25),
)

#: Пять точек на руку для рендера и калибровки (03c→08a): подмножество полного набора.
#: Имена и индексы совпадают с ``HAND_JOINTS_FULL``, поэтому старые колонки CSV и
#: ``sync.POINT_KEYS`` не меняются при расширении данных до 26 джойнтов (08d).
HAND_JOINTS = (("palm", 0), ("wrist", 1), ("thumb", 5), ("index", 10), ("middle", 15))

#: Локти из ``Body.joints`` (задача 08h): 18 — левый, 19 — правый. В Hand локтя нет
#: (26 джойнтов — запястье и пальцы), поэтому единственный источник локтя — Body.
#: Запястья Body (20/21) в данные **не** добавляются: концом ребра служит точный
#: ``Hand.wrist`` (Body.wrist — его же аппроксимация, см. ``docs/NOTES.md``, 08h).
#: ``CSV_COLUMNS`` от этого не меняется: elbow живёт только в пути рендера, иначе
#: поехали бы sha256-эталоны 08d/08g.
BODY_ELBOW_JOINTS = (("left_elbow", 18), ("right_elbow", 19))

#: Плоские ключи локтей: ``left_elbow``/``right_elbow`` (для ``sync.extra_keys``).
ELBOW_KEYS = tuple(name for name, _ in BODY_ELBOW_JOINTS)

#: Колонки tracking.csv: время, поза головы, по 26 джойнтов на руку, флаги активности.
CSV_COLUMNS = (
    "t_ns",
    "t_rel_s",
    "head_x",
    "head_y",
    "head_z",
    "head_qx",
    "head_qy",
    "head_qz",
    "head_qw",
    *(
        f"{side}_{name}_{axis}"
        for side in ("left", "right")
        for name, _ in HAND_JOINTS_FULL
        for axis in ("x", "y", "z")
    ),
    "left_active",
    "right_active",
)

#: Сколько строк пишем в CSV за один проход (08d: 167 колонок × 21 787 строк ≈ 44 МБ,
#: держать все строки в памяти незачем).
CSV_CHUNK_ROWS = 2000


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


def timestamp_range(path: str | Path) -> tuple[int, int]:
    """Метки времени первого и последнего сэмплов (Unix ns), потоково.

    Нужна синхронизации видео↔трекинг (задача 04a): конец трекинга — это
    ``timeStampNs`` последнего сэмпла, а не ``header + tracking_duration_s``
    (в ``video_meta.json`` это span first→last; разница 18.4 мс).
    """
    first: int | None = None
    last: int | None = None
    for sample in iter_samples(path):
        timestamp = sample.get("timeStampNs")
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise ValueError(f"{path}: сэмпл без целого timeStampNs: {timestamp!r}")
        if first is None:
            first = timestamp
        last = timestamp
    if first is None or last is None:
        raise ValueError(f"{path}: нет сэмплов")
    return first, last


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


def _joint_xyz(
    joints: list[dict], index: int, active: bool, where: str | None
) -> tuple[float, float, float]:
    """Позиция сустава (x, y, z) или NaN×3, если рука/сустав невалидны."""
    if not active or index >= len(joints) or joints[index].get("s", 0) == 0:
        return (math.nan, math.nan, math.nan)
    pose = parse_pose(joints[index]["p"], where=where)
    return (pose[0], pose[1], pose[2])


def _body_joint_xyz(
    joints: list[dict], index: int, where: str | None
) -> tuple[float, float, float]:
    """Позиция сустава ``Body.joints[index]`` или NaN×3, если её нет/она невалидна (08h).

    У суставов Body **нет** флагов валидности (в отличие от ``HandJointLocations.s``):
    в файле каждый джойнт — ``{p, t, va, wva}``, где ``p`` — позиция и кватернион, ``t`` —
    метка времени. Поэтому валидность определяется по данным: конечность трёх координат и
    защита от «все нули» (нулевая позиция — признак незаполненного джойнта, а не позы).
    """
    if not 0 <= index < len(joints):
        return (math.nan, math.nan, math.nan)
    joint = joints[index]
    if not isinstance(joint, dict) or "p" not in joint:
        return (math.nan, math.nan, math.nan)
    pose = parse_pose(joint["p"], where=where)
    if len(pose) < 3:
        return (math.nan, math.nan, math.nan)
    x, y, z = pose[0], pose[1], pose[2]
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return (math.nan, math.nan, math.nan)
    if x == 0.0 and y == 0.0 and z == 0.0:
        # «все нули» — незаполненный джойнт; рисовать ребро от мировой точки (0, 0, 0) нельзя
        return (math.nan, math.nan, math.nan)
    return (x, y, z)


def extract_forearm(sample: dict, where: str | None = None) -> dict[str, float]:
    """Локти из ``Body.joints[18/19]``: ``{left,right}_elbow_{x,y,z}`` (задача 08h).

    Возвращает только 6 координат — без запястий Body (20/21) и без остальных 24 джойнтов.
    Невалидный/отсутствующий локоть — ``NaN`` (те же правила, что у рук: не рисовать, а не
    подставлять ноль). Валидность Body-джойнта считается по значениям
    (:func:`_body_joint_xyz`): флагов ``s`` у Body нет.
    """
    body = sample.get("Body")
    joints = body.get("joints") if isinstance(body, dict) else None
    joints = joints if isinstance(joints, list) else []
    record: dict[str, float] = {}
    for name, index in BODY_ELBOW_JOINTS:
        joint_where = f"{where}, Body.joints[{index}]" if where else f"Body.joints[{index}]"
        x, y, z = _body_joint_xyz(joints, index, joint_where)
        record[f"{name}_x"] = x
        record[f"{name}_y"] = y
        record[f"{name}_z"] = z
    return record


def extract_sample(
    sample: dict, where: str | None = None, *, include_forearm: bool = False
) -> dict:
    """Плоская запись сэмпла: поза головы (3+4) + все 26 джойнтов на руку (x, y, z) + isActive.

    Невалидные точки (``isActive == 0`` у руки или ``s == 0`` у сустава) — NaN, а не нули:
    у неактивной правой руки ``p`` содержит «замороженную» позу с |q| ≈ 1, так что без
    проверки флагов скелет нарисовался бы где-то в стороне от руки.

    Набор джойнтов — :data:`HAND_JOINTS_FULL` (26 на руку, индексы OpenXR 0…25); рендер
    использует его 5-точечное подмножество :data:`HAND_JOINTS`, но колонки с теми же
    именами в плоской записи совпадают.

    ``include_forearm`` (08h) добавляет к записи 6 координат локтей из ``Body.joints[18/19]``
    (:func:`extract_forearm`). По умолчанию **выключено**, поэтому ``tracking.csv`` (08d) и
    его sha256 не меняются: локти нужны только пути рендера, а ``sync.load_poses`` включает
    их, когда среди ``extra_keys`` есть :data:`ELBOW_KEYS`.
    """
    head = sample["Head"]
    head_where = f"{where}, Head.pose" if where else "Head.pose"
    pose = parse_pose(head["pose"], where=head_where)

    record: dict[str, float | int] = {"timeStampNs": int(sample["timeStampNs"])}
    for key, value in zip(("x", "y", "z", "qx", "qy", "qz", "qw"), pose):
        record[f"head_{key}"] = value
    if include_forearm:
        record.update(extract_forearm(sample, where=where))

    for side, hand_key in (("left", "leftHand"), ("right", "rightHand")):
        hand = sample["Hand"].get(hand_key, {})
        active = bool(hand.get("isActive"))
        joints = hand.get("HandJointLocations", [])
        record[f"{side}_active"] = 1 if active else 0
        for name, index in HAND_JOINTS_FULL:
            joint_where = f"{where}, {side} {name}" if where else f"{side} {name}"
            x, y, z = _joint_xyz(joints, index, active, joint_where)
            record[f"{side}_{name}_x"] = x
            record[f"{side}_{name}_y"] = y
            record[f"{side}_{name}_z"] = z
    return record


def _csv_row(record: dict, base_ts: int) -> dict:
    """Строка CSV: t_ns/t_rel_s + остальные поля плоской записи."""
    row = {
        "t_ns": record["timeStampNs"],
        "t_rel_s": round((record["timeStampNs"] - base_ts) / 1e9, 6),
    }
    for column in CSV_COLUMNS:
        if column not in row:
            row[column] = record[column]
    return row


def frame_stats(frame) -> dict:
    """Сводка по DataFrame: строки, длительность, медиана dt, невалидные по рукам."""
    timestamps = frame["t_ns"].tolist()
    rows = len(timestamps)
    dt_ms = [(b - a) / 1e6 for a, b in zip(timestamps, timestamps[1:])]
    left_invalid = int((frame["left_active"] == 0).sum())
    right_invalid = int((frame["right_active"] == 0).sum())
    return {
        "rows": rows,
        "duration_s": (timestamps[-1] - timestamps[0]) / 1e9 if rows > 1 else 0.0,
        "median_dt_ms": statistics.median(dt_ms) if dt_ms else 0.0,
        "left_invalid": left_invalid,
        "right_invalid": right_invalid,
        "left_invalid_pct": 100.0 * left_invalid / rows if rows else 0.0,
        "right_invalid_pct": 100.0 * right_invalid / rows if rows else 0.0,
    }


def write_tracking_csv(tracking_path: str | Path, out_path: str | Path, header: dict | None = None) -> dict:
    """Читает txt потоково, пишет tracking.csv (NaN → пустые поля) и возвращает статистику.

    Строки пишутся порциями по :data:`CSV_CHUNK_ROWS`: после 08d в строке 167 колонок
    (26 джойнтов × 2 руки), и держать все 21 787 строк-словарей в памяти незачем.
    Выход собирается в ``<out>.part`` и подменяет целевой файл атомарно; недописанный
    ``.part`` удаляется при ошибке.

    ``frame_stats`` получает только три нужные колонки (время и флаги рук), а не весь
    кадр, — сводка считается по полным данным, память остаётся ограниченной.

    pandas импортируется локально: read_header/parse_pose/extract_sample остаются
    без зависимости от него.
    """
    import pandas as pd

    header = header or read_header(tracking_path)
    base_ts = header["timeStampNs"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = out_path.with_name(f"{out_path.name}.part")

    columns = list(CSV_COLUMNS)
    timestamps: list[int] = []
    left_active: list[int] = []
    right_active: list[int] = []
    chunk: list[dict] = []

    def write_chunk(rows: list[dict], first: bool) -> None:
        chunk_frame = pd.DataFrame(rows, columns=columns)
        chunk_frame.to_csv(fh, index=False, header=first)

    try:
        with part_path.open("w", encoding="utf-8", newline="") as fh:
            first_chunk = True
            for sample in iter_samples(tracking_path):
                record = extract_sample(sample)
                timestamps.append(record["timeStampNs"])
                left_active.append(record["left_active"])
                right_active.append(record["right_active"])
                chunk.append(_csv_row(record, base_ts))
                if len(chunk) >= CSV_CHUNK_ROWS:
                    write_chunk(chunk, first_chunk)
                    first_chunk = False
                    chunk.clear()
            if chunk or first_chunk:
                write_chunk(chunk, first_chunk)
        part_path.replace(out_path)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise

    frame = pd.DataFrame(
        {"t_ns": timestamps, "left_active": left_active, "right_active": right_active}
    )
    return frame_stats(frame)


def main() -> None:
    """CLI: печатает header и сводку; с -o пишет tracking.csv."""
    parser = argparse.ArgumentParser(description="Прочитать файл трекинга; с -o записать tracking.csv.")
    parser.add_argument("tracking", type=Path, help="путь к txt, например data/trackingData_*.txt")
    parser.add_argument("-o", "--out", type=Path, default=None, help="куда записать CSV (output/tracking.csv)")
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

    if args.out is not None:
        stats = write_tracking_csv(args.tracking, args.out, header)
        print(
            f"  строк: {stats['rows']}, длительность: {stats['duration_s']:.2f} с, "
            f"медиана dt: {stats['median_dt_ms']:.2f} мс"
        )
        print(
            f"  невалидные: левая {stats['left_invalid']} ({stats['left_invalid_pct']:.1f} %), "
            f"правая {stats['right_invalid']} ({stats['right_invalid_pct']:.1f} %)"
        )
        print(f"  -> {args.out} ({len(CSV_COLUMNS)} колонок)")
    else:
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

        record = extract_sample(first_sample, where="первый сэмпл")
        right_nan = sum(
            math.isnan(record[f"right_{name}_{axis}"])
            for name, _ in HAND_JOINTS_FULL
            for axis in "xyz"
        )
        print(
            f"  извлечение: head=({record['head_x']:.3f}, {record['head_y']:.3f}, {record['head_z']:.3f}), "
            f"left_active={record['left_active']}, right_active={record['right_active']}, "
            f"right NaN = {right_nan}/{3 * len(HAND_JOINTS_FULL)}"
        )


if __name__ == "__main__":
    main()
