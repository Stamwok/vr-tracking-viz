"""sync.py — время кадра видео в шкале трекинга (задача 04a).

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

Запуск:
    python src/sync.py output/video_meta.json data/trackingData_20260505_165740.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.parse_tracking import read_header, timestamp_range  # noqa: E402

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


def _load_meta(path: Path) -> dict:
    """Читает video_meta.json; не-объект — ValueError."""
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: ожидался JSON-объект video_meta, получено {type(meta).__name__}")
    return meta


def main() -> None:
    """CLI: граница трекинга, первые/последние кадры и число кадров без данных."""
    parser = argparse.ArgumentParser(description="Время кадра видео в шкале трекинга (задача 04a).")
    parser.add_argument("meta", type=Path, help="output/video_meta.json")
    parser.add_argument("tracking", type=Path, help="data/trackingData_*.txt")
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


if __name__ == "__main__":
    main()
