"""video_meta.py — метаданные видео: разрешение, fps, число кадров, длительность, SBS.

Задача 02. Пишет output/video_meta.json и output/debug/first_frame.png
(первый кадр с линией шва стерео-пары).

Пиксели / fps / число кадров берём из cv2.VideoCapture. Точную длительность — из
контейнера (mvhd: timescale / duration), потому что CAP_PROP_FPS отдаёт 29.9976,
а frame_count / fps даёт 242.653 с вместо реальных 242.661 с (расхождение 8 мс).

Запуск:
    python src/video_meta.py data/CameraRecord_20260505_165740.mp4
Результат:
    output/video_meta.json, output/debug/first_frame.png
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

# Длительность трекинговой части записи (21 787 сэмплов, 242.542 с) — измерение задачи 03,
# здесь нужно для сверки «видео длиннее трекинга на ~119 мс» (карточка 02, критерий).
TRACKING_DURATION_S = 242.542

SEAM_COLOR = (0, 255, 0)  # BGR: зелёный
SEAM_THICKNESS = 3


def _iter_boxes(fh, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """Итерирует ISO-BMFF боксы в диапазоне [start, end): (тип, начало данных, конец бокса)."""
    while start + 8 <= end:
        fh.seek(start)
        header = fh.read(16)
        if len(header) < 8:
            return
        size, kind = struct.unpack(">I4s", header[:8])
        header_len = 8
        if size == 1:  # 64-битный размер: size + type + largesize
            if len(header) < 16:
                return
            size = struct.unpack(">Q", header[8:16])[0]
            header_len = 16
        elif size == 0:  # бокс до конца файла
            size = end - start
        if size < header_len or start + size > end:
            return
        yield kind, start + header_len, start + size
        start += size


def container_duration(path: Path) -> float | None:
    """Длительность из moov → mvhd (секунды) или None, если бокс не найден."""
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        for kind, payload, box_end in _iter_boxes(fh, 0, file_size):
            if kind != b"moov":
                continue
            for sub, sub_payload, sub_end in _iter_boxes(fh, payload, box_end):
                if sub != b"mvhd" or sub_payload + 4 > sub_end:
                    continue
                fh.seek(sub_payload)
                head = fh.read(32)
                version = head[0]
                if version == 0 and len(head) >= 20:
                    timescale, duration = struct.unpack(">II", head[12:20])
                elif version == 1 and len(head) >= 32:
                    timescale = struct.unpack(">I", head[20:24])[0]
                    duration = struct.unpack(">Q", head[24:32])[0]
                else:
                    return None
                return duration / timescale if timescale else None
    return None


def detect_layout(width: int, height: int, frame: np.ndarray) -> tuple[str, list[int], float]:
    """Определяет SBS по пропорциям кадра, возвращает (layout, eye_size, |left-right|).

    Признак SBS: ширина делится на два, и глаз получается «шире, чем выше»
    (2160×810 → глаз 1080×810, ratio 1.33). Для обычного 16:9 mono (1920×1080)
    глаз 960×1080 даёт ratio 0.89 и layout остаётся mono.
    """
    if width % 2:
        return "mono", [width, height], 0.0

    eye_w = width // 2
    left = frame[:, :eye_w].astype(np.int16)
    right = frame[:, eye_w : 2 * eye_w].astype(np.int16)
    halves_diff = float(np.abs(left - right).mean())

    layout = "sbs" if eye_w / height >= 1.2 else "mono"
    eye_size = [eye_w, height] if layout == "sbs" else [width, height]
    return layout, eye_size, halves_diff


def render_first_frame(frame: np.ndarray, eye_width: int, out_path: Path) -> None:
    """Сохраняет первый кадр с линией шва стерео-пары."""
    marked = frame.copy()
    cv2.line(marked, (eye_width, 0), (eye_width, marked.shape[0] - 1), SEAM_COLOR, SEAM_THICKNESS)
    cv2.putText(
        marked,
        f"SBS seam x={eye_width}",
        (eye_width + 12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        SEAM_COLOR,
        2,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), marked):
        raise SystemExit(f"не удалось записать кадр: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Снять метаданные видео и сохранить debug-кадр.")
    parser.add_argument("video", type=Path, help="путь к mp4, например data/CameraRecord_*.mp4")
    parser.add_argument("--out-json", type=Path, default=Path("output/video_meta.json"))
    parser.add_argument("--out-frame", type=Path, default=Path("output/debug/first_frame.png"))
    args = parser.parse_args()

    if not args.video.is_file():
        raise SystemExit(f"нет файла: {args.video}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cv2.VideoCapture не открыл {args.video}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cv2 = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        raise SystemExit("не удалось прочитать первый кадр")
    if frame.shape[:2] != (height, width):
        raise SystemExit(f"кадр {frame.shape[1]}x{frame.shape[0]} не совпал с заголовком {width}x{height}")

    layout, eye_size, halves_diff = detect_layout(width, height, frame)
    duration_box = container_duration(args.video)
    duration = duration_box if duration_box is not None else frame_count / fps_cv2

    meta = {
        "source": str(args.video),
        "width": width,
        "height": height,
        "fps": round(fps_cv2),  # номинал 30.0; cv2 отдаёт 29.9976 (см. fps_measured_cv2)
        "fps_measured_cv2": round(fps_cv2, 4),
        "frame_count": frame_count,
        "duration": round(duration, 3),  # секунды
        "duration_source": "mvhd" if duration_box is not None else "frame_count/fps",
        "stereo_layout": layout,
        "eye_size": eye_size,
        "eye_halves_mean_abs_diff": round(halves_diff, 2),
        "tracking_duration_s": TRACKING_DURATION_S,
        "video_minus_tracking_s": round(duration - TRACKING_DURATION_S, 3),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_first_frame(frame, eye_size[0], args.out_frame)

    print(f"{args.video}")
    print(f"  {width}x{height}, fps={meta['fps']} (cv2 {meta['fps_measured_cv2']}), кадров {frame_count}")
    print(f"  duration={meta['duration']} с ({meta['duration_source']}), layout={layout}, eye={eye_size[0]}x{eye_size[1]}")
    print(f"  видео длиннее трекинга на {meta['video_minus_tracking_s']} с")
    print(f"  -> {args.out_json}, {args.out_frame}")


if __name__ == "__main__":
    main()
