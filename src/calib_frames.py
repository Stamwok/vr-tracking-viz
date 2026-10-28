"""calib_frames.py — рабочий набор кадров и пиксельный эталон для калибровки (задача 07a).

Задача 07b подбирает конвенцию осей и ``T_head_cam``, сравнивая проекцию трекинга с видео.
Чтобы это сравнение было не «на глаз», нужен эталон, **независимый от проекции**: кадры,
на которых положение кистей в пикселях известно из самого видео. Он и лежит здесь — пять
кадров из 02a плюс прямоугольники кистей в координатах **левого глаза** (1080×810).

Почему прямоугольник, а не точка: точность ручной разметки по кадру 1080×810 — примерно
полсотни пикселей (сетка 100 px, см. ``TOLERANCE_PX``). Требовать попадания в точку
бессмысленно, а «точка попала в область кисти» — это как раз тот уровень строгости,
который отличает верную конвенцию осей от неверной: у неверной проекция уезжает на сотни
пикселей (см. ``docs/NOTES.md`` за 05b/05c).

Координаты — левый глаз обрезанного SBS-кадра (``frame[:, :1080]``, задача 05c): ``x``
вправо, ``y`` вниз, границы ``0 ≤ x < 1080``, ``0 ≤ y < 810``. Правый глаз — те же пиксели
плюс 1080 по ``x``, поэтому эталон задаётся один раз.

Разметка сделана по видео, а не по данным: прямоугольник обводит видимую кисть (пальцы +
ладонь + запястье), включая ремешок трекера, потому что сустав ``wrist`` в данных — это
именно запястье под ремешком. Черновые рамки (02a) проверены и уточнены наложением на
кадры; полученные PNG — ``output/debug/calib/`` (``output/`` в ``.gitignore``, поэтому
файлы восстанавливаются командой ``python src/calib_frames.py data/CameraRecord_*.mp4``).

Разбор эталона для 07b:

* ``CalibFrame.hands`` — прямоугольники видимых кистей; ``hand`` = ``"left"``/``"right"``
  (гипотеза разметки) или ``"unknown"``, если рука видна, но идентичность не установлена;
* ``peak`` — ``partial``: кисть обрезана краем кадра, часть суставов за пределами;
* ``ignore`` — стороны, чьи точки на этом кадре **не** считаются эталоном (рука за
  пределами FOV): на k = 30 правая кисть проецируется у самого края кадра, а в видео её
  нет — по ней нельзя судить о калибровке;
* ``TOLERANCE_PX`` — допуск разметки (±50 px).

Метрика, для которой это годится: проекция сустава, попавшая внутрь кадра, должна лежать
внутри прямоугольника **своей** руки; отдельно считается доля попаданий и число
перепутанных рук (``left`` в правом прямоугольнике и наоборот).

Запуск:
    python src/calib_frames.py --show                 # таблица эталона
    python src/calib_frames.py --json                 # эталон для 07b
    python src/calib_frames.py data/CameraRecord_*.mp4  # сохранить PNG + frames.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Размер кадра одного глаза (левая половина SBS 2160×810): система координат разметки.
EYE_WIDTH = 1080
EYE_HEIGHT = 810

#: Точность ручной разметки по сетке 100 px, пиксели. Прямоугольник — «плюс-минус столько».
TOLERANCE_PX = 50

#: Допустимые значения ``HandBox.hand``: две стороны и «видна, но кто — неизвестно».
HANDS = ("left", "right", "unknown")

#: Допустимые уровни уверенности разметки (см. ``HandBox.confidence``).
CONFIDENCE = ("high", "medium", "low")


@dataclass(frozen=True)
class HandBox:
    """Прямоугольник видимой кисти на кадре (координаты левого глаза, нижняя граница включительно).

    ``bbox`` — ``(x0, y0, x1, y1)`` в пикселях, ``x0 < x1`` и ``y0 < y1``. Прямоугольник
    включает запястье с ремешком трекера: сустав ``wrist`` в данных находится именно там.

    ``confidence`` — насколько уверенно установлены границы: ``high`` (кисть целиком в
    кадре, контур читается), ``medium`` (границы по видимой части), ``low`` (узкая полоска
    у края кадра). ``hand = "unknown"`` означает, что рука видна, но отнесение её к
    левой/правой не установлено (в 02a такие кадры могли быть подписаны неверно).

    ``partial`` — кисть обрезана краем кадра, поэтому часть её суставов за пределами
    разметки: при оценке такие прямоугольники нельзя требовать целиком.
    """

    hand: str
    bbox: tuple[int, int, int, int]
    confidence: str = "high"
    partial: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        if self.hand not in HANDS:
            raise ValueError(f"HandBox.hand: ожидалось одно из {HANDS}, получено {self.hand!r}")
        if self.confidence not in CONFIDENCE:
            raise ValueError(
                f"HandBox.confidence: ожидалось одно из {CONFIDENCE}, получено {self.confidence!r}"
            )
        if not isinstance(self.partial, bool):
            raise ValueError(f"HandBox.partial: ожидалось bool, получено {self.partial!r}")
        bbox = self.bbox
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            raise ValueError(f"HandBox.bbox: ожидалось 4 числа (x0, y0, x1, y1), получено {bbox!r}")
        for name, value in zip(("x0", "y0", "x1", "y1"), bbox):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"HandBox.bbox.{name}: ожидалось целое число, получено {value!r}")
        x0, y0, x1, y1 = bbox
        if not (x0 < x1 and y0 < y1):
            raise ValueError(f"HandBox.bbox: нужны x0 < x1 и y0 < y1, получено {bbox!r}")
        if not (0 <= x0 and x1 <= EYE_WIDTH and 0 <= y0 and y1 <= EYE_HEIGHT):
            raise ValueError(
                f"HandBox.bbox: {bbox!r} выходит за кадр левого глаза "
                f"{EYE_WIDTH}×{EYE_HEIGHT}"
            )

    @property
    def center(self) -> tuple[float, float]:
        """Центр прямоугольника, пиксели."""
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    def contains(self, u: float, v: float) -> bool:
        """Пиксель ``(u, v)`` внутри прямоугольника (границы включительно)."""
        x0, y0, x1, y1 = self.bbox
        return x0 <= u <= x1 and y0 <= v <= y1


@dataclass(frozen=True)
class CalibFrame:
    """Один кадр эталона: номер, время и прямоугольники кистей.

    ``t_s`` — время кадра относительно ``header.timeStampNs`` (``sync.frame_time(k)``,
    шаг 33.3371 мс); хранится здесь, чтобы эталон читался без ``video_meta.json``, а тест
    сверяет его с реальным ``sync`` (в репозитории без ``data/`` тест пропускается).
    """

    index: int
    t_s: float
    hands: tuple[HandBox, ...]
    ignore: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise ValueError(f"CalibFrame.index: ожидалось целое число, получено {self.index!r}")
        if self.index < 0:
            raise ValueError(f"CalibFrame.index: ожидалось ≥ 0, получено {self.index}")
        if not isinstance(self.t_s, (int, float)) or isinstance(self.t_s, bool):
            raise ValueError(f"CalibFrame.t_s: ожидалось число секунд, получено {self.t_s!r}")
        if not math.isfinite(self.t_s) or self.t_s < 0:
            raise ValueError(f"CalibFrame.t_s: ожидалось конечное число ≥ 0, получено {self.t_s}")
        if not self.hands:
            raise ValueError(f"CalibFrame {self.index}: нужен хотя бы один прямоугольник кисти")
        labels = [box.hand for box in self.hands if box.hand != "unknown"]
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"CalibFrame {self.index}: сторона встречается дважды ({labels}) — "
                "на кадре одна кисть каждой стороны"
            )
        for side in self.ignore:
            if side not in ("left", "right"):
                raise ValueError(
                    f"CalibFrame {self.index}: ignore — ожидалось 'left'/'right', получено {side!r}"
                )
            if side in labels:
                raise ValueError(
                    f"CalibFrame {self.index}: сторона {side!r} и размечена, и в ignore — "
                    "неоднозначный эталон"
                )

    @property
    def labeled_boxes(self) -> tuple[HandBox, ...]:
        """Прямоугольники с установленной стороной (``unknown`` не участвуют в проверке меток)."""
        return tuple(box for box in self.hands if box.hand != "unknown")

    def box_for(self, hand: str) -> HandBox | None:
        """Прямоугольник указанной стороны (``None``, если её нет на кадре/не размечена)."""
        for box in self.hands:
            if box.hand == hand:
                return box
        return None

    def as_dict(self) -> dict:
        """Плоское представление для ``--json`` (им же пользуется 07b)."""
        return {
            "index": self.index,
            "t_s": self.t_s,
            "ignore": list(self.ignore),
            "note": self.note,
            "hands": [
                {
                    "hand": box.hand,
                    "bbox": list(box.bbox),
                    "confidence": box.confidence,
                    "partial": box.partial,
                    "note": box.note,
                }
                for box in self.hands
            ],
        }


#: Рабочий набор кадров (07a): пять из 02a, разметка — по видео, левый глаз 1080×810.
#:
#: Кисти: рабочая рука приходит из правого края кадра (ремешок трекера на запястье виден
#: на k = 1800, 3600, 5400, 7200), вторая — покоится в левом нижнем углу. На k = 30 видна
#: только она: в 02a этот кадр был подписан «правая кисть», но та же поза в кадрах 1800 и
#: 7200 — это левая (покойная) рука, поэтому сторона помечена как ``unknown``.
CALIB_FRAMES: tuple[CalibFrame, ...] = (
    CalibFrame(
        index=30,
        t_s=1.000114027,
        hands=(
            HandBox(
                hand="unknown",
                bbox=(60, 350, 430, 810),
                confidence="medium",
                partial=True,
                note=(
                    "крупный кулак у камеры, предплечье уходит за нижний край; "
                    "вторая кисть за пределами FOV"
                ),
            ),
        ),
        ignore=("right",),
        note="близкая кисть — проверка масштаба; правая рука в кадр не попадает",
    ),
    CalibFrame(
        index=1800,
        t_s=60.006841599,
        hands=(
            HandBox(
                hand="right",
                bbox=(440, 130, 900, 340),
                confidence="high",
                note="пальцы над корзиной, ремешок трекера справа",
            ),
            HandBox(
                hand="left",
                bbox=(0, 520, 250, 810),
                confidence="high",
                partial=True,
                note="покойная кисть в нижнем левом углу, обрезана левым краем",
            ),
        ),
    ),
    CalibFrame(
        index=3600,
        t_s=120.013683198,
        hands=(
            HandBox(
                hand="right",
                bbox=(545, 190, 1080, 460),
                confidence="high",
                partial=True,
                note="кулак в центре-справа, запястье и трекер уходят за правый край",
            ),
            HandBox(
                hand="left",
                bbox=(0, 520, 100, 810),
                confidence="low",
                partial=True,
                note="от покойной кисти видна только полоска у левого края",
            ),
        ),
    ),
    CalibFrame(
        index=5400,
        t_s=180.020524797,
        hands=(
            HandBox(
                hand="right",
                bbox=(660, 290, 1060, 560),
                confidence="high",
                note="кисть на уровне стола, трекер за правым краем",
            ),
            HandBox(
                hand="left",
                bbox=(0, 540, 95, 810),
                confidence="low",
                partial=True,
                note="покойная кисть — узкая полоска у левого края",
            ),
        ),
    ),
    CalibFrame(
        index=7200,
        t_s=240.027366396,
        hands=(
            HandBox(
                hand="right",
                bbox=(420, 180, 845, 415),
                confidence="high",
                note="держит предмет над корзиной, оба сустава кисти в кадре",
            ),
            HandBox(
                hand="left",
                bbox=(10, 340, 300, 700),
                confidence="high",
                note="покойная кисть на левом краю корзины, предплечье вниз",
            ),
        ),
    ),
)


def validate_frames(frames: Sequence[CalibFrame] = CALIB_FRAMES) -> None:
    """Проверка набора целиком: номера и время возрастают, стороны не дублируются.

    Покрытие положений рук проверяется по данным трекинга (в тестах, где есть ``data/``):
    здесь проверяется только внутренняя согласованность разметки.
    """
    if len(frames) < 3:
        raise ValueError(f"эталон калибровки: нужно ≥ 3 кадров, получено {len(frames)}")
    for previous, current in zip(frames, frames[1:]):
        if current.index <= previous.index:
            raise ValueError(
                f"эталон калибровки: номера кадров должны возрастать "
                f"({previous.index} → {current.index})"
            )
        if current.t_s <= previous.t_s:
            raise ValueError(
                f"эталон калибровки: время кадров должно возрастать "
                f"(k={previous.index}: {previous.t_s} → k={current.index}: {current.t_s})"
            )
    indexes = {frame.index for frame in frames}
    if len(indexes) != len(frames):
        raise ValueError("эталон калибровки: номера кадров повторяются")
    if not any(box.confidence == "high" for frame in frames for box in frame.hands):
        raise ValueError("эталон калибровки: нет ни одного прямоугольника с уверенностью high")


def in_eye_frame(u: float, v: float) -> bool:
    """Пиксель внутри кадра левого глаза (полуоткрытый интервал, как ``camera_model.in_frame``)."""
    return 0 <= u < EYE_WIDTH and 0 <= v < EYE_HEIGHT


def as_json(frames: Sequence[CalibFrame] = CALIB_FRAMES) -> str:
    """Эталон в JSON (для 07b и внешних скриптов): кадры, рамки, допуск разметки."""
    return json.dumps(
        {
            "eye_size": [EYE_WIDTH, EYE_HEIGHT],
            "tolerance_px": TOLERANCE_PX,
            "frames": [frame.as_dict() for frame in frames],
        },
        ensure_ascii=False,
        indent=2,
    )


def _describe(frames: Sequence[CalibFrame]) -> str:
    """Человекочитаемая таблица эталона (та же, что пишется в ``frames.md``)."""
    lines = [
        "# Рабочий набор кадров для калибровки (задача 07a)",
        "",
        f"Координаты — левый глаз SBS-кадра {EYE_WIDTH}×{EYE_HEIGHT} (`frame[:, :1080]`), "
        "x вправо, y вниз.",
        f"Точность разметки — ±{TOLERANCE_PX} px (сетка 100 px по кадру).",
        "",
        "| # | k | t, с | Сторона | bbox (x0, y0, x1, y1) | Уверенность | Обрезана краем | Комментарий |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for number, frame in enumerate(frames, start=1):
        for box in frame.hands:
            x0, y0, x1, y1 = box.bbox
            lines.append(
                f"| {number} | {frame.index} | {frame.t_s:.3f} | {box.hand} | "
                f"({x0}, {y0}, {x1}, {y1}) | {box.confidence} | "
                f"{'да' if box.partial else 'нет'} | {box.note or '—'} |"
            )
        if frame.ignore:
            lines.append(
                f"| {number} | {frame.index} | {frame.t_s:.3f} | — | — | — | — | "
                f"не эталон: {', '.join(frame.ignore)} в кадре нет |"
            )
    lines.append("")
    lines.append("Кадры сохранены полностью (2160×810, SBS): `frame_{k:05d}.png`.")
    return "\n".join(lines)


def _save_frames(video: Path, out_dir: Path, frames: Sequence[CalibFrame], overlay: bool) -> int:
    """Сохраняет кадры набора (и, при ``overlay``, разметку) в ``out_dir``.

    Возвращает число записанных кадров. ``cv2`` импортируется локально: модуль эталона
    нужен 07b и тестам без OpenCV. Размер кадра проверяется до записи: если видео не
    SBS 2160×810, разметка в координатах левого глаза недействительна.
    """
    import cv2  # noqa: PLC0415 — локальный импорт: без OpenCV эталон тоже нужен

    if not video.is_file():
        raise SystemExit(f"нет файла: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"не открывается видео: {video}")
    full_size = (2 * EYE_WIDTH, EYE_HEIGHT)
    written = 0
    try:
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame.index)
            ok, image = cap.read()
            if not ok:
                raise SystemExit(f"не читается кадр k={frame.index}")
            height, width = image.shape[:2]
            if (width, height) != full_size:
                raise SystemExit(
                    f"кадр k={frame.index}: {width}×{height}, ожидалось "
                    f"{full_size[0]}×{full_size[1]} (SBS) — разметка в координатах левого глаза "
                    "недействительна"
                )
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"frame_{frame.index:05d}.png"), image)
            written += 1
            if overlay:
                marked = image[:, :EYE_WIDTH].copy()
                for box in frame.hands:
                    x0, y0, x1, y1 = box.bbox
                    color = {"left": (255, 0, 0), "right": (0, 0, 255)}.get(box.hand, (0, 200, 0))
                    cv2.rectangle(marked, (x0, y0), (x1, y1), color, 2)
                    cv2.putText(
                        marked,
                        f"{box.hand} {x0},{y0},{x1},{y1}",
                        (x0 + 4, max(y0 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                annotated = out_dir / "annotated"
                annotated.mkdir(exist_ok=True)
                cv2.imwrite(str(annotated / f"frame_{frame.index:05d}_left.png"), marked)
    finally:
        cap.release()
    return written


def main(argv: Iterable[str] | None = None) -> int:
    """CLI: таблица эталона, JSON для 07b или сохранение PNG + ``frames.md``."""
    parser = argparse.ArgumentParser(
        description="Рабочий набор кадров и пиксельный эталон кистей для калибровки (07a)."
    )
    parser.add_argument("video", nargs="?", type=Path, help="data/CameraRecord_*.mp4 (для сохранения PNG)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/debug/calib"),
        help="куда сохранять кадры (по умолчанию output/debug/calib)",
    )
    parser.add_argument("--show", action="store_true", help="напечатать таблицу эталона")
    parser.add_argument("--json", action="store_true", help="напечатать эталон в JSON")
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="вместе с кадрами сохранить разметку (annotated/frame_*_left.png)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    frames = CALIB_FRAMES
    validate_frames(frames)
    if args.json:
        print(as_json(frames))
        return 0
    if args.show or args.video is None:
        print(_describe(frames))
    if args.video is None:
        if not args.show and not args.json:
            print("\n(кадры: укажите путь к видео, чтобы сохранить PNG)")
        return 0
    if not args.video.is_file():
        raise SystemExit(f"нет файла: {args.video}")
    written = _save_frames(args.video, args.out, frames, args.overlay)
    (args.out / "frames.md").write_text(_describe(frames) + "\n", encoding="utf-8")
    print(f"\nсохранено кадров: {written} → {args.out}")
    print(f"разметка: {args.out / 'frames.md'}")
    if args.overlay:
        print(f"с разметкой: {args.out / 'annotated'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
