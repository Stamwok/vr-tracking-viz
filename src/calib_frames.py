"""calib_frames.py — рабочий набор кадров и пиксельный эталон для калибровки (задачи 07a, 08e).

Задача 07b подбирает конвенцию осей и ``T_head_cam``, сравнивая проекцию трекинга с видео.
Чтобы это сравнение было не «на глаз», нужен эталон, **независимый от проекции**: кадры,
на которых положение кистей в пикселях известно из самого видео. Он и лежит здесь — пять
кадров из 02a плюс прямоугольники кистей в координатах **левого глаза** (1080×810).

Задача 08e добавила рядом второй, более тонкий эталон — костную разметку :data:`BONE_FRAMES`:
7–9 пиксельных точек на руку (запястье, ладонь, пястная кость, фаланги указательного и
среднего пальцев) и цепочки отрезков между ними. Рамки :data:`CALIB_FRAMES` при этом не
менялись: метрика 07a остаётся критерием, а кости — дополнение для густого скелета (08f),
которое считает ``src/calib_bones.py`` (расстояние от проекции отрезков до размеченных
отрезков + счётчик изломов). Разметка костей сделана по видео с точностью ±30–50 px (кисть
на кадре занимает 150–370 px), поэтому вердикт костной метрики справочный.

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
    python src/calib_frames.py --bones                # костная разметка 08e (7–9 точек на руку)
    python src/calib_frames.py --bones-json           # то же в JSON
    python src/calib_frames.py data/CameraRecord_*.mp4  # сохранить PNG + frames.md + bones.md
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

from src.parse_tracking import HAND_JOINTS_FULL

#: Размер кадра одного глаза (левая половина SBS 2160×810): система координат разметки.
EYE_WIDTH = 1080
EYE_HEIGHT = 810

#: Точность ручной разметки по сетке 100 px, пиксели. Прямоугольник — «плюс-минус столько».
TOLERANCE_PX = 50

#: Допустимые значения ``HandBox.hand``: две стороны и «видна, но кто — неизвестно».
HANDS = ("left", "right", "unknown")

#: Допустимые уровни уверенности разметки (см. ``HandBox.confidence``).
CONFIDENCE = ("high", "medium", "low")

#: Допуск костной разметки (08e), пиксели: та же ручная сетка, что у рамок 07a.
BONE_TOLERANCE_PX = TOLERANCE_PX

#: Порог «излома» (08e): кость, направление которой ушло от размеченного на столько
#: градусов, считается изломом — скелет визуально сломан относительно эталона.
BONE_KINK_ANGLE_DEG = 45.0

#: Сколько точек на руку допускает костная разметка (08e), включительно.
BONE_POINTS_PER_HAND = (7, 9)

#: Стороны, которые можно размечать костями: для метрики нужна конкретная рука
#: (в отличие от рамки 07a, где допустимо ``unknown``).
BONE_HANDS = ("left", "right")


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


# --------------------------------------------------------------------------------------
# Костная разметка (задача 08e): 7–9 точек на руку и отрезки между ними.
# --------------------------------------------------------------------------------------

#: Имена суставов, которые вообще бывают в данных: 26 джойнтов OpenXR на руку (08d).
#: Костная разметка берёт имена отсюда, поэтому промежуточные фаланги называются так же,
#: как колонки ``tracking.csv`` (``index_proximal``, ``middle_intermediate`` …).
JOINT_NAMES: tuple[str, ...] = tuple(name for name, _ in HAND_JOINTS_FULL)

#: Топология эталона 08e — одинаковая для каждой размеченной кисти.
#:
#: Соседние имена в цепочке задают «кость» — отрезок эталонного скелета, до которого потом
#: меряется проекция (``src/calib_bones.py``). 9 точек и 8 отрезков:
#:
#: * ``wrist → palm`` — ось запястья (обе точки уже есть в 5-точечном наборе);
#: * ``palm → index_metacarpal → index_proximal → index_intermediate → index`` — цепочка
#:   указательного пальца: пястная кость, основная, средняя и ногтевая фаланги;
#: * ``palm → middle_proximal → middle_intermediate → middle`` — средний палец.
#:
#: Почему только указательный и средний: на всех кадрах записи кисть сжата (кулак, обхват
#: предмета), и именно эти два пальца читаются по видео лучше остальных; безымянный и
#: мизинец в такой позе перекрыты ладонью. Почему 9, а не 5 точек: короткие отрезки
#: (средняя фаланга ≈30–40 px) проверяют форму пальца, а не только «точка где-то на кисти».
BONE_CHAINS: tuple[tuple[str, ...], ...] = (
    ("wrist", "palm"),
    ("palm", "index_metacarpal", "index_proximal", "index_intermediate", "index"),
    ("palm", "middle_proximal", "middle_intermediate", "middle"),
)


@dataclass(frozen=True)
class JointMark:
    """Пиксельная разметка одного сустава на кадре (08e), координаты левого глаза.

    ``joint`` — имя из :data:`JOINT_NAMES`; ``point`` — ``(u, v)`` в пикселях кадра
    1080×810; ``confidence`` — насколько точка читается по видео. ``low`` — не «неверно»,
    а «плохо видно»: сжатый кулак не разбирается на фаланги, и такая точка годится только
    как грубая привязка (метрика считает её отдельно — см. ``strict`` в ``calib_bones``).
    """

    joint: str
    point: tuple[int, int]
    confidence: str = "high"
    note: str = ""

    def __post_init__(self) -> None:
        if self.joint not in JOINT_NAMES:
            raise ValueError(
                f"JointMark.joint: ожидалось имя из HAND_JOINTS_FULL, получено {self.joint!r}"
            )
        if self.confidence not in CONFIDENCE:
            raise ValueError(
                f"JointMark.confidence: ожидалось одно из {CONFIDENCE}, "
                f"получено {self.confidence!r}"
            )
        point = self.point
        if not isinstance(point, tuple) or len(point) != 2:
            raise ValueError(f"JointMark.point: ожидалось (u, v), получено {point!r}")
        for name, value in zip(("u", "v"), point):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"JointMark.point.{name}: ожидалось целое, получено {value!r}")
        u, v = point
        if not in_eye_frame(u, v):
            raise ValueError(
                f"JointMark.point: {point!r} выходит за кадр левого глаза {EYE_WIDTH}×{EYE_HEIGHT}"
            )


@dataclass(frozen=True)
class HandMarks:
    """Костная разметка одной кисти (08e): 7–9 суставов и цепочки отрезков между ними.

    ``chains`` — цепочки имён суставов; каждая соседняя пара — отрезок эталона. Все имена
    цепочек обязаны быть размечены (``marks``), и каждая точка должна входить хотя бы в
    одну цепочку: иначе точка ни на что не влияет и её легко потерять при правках.
    """

    hand: str
    chains: tuple[tuple[str, ...], ...]
    marks: tuple[JointMark, ...]
    note: str = ""

    def __post_init__(self) -> None:
        if self.hand not in BONE_HANDS:
            raise ValueError(
                f"HandMarks.hand: ожидалось 'left'/'right', получено {self.hand!r} "
                "(для костей сторона обязательна — рамка 07a может быть 'unknown')"
            )
        if not self.marks:
            raise ValueError(f"HandMarks {self.hand}: нужна хотя бы одна размеченная точка")
        low, high = BONE_POINTS_PER_HAND
        if not low <= len(self.marks) <= high:
            raise ValueError(
                f"HandMarks {self.hand}: ожидалось {low}–{high} точек, получено {len(self.marks)}"
            )
        names = [mark.joint for mark in self.marks]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"HandMarks {self.hand}: сустав размечен дважды ({duplicates})")
        if not self.chains:
            raise ValueError(f"HandMarks {self.hand}: нужна хотя бы одна цепочка")
        for chain in self.chains:
            if not isinstance(chain, tuple) or len(chain) < 2:
                raise ValueError(
                    f"HandMarks {self.hand}: цепочка {chain!r} должна быть кортежем ≥ 2 имён"
                )
            unknown = [name for name in chain if name not in names]
            if unknown:
                raise ValueError(
                    f"HandMarks {self.hand}: в цепочке {chain!r} есть неразмеченные {unknown}"
                )
        used = {name for chain in self.chains for name in chain}
        unused = sorted(set(names) - used)
        if unused:
            raise ValueError(
                f"HandMarks {self.hand}: точки {unused} не входят ни в одну цепочку"
            )

    @property
    def bones(self) -> tuple[tuple[str, str], ...]:
        """Отрезки эталона: соседние пары во всех цепочках."""
        return tuple(
            (first, second)
            for chain in self.chains
            for first, second in zip(chain, chain[1:])
        )

    @property
    def points(self) -> dict[str, tuple[int, int]]:
        """Пиксели разметки по имени сустава."""
        return {mark.joint: mark.point for mark in self.marks}

    def mark_for(self, joint: str) -> JointMark:
        """Разметка конкретного сустава (``KeyError``, если его нет)."""
        for mark in self.marks:
            if mark.joint == joint:
                return mark
        raise KeyError(f"HandMarks {self.hand}: сустав {joint!r} не размечен")

    def as_dict(self) -> dict:
        """Плоское представление для JSON."""
        return {
            "hand": self.hand,
            "note": self.note,
            "chains": [list(chain) for chain in self.chains],
            "marks": [
                {
                    "joint": mark.joint,
                    "point": list(mark.point),
                    "confidence": mark.confidence,
                    "note": mark.note,
                }
                for mark in self.marks
            ],
        }


@dataclass(frozen=True)
class BoneFrame:
    """Кадр костной разметки (08e): номер кадра видео и размеченные кисти."""

    index: int
    hands: tuple[HandMarks, ...]
    note: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise ValueError(f"BoneFrame.index: ожидалось целое число, получено {self.index!r}")
        if self.index < 0:
            raise ValueError(f"BoneFrame.index: ожидалось ≥ 0, получено {self.index}")
        if not self.hands:
            raise ValueError(f"BoneFrame {self.index}: нужна хотя бы одна размеченная кисть")
        sides = [hand.hand for hand in self.hands]
        if len(set(sides)) != len(sides):
            raise ValueError(
                f"BoneFrame {self.index}: сторона встречается дважды ({sides}) — "
                "на кадре одна кисть каждой стороны"
            )

    def hand_for(self, hand: str) -> HandMarks | None:
        """Разметка указанной стороны (``None``, если она не размечена)."""
        for marks in self.hands:
            if marks.hand == hand:
                return marks
        return None

    @property
    def bones(self) -> int:
        """Сколько отрезков размечено на кадре."""
        return sum(len(marks.bones) for marks in self.hands)

    def as_dict(self) -> dict:
        """Плоское представление для JSON."""
        return {
            "index": self.index,
            "note": self.note,
            "hands": [marks.as_dict() for marks in self.hands],
        }


#: Костная разметка (08e): 6 кистей на 5 кадрах, по 9 точек и 8 отрезков на кисть.
#:
#: Кадры и руки выбраны по видимости, а не по удобству: на кадре должна читаться кисть
#: целиком (тыл кисти, пястные кости, указательный и средний пальцы). Поэтому размечены:
#:
#: * k = 30 — левая (в 07a рамка ``unknown``; см. ``note`` кадра), уверенность ``low``:
#:   кисть вплотную к камере и обхватывает край корзины, фаланги читаются только по буграм;
#: * k = 1800 — правая: пальцы над корзиной видны сверху;
#: * k = 3600 — правая: кулак, видны тыл кисти и согнутые фаланги;
#: * k = 5400 — правая: самый ясный кадр, фаланги читаются по отдельности;
#: * k = 7200 — правая (держит предмет) и левая (на краю корзины).
#:
#: Чего здесь нет и почему (честно, а не «забыли»):
#:
#: * k = 1800, левая кисть — обрезана левым краем кадра, кончики пальцев за границей
#:   (это записано и в рамке 07a: ``partial=True``); размечено 0 точек, рамка остаётся;
#: * k = 3600 и 5400, левая кисть — от неё видна полоска 95–100 px у левого края
#:   (``confidence="low"`` в 07a): размечать фаланги не по чему;
#: * безымянный палец и мизинец не размечены нигде: в этой записи кисть всегда сжата, и
#:   они перекрыты ладонью (та же причина, по которой BONE_CHAINS короче 26 джойнтов).
BONE_FRAMES: tuple[BoneFrame, ...] = (
    BoneFrame(
        index=30,
        note=(
            "рамка 07a помечена 'unknown'; 07a установила, что это покоящаяся левая кисть — "
            "для костей сторона обязательна, поэтому left"
        ),
        hands=(
            HandMarks(
                hand="left",
                chains=BONE_CHAINS,
                marks=(
                    JointMark("wrist", (210, 700), "high"),
                    JointMark("palm", (285, 555), "high"),
                    JointMark("index_metacarpal", (307, 525), "low"),
                    JointMark("index_proximal", (330, 495), "low"),
                    JointMark("index_intermediate", (335, 530), "low"),
                    JointMark("index", (340, 568), "low"),
                    JointMark("middle_proximal", (355, 512), "low"),
                    JointMark("middle_intermediate", (362, 555), "low"),
                    JointMark("middle", (368, 600), "low"),
                ),
                note=(
                    "кисть вплотную к камере, сжата в кулак и обхватывает край корзины: "
                    "фаланги угадываются по буграм, поэтому low"
                ),
            ),
        ),
    ),
    BoneFrame(
        index=1800,
        note="правая кисть над корзиной: указательный и средний пальцы видны сверху",
        hands=(
            HandMarks(
                hand="right",
                chains=BONE_CHAINS,
                marks=(
                    JointMark("wrist", (700, 195), "high"),
                    JointMark("palm", (620, 180), "high"),
                    JointMark("index_metacarpal", (605, 207), "medium"),
                    JointMark("index_proximal", (590, 235), "medium"),
                    JointMark("index_intermediate", (562, 262), "medium"),
                    JointMark("index", (545, 285), "medium"),
                    JointMark("middle_proximal", (630, 245), "medium"),
                    JointMark("middle_intermediate", (600, 270), "medium"),
                    JointMark("middle", (578, 292), "medium"),
                ),
                note="кисть мелкая (≈150 px): точность разметки фаланг здесь хуже всего",
            ),
        ),
    ),
    BoneFrame(
        index=3600,
        note=(
            "правая кисть сжата в кулак: видны тыл кисти, бугры пястных костей и согнутые "
            "фаланги"
        ),
        hands=(
            HandMarks(
                hand="right",
                chains=BONE_CHAINS,
                marks=(
                    JointMark("wrist", (830, 270), "high"),
                    JointMark("palm", (701, 245), "high"),
                    JointMark("index_metacarpal", (676, 273), "medium"),
                    JointMark("index_proximal", (651, 301), "medium"),
                    JointMark("index_intermediate", (636, 341), "medium"),
                    JointMark("index", (628, 380), "medium"),
                    JointMark("middle_proximal", (684, 313), "medium"),
                    JointMark("middle_intermediate", (679, 346), "medium"),
                    JointMark("middle", (679, 380), "medium"),
                ),
            ),
        ),
    ),
    BoneFrame(
        index=5400,
        note="самый ясный кадр: тыл правой кисти, указательный и средний пальцы разведены",
        hands=(
            HandMarks(
                hand="right",
                chains=BONE_CHAINS,
                marks=(
                    JointMark("wrist", (880, 415), "high"),
                    JointMark("palm", (820, 390), "high"),
                    JointMark("index_metacarpal", (790, 415), "medium"),
                    JointMark("index_proximal", (760, 440), "medium"),
                    JointMark("index_intermediate", (750, 465), "medium"),
                    JointMark("index", (750, 490), "medium"),
                    JointMark("middle_proximal", (790, 445), "medium"),
                    JointMark("middle_intermediate", (780, 470), "medium"),
                    JointMark("middle", (785, 495), "medium"),
                ),
            ),
        ),
    ),
    BoneFrame(
        index=7200,
        note="правая кисть держит предмет, левая — на краю корзины: фаланги видны сбоку",
        hands=(
            HandMarks(
                hand="left",
                chains=BONE_CHAINS,
                marks=(
                    JointMark("wrist", (165, 620), "high"),
                    JointMark("palm", (158, 541), "high"),
                    JointMark("index_metacarpal", (163, 500), "medium"),
                    JointMark("index_proximal", (168, 458), "medium"),
                    JointMark("index_intermediate", (200, 425), "medium"),
                    JointMark("index", (234, 389), "medium"),
                    JointMark("middle_proximal", (190, 478), "medium"),
                    JointMark("middle_intermediate", (224, 454), "medium"),
                    JointMark("middle", (254, 422), "medium"),
                ),
                note="кисть боком, пальцы прижаты к краю корзины: видны бугры фаланг",
            ),
            HandMarks(
                hand="right",
                chains=BONE_CHAINS,
                marks=(
                    JointMark("wrist", (610, 310), "high"),
                    JointMark("palm", (550, 286), "high"),
                    JointMark("index_metacarpal", (525, 310), "medium"),
                    JointMark("index_proximal", (500, 335), "medium"),
                    JointMark("index_intermediate", (490, 360), "medium"),
                    JointMark("index", (485, 385), "medium"),
                    JointMark("middle_proximal", (525, 345), "medium"),
                    JointMark("middle_intermediate", (515, 370), "medium"),
                    JointMark("middle", (512, 395), "medium"),
                ),
            ),
        ),
    ),
)


def validate_bone_frames(
    frames: Sequence[BoneFrame] = BONE_FRAMES,
    calib: Sequence[CalibFrame] = CALIB_FRAMES,
) -> None:
    """Проверка костной разметки целиком и её согласованности с рамками 07a.

    Две разметки живут раздельно (рамки не трогаются), поэтому проверяется их стык:
    каждый кадр костей есть в 07a, на нём действительно видна эта рука (не в ``ignore``),
    для неё есть прямоугольник, а кадры 07a покрыты костями все — иначе «эталон на 5
    кадрах» тихо превратился бы в эталон на трёх.
    """
    if not frames:
        raise ValueError("костная разметка: пусто")
    by_index = {frame.index: frame for frame in calib}
    for previous, current in zip(frames, frames[1:]):
        if current.index <= previous.index:
            raise ValueError(
                f"костная разметка: номера кадров должны возрастать "
                f"({previous.index} → {current.index})"
            )
    for frame in frames:
        reference = by_index.get(frame.index)
        if reference is None:
            raise ValueError(f"костная разметка: кадра k={frame.index} нет в CALIB_FRAMES (07a)")
        for marks in frame.hands:
            if marks.hand in reference.ignore:
                raise ValueError(
                    f"костная разметка k={frame.index}: рука {marks.hand!r} помечена в 07a как "
                    "невидимая (ignore) — размечать её кости нельзя"
                )
            if reference.box_for(marks.hand) is None and reference.box_for("unknown") is None:
                raise ValueError(
                    f"костная разметка k={frame.index}: для руки {marks.hand!r} нет рамки в 07a"
                )
    missing = [frame.index for frame in calib if frame.index not in {f.index for f in frames}]
    if missing:
        raise ValueError(f"костная разметка: кадры 07a без костей — {missing}")
    if not any(mark.confidence == "high" for frame in frames for hand in frame.hands
               for mark in hand.marks):
        raise ValueError("костная разметка: нет ни одной точки с уверенностью high")


def bones_as_json(frames: Sequence[BoneFrame] = BONE_FRAMES) -> str:
    """Костная разметка в JSON: точки, цепочки, допуск и порог излома."""
    return json.dumps(
        {
            "eye_size": [EYE_WIDTH, EYE_HEIGHT],
            "tolerance_px": BONE_TOLERANCE_PX,
            "kink_angle_deg": BONE_KINK_ANGLE_DEG,
            "points_per_hand": list(BONE_POINTS_PER_HAND),
            "frames": [frame.as_dict() for frame in frames],
        },
        ensure_ascii=False,
        indent=2,
    )


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


def _describe_bones(frames: Sequence[BoneFrame] = BONE_FRAMES) -> str:
    """Человекочитаемая таблица костной разметки (пишется в ``bones.md``)."""
    low, high = BONE_POINTS_PER_HAND
    lines = [
        "# Костная разметка: 7–9 точек на руку (задача 08e)",
        "",
        f"Координаты — левый глаз SBS-кадра {EYE_WIDTH}×{EYE_HEIGHT}; допуск "
        f"±{BONE_TOLERANCE_PX} px, порог излома {BONE_KINK_ANGLE_DEG:g}°.",
        f"На кисть {low}–{high} точек; цепочки отрезков — `BONE_CHAINS`:",
        "",
        "```",
        *("  " + " → ".join(chain) for chain in BONE_CHAINS),
        "```",
        "",
        "| k | сторона | сустав | u | v | уверенность | в цепочках |",
        "|---|---|---|---|---|---|---|",
    ]
    for frame in frames:
        for hand in frame.hands:
            for mark in hand.marks:
                groups = [
                    f"#{i}" for i, chain in enumerate(hand.chains) if mark.joint in chain
                ]
                lines.append(
                    f"| {frame.index} | {hand.hand} | {mark.joint} | {mark.point[0]} | "
                    f"{mark.point[1]} | {mark.confidence} | {', '.join(groups)} |"
                )
    lines.append("")
    lines.append(
        "Не размечены: k = 1800 (левая), k = 3600 и k = 5400 (левая) — кисть обрезана краем "
        "кадра, кончиков пальцев в кадре нет."
    )
    return "\n".join(lines)


def _save_frames(
    video: Path,
    out_dir: Path,
    frames: Sequence[CalibFrame],
    overlay: bool,
    bones: Sequence[BoneFrame] = BONE_FRAMES,
) -> int:
    """Сохраняет кадры набора (и, при ``overlay``, разметку) в ``out_dir``.

    Возвращает число записанных кадров. ``cv2`` импортируется локально: модуль эталона
    нужен 07b и тестам без OpenCV. Размер кадра проверяется до записи: если видео не
    SBS 2160×810, разметка в координатах левого глаза недействительна.

    При ``overlay`` на кадр левого глаза рисуются и рамки 07a, и костная разметка 08e
    (точки + цепочки): это артефакт для глаз, метрика его не использует.
    """
    import cv2  # noqa: PLC0415 — локальный импорт: без OpenCV эталон тоже нужен

    if not video.is_file():
        raise SystemExit(f"нет файла: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"не открывается видео: {video}")
    full_size = (2 * EYE_WIDTH, EYE_HEIGHT)
    bones_by_index = {frame.index: frame for frame in bones}
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
                bone_frame = bones_by_index.get(frame.index)
                if bone_frame is not None:
                    _draw_bone_marks(cv2, marked, bone_frame)
                annotated = out_dir / "annotated"
                annotated.mkdir(exist_ok=True)
                cv2.imwrite(str(annotated / f"frame_{frame.index:05d}_left.png"), marked)
    finally:
        cap.release()
    return written


def _draw_bone_marks(cv2, image, frame: BoneFrame) -> None:
    """Нарисовать костную разметку кадра: отрезки цепочек и точки с подписями."""
    colors = {"left": (255, 128, 0), "right": (0, 128, 255)}
    for hand in frame.hands:
        color = colors.get(hand.hand, (0, 200, 0))
        points = hand.points
        for chain in hand.chains:
            for first, second in zip(chain, chain[1:]):
                cv2.line(image, points[first], points[second], color, 2, cv2.LINE_AA)
        for mark in hand.marks:
            u, v = mark.point
            radius = 3 if mark.confidence == "high" else 2
            cv2.circle(image, (u, v), radius, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                mark.joint,
                (u + 5, v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                image,
                mark.joint,
                (u + 5, v - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )


def main(argv: Iterable[str] | None = None) -> int:
    """CLI: таблица эталона, JSON для 07b, костная разметка 08e или сохранение PNG."""
    parser = argparse.ArgumentParser(
        description=(
            "Рабочий набор кадров и пиксельный эталон кистей для калибровки (07a) "
            "с костной разметкой 08e."
        )
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
        "--bones",
        action="store_true",
        help="напечатать костную разметку 08e (7–9 точек на руку) вместо таблицы рамок",
    )
    parser.add_argument(
        "--bones-json",
        action="store_true",
        help="напечатать костную разметку 08e в JSON",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="вместе с кадрами сохранить разметку: рамки 07a + кости 08e",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    frames = CALIB_FRAMES
    validate_frames(frames)
    validate_bone_frames()
    if args.bones_json:
        print(bones_as_json())
        return 0
    if args.bones:
        print(_describe_bones())
        return 0
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
    (args.out / "bones.md").write_text(_describe_bones() + "\n", encoding="utf-8")
    print(f"\nсохранено кадров: {written} → {args.out}")
    print(f"разметка: {args.out / 'frames.md'}")
    print(f"кости: {args.out / 'bones.md'}")
    if args.overlay:
        print(f"с разметкой: {args.out / 'annotated'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
