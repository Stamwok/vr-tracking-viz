"""overlay_video.py — покадровый рендер скелета кистей поверх видео камеры шлема (08a, 08b).

Что делает кадр: берёт левую половину SBS-кадра (левый глаз, 1080×810), для номера кадра
``k`` считает время (:func:`sync.frame_time`, 04a) и позу на этот момент
(:meth:`sync.PoseTrack.pose_at`: линейная интерполяция позиций + SLERP ориентации, 04b/04c),
переводит 10 точек кистей в координаты камеры (:func:`camera_model.world_to_camera_matrix`
с константами из :mod:`src.config`) и в пиксели (:func:`camera_model.project` по per-eye
интринсикам, 05a–05c), затем рисует по 5 точек на руку и линии «ладонь → запястье /
большой / указательный / средний». Левая и правая руки — разные цвета.

Задача 08b добавляет след кистей и HUD (оба включаются флагами ``--no-trail``/``--no-hud``):

* **След** — мировой путь ладони за последние :data:`TRAIL_SECONDS` секунд, спроецированный
  матрицей **текущего** кадра: шлейф показывает, где кисть была, с точки зрения нынешней
  камеры. Выборка идёт **по времени** (шаг :data:`TRAIL_STEP_S`), а не по номерам кадров,
  поэтому окно следа не зависит от fps рендера. Прозрачность растёт к текущему моменту;
  если данных нет (``isActive = 0``, NaN сустава, точка за камерой), линия **рвётся**, а не
  соединяет «до» и «после» — иначе след залипал бы на переходах и на старте записи.
* **HUD** — неподвижная полупрозрачная панель в левом верхнем углу: время, номер кадра,
  состояние каждой руки («нет данных» при ``isActive = 0``) и окно следа. Панель не
  сдвигается от кадра к кадру, поэтому её положение проверяемо: на 5 кадрах эталона 07a она
  не перекрывает ни одной размеченной кисти (:func:`hud_rect`). Подписи латиницей, потому
  что ``cv2.putText`` с Hershey-шрифтами кириллицу не рисует.

Задача 08f рисует **густой скелет** вместо 5 точек: список рёбер задан явно в коде
(:data:`DENSE_BONES` для 17 суставов по умолчанию, :data:`FULL_BONES` — все 26 джойнтов
OpenXR с пястными костями). Политика разрывов та же, что у 5-точечного варианта: кость
рисуется, только если **оба** её конца видны; ``NaN`` («нет данных», правило 04b) и точка
за камерой дают ``visible = False``, поэтому через дырку линия не проводится и не
интерполируется. Пять опорных суставов берутся из :class:`sync.Pose` (как в 08a), остальные —
через :meth:`sync.PoseTrack.joint_at` с той же интерполяцией и тем же моментом времени
(:func:`src.calib_bones` использует этот путь с 08e). CLI: по умолчанию густой скелет,
``--all-joints`` — все 26 суставов, ``--five-points`` — прежний 5-точечный кадр 08a
(для сравнений «до/после»).

Задача 08g кодирует **глубину цветом**: множитель к цвету руки зависит от ``z`` сустава
в системе камеры (``p_cam.z``, а не от мировой координаты и не от min–max по кадру — окно
абсолютное, :data:`DEPTH_NEAR_M`…:data:`DEPTH_FAR_M`, поэтому яркость не «дышит» и не
мигает). Ближние кости и точки рисуются полным цветом, дальние — не темнее
:data:`DEPTH_MIN_FACTOR`; оттенок руки и геометрия не меняются, калибровка и время кадра
не трогаются. 5-точечный кадр 08a по умолчанию остаётся плоским, режим переключается
``--depth-color``/``--no-depth-color``.

Доработка 08g-fix2 добавляет **второй канал глубины — прозрачность** (``alpha``): ближняя
кость/точка непрозрачна, дальняя полупрозрачна (:func:`depth_alpha`, пол
:data:`DEPTH_MIN_ALPHA`), поэтому глубина читается не только по яркости, но и по тому, что
сквозь дальнюю кость виден фон. Кость по-прежнему рисуется :data:`DEPTH_BONE_SEGMENTS`
отрезками, и каждый отрезок получает свою alpha по средней глубине своих концов; у точки —
alpha по её глубине, а чёрная обводка остаётся полностью непрозрачной (иначе точка исчезает
на тёмном фоне). Так как ``cv2.line``/``cv2.circle`` не умеют alpha, отрезок и круг рисуются
в маску размером со свой bbox (:func:`_blend_line`, :func:`_blend_point`) — тот же приём, что
у следа 08b, иначе на кадр лёг бы полупрозрачный тёмный прямоугольник. В режиме alpha
дальние кости и точки рисуются **до** ближних (сортировка по ``p_cam.z`` по убыванию), чтобы
полупрозрачная дальняя кость не просвечивала поверх непрозрачной ближней. Флаги —
``--depth-alpha``/``--no-depth-alpha`` (по умолчанию включено для густого скелета и
``--all-joints``, выключено для 5-точечного 08a), ``--no-depth-color`` отключает и яркость,
и alpha; ``--no-depth-alpha`` воспроизводит кадр 08g-fix (только яркость).

Поза головы = поза камеры (камера в голове, ``T_head_cam`` из ``config.py``), поэтому
отдельного маркера головы в кадре нет — о ней говорит HUD (время и номер кадра).

Почему на входе сырой файл трекинга (JSON Lines), а не ``tracking.csv``: для синхронизации
нужен ``header.timeStampNs`` — метка первого кадра видео (04a). В ``tracking.csv`` её нет
(там только ``t_ns`` сэмплов), поэтому по CSV нельзя восстановить время кадра точно.
CSV остаётся входом графиков (09), а рендер читает ``data/trackingData_*.txt`` потоково
(:func:`sync.load_poses`, ≈6 МБ в памяти, ≈3 с на 21 787 сэмплов).

Задержка видеотракта. Поза кадра берётся не в ``frame_time(k)``, а со сдвигом
``config.VIDEO_TIME_OFFSET_MS`` (:func:`sync.pose_at_frame`): кадр показывает сцену раньше
своей метки, и без поправки скелет уезжает с кисти на быстрых движениях (диагностика —
``src/drift_diag.py``, 29.10). Сдвиг применяется и в проверке калибровки (``calib_search``),
поэтому метрика 07a описывает именно то, что рисует рендер. ``--time-offset-ms`` перекрывает
константу для сравнений.

Кодеки. Пишем H.264 (``avc1``), при недоступности — MPEG-4 part 2 (``mp4v``): в сборке
OpenCV без OpenH264 первый вариант просто не откроется. Фактический файл после рендера
проверяется :func:`probe_output` (открывается заново, число кадров и размер сверяются).

Запуск (весь клип, 7279 кадров, ≈242.66 с):
    python src/overlay_video.py data/CameraRecord_20260505_165740.mp4 \\
        --tracking data/trackingData_20260505_165740.txt --out output/overlay.mp4

Короткий фрагмент, PNG-проверка на 5 отладочных кадрах 07a и рендер «как было в 08a»
(без следа и HUD — этот кадр идёт в сравнение «до/после»):
    python src/overlay_video.py data/CameraRecord_*.mp4 --start 1790 --end 1810 --scale 1.5
    python src/overlay_video.py data/CameraRecord_*.mp4 --stills output/debug/overlay
    python src/overlay_video.py data/CameraRecord_*.mp4 --stills output/debug/overlay_before \\
        --frames 1800 --no-trail --no-hud
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import camera_model, sync
from src.parse_tracking import HAND_JOINTS, HAND_JOINTS_FULL, read_header, timestamp_range

#: Имена 5 суставов на руку в порядке отрисовки (те же, что в ``tracking.csv``).
JOINTS = tuple(name for name, _ in HAND_JOINTS)

#: Имена всех 26 джойнтов OpenXR на руку (порядок и индексы — ``parse_tracking.HAND_JOINTS_FULL``).
FULL_JOINTS = tuple(name for name, _ in HAND_JOINTS_FULL)

#: Кости скелета: ладонь → запястье и три «пальцевые» точки (ТЗ: 3–5 точек на руку).
HAND_CHAINS = (
    ("palm", "wrist"),
    ("palm", "thumb"),
    ("palm", "index"),
    ("palm", "middle"),
)

# --- 08f: густой скелет ---------------------------------------------------------------

#: Суставы густого скелета по умолчанию (17 на руку). Указательный и средний пальцы —
#: целиком, большой — без пястной кости, безымянный и мизинец — проксимальная фаланга и
#: кончик (иначе не уложиться в бюджет карточки «15–18 точек»). Пястные кости в кадре
#: лежат внутри силуэта ладони, поэтому в наборе по умолчанию их нет; ``--all-joints``
#: рисует все 26 суставов и полные цепочки.
DENSE_JOINTS = (
    "palm",
    "wrist",
    "thumb_proximal",
    "thumb_distal",
    "thumb",
    "index_proximal",
    "index_intermediate",
    "index_distal",
    "index",
    "middle_proximal",
    "middle_intermediate",
    "middle_distal",
    "middle",
    "ring_proximal",
    "ring",
    "little_proximal",
    "little",
)

#: Рёбра густого скелета — та же структура, что :data:`HAND_CHAINS`: (начало, конец).
#: У безымянного и мизинца в наборе по умолчанию нет промежуточной и дистальной фаланг,
#: поэтому ребро «проксимальная → кончик» — прямая через согнутый палец: это осознанное
#: упрощение ради бюджета точек, а не точная кость (полные цепочки — :data:`FULL_BONES`).
DENSE_BONES = (
    ("palm", "wrist"),
    ("palm", "thumb_proximal"),
    ("thumb_proximal", "thumb_distal"),
    ("thumb_distal", "thumb"),
    ("palm", "index_proximal"),
    ("index_proximal", "index_intermediate"),
    ("index_intermediate", "index_distal"),
    ("index_distal", "index"),
    ("palm", "middle_proximal"),
    ("middle_proximal", "middle_intermediate"),
    ("middle_intermediate", "middle_distal"),
    ("middle_distal", "middle"),
    ("palm", "ring_proximal"),
    ("ring_proximal", "ring"),
    ("palm", "little_proximal"),
    ("little_proximal", "little"),
)

#: Рёбра полного скелета 26 джойнтов OpenXR: запястье → ладонь, ладонь → пястная кость
#: каждого пальца, дальше цепочка фаланг до кончика (у большого пальца нет промежуточной).
FULL_BONES = (
    ("palm", "wrist"),
    ("palm", "thumb_metacarpal"),
    ("thumb_metacarpal", "thumb_proximal"),
    ("thumb_proximal", "thumb_distal"),
    ("thumb_distal", "thumb"),
    ("palm", "index_metacarpal"),
    ("index_metacarpal", "index_proximal"),
    ("index_proximal", "index_intermediate"),
    ("index_intermediate", "index_distal"),
    ("index_distal", "index"),
    ("palm", "middle_metacarpal"),
    ("middle_metacarpal", "middle_proximal"),
    ("middle_proximal", "middle_intermediate"),
    ("middle_intermediate", "middle_distal"),
    ("middle_distal", "middle"),
    ("palm", "ring_metacarpal"),
    ("ring_metacarpal", "ring_proximal"),
    ("ring_proximal", "ring_intermediate"),
    ("ring_intermediate", "ring_distal"),
    ("ring_distal", "ring"),
    ("palm", "little_metacarpal"),
    ("little_metacarpal", "little_proximal"),
    ("little_proximal", "little_intermediate"),
    ("little_intermediate", "little_distal"),
    ("little_distal", "little"),
)


@dataclass(frozen=True)
class Skeleton:
    """Набор суставов и рёбер одной руки, который рисует :func:`draw_skeleton`.

    Проверки в :meth:`__post_init__` — не формальность: список рёбер задан руками, и
    опечатка в имени сустава иначе проявилась бы молчаливым ``KeyError`` на середине
    рендера 7279 кадров. Ошибка здесь падает сразу при импорте модуля.
    """

    #: Короткое имя для CLI и статистики: ``five``, ``dense``, ``all``.
    name: str
    #: Человекочитаемое описание для логов: «густой (17 суставов)».
    title: str
    #: Имена суставов (без префикса руки) в порядке отрисовки точек.
    joints: tuple[str, ...]
    #: Рёбра ``(начало, конец)`` — тоже без префикса руки.
    bones: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Skeleton.name: ожидалось непустое имя, получено {self.name!r}")
        if not isinstance(self.title, str) or not self.title:
            raise ValueError(f"Skeleton.title: ожидалось непустое описание, получено {self.title!r}")
        if not self.joints:
            raise ValueError(f"Skeleton {self.name!r}: пустой список суставов")
        seen: set[str] = set()
        for joint in self.joints:
            if joint not in FULL_JOINTS:
                raise ValueError(f"Skeleton {self.name!r}: неизвестный сустав {joint!r}")
            if joint in seen:
                raise ValueError(f"Skeleton {self.name!r}: сустав {joint!r} повторяется")
            seen.add(joint)
        used: set[frozenset[str]] = set()
        for bone in self.bones:
            if not isinstance(bone, tuple) or len(bone) != 2:
                raise ValueError(f"Skeleton {self.name!r}: ребро {bone!r} — не пара суставов")
            start, end = bone
            for joint in (start, end):
                if joint not in seen:
                    raise ValueError(
                        f"Skeleton {self.name!r}: ребро {start!r}→{end!r} ссылается на "
                        f"сустав {joint!r}, которого нет в наборе"
                    )
            if start == end:
                raise ValueError(f"Skeleton {self.name!r}: ребро {start!r}→{end!r} вырождено")
            key = frozenset(bone)
            if key in used:
                raise ValueError(f"Skeleton {self.name!r}: ребро {start!r}→{end!r} повторяется")
            used.add(key)

    @property
    def extra_keys(self) -> tuple[str, ...]:
        """Ключи ``sync.PoseTrack.extra_keys`` для этого скелета (для обеих рук).

        Пять опорных суставов уже лежат в :class:`sync.Pose`; остальные нужно догрузить
        в трек: ``sync.load_poses(tracking, extra_keys=skeleton.extra_keys)`` (или
        ``load_scene(..., extra_keys=...)``), иначе :meth:`sync.PoseTrack.joint_at`
        не найдёт колонку.
        """
        names = tuple(joint for joint in self.joints if joint not in JOINTS)
        return tuple(f"{side}_{joint}" for side in ("left", "right") for joint in names)


#: Прежний 5-точечный скелет 08a (значения по умолчанию у функций рисования).
FIVE_POINT = Skeleton("five", "5 точек", JOINTS, HAND_CHAINS)

#: Густой скелет по умолчанию: 17 суставов на руку (задача 08f).
DENSE = Skeleton("dense", "густой (17 суставов)", DENSE_JOINTS, DENSE_BONES)

#: Полный скелет: все 26 джойнтов OpenXR на руку (``--all-joints``).
ALL_JOINTS = Skeleton("all", "все 26 суставов", FULL_JOINTS, FULL_BONES)

#: Все скелеты по имени (для CLI и тестов).
SKELETONS: Mapping[str, Skeleton] = {
    skeleton.name: skeleton for skeleton in (FIVE_POINT, DENSE, ALL_JOINTS)
}

#: Скелет, который рисует CLI без флагов (задача 08f).
DEFAULT_SKELETON = DENSE

#: Скелет, кадр которого должен воспроизводиться без новых эффектов (задача 08a).
FLAT_SKELETON = FIVE_POINT


def resolve_depth_shading(value: bool | None, skeleton: Skeleton) -> bool:
    """Включено ли кодирование глубины цветом (08g) для выбранного скелета.

    ``None`` — авторежим: включено для любого скелета, кроме прежнего 5-точечного
    (:data:`FLAT_SKELETON`). Так густой скелет по умолчанию получает подсказку глубины, а
    кадр 08a (``--five-points``) остаётся прежним. Явные ``True``/``False`` (CLI
    ``--depth-color``/``--no-depth-color``) перекрывают авторежим. Сравнение по имени, а не
    по идентичности объекта: набор ``Skeleton("five", …)``, собранный заново, — тот же кадр.
    """
    if not isinstance(skeleton, Skeleton):
        raise ValueError(f"skeleton: ожидался Skeleton, получено {skeleton!r}")
    if value is None:
        return skeleton.name != FLAT_SKELETON.name
    if not isinstance(value, bool):
        raise ValueError(f"depth_shading: ожидалось bool или None, получено {value!r}")
    return value


def resolve_depth_alpha(value: bool | None, skeleton: Skeleton) -> bool:
    """Включена ли прозрачность по глубине (08g-fix2) для выбранного скелета.

    Правило авторежима то же, что у яркости (:func:`resolve_depth_shading`): ``None`` —
    включено для любого скелета, кроме прежнего 5-точечного (:data:`FLAT_SKELETON`); явные
    ``True``/``False`` (CLI ``--depth-alpha``/``--no-depth-alpha``) перекрывают авторежим.
    Сравнение по имени набора — см. :func:`resolve_depth_shading`.

    Отдельная функция, а не флаг внутри :func:`resolve_depth_shading`: яркость и alpha —
    независимые каналы, и их можно включать порознь (сравнение «только яркость» / «только
    alpha» / «оба» — это проверка доработки). ``--no-depth-color`` отключает оба канала сразу,
    поэтому при выключенной яркости alpha не включается (:func:`render_video`).
    """
    if not isinstance(skeleton, Skeleton):
        raise ValueError(f"skeleton: ожидался Skeleton, получено {skeleton!r}")
    if value is None:
        return skeleton.name != FLAT_SKELETON.name
    if not isinstance(value, bool):
        raise ValueError(f"depth_alpha: ожидалось bool или None, получено {value!r}")
    return value


#: Ширина кадра глаза, к которой отнесены радиус и толщина по умолчанию.
REFERENCE_WIDTH = 1080

#: Радиус точки и толщина линии при ``scale = 1`` на кадре 1080×810, px.
POINT_RADIUS_PX = 4.0
LINE_THICKNESS_PX = 2.0

#: Цвета рук (BGR): левая — голубая, правая — красная. Те же цвета, что в разметке
#: эталона 07a (``calib_frames``/``calib_search``), поэтому PNG наложения и разметка
#: читаются одинаково; между собой и с фоном сцены цвета различимы.
COLORS: Mapping[str, tuple[int, int, int]] = {"left": (255, 140, 0), "right": (0, 0, 255)}

#: Тёмная обводка точки и её толщина: цвет не теряется на светлых предметах (губки, бумага).
OUTLINE_COLOR = (0, 0, 0)
OUTLINE_THICKNESS = 1

#: Кодеки для :class:`cv2.VideoWriter` в порядке предпочтения: H.264, затем MPEG-4 part 2.
FOURCC_CANDIDATES = ("avc1", "mp4v")

#: Предел координаты при отрисовке: cv2 ждёт int32, а проекция «видимой» точки у края кадра
#: может уйти на тысячи пикселей. Обрезку отрезка по краю кадра cv2 делает сам, поэтому
#: координата не отбрасывается, а только ограничивается по модулю.
COORD_LIMIT = 1_000_000

#: Кадры 07a по умолчанию для режима ``--stills`` (совпадают с ``camera_model.DEBUG_FRAMES``).
DEBUG_FRAMES = (30, 1800, 3600, 5400, 7200)

# --- 08b: след (trail) и HUD -------------------------------------------------------------

#: Окно следа в секундах: видно движение кисти за последние столько секунд до кадра.
#: Именно время, а не число кадров: окно одинаково при любом fps и не «плывёт» на паузах.
TRAIL_SECONDS = 0.6

#: Шаг выборки следа, секунды (1/30 — плотность кадров записи: шлейф гладкий, но не гуще,
#: 19 выборок на окно 0.6 с). Шаг по времени, поэтому след не зависит от ``--start/--end``.
TRAIL_STEP_S = 1.0 / 30.0

#: Прозрачность следа: у текущего момента и у самого старого конца окна.
TRAIL_ALPHA_NEW = 0.80
TRAIL_ALPHA_OLD = 0.12

#: HUD — фиксированная панель в левом верхнем углу (размеры для кадра 1080×810, дальше
#: масштабируются шириной кадра). Постоянный размер — чтобы положение панели можно было
#: проверить на эталоне 07a: ``(12, 12, 12+330, 12+136)`` не пересекает размеченных кистей.
HUD_MARGIN_PX = 12
HUD_WIDTH_PX = 330
HUD_HEIGHT_PX = 136
HUD_PADDING_PX = 10
HUD_LINE_STEP_PX = 22
HUD_BASELINE_OFFSET_PX = 5
HUD_FONT_SCALE = 0.5
#: ``cv2.FONT_HERSHEY_SIMPLEX`` = 0: константа без импорта OpenCV (модуль импортируем без него).
HUD_FONT_FACE = 0
HUD_PANEL_COLOR = (28, 28, 28)
HUD_ALPHA = 0.6
HUD_TEXT_COLOR = (238, 238, 238)
#: Цвет предупреждения «нет данных» (BGR: оранжево-жёлтый, заметен на панели и на видео).
HUD_WARNING_COLOR = (0, 200, 255)


# --- 08g: глубина → интенсивность цвета -------------------------------------------------

#: Абсолютное окно глубины (``z`` в системе камеры), м: у ближней границы цвет полный, у
#: дальней и за ней — приглушённый (не темнее :data:`DEPTH_MIN_FACTOR`). Именно абсолютные
#: метры, а не min–max по кадру: иначе множитель пересчитывался бы каждый кадр и на паузе
#: скелет мигал бы. Дооборудовано в доработке 08g: окно сужено с 0.22…0.58 до 0.22…0.48 м,
#: а нижняя граница множителя опущена с 0.45 до 0.30 — иначе разброс внутри одной кисти
#: (p50 0.10 м по суставам) давал контраст ≈1.2× и был неразличим. Замер: 14 507 пар
#: «кадр × рука», 246 619 видимых суставов; z суставов 0.072…0.640, z средних по костям
#: 0.224…0.556; кости за 0.48 м упираются в пол 0.30 (около 25 % пар) — дальняя рука тёмная,
#: но остаётся различимой. Типичные кадры дают контраст ближняя/дальняя кость одной руки
#: ≥ 1.6× примерно в половине случаев (p50 1.61 по суставам), на крайних кадрах (k = 4285,
#: рука на вытяжку за 0.5 м) дальняя кисть упирается в пол — там глубина читается как
#: «вся рука дальше», а не как тонкая градация внутри неё.
DEPTH_NEAR_M = 0.22
DEPTH_FAR_M = 0.48

#: Нижняя граница множителя: дальняя кость приглушается, но не исчезает — на тёмном фоне
#: (корзина, тени) линия остаётся различимой. Верхняя граница — 1.0 (полный цвет руки).
DEPTH_MIN_FACTOR = 0.30

#: Нижняя граница прозрачности в режиме глубины (доработка 08g-fix2), alpha: ближняя кость
#: или точка полностью непрозрачна (1.0), дальняя — не прозрачнее 0.45. Это второй канал,
#: независимый от яркости: сквозь дальнюю кость виден фон, поэтому глубина читается и на
#: пёстром фоне, где приглушение насыщенного цвета руки само по себе почти незаметно
#: (наблюдение пользователя по 08g-fix). Окно по умолчанию — то же :data:`DEPTH_NEAR_M`…
#: :data:`DEPTH_FAR_M`: одна шкала для обоих каналов проще для интерпретации.
DEPTH_MIN_ALPHA = 0.45

#: Сколько отрезков рисуется на одну кость в режиме глубины (градиент вдоль кости, 08g).
#: 5 — компромисс: градиент читается на линии толщиной 3 px, но не рвёт её на «рябь»;
#: цвет отрезка берётся по множителю в его середине. При ``depths=None`` кость, как раньше,
#: рисуется одной линией.
DEPTH_BONE_SEGMENTS = 5

#: Толщина кости в режиме глубины на кадре 1080×810, px (доработка 08g). Толще обычных
#: :data:`LINE_THICKNESS_PX`, потому что градиент на линии 2 px почти не читается; только
#: для густого скелета/всех 26 джойнтов — 5-точечный кадр 08a не трогается.
DEPTH_LINE_THICKNESS_PX = 3


@dataclass(frozen=True)
class Style:
    """Толщины отрисовки в пикселях уже для конкретного кадра (масштаб применён)."""

    point_radius: int
    line_thickness: int

    def __post_init__(self) -> None:
        for name in ("point_radius", "line_thickness"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Style.{name}: ожидалось целое число, получено {value!r}")
            if value < 1:
                raise ValueError(f"Style.{name}: должно быть ≥ 1, получено {value!r}")


def style_for(width: int, scale: float = 1.0) -> Style:
    """Стиль для кадра шириной ``width``: радиус/толщина растут вместе с кадром.

    ``scale`` — пользовательский множитель (CLI ``--scale``): ``1.0`` — значения по
    умолчанию для 1080×810, ``2.0`` — вдвое толще. Итог округляется и не может быть
    меньше 1 px, иначе на уменьшенном кадре точки исчезают.
    """
    if isinstance(width, bool) or not isinstance(width, int):
        raise ValueError(f"width: ожидалось целое число, получено {width!r}")
    if width <= 0:
        raise ValueError(f"width: должно быть > 0, получено {width!r}")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ValueError(f"scale: ожидалось число, получено {scale!r}")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale: должно быть конечным и > 0, получено {scale!r}")
    factor = width / REFERENCE_WIDTH * float(scale)
    return Style(
        point_radius=max(1, int(round(POINT_RADIUS_PX * factor))),
        line_thickness=max(1, int(round(LINE_THICKNESS_PX * factor))),
    )


def frame_range(meta: Mapping[str, Any], start: int = 0, end: int | None = None) -> tuple[int, int]:
    """Диапазон кадров для рендера: ``(start, end)`` включительно, проверенный по метаданным.

    ``end=None`` — до последнего кадра видео. Границы — номера кадров 0…frame_count−1;
    ``start > end``, выход за пределы и нецелые значения — ``ValueError`` (bool тоже).
    """
    count = meta.get("frame_count") if isinstance(meta, Mapping) else None
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"meta.frame_count: ожидалось целое > 0, получено {count!r}")
    for name, value in (("start", start), ("end", end)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{name}: ожидался номер кадра (int), получено {value!r}")
    last = count - 1 if end is None else end
    if not 0 <= start <= last <= count - 1:
        raise ValueError(
            f"кадры {start}…{last}: ожидалось 0 ≤ start ≤ end ≤ {count - 1} "
            f"(в видео {count} кадров)"
        )
    return start, last


def parse_frames(text: str) -> tuple[int, ...]:
    """Разобрать список кадров ``"30, 1800, 3600"`` в кортеж int (для ``--frames``)."""
    if not isinstance(text, str):
        raise ValueError(f"frames: ожидалась строка, получено {text!r}")
    frames = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            frames.append(int(chunk))
        except ValueError:
            raise ValueError(f"frames: {chunk!r} — не номер кадра") from None
    if not frames:
        raise ValueError("frames: пустой список")
    if len(set(frames)) != len(frames):
        raise ValueError(f"frames: повторяющиеся кадры в {text!r}")
    return tuple(frames)


@dataclass(frozen=True)
class Scene:
    """Всё, что нужно рендеру: метаданные видео, синхронизация и позы трекинга.

    Собирается один раз (:func:`load_scene`): загрузка трекинга занимает ≈3 с, а вызов
    :func:`render_video`/`render_stills` её уже не повторяет.
    """

    meta: dict
    header_ns: int
    tracking_end_ns: int
    track: sync.PoseTrack
    intrinsics: camera_model.Intrinsics
    #: Поправка времени кадра (нс): поза берётся из ``frame_time(k).ts_ns + time_offset_ns``.
    #: Значение — :func:`sync.video_time_offset_ns` (``config.VIDEO_TIME_OFFSET_MS``).
    time_offset_ns: int

    @property
    def frame_count(self) -> int:
        """Сколько кадров в видео (из ``video_meta.json``)."""
        return int(self.meta["frame_count"])

    @property
    def duration_s(self) -> float:
        """Длительность видео в секундах (из ``video_meta.json``)."""
        return float(self.meta["duration"])

    @property
    def fps(self) -> float:
        """Кадров в секунду для записи: ``frame_count / duration`` (29.9976…).

        Не номинальные 30: при записи 30 fps контейнер дал бы 7279/30 = 242.63 с вместо
        242.661 с, то есть видео «сжалось» бы на 30 мс.
        """
        return self.frame_count / self.duration_s

    @property
    def eye_size(self) -> tuple[int, int]:
        """Размер одного глаза SBS-пары (``1080, 810``) — размер выходного кадра."""
        return self.intrinsics.eye_w, self.intrinsics.eye_h


def load_scene(
    meta_path: str | Path = "output/video_meta.json",
    tracking_path: str | Path | None = None,
    *,
    data_dir: str | Path = "data",
    extra_keys: Sequence[str] = (),
) -> Scene:
    """Прочитать метаданные видео и трекинг, проверить согласованность.

    ``tracking_path=None`` — единственный ``data/trackingData_*.txt`` (иначе ``ValueError``:
    молча выбрать один из нескольких файлов значило бы отрендерить не ту запись).
    Размер кадра из ``video_meta.json`` передаётся в :meth:`Intrinsics.from_header`, поэтому
    расхождение интринсиков с видео (не тот файл, другая раскладка) падает сразу.

    ``extra_keys`` (08f) — дополнительные суставы густого скелета
    (``skeleton.extra_keys``, ``None`` не принимается): без них рисуется только 5-точечный
    скелет, а густой упадёт с понятной ошибкой до первого кадра, а не на середине рендера.
    """
    if extra_keys is None:
        raise ValueError("extra_keys: ожидался список ключей, получено None")
    meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"{meta_path}: ожидался JSON-объект video_meta")
    frame_count = meta.get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError(f"{meta_path}: frame_count должен быть целым > 0")
    duration = meta.get("duration")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        raise ValueError(f"{meta_path}: duration должен быть числом > 0")

    tracking = Path(tracking_path) if tracking_path is not None else find_tracking(data_dir)
    if not tracking.is_file():
        raise FileNotFoundError(f"нет файла трекинга: {tracking}")

    frame_size = None
    if isinstance(meta.get("width"), int) and isinstance(meta.get("height"), int):
        frame_size = (meta["width"], meta["height"])
    header = read_header(tracking)
    intrinsics = camera_model.Intrinsics.from_header(header, frame_size)
    if frame_size is not None and (intrinsics.full_w, intrinsics.full_h) != frame_size:
        raise ValueError("интринсики не описывают полный кадр видео")
    return Scene(
        meta=meta,
        header_ns=int(header["timeStampNs"]),
        tracking_end_ns=int(timestamp_range(tracking)[1]),
        track=sync.load_poses(tracking, extra_keys=extra_keys),
        intrinsics=intrinsics,
        time_offset_ns=sync.video_time_offset_ns(),
    )


def find_tracking(data_dir: str | Path = "data") -> Path:
    """Единственный файл ``trackingData_*.txt`` в ``data_dir`` (иначе понятная ошибка)."""
    candidates = sorted(Path(data_dir).glob("trackingData_*.txt"))
    if not candidates:
        raise ValueError(f"в {data_dir}/ нет trackingData_*.txt — укажите --tracking")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"в {data_dir}/ несколько файлов трекинга ({names}) — укажите --tracking")
    return candidates[0]


def project_hands(
    pose: sync.Pose,
    intrinsics: camera_model.Intrinsics,
    side: str = "left",
) -> dict[str, dict[str, camera_model.ProjectedPoint]]:
    """10 точек кистей кадра в пикселях глаза: ``side → joint → ProjectedPoint``.

    Матрица ``мир → камера`` считается один раз на кадр (поза головы + константы
    ``src/config.py``), дальше каждая точка переводится той же матрицей. Точка за камерой,
    ближе 1 см или с ``NaN`` («нет данных» у неактивной руки) даёт ``visible = False`` —
    рендер такую точку и линии к ней пропускает.
    """
    eye = camera_model.eye_intrinsics(intrinsics, side)
    matrix = camera_model.world_to_camera_matrix(pose)
    result: dict[str, dict[str, camera_model.ProjectedPoint]] = {}
    for hand in ("left", "right"):
        pixels: dict[str, camera_model.ProjectedPoint] = {}
        for joint in JOINTS:
            world = pose.hands[f"{hand}_{joint}"]
            if not all(math.isfinite(value) for value in world):
                pixels[joint] = camera_model.ProjectedPoint(math.nan, math.nan, False)
                continue
            pixels[joint] = camera_model.project(
                camera_model.transform_point(matrix, world), eye
            )
        result[hand] = pixels
    return result


def _require_skeleton_track(track: sync.PoseTrack | None, skeleton: Skeleton) -> None:
    """Проверить, что в треке есть дополнительные суставы скелета (08f).

    Ошибка до первого кадра: иначе :meth:`sync.PoseTrack.joint_at` упал бы на середине
    рендера 7279 кадров, уже записав часть файла.
    """
    if not isinstance(skeleton, Skeleton):
        raise ValueError(f"skeleton: ожидался Skeleton, получено {skeleton!r}")
    extra = skeleton.extra_keys
    if not extra:
        return
    if track is None:
        raise ValueError(
            f"скелет {skeleton.name!r}: нужны дополнительные суставы {extra[:3]}… — "
            "передайте track с extra_keys"
        )
    missing = [key for key in extra if key not in track.extra_keys]
    if missing:
        raise ValueError(
            f"скелет {skeleton.name!r}: в треке нет {len(missing)} ключей (например "
            f"{missing[0]!r}); загрузите сцену с load_scene(..., "
            f"extra_keys=SKELETONS[{skeleton.name!r}].extra_keys)"
        )


def _skeleton_world_points(
    pose: sync.Pose, skeleton: Skeleton, track: sync.PoseTrack | None
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Мировые координаты суставов скелета кадра: ``side → joint → (x, y, z)``.

    Пять опорных суставов лежат в :class:`sync.Pose` (интерполяция 04b/04c), остальные
    берутся из ``track.joint_at(pose.ts_ns, key)`` — та же интерполяция и тот же момент
    времени. Вынесено из :func:`project_skeleton`, чтобы глубина (08g) считалась по тем же
    самым точкам, что и пиксели, — иначе два прохода могли бы разойтись.
    """
    _require_skeleton_track(track, skeleton)
    result: dict[str, dict[str, tuple[float, float, float]]] = {}
    for hand in ("left", "right"):
        joints: dict[str, tuple[float, float, float]] = {}
        for joint in skeleton.joints:
            key = f"{hand}_{joint}"
            if key in pose.hands:
                world = pose.hands[key]
            else:
                world = track.joint_at(pose.ts_ns, key)  # type: ignore[union-attr]
            joints[joint] = (world[0], world[1], world[2])
        result[hand] = joints
    return result


def project_skeleton_depth(
    pose: sync.Pose,
    intrinsics: camera_model.Intrinsics,
    skeleton: Skeleton = DENSE,
    *,
    track: sync.PoseTrack | None = None,
    side: str = "left",
) -> tuple[
    dict[str, dict[str, camera_model.ProjectedPoint]],
    dict[str, dict[str, float]],
]:
    """Пиксели скелета кадра **и** глубина каждого сустава в системе камеры (08g).

    Возвращает ``(pixels, depths)``: ``pixels`` — ровно то же, что у
    :func:`project_skeleton` (``side → joint → ProjectedPoint``), ``depths`` —
    ``side → joint → z`` точки камеры в метрах: ``z`` вперёд по взгляду, из той же матрицы
    ``мир → камера``, что даёт пиксели. Глубина не кладётся в :class:`ProjectedPoint`
    намеренно: у него контракт «пиксель + видимость» (тест 05c), а рендеру 08a глубина не
    нужна. ``NaN`` вместо глубины — «нет данных» у сустава (как ``visible = False`` у
    пикселя): у неактивной руки и невалидных джойнтов ``NaN`` в данных, у точки за камерой
    ``z`` бывает конечным, но она невидима и не рисуется.

    Калибровка и :func:`camera_model.project` не меняются: глубина — просто ``z`` уже
    посчитанной точки камеры, а не второй источник проекции.
    """
    world = _skeleton_world_points(pose, skeleton, track)
    eye = camera_model.eye_intrinsics(intrinsics, side)
    matrix = camera_model.world_to_camera_matrix(pose)
    pixels: dict[str, dict[str, camera_model.ProjectedPoint]] = {}
    depths: dict[str, dict[str, float]] = {}
    for hand, joints in world.items():
        hand_pixels: dict[str, camera_model.ProjectedPoint] = {}
        hand_depths: dict[str, float] = {}
        for joint, point in joints.items():
            if not all(math.isfinite(value) for value in point):
                hand_pixels[joint] = camera_model.ProjectedPoint(math.nan, math.nan, False)
                hand_depths[joint] = math.nan
                continue
            p_cam = camera_model.transform_point(matrix, point)
            hand_pixels[joint] = camera_model.project(p_cam, eye)
            hand_depths[joint] = p_cam[2]
        pixels[hand] = hand_pixels
        depths[hand] = hand_depths
    return pixels, depths


def project_skeleton(
    pose: sync.Pose,
    intrinsics: camera_model.Intrinsics,
    skeleton: Skeleton = DENSE,
    *,
    track: sync.PoseTrack | None = None,
    side: str = "left",
) -> dict[str, dict[str, camera_model.ProjectedPoint]]:
    """Пиксели выбранного скелета на кадре: ``side → joint → ProjectedPoint`` (08f).

    Пять опорных суставов берутся из позы (:class:`sync.Pose`, интерполяция 04b/04c),
    остальные — ``track.joint_at(pose.ts_ns, key)``: та же интерполяция, тот же момент
    времени (этим же путём пользуется метрика 08e). Матрица ``мир → камера`` считается
    один раз на кадр. ``NaN`` («нет данных») и точка за камерой дают ``visible = False`` —
    :func:`draw_skeleton` не рисует такую точку и не проводит кости к ней.

    Глубины не возвращает (это :func:`project_skeleton_depth`): путь 08f и его числа
    остаются ровно прежними.
    """
    pixels, _depths = project_skeleton_depth(
        pose, intrinsics, skeleton, track=track, side=side
    )
    return pixels


def _coord(value: float) -> int:
    """Координата для cv2: целое, ограниченное :data:`COORD_LIMIT` (защита от int32)."""
    if not math.isfinite(value):
        raise ValueError(f"координата: нефинитное значение {value!r}")
    return max(-COORD_LIMIT, min(COORD_LIMIT, int(round(value))))


def _check_depth_window(near: float, far: float, minimum: float, floor_name: str) -> None:
    """Проверить окно глубины и нижнюю границу шкалы (общая для множителя и alpha).

    ``minimum`` — значение шкалы на дальнем конце окна (множитель яркости или alpha),
    ``floor_name`` — его имя для текста ошибки. Окно ``near``…``far`` — метры системы камеры.
    """
    for name, value in (("near", near), ("far", far), (floor_name, minimum)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}: ожидалось число, получено {value!r}")
        if not math.isfinite(value):
            raise ValueError(f"{name}: должно быть конечным, получено {value!r}")
    if near <= 0:
        raise ValueError(f"near: должно быть > 0, получено {near!r}")
    if far <= near:
        raise ValueError(f"far: должно быть больше near ({near!r}), получено {far!r}")
    if not 0.0 < minimum <= 1.0:
        raise ValueError(f"{floor_name}: ожидалось в (0, 1], получено {minimum!r}")


def depth_factor(
    depth: float,
    *,
    near: float = DEPTH_NEAR_M,
    far: float = DEPTH_FAR_M,
    min_factor: float = DEPTH_MIN_FACTOR,
) -> float:
    """Множитель к цвету руки по глубине: 1.0 ближе ``near``, ``min_factor`` дальше ``far``.

    Линейно по глубине (без гаммы и ступеней) и монотонно не возрастает с ``z``: ближе —
    ярче и насыщеннее, дальше — приглушённее. Окно **абсолютное**, в метрах системы камеры
    (:data:`DEPTH_NEAR_M`…:data:`DEPTH_FAR_M` замерены по всей записи), поэтому множитель не
    зависит от того, что попало в кадр, и не «дышит» от кадра к кадру.

    Неизвестная глубина (``NaN``, ``±inf``) даёт 1.0 — полный цвет: невидимая точка всё
    равно не рисуется, а выцветать из-за отсутствия данных скелет не должен. Глубина за
    камерой (``z ≤ 0``) тоже упирается в ближнюю границу окна и даёт 1.0 — такая точка в
    кадр не попадает, множитель для неё смысла не имеет.
    """
    _check_depth_window(near, far, min_factor, "min_factor")
    if isinstance(depth, bool) or not isinstance(depth, (int, float)):
        raise ValueError(f"depth: ожидалось число, получено {depth!r}")
    if not math.isfinite(depth):
        return 1.0
    t = (float(depth) - near) / (far - near)
    if t <= 0.0:
        return 1.0
    if t >= 1.0:
        return min_factor
    return 1.0 - (1.0 - min_factor) * t


def depth_alpha(
    depth: float,
    *,
    near: float = DEPTH_NEAR_M,
    far: float = DEPTH_FAR_M,
    min_alpha: float = DEPTH_MIN_ALPHA,
) -> float:
    """Прозрачность по глубине: 1.0 ближе ``near``, ``min_alpha`` дальше ``far`` (08g-fix2).

    Второй, независимый от яркости канал: ближняя кость или точка рисуется полностью
    непрозрачной, дальняя — полупрозрачной, сквозь неё виден фон. Шкала линейная и
    монотонно не возрастает с ``z``, окно по умолчанию — **то же**, что у
    :func:`depth_factor` (:data:`DEPTH_NEAR_M`…:data:`DEPTH_FAR_M`): одна шкала на оба канала
    проще для интерпретации, а ``near``/``far``/``min_alpha`` можно перекрыть (например, для
    сравнения вариантов), не меняя яркость.

    Неизвестная глубина (``NaN``, ``±inf``) и глубина за камерой (``z ≤ 0``) дают 1.0 —
    полную непрозрачность: отсутствие данных не повод «выцветать» или прятать точку, а такая
    точка в кадр не попадает. Крайние значения окна возвращают ровно 1.0 / ``min_alpha``
    (без ``0.44999999999999996``), дальше — зажим, а не экстраполяция.
    """
    _check_depth_window(near, far, min_alpha, "min_alpha")
    if isinstance(depth, bool) or not isinstance(depth, (int, float)):
        raise ValueError(f"depth: ожидалось число, получено {depth!r}")
    if not math.isfinite(depth):
        return 1.0
    if depth <= 0.0:
        return 1.0
    t = (float(depth) - near) / (far - near)
    if t <= 0.0:
        return 1.0
    if t >= 1.0:
        return min_alpha
    return 1.0 - (1.0 - min_alpha) * t



def shade_color(color: Sequence[int], factor: float) -> tuple[int, int, int]:
    """Умножить цвет BGR на ``factor`` (0…1) с округлением и зажимом в 0…255.

    Умножение сохраняет оттенок руки (все каналы масштабируются одинаково) и меняет только
    яркость/насыщенность — вторая палитра не появляется, левая/правая рука по-прежнему
    различимы. Множитель обычно приходит из :func:`depth_factor`, где снизу ограничен
    :data:`DEPTH_MIN_FACTOR`, поэтому дальняя кость не превращается в чёрную.
    """
    channels = tuple(color)
    if len(channels) != 3:
        raise ValueError(f"color: ожидался BGR из 3 каналов, получено {color!r}")
    for value in channels:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
            raise ValueError(f"color: канал должен быть int 0…255, получено {value!r}")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise ValueError(f"factor: ожидалось число, получено {factor!r}")
    if not math.isfinite(factor) or not 0.0 <= factor <= 1.0:
        raise ValueError(f"factor: ожидалось в 0…1, получено {factor!r}")
    shaded = (min(255, max(0, int(round(value * factor)))) for value in channels)
    return tuple(shaded)  # type: ignore[return-value]


def _check_segments(segments: int) -> None:
    """Проверить число отрезков кости (общая проверка цветов и alpha по глубине)."""
    if isinstance(segments, bool) or not isinstance(segments, int):
        raise ValueError(f"segments: ожидалось целое число, получено {segments!r}")
    if segments < 1:
        raise ValueError(f"segments: должно быть ≥ 1, получено {segments!r}")


def bone_gradient_colors(
    color: Sequence[int],
    depth_start: float,
    depth_end: float,
    *,
    segments: int = DEPTH_BONE_SEGMENTS,
) -> tuple[tuple[int, int, int], ...]:
    """Цвета ``segments`` отрезков кости: множитель плавно идёт от одного конца к другому.

    Порядок цветов — от сустава ``depth_start`` к ``depth_end`` (это начало и конец ребра).
    Внутри отрезка цвет постоянен и берётся по множителю в его середине, ``t = (i + 0.5)/N``,
    — так градиент вдоль кости не зависит от длины в пикселях и от толщины линии.
    Множители концов — :func:`depth_factor` от ``z``: ближний конец ярче, дальний темнее,
    поэтому на экране видно, каким концом кость смотрит в камеру.

    Глубина хотя бы одного конца неизвестна (``NaN``/``±inf``) — все отрезки получают
    полный цвет руки: кость выглядит как прежняя плоская линия. Неположительное число
    отрезков — ``ValueError``.
    """
    _check_segments(segments)
    if not (math.isfinite(depth_start) and math.isfinite(depth_end)):
        base = shade_color(color, 1.0)
        return (base,) * segments
    start_factor = depth_factor(depth_start)
    end_factor = depth_factor(depth_end)
    step = end_factor - start_factor
    return tuple(
        shade_color(color, start_factor + step * ((index + 0.5) / segments))
        for index in range(segments)
    )


def bone_gradient_alphas(
    depth_start: float,
    depth_end: float,
    *,
    segments: int = DEPTH_BONE_SEGMENTS,
) -> tuple[float, ...]:
    """Alpha ``segments`` отрезков кости — по средней глубине концов отрезка (08g-fix2).

    Ровно как у :func:`bone_gradient_colors`, но для прозрачности: параметр вдоль кости
    ``t = (i + 0.5)/N`` переводится в глубину (линейно между глубинами концов), а alpha
    берётся :func:`depth_alpha` в этой точке. Средняя глубина концов отрезка — это и есть
    глубина его середины, поэтому вдоль кости виден не только градиент яркости, но и
    градиент плотности: ближний конец плотный, дальний просвечивает.

    Глубина хотя бы одного конца неизвестна (``NaN``/``±inf``) — все отрезки непрозрачны
    (1.0): отсутствие данных не делает кость полупрозрачной. Неположительное число отрезков —
    ``ValueError``.
    """
    _check_segments(segments)
    if not (math.isfinite(depth_start) and math.isfinite(depth_end)):
        return (1.0,) * segments
    step = (float(depth_end) - float(depth_start)) / segments
    return tuple(
        depth_alpha(depth_start + step * (index + 0.5)) for index in range(segments)
    )


def _lerp_point(
    first: camera_model.ProjectedPoint, second: camera_model.ProjectedPoint, t: float
) -> tuple[int, int]:
    """Точка на отрезке ``first → second`` в параметре ``t`` (0 — first, 1 — second)."""
    return (
        _coord(first.u + (second.u - first.u) * t),
        _coord(first.v + (second.v - first.v) * t),
    )


def _draw_segment(
    image: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    alpha: float = 1.0,
) -> None:
    """Отрезок между пикселями: alpha 1.0 — обычный ``cv2.line``, иначе смешивание по маске.

    ``cv2.line`` не умеет alpha, поэтому прозрачный отрезок рисуется в маску размером со свой
    ограничивающий прямоугольник (:func:`_blend_line`), иначе на кадр лёг бы полупрозрачный
    тёмный прямоугольник вокруг линии (та же ловушка, что у следа 08b).
    """
    import cv2  # noqa: PLC0415

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha: ожидалось число, получено {alpha!r}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha: должно быть в (0, 1], получено {alpha!r}")
    if alpha >= 1.0:
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)
        return
    _blend_line(image, start, end, color, thickness, alpha)


def _draw_bone(
    image: Any,
    first: camera_model.ProjectedPoint,
    second: camera_model.ProjectedPoint,
    color: tuple[int, int, int],
    style: Style,
    alpha: float = 1.0,
) -> None:
    """Одна линия ``first → second`` цветом ``color`` (плоский режим и кость в одном множителе)."""
    _draw_segment(
        image,
        (_coord(first.u), _coord(first.v)),
        (_coord(second.u), _coord(second.v)),
        color,
        style.line_thickness,
        alpha,
    )


def _blend_bone_gradient(
    image: Any,
    first: camera_model.ProjectedPoint,
    second: camera_model.ProjectedPoint,
    segment_colors: Sequence[tuple[int, int, int]],
    segment_alphas: Sequence[float],
    thickness: int,
) -> bool:
    """Кость с прозрачностью: цвет и alpha на отрезок, но смешивание **один раз** (08g-fix2).

    Если рисовать каждый из :data:`DEPTH_BONE_SEGMENTS` отрезков отдельным
    :func:`_blend_line`, то на стыках два антиалиасных конца ложатся друг на друга и смешивание
    идёт дважды: на кадре появляются «бусины» — более плотные точки через каждые 20 % кости
    (видно на 6× зуме). Поэтому кость кладётся в **одну** маску (cv2.line с AA), а затем
    каждый пиксель маски относится к своему отрезку по проекции на ось кости
    ``t = (P−A)·(B−A)/|B−A|²``; цвет и alpha берутся по номеру отрезка, и смешивание с кадром
    происходит ровно один раз. Швов между отрезками нет, а ступенька цвета на границе
    отрезков остаётся (это и есть градиент из 5 отрезков).

    Возврат — было ли что рисовать (кость целиком вне кадра — ``False``).
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if len(segment_colors) != len(segment_alphas):
        raise ValueError("segment_colors и segment_alphas: разная длина")
    start = (_coord(first.u), _coord(first.v))
    end = (_coord(second.u), _coord(second.v))
    height, width = image.shape[:2]
    pad = thickness + 2
    left = max(0, min(start[0], end[0]) - pad)
    top = max(0, min(start[1], end[1]) - pad)
    right = min(width, max(start[0], end[0]) + pad)
    bottom = min(height, max(start[1], end[1]) + pad)
    if right <= left or bottom <= top:
        return False
    mask = np.zeros((bottom - top, right - left), np.uint8)
    cv2.line(
        mask,
        (start[0] - left, start[1] - top),
        (end[0] - left, end[1] - top),
        255,
        thickness,
        cv2.LINE_AA,
    )
    if not mask.any():
        return False
    ax, ay = start[0] - left, start[1] - top
    bx, by = end[0] - left, end[1] - top
    dx, dy = bx - ax, by - ay
    rows, cols = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    length2 = dx * dx + dy * dy
    if length2:
        t = ((cols - ax) * dx + (rows - ay) * dy) / float(length2)
    else:
        t = np.zeros_like(cols, dtype=np.float32)
    np.clip(t, 0.0, 1.0, out=t)
    count = len(segment_colors)
    index = np.minimum((t * count).astype(np.int32), count - 1)
    colors = np.asarray(segment_colors, np.float32)[index]
    alphas = np.asarray(segment_alphas, np.float32)[index]
    weight = ((mask.astype(np.float32) / 255.0) * alphas)[..., None]
    roi = image[top:bottom, left:right]
    blended = roi.astype(np.float32) * (1.0 - weight) + colors * weight
    roi[:] = np.rint(blended).astype(np.uint8)
    return True


def _depth_sort_key(depth: float) -> float:
    """Ключ сортировки «дальние раньше»: неизвестная глубина — раньше всех (она непрозрачная)."""
    return depth if math.isfinite(depth) else math.inf


def _bone_mean_depth(depths: Mapping[str, float], bone: tuple[str, str]) -> float:
    """Средняя глубина концов ребра (глубина его середины); без данных — ``inf``."""
    start, end = bone
    first, second = depths[start], depths[end]
    if not (math.isfinite(first) and math.isfinite(second)):
        return math.inf
    return (first + second) / 2.0


def _paint_bone(
    image: Any,
    first: camera_model.ProjectedPoint,
    second: camera_model.ProjectedPoint,
    color: tuple[int, int, int],
    style: Style,
    depth_start: float | None = None,
    depth_end: float | None = None,
    *,
    shade: bool = True,
    alpha: bool = False,
) -> None:
    """Нарисовать ребро: плоское, с градиентом яркости и/или прозрачности по глубине.

    ``depth_start``/``depth_end`` — ``None`` в плоском режиме (одна линия полного цвета).
    Иначе цвет и alpha идут вдоль кости :data:`DEPTH_BONE_SEGMENTS` отрезками: цвет — по
    множителю в середине отрезка (:func:`bone_gradient_colors`), alpha — по средней глубине
    его концов (:func:`bone_gradient_alphas`). Если вся кость в одном множителе и одной alpha
    (дальний конец упёрся в пол, глубина неизвестна, у `--no-depth-alpha`), она рисуется одной
    линией — ровно как плоская, без швов антиалиасинга между стыкующимися отрезками.
    """
    if depth_start is None or depth_end is None:
        _draw_bone(image, first, second, color, style)
        return
    segment_colors = (
        bone_gradient_colors(color, depth_start, depth_end)
        if shade
        else (shade_color(color, 1.0),) * DEPTH_BONE_SEGMENTS
    )
    segment_alphas = bone_gradient_alphas(depth_start, depth_end) if alpha else None
    alphas_uniform = segment_alphas is None or len(set(segment_alphas)) == 1
    if len(set(segment_colors)) == 1 and alphas_uniform:
        _draw_bone(
            image,
            first,
            second,
            segment_colors[0],
            style,
            1.0 if segment_alphas is None else segment_alphas[0],
        )
        return
    if segment_alphas is not None and any(value < 1.0 for value in segment_alphas):
        if len(set(segment_colors)) == 1 and len(set(segment_alphas)) == 1:
            _draw_bone(image, first, second, segment_colors[0], style, segment_alphas[0])
        else:
            _blend_bone_gradient(
                image, first, second, segment_colors, segment_alphas, style.line_thickness
            )
        return
    segments = len(segment_colors)
    for index, segment_color in enumerate(segment_colors):
        _draw_segment(
            image,
            _lerp_point(first, second, index / segments),
            _lerp_point(first, second, (index + 1) / segments),
            segment_color,
            style.line_thickness,
            1.0 if segment_alphas is None else segment_alphas[index],
        )


def _paint_point(
    image: Any,
    point: camera_model.ProjectedPoint,
    color: tuple[int, int, int],
    style: Style,
    depth: float | None = None,
    *,
    shade: bool = True,
    alpha: bool = False,
) -> None:
    """Нарисовать точку сустава: цвет и прозрачность по её глубине, обводка всегда чёрная.

    Обводка не прозрачная намеренно: иначе дальняя точка «исчезает» на тёмном фоне, а на
    светлом остаётся цветное пятно без границы. ``depth = None`` — плоский кадр 08a/08f.
    """
    import cv2  # noqa: PLC0415

    center = (_coord(point.u), _coord(point.v))
    fill_color = color
    fill_alpha = 1.0
    if depth is not None and math.isfinite(depth):
        if shade:
            fill_color = shade_color(color, depth_factor(depth))
        if alpha:
            fill_alpha = depth_alpha(depth)
    if fill_alpha >= 1.0:
        cv2.circle(image, center, style.point_radius, fill_color, -1, cv2.LINE_AA)
    else:
        _blend_point(image, center, style.point_radius, fill_color, fill_alpha)
    cv2.circle(image, center, style.point_radius, OUTLINE_COLOR, OUTLINE_THICKNESS, cv2.LINE_AA)


def draw_skeleton(
    image: Any,
    pixels: Mapping[str, Mapping[str, camera_model.ProjectedPoint]],
    style: Style,
    *,
    colors: Mapping[str, tuple[int, int, int]] = COLORS,
    joints: Sequence[str] = JOINTS,
    bones: Sequence[tuple[str, str]] = HAND_CHAINS,
    depths: Mapping[str, Mapping[str, float]] | None = None,
    alpha: bool = False,
    shade: bool = True,
) -> int:
    """Нарисовать скелет на кадре (in place) и вернуть число видимых точек.

    ``joints``/``bones`` (08f) — набор суставов и рёбер одной руки; по умолчанию прежний
    5-точечный скелет 08a. Порядок: сначала линии (под точками), потом точки: точка сустава
    всегда остаётся видимой поверх своей кости. Пара, в которой хотя бы одна точка невидима,
    линией не соединяется — иначе отрезок «прибил» бы руку к точке за камерой, а ``NaN``
    («нет данных», 04b) дал бы линию через дырку. Точки, вышедшие за край кадра, всё равно
    рисуются: cv2 обрежет круг/отрезок по границе, и видимая часть кисти останется на месте.

    ``depths`` (08g) — глубина суставов в системе камеры, ``side → joint → z`` (как у
    :func:`project_skeleton_depth`). Если передана, цвет руки умножается на
    :func:`depth_factor`: у точки — по её глубине, у кости — **градиентом вдоль кости**:
    ребро рисуется :data:`DEPTH_BONE_SEGMENTS` отрезками, множитель линейно идёт от
    множителя начала к множителю конца (:func:`bone_gradient_colors`), поэтому видно,
    каким концом кость ближе к камере. ``NaN`` у конца кости или у точки даёт полный цвет
    (глубина неизвестна). Обводка точки остаётся чёрной при любом множителе — контраст на
    светлом фоне сохраняется. ``None`` (по умолчанию) — плоские цвета 08a: одна линия на
    кость, та же картинка, что до 08g.

    ``alpha`` (08g-fix2) — второй канал глубины: у точки прозрачность по её глубине
    (:func:`depth_alpha`), у кости — по средней глубине концов каждого отрезка
    (:func:`bone_gradient_alphas`). Чтобы полупрозрачная дальняя кость не просвечивала поверх
    непрозрачной ближней, в этом режиме кости и точки рисуются **от дальних к ближним**
    (сортировка по ``z`` по убыванию, в пределах кадра обе руки): неизвестная глубина идёт
    первой, точки — после всех костей. Без ``alpha`` порядок остаётся прежним (список
    ``bones``, затем точки, левая рука, правая), поэтому кадр 08g-fix воспроизводится
    побайтово.

    ``shade=False`` — отключить множитель яркости, оставив прозрачность: это режим «только
    alpha» для сравнения каналов (CLI такого флага не имеет, см. ``docs/overlay.md``).
    Плоский кадр (``depths=None``) он не трогает.
    """
    for name, value in (("alpha", alpha), ("shade", shade)):
        if not isinstance(value, bool):
            raise ValueError(f"{name}: ожидалось bool, получено {value!r}")
    if alpha and depths is None:
        raise ValueError("alpha: прозрачность по глубине невозможна без depths")
    if not isinstance(style, Style):
        raise ValueError(f"style: ожидался Style, получено {style!r}")
    if image is None or getattr(image, "ndim", 0) != 3:
        raise ValueError("image: ожидался BGR-кадр (H×W×3)")
    names = tuple(joints)
    if not names:
        raise ValueError("joints: пустой список суставов")
    pairs = tuple(bones)
    for start, end in pairs:
        for joint in (start, end):
            if joint not in names:
                raise ValueError(
                    f"bones: ребро {start!r}→{end!r} ссылается на сустав {joint!r}, "
                    "которого нет в joints"
                )
    if depths is not None and not isinstance(depths, Mapping):
        raise ValueError(f"depths: ожидался Mapping side → joint → z, получено {depths!r}")

    hands: dict[str, Mapping[str, camera_model.ProjectedPoint]] = {}
    hand_depths: dict[str, Mapping[str, float]] = {}
    for side in ("left", "right"):
        hand = pixels.get(side)
        if hand is None:
            raise ValueError(f"pixels: нет руки {side!r}")
        missing = [joint for joint in names if joint not in hand]
        if missing:
            raise ValueError(f"pixels[{side!r}]: нет суставов {missing}")
        hands[side] = hand
        if depths is not None:
            side_depths = depths.get(side)
            if side_depths is None:
                raise ValueError(f"depths: нет руки {side!r}")
            missing = [joint for joint in names if joint not in side_depths]
            if missing:
                raise ValueError(f"depths[{side!r}]: нет суставов {missing}")
            hand_depths[side] = side_depths

    if depths is None:
        # плоский кадр 08a/08f: кости в порядке списка, затем точки; руки — левая, правая
        drawn = 0
        for side in ("left", "right"):
            for start, end in pairs:
                first, second = hands[side][start], hands[side][end]
                if first.visible and second.visible:
                    _paint_bone(image, first, second, colors[side], style)
            for joint in names:
                point = hands[side][joint]
                if point.visible:
                    _paint_point(image, point, colors[side], style)
                    drawn += 1
        return drawn

    if not alpha:
        # 08g-fix: только яркость, порядок костей — как в списке ``bones``
        drawn = 0
        for side in ("left", "right"):
            for start, end in pairs:
                first, second = hands[side][start], hands[side][end]
                if first.visible and second.visible:
                    _paint_bone(
                        image,
                        first,
                        second,
                        colors[side],
                        style,
                        hand_depths[side][start],
                        hand_depths[side][end],
                        shade=shade,
                    )
            for joint in names:
                point = hands[side][joint]
                if point.visible:
                    _paint_point(
                        image,
                        point,
                        colors[side],
                        style,
                        hand_depths[side][joint],
                        shade=shade,
                    )
                    drawn += 1
        return drawn

    # 08g-fix2: alpha как второй канал — сначала все кости от дальних к ближним, затем точки
    # от дальних к ближним (точка сустава должна остаться видимой поверх своей кости)
    bones_plan: list[
        tuple[float, str, camera_model.ProjectedPoint, camera_model.ProjectedPoint, float, float]
    ] = []
    for side in ("left", "right"):
        for start, end in pairs:
            first, second = hands[side][start], hands[side][end]
            if not (first.visible and second.visible):
                continue
            depth_start = hand_depths[side][start]
            depth_end = hand_depths[side][end]
            bones_plan.append(
                (
                    _bone_mean_depth(hand_depths[side], (start, end)),
                    side,
                    first,
                    second,
                    depth_start,
                    depth_end,
                )
            )
    bones_plan.sort(key=lambda item: item[0], reverse=True)
    for _, side, first, second, depth_start, depth_end in bones_plan:
        _paint_bone(
            image,
            first,
            second,
            colors[side],
            style,
            depth_start,
            depth_end,
            shade=shade,
            alpha=True,
        )
    points_plan: list[tuple[float, str, str]] = []
    for side in ("left", "right"):
        for joint in names:
            if hands[side][joint].visible:
                points_plan.append((_depth_sort_key(hand_depths[side][joint]), side, joint))
    points_plan.sort(key=lambda item: item[0], reverse=True)
    for _, side, joint in points_plan:
        _paint_point(
            image,
            hands[side][joint],
            colors[side],
            style,
            hand_depths[side][joint],
            shade=shade,
            alpha=True,
        )
    return len(points_plan)


def _trail_window(seconds: float, step_s: float) -> int:
    """Число выборок следа: ``round(seconds/step_s) + 1`` (старейшая … текущий момент).

    Проверки здесь, а не у вызывающих: одно место для ``seconds``/``step_s`` — иначе при
    неположительном шаге след молча рисовал бы одну точку.
    """
    for name, value in (("seconds", seconds), ("step_s", step_s)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"trail {name}: ожидалось число, получено {value!r}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"trail {name}: должно быть конечным и > 0, получено {value!r}")
    if step_s > seconds:
        raise ValueError(
            f"trail: шаг {step_s:g} с больше окна {seconds:g} с — линию не построить"
        )
    return int(round(seconds / step_s)) + 1


def trail_times(
    ts_ns: int, *, seconds: float = TRAIL_SECONDS, step_s: float = TRAIL_STEP_S
) -> tuple[int, ...]:
    """Моменты выборки следа: по возрастанию, от ``ts_ns − окно`` до ``ts_ns``.

    Шаг и окно — в секундах и переводятся в наносекунды здесь, поэтому след не зависит от
    того, с какой частотой рендерятся кадры (в отличие от «следа по номеру кадра»).
    """
    if isinstance(ts_ns, bool) or not isinstance(ts_ns, int):
        raise ValueError(f"ts_ns: ожидалось целое число, получено {ts_ns!r}")
    count = _trail_window(seconds, step_s)
    step_ns = int(round(step_s * 1e9))
    return tuple(ts_ns - index * step_ns for index in range(count - 1, -1, -1))


def trail_points(
    track: sync.PoseTrack,
    ts_ns: int,
    *,
    seconds: float = TRAIL_SECONDS,
    step_s: float = TRAIL_STEP_S,
    joint: str = "palm",
) -> dict[str, list[tuple[float, float, float]]]:
    """Мировые позиции сустава каждой руки на моментах следа (старые → новые).

    Выборки вне диапазона трекинга **не добавляются**: до первого сэмпла следа ещё нет
    (иначе «замороженная» поза растянула бы точку в линию), а после последнего добавить
    нечего. ``NaN`` («нет данных» руки) остаётся в списке как есть — рвать линию на нём
    должен рендер (:func:`draw_trail`), а не источник данных.
    """
    if not hasattr(track, "pose_at") or not hasattr(track, "ts"):
        raise ValueError(f"track: ожидался sync.PoseTrack, получено {type(track).__name__}")
    if joint not in JOINTS:
        raise ValueError(f"joint: ожидался один из {JOINTS}, получено {joint!r}")
    times = trail_times(ts_ns, seconds=seconds, step_s=step_s)
    if len(track.ts) == 0:
        raise ValueError("track: пустой трек — след не построить")
    first, last = int(track.ts[0]), int(track.ts[-1])
    points: dict[str, list[tuple[float, float, float]]] = {"left": [], "right": []}
    for moment_ns in times:
        if not first <= moment_ns <= last:
            continue
        pose = track.pose_at(moment_ns)
        for side in points:
            points[side].append(pose.hands[f"{side}_{joint}"])
    return points


def _blend_line(
    image: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    alpha: float,
) -> bool:
    """Отрезок с прозрачностью ``alpha``, только по своей маске (возврат — нарисован ли).

    OpenCV не рисует с alpha, поэтому отрезок кладётся в маску размером с его ограничивающий
    прямоугольник, а пиксели кадра смешиваются с цветом лишь там, где маска ненулевая:
    иначе на кадр лёг бы полупрозрачный **тёмный прямоугольник** вокруг линии.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if isinstance(start, bool) or not isinstance(start, tuple) or len(start) != 2:
        raise ValueError(f"start: ожидалась пара (x, y), получено {start!r}")
    if isinstance(end, bool) or not isinstance(end, tuple) or len(end) != 2:
        raise ValueError(f"end: ожидалась пара (x, y), получено {end!r}")
    if isinstance(thickness, bool) or not isinstance(thickness, int) or thickness < 1:
        raise ValueError(f"thickness: ожидалось целое ≥ 1, получено {thickness!r}")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha: ожидалось число, получено {alpha!r}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha: должно быть в (0, 1], получено {alpha!r}")

    height, width = image.shape[:2]
    pad = thickness + 2
    left = max(0, min(start[0], end[0]) - pad)
    top = max(0, min(start[1], end[1]) - pad)
    right = min(width, max(start[0], end[0]) + pad)
    bottom = min(height, max(start[1], end[1]) + pad)
    if right <= left or bottom <= top:
        return False  # отрезок целиком вне кадра (cv2 обрезал бы его в пустоту)

    mask = np.zeros((bottom - top, right - left), np.uint8)
    cv2.line(
        mask,
        (start[0] - left, start[1] - top),
        (end[0] - left, end[1] - top),
        255,
        thickness,
        cv2.LINE_AA,
    )
    return _blend_mask(image, mask, left, top, color, alpha)


def _blend_mask(
    image: Any,
    mask: Any,
    left: int,
    top: int,
    color: tuple[int, int, int],
    alpha: float,
) -> bool:
    """Смешать пиксели кадра с ``color`` по маске в позиции ``(left, top)`` (08b, 08g-fix2).

    Общая часть :func:`_blend_line` и :func:`_blend_point`: OpenCV не умеет alpha в
    примитивах, поэтому сначала примитив рисуется в одноканальную маску, а затем кадр
    смешивается с цветом только там, где маска ненулевая. Иначе вокруг линии или круга лёг
    бы полупрозрачный тёмный прямоугольник (ловушка из NOTES 08b). Возврат — было ли что
    рисовать (маска не пуста).
    """
    import numpy as np  # noqa: PLC0415

    if mask is None or getattr(mask, "ndim", 0) != 2:
        raise ValueError("mask: ожидалась одноканальная маска")
    if not mask.any():
        return False
    weight = (mask.astype(np.float32) * (alpha / 255.0))[..., None]
    height, width = mask.shape[:2]
    roi = image[top : top + height, left : left + width]
    blended = roi.astype(np.float32) * (1.0 - weight) + np.asarray(color, np.float32) * weight
    roi[:] = np.rint(blended).astype(np.uint8)
    return True


def _blend_point(
    image: Any,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    alpha: float,
) -> bool:
    """Круг с прозрачностью ``alpha``, только по своей маске (возврат — нарисован ли).

    Аналог :func:`_blend_line` для точки сустава (08g-fix2): ``cv2.circle`` не умеет alpha,
    поэтому круг кладётся в маску размером со свой bbox, и кадр смешивается с цветом только
    там, где маска ненулевая — иначе вокруг точки появилось бы полупрозрачное тёмное пятно
    на пёстром фоне. Антиалиасинг края сохраняется (``cv2.LINE_AA`` в маске).
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if isinstance(center, bool) or not isinstance(center, tuple) or len(center) != 2:
        raise ValueError(f"center: ожидалась пара (x, y), получено {center!r}")
    for value in center:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"center: ожидались целые координаты, получено {center!r}")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError(f"radius: ожидалось целое ≥ 1, получено {radius!r}")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError(f"alpha: ожидалось число, получено {alpha!r}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha: должно быть в (0, 1], получено {alpha!r}")

    height, width = image.shape[:2]
    pad = radius + 2
    left = max(0, center[0] - pad)
    top = max(0, center[1] - pad)
    right = min(width, center[0] + pad)
    bottom = min(height, center[1] + pad)
    if right <= left or bottom <= top:
        return False  # точка целиком вне кадра

    mask = np.zeros((bottom - top, right - left), np.uint8)
    cv2.circle(
        mask,
        (center[0] - left, center[1] - top),
        radius,
        255,
        -1,
        cv2.LINE_AA,
    )
    return _blend_mask(image, mask, left, top, color, alpha)


def draw_trail(
    image: Any,
    pose: sync.Pose,
    track: sync.PoseTrack,
    intrinsics: camera_model.Intrinsics,
    style: Style,
    *,
    seconds: float = TRAIL_SECONDS,
    step_s: float = TRAIL_STEP_S,
    joint: str = "palm",
    colors: Mapping[str, tuple[int, int, int]] = COLORS,
    alpha_new: float = TRAIL_ALPHA_NEW,
    alpha_old: float = TRAIL_ALPHA_OLD,
) -> int:
    """Нарисовать след обеих кистей (in place) и вернуть число нарисованных отрезков.

    Мировые точки прошлых моментов проецируются матрицей **текущего** кадра, иначе след
    «отставал» бы от собственной камеры. Сегмент рисуется, только если оба его конца видимы:
    разрыв на ``NaN``/невидимой точке — это и есть обрыв следа на невалидных данных и на
    старте записи, где правая рука ещё не трекается.
    """
    if not isinstance(style, Style):
        raise ValueError(f"style: ожидался Style, получено {style!r}")
    if image is None or getattr(image, "ndim", 0) != 3:
        raise ValueError("image: ожидался BGR-кадр (H×W×3)")
    if not 0.0 < alpha_old <= alpha_new <= 1.0:
        raise ValueError(
            f"alpha: ожидалось 0 < alpha_old ≤ alpha_new ≤ 1, "
            f"получено alpha_old={alpha_old!r}, alpha_new={alpha_new!r}"
        )
    matrix = camera_model.world_to_camera_matrix(pose)
    eye = camera_model.eye_intrinsics(intrinsics, "left")
    by_side = trail_points(track, pose.ts_ns, seconds=seconds, step_s=step_s, joint=joint)
    thickness = max(1, style.line_thickness)
    drawn = 0
    for side in ("left", "right"):
        pixels: list[camera_model.ProjectedPoint | None] = []
        for world in by_side[side]:
            if not all(math.isfinite(value) for value in world):
                pixels.append(None)  # нет данных: линия здесь рвётся
                continue
            point = camera_model.project(camera_model.transform_point(matrix, world), eye)
            pixels.append(point if point.visible else None)
        last = len(pixels) - 1
        for index in range(1, len(pixels)):
            first, second = pixels[index - 1], pixels[index]
            if first is None or second is None:
                continue
            age = (last - index) / last if last else 0.0  # 0 — текущий момент, 1 — старый конец
            alpha = alpha_new + (alpha_old - alpha_new) * age
            if _blend_line(
                image,
                (_coord(first.u), _coord(first.v)),
                (_coord(second.u), _coord(second.v)),
                colors[side],
                thickness,
                alpha,
            ):
                drawn += 1
    return drawn


def hud_rect(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Прямоугольник HUD-панели ``(x0, y0, x1, y1)`` для кадра ``size`` (левый глаз).

    Размер фиксирован (``HUD_*`` для 1080×810, масштабируется шириной), поэтому панель
    занимает одно и то же место на всех кадрах — это позволяет проверить, что она не лежит
    на кистях (``tests/test_overlay_video.py`` сверяет с рамками эталона 07a).
    """
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in size)
    ):
        raise ValueError(f"size: ожидалась пара целых (width, height), получено {size!r}")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"size: должно быть > 0, получено {size!r}")
    factor = width / REFERENCE_WIDTH
    margin = int(round(HUD_MARGIN_PX * factor))
    panel_w = int(round(HUD_WIDTH_PX * factor))
    panel_h = int(round(HUD_HEIGHT_PX * factor))
    if panel_w < 1 or panel_h < 1 or margin + panel_w > width or margin + panel_h > height:
        raise ValueError(f"HUD не помещается в кадр {width}×{height}")
    return margin, margin, margin + panel_w, margin + panel_h


def hud_items(
    moment: sync.FrameTime,
    pose: sync.Pose,
    frame_count: int,
    *,
    trail_seconds: float | None = TRAIL_SECONDS,
) -> list[tuple[str, tuple[int, int, int]]]:
    """Строки HUD: время, кадр, состояние рук, след. Подписи латиницей (см. модуль)."""
    if not isinstance(moment, sync.FrameTime):
        raise ValueError(f"moment: ожидался sync.FrameTime, получено {type(moment).__name__}")
    if not isinstance(pose, sync.Pose):
        raise ValueError(f"pose: ожидался sync.Pose, получено {type(pose).__name__}")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError(f"frame_count: ожидалось целое > 0, получено {frame_count!r}")
    if not 0 <= moment.index < frame_count:
        raise ValueError(f"кадр {moment.index} вне 0…{frame_count - 1}")

    def hand_row(side: str) -> tuple[str, tuple[int, int, int]]:
        active = pose.active.get(side)
        text = f"{side}: {'OK' if active else 'NO DATA'}"
        return text, HUD_TEXT_COLOR if active else HUD_WARNING_COLOR

    rows = [
        (f"t = {moment.t_rel_s:8.3f} s", HUD_TEXT_COLOR),
        (f"frame {moment.index} / {frame_count - 1}", HUD_TEXT_COLOR),
        hand_row("left"),
        hand_row("right"),
    ]
    if trail_seconds is None:
        trail_text = "trail off"
    else:
        _trail_window(trail_seconds, TRAIL_STEP_S)  # валидация окна — здесь же, что и у рендера
        trail_text = f"trail {trail_seconds:.1f} s"
    if not moment.in_tracking:
        trail_text += " | pose frozen"
    rows.append((trail_text, HUD_TEXT_COLOR))
    return rows


def draw_hud(
    image: Any,
    moment: sync.FrameTime,
    pose: sync.Pose,
    frame_count: int,
    *,
    trail_seconds: float | None = TRAIL_SECONDS,
) -> tuple[int, int, int, int]:
    """Нарисовать HUD (in place) и вернуть его прямоугольник.

    Панель полупрозрачная и стоит в левом верхнем углу; текст проверяется на вписывание в
    панель до отрисовки, чтобы длинная строка не вылезала за её край.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    if image is None or getattr(image, "ndim", 0) != 3:
        raise ValueError("image: ожидался BGR-кадр (H×W×3)")
    height, width = image.shape[:2]
    x0, y0, x1, y1 = hud_rect((width, height))
    factor = width / REFERENCE_WIDTH
    scale = HUD_FONT_SCALE * factor
    thickness = max(1, int(round(factor)))
    step = int(round(HUD_LINE_STEP_PX * factor))
    pad = int(round(HUD_PADDING_PX * factor))
    rows = hud_items(moment, pose, frame_count, trail_seconds=trail_seconds)

    if pad * 2 + len(rows) * step > y1 - y0:
        raise ValueError(f"HUD: {len(rows)} строк не влезают в панель {x1 - x0}×{y1 - y0}")
    for text, _color in rows:
        (text_w, _text_h), _baseline = cv2.getTextSize(text, HUD_FONT_FACE, scale, thickness)
        if pad + text_w > x1 - x0 - pad:
            raise ValueError(f"HUD: строка {text!r} шире панели {x1 - x0} px")

    roi = image[y0:y1, x0:x1]
    blended = roi.astype(np.float32) * (1.0 - HUD_ALPHA) + np.asarray(
        HUD_PANEL_COLOR, np.float32
    ) * HUD_ALPHA
    roi[:] = np.rint(blended).astype(np.uint8)

    offset = int(round(HUD_BASELINE_OFFSET_PX * factor))
    for row, (text, color) in enumerate(rows):
        baseline = y0 + pad + (row + 1) * step - offset
        cv2.putText(
            image,
            text,
            (x0 + pad, baseline),
            HUD_FONT_FACE,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return x0, y0, x1, y1


def _eye_view(image: Any, scene: Scene) -> Any:
    """Левая половина SBS-кадра (левый глаз) — копия под рисование.

    Размер проверяется до рисования: если видео не то (или не SBS), координаты левого
    глаза недействительны, и молча нарисовать «съехавший» скелет хуже, чем упасть.
    """
    height, width = image.shape[:2]
    expected = (scene.intrinsics.full_w, scene.intrinsics.full_h)
    if (width, height) != expected:
        raise ValueError(
            f"кадр {width}×{height}, ожидался полный SBS-кадр {expected[0]}×{expected[1]}"
        )
    eye_width, eye_height = scene.eye_size
    return image[:eye_height, :eye_width].copy()


def _draw_pose(
    image: Any,
    pose: sync.Pose,
    scene: Scene,
    style: Style,
    skeleton: Skeleton = FIVE_POINT,
    *,
    depth_shading: bool = False,
    depth_alpha: bool = False,
) -> int:
    """Спроецировать скелет кадра и нарисовать его на уже обрезанном кадре глаза.

    ``depth_shading`` (08g) — красить ли по глубине: тогда вместе с пикселями считается
    ``z`` в системе камеры (:func:`project_skeleton_depth`), и цвет кости/точки гаснет с
    расстоянием. По умолчанию выключено: кадр 08a воспроизводится без изменений.

    ``depth_alpha`` (08g-fix2) — прозрачность по глубине (только вместе с ``depth_shading``:
    без глубин её не из чего считать). Порядок отрисовки в этом режиме — от дальних костей и
    точек к ближним, см. :func:`draw_skeleton`.
    """
    if depth_shading:
        pixels, depths = project_skeleton_depth(
            pose, scene.intrinsics, skeleton, track=scene.track
        )
    else:
        pixels, depths = project_skeleton(
            pose, scene.intrinsics, skeleton, track=scene.track
        ), None
    draw_style = style
    if depth_shading and skeleton.name != FLAT_SKELETON.name:
        # градиент вдоль кости виден только на линии потолще; 5-точечный кадр 08a не трогаем
        target = max(1, int(round(DEPTH_LINE_THICKNESS_PX * scene.eye_size[0] / REFERENCE_WIDTH)))
        if target > style.line_thickness:
            draw_style = Style(point_radius=style.point_radius, line_thickness=target)
    return draw_skeleton(
        image,
        pixels,
        draw_style,
        joints=skeleton.joints,
        bones=skeleton.bones,
        depths=depths,
        alpha=depth_shading and depth_alpha,
    )


def _codec_name(path: Path) -> str:
    """Кодек из заголовка mp4 (по четырёхбуквенному тегу в первых мегабайтах)."""
    tags = {
        b"avc1": "H.264 (avc1)",
        b"avc3": "H.264 (avc3)",
        b"hvc1": "H.265 (hvc1)",
        b"hev1": "H.265 (hev1)",
        b"mp4v": "MPEG-4 part 2 (mp4v)",
        b"vp09": "VP9",
        b"av01": "AV1",
    }
    with path.open("rb") as handle:
        head = handle.read(4 * 1024 * 1024)
    for tag, name in tags.items():
        if tag in head:
            return name
    return "unknown"


def probe_output(
    path: str | Path, expected_frames: int, expected_size: tuple[int, int]
) -> dict[str, Any]:
    """Прочитать записанный файл заново: число кадров, размер, кодек.

    Это не формальность: ``VideoWriter.isOpened()`` у части сборок OpenCV возвращает True,
    а файл получается пустым или с обрезанным числом кадров. Ошибка здесь означает, что
    рендер надо повторять, а не публиковать.
    """
    import cv2  # noqa: PLC0415

    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"файл не записан (или пуст): {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"записанный файл не открывается: {path}")
    try:
        frames = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        size = (
            int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
            int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
        ok, image = cap.read()
    finally:
        cap.release()
    if not ok or image is None:
        raise ValueError(f"из {path} не читается ни одного кадра")
    if size != expected_size:
        raise ValueError(f"размер {size}, ожидался {expected_size}")
    if frames != expected_frames:
        raise ValueError(f"кадров {frames}, ожидалось {expected_frames}")
    return {"frame_count": frames, "size": size, "codec": _codec_name(path)}


@dataclass(frozen=True)
class RenderStats:
    """Итог рендера: что записано, чем, за сколько и сколько точек нарисовано."""

    out: Path
    first_frame: int
    last_frame: int
    size: tuple[int, int]
    fps: float
    codec: str
    elapsed_s: float
    bytes: int
    joints_drawn: int
    frames_after_tracking: int
    time_offset_ms: float = 0.0
    #: 08b: окно следа (0 — след выключен), сколько отрезков следа нарисовано за прогон.
    trail_seconds: float = 0.0
    trail_segments: int = 0
    #: 08b: рисовался ли HUD.
    hud: bool = True
    #: 08f: имя скелета (``five``/``dense``/``all``) и число суставов на руку в нём.
    skeleton: str = "five"
    joints_per_hand: int = 5
    #: 08g: красился ли скелет по глубине (множитель к цвету руки).
    depth_shading: bool = False
    #: 08g-fix2: включалась ли прозрачность по глубине (второй канал, сортировка по z).
    depth_alpha: bool = False

    @property
    def frames(self) -> int:
        """Сколько кадров записано."""
        return self.last_frame - self.first_frame + 1

    @property
    def duration_s(self) -> float:
        """Длительность записанного файла по метаданным записи."""
        return self.frames / self.fps if self.fps else 0.0

    @property
    def frames_per_s(self) -> float:
        """Скорость рендера, кадр/с."""
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0


def _open_writer(
    out: Path, fps: float, size: tuple[int, int], candidates: Sequence[str], log: Callable[[str], None]
) -> tuple[Any, str]:
    """Первый открывшийся ``VideoWriter`` из списка кодеков (H.264 → mp4v)."""
    import cv2  # noqa: PLC0415

    if not candidates:
        raise ValueError("fourcc_candidates: пустой список")
    for fourcc in candidates:
        writer = cv2.VideoWriter(
            str(out), cv2.VideoWriter_fourcc(*fourcc), fps, size
        )
        if writer.isOpened():
            if fourcc != candidates[0]:
                log(f"  кодек {candidates[0]!r} недоступен, пишу {fourcc!r}")
            return writer, fourcc
        writer.release()
    raise ValueError(f"ни один кодек не открылся: {', '.join(candidates)}")


def render_video(
    video: str | Path,
    out: str | Path,
    scene: Scene,
    *,
    start: int = 0,
    end: int | None = None,
    scale: float = 1.0,
    fourcc_candidates: Sequence[str] = FOURCC_CANDIDATES,
    progress_every: int = 300,
    trail: bool = True,
    hud: bool = True,
    trail_seconds: float = TRAIL_SECONDS,
    skeleton: Skeleton = FIVE_POINT,
    depth_shading: bool | None = None,
    depth_alpha: bool | None = None,
    log: Callable[[str], None] = print,
) -> RenderStats:
    """Отрендерить диапазон кадров ``start…end`` в ``out`` и вернуть :class:`RenderStats`.

    Пишется только левый глаз (1080×810). ``progress_every`` — как часто печатать строку
    прогресса (0 — молча); итоговая строка печатается всегда. Скорость рендера в статистике
    — фактическое время (от открытия видео до закрытия файла), а не оценка.

    ``trail``/``hud`` — элементы 08b (след и панель состояния); ``trail_seconds`` — окно
    следа в секундах (выборка по времени, не по кадрам). Порядок отрисовки: след под
    скелетом, HUD поверх всего — так подписи остаются читаемыми.

    ``skeleton`` (08f) — какой скелет рисовать: по умолчанию прежний 5-точечный (08a);
    CLI передаёт :data:`DENSE` или :data:`ALL_JOINTS`. Для густого скелета сцена должна
    быть загружена с ``extra_keys=skeleton.extra_keys`` — иначе ``ValueError`` до записи.

    ``depth_shading`` (08g) — кодировать ли глубину цветом: ``None`` — авторежим
    (:func:`resolve_depth_shading`: да для густого скелета, нет для 5-точечного 08a),
    ``True``/``False`` — принудительно.

    ``depth_alpha`` (08g-fix2) — прозрачность по глубине (второй канал): авторежим
    (:func:`resolve_depth_alpha`) включается вместе с яркостью для густого скелета и
    ``--all-joints``; без ``depth_shading`` не включается, потому что глубины не считаются.
    Явное ``False`` — «только яркость», кадр 08g-fix.
    """
    import cv2  # noqa: PLC0415

    if isinstance(progress_every, bool) or not isinstance(progress_every, int):
        raise ValueError(f"progress_every: ожидалось целое число, получено {progress_every!r}")
    if progress_every < 0:
        raise ValueError(f"progress_every: должно быть ≥ 0, получено {progress_every!r}")
    for name, value in (("trail", trail), ("hud", hud)):
        if not isinstance(value, bool):
            raise ValueError(f"{name}: ожидалось bool, получено {value!r}")
    _require_skeleton_track(scene.track, skeleton)
    shading = resolve_depth_shading(depth_shading, skeleton)
    transparency = shading and resolve_depth_alpha(depth_alpha, skeleton)
    if trail:
        _trail_window(trail_seconds, TRAIL_STEP_S)  # проверка окна до открытия файлов
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"нет видео: {video}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    first, last = frame_range(scene.meta, start, end)
    size = scene.eye_size
    style = style_for(size[0], scale)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cv2.VideoCapture не открывает {video}")
    writer = None
    try:
        if first:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, first):
                raise ValueError(f"не удалось перейти к кадру {first}")
        writer, _fourcc = _open_writer(out, scene.fps, size, fourcc_candidates, log)
        total = last - first + 1
        started = time.perf_counter()
        joints = 0
        trail_segments = 0
        after_tracking = 0
        completed = False
        try:
            for k in range(first, last + 1):
                ok, image = capture.read()
                if not ok or image is None:
                    raise ValueError(f"видео кончилось на кадре {k} (ожидались по {last})")
                view = _eye_view(image, scene)
                moment = sync.frame_time(k, scene.meta, scene.header_ns, scene.tracking_end_ns)
                pose = sync.pose_at_frame(scene.track, moment, scene.time_offset_ns)
                if trail:
                    trail_segments += draw_trail(
                        view, pose, scene.track, scene.intrinsics, style, seconds=trail_seconds
                    )
                joints += _draw_pose(
                    view,
                    pose,
                    scene,
                    style,
                    skeleton,
                    depth_shading=shading,
                    depth_alpha=transparency,
                )
                if hud:
                    draw_hud(
                        view,
                        moment,
                        pose,
                        scene.frame_count,
                        trail_seconds=trail_seconds if trail else None,
                    )
                if not moment.in_tracking:
                    after_tracking += 1
                writer.write(view)
                done = k - first + 1
                if progress_every and (done % progress_every == 0 or done == total):
                    elapsed = time.perf_counter() - started
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (total - done) / rate if rate else 0.0
                    log(
                        f"  кадр {k:5d}/{last} ({100.0 * done / total:5.1f} %) — "
                        f"{elapsed:6.1f} с, {rate:5.1f} кадр/с, осталось ~{eta:5.1f} с"
                    )
            completed = True
        finally:
            writer.release()
            writer = None
            elapsed = time.perf_counter() - started
            if not completed:
                out.unlink(missing_ok=True)
    finally:
        if writer is not None:
            writer.release()
        capture.release()

    probe = probe_output(out, total, size)
    return RenderStats(
        out=out,
        first_frame=first,
        last_frame=last,
        size=probe["size"],
        fps=scene.fps,
        codec=probe["codec"],
        elapsed_s=elapsed,
        bytes=out.stat().st_size,
        joints_drawn=joints,
        frames_after_tracking=after_tracking,
        time_offset_ms=scene.time_offset_ns / 1e6,
        trail_seconds=float(trail_seconds) if trail else 0.0,
        trail_segments=trail_segments,
        hud=hud,
        skeleton=skeleton.name,
        joints_per_hand=len(skeleton.joints),
        depth_shading=shading,
        depth_alpha=transparency,
    )


def render_stills(
    video: str | Path,
    scene: Scene,
    out_dir: str | Path,
    *,
    frames: Iterable[int] = DEBUG_FRAMES,
    scale: float = 1.0,
    trail: bool = True,
    hud: bool = True,
    trail_seconds: float = TRAIL_SECONDS,
    skeleton: Skeleton = FIVE_POINT,
    depth_shading: bool | None = None,
    depth_alpha: bool | None = None,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Сохранить PNG-кадры с наложением (по умолчанию 5 отладочных кадров 07a).

    Это визуальная проверка наложения перед полным рендером: PNG 1080×810, по одному на
    кадр, ``frame_{k:05d}.png`` в ``out_dir``. Артефакт регенерируемый (``output/`` в
    ``.gitignore``), поэтому коммитится не он, а выводы о проверке.

    ``trail``/``hud`` — как у :func:`render_video`: ``--no-trail --no-hud`` даёт кадр «как в
    08a», он идёт в сравнение «до/после» (``docs/overlay.md``). ``skeleton`` — как у
    :func:`render_video`: по умолчанию 5 точек, CLI передаёт густой скелет.
    ``depth_shading`` (08g) — как у :func:`render_video`: ``None`` — авторежим (да для густого
    скелета, нет для 5-точечного 08a), ``True``/``False`` — принудительно; на PNG это
    видно без кодирования видео, поэтому «до/после» снимается именно ``--stills``.
    ``depth_alpha`` (08g-fix2) — как у :func:`render_video`: прозрачность по глубине
    (авторежим как у яркости, без яркости не включается).
    """
    import cv2  # noqa: PLC0415

    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"нет видео: {video}")
    indices = sorted(set(frames))
    style = style_for(scene.eye_size[0], scale)
    _require_skeleton_track(scene.track, skeleton)
    shading = resolve_depth_shading(depth_shading, skeleton)
    transparency = shading and resolve_depth_alpha(depth_alpha, skeleton)
    if trail:
        _trail_window(trail_seconds, TRAIL_STEP_S)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cv2.VideoCapture не открывает {video}")
    written: list[Path] = []
    try:
        for k in indices:
            frame_range(scene.meta, k, k)  # проверка номера кадра
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, k):
                raise ValueError(f"не удалось перейти к кадру {k}")
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"не читается кадр {k}")
            view = _eye_view(image, scene)
            moment = sync.frame_time(k, scene.meta, scene.header_ns, scene.tracking_end_ns)
            pose = sync.pose_at_frame(scene.track, moment, scene.time_offset_ns)
            segments = 0
            if trail:
                segments = draw_trail(
                    view, pose, scene.track, scene.intrinsics, style, seconds=trail_seconds
                )
            drawn = _draw_pose(
                view,
                pose,
                scene,
                style,
                skeleton,
                depth_shading=shading,
                depth_alpha=transparency,
            )
            if hud:
                draw_hud(
                    view,
                    moment,
                    pose,
                    scene.frame_count,
                    trail_seconds=trail_seconds if trail else None,
                )
            if not moment.in_tracking:
                log(f"  кадр {k:5d}: трекинга уже нет — поза заморожена (по 04a)")
            path = out / f"frame_{k:05d}.png"
            if not cv2.imwrite(str(path), view):
                raise ValueError(f"не удалось записать {path}")
            written.append(path)
            trail_note = f", сегментов следа {segments}" if trail else ""
            log(
                f"  кадр {k:5d} (t = {moment.t_rel_s:9.3f} с): точек "
                f"{drawn}/{2 * len(skeleton.joints)}{trail_note} → {path}"
            )
    finally:
        capture.release()
    return written


def _format_stats(stats: RenderStats) -> str:
    """Итог рендера одной таблицей — эти числа идут в README/заметки."""
    megabytes = stats.bytes / (1024 * 1024)
    trail = (
        f"след {stats.trail_seconds:g} с, сегментов {stats.trail_segments}; "
        f"HUD: {'да' if stats.hud else 'нет'}"
        if stats.trail_seconds > 0
        else f"след выключен; HUD: {'да' if stats.hud else 'нет'}"
    )
    if stats.depth_shading:
        shading_line = (
            f"яркость: окно {DEPTH_NEAR_M:g}…{DEPTH_FAR_M:g} м, множитель "
            f"{DEPTH_MIN_FACTOR:g}…1.0"
        )
        if stats.depth_alpha:
            shading_line += (
                f"; alpha: {DEPTH_MIN_ALPHA:g}…1.0, дальние раньше ближних"
            )
        else:
            shading_line += "; только яркость, без alpha (кадр 08g-fix)"
    else:
        shading_line = "нет (плоские цвета рук)"
    lines = [
        f"записано: {stats.out}",
        f"  кадры: {stats.first_frame}…{stats.last_frame} ({stats.frames}), "
        f"{stats.size[0]}×{stats.size[1]}, {stats.fps:.4f} fps → {stats.duration_s:.3f} с",
        f"  кодек: {stats.codec}; файл {megabytes:.1f} МиБ; "
        f"поправка времени {stats.time_offset_ms:+.0f} мс",
        f"  скелет: {stats.skeleton} ({stats.joints_per_hand} суставов на руку)",
        f"  глубина → цвет: {shading_line}",
        f"  рендер: {stats.elapsed_s:.1f} с, {stats.frames_per_s:.1f} кадр/с "
        f"(видимых суставов {stats.joints_drawn})",
        f"  {trail}",
        f"  кадров вне трекинга (поза заморожена): {stats.frames_after_tracking}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: полный рендер, фрагмент (``--start/--end``) или PNG-проверка (``--stills``)."""
    parser = argparse.ArgumentParser(
        description="Рендер видео с наложением скелета кистей, следа и HUD (08a/08b/08f)."
    )
    parser.add_argument("video", type=Path, help="data/CameraRecord_*.mp4 (полный SBS-кадр)")
    parser.add_argument(
        "--tracking",
        type=Path,
        default=None,
        help="data/trackingData_*.txt (по умолчанию — единственный в data/)",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("output/video_meta.json"),
        help="video_meta.json (по умолчанию output/video_meta.json)",
    )
    parser.add_argument("--out", type=Path, default=Path("output/overlay.mp4"),
                        help="куда писать видео (по умолчанию output/overlay.mp4)")
    parser.add_argument("--start", type=int, default=0, help="первый кадр (0-based, по умолчанию 0)")
    parser.add_argument("--end", type=int, default=None, help="последний кадр включительно")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="множитель радиуса точек и толщины линий (по умолчанию 1.0)")
    parser.add_argument("--fourcc", default=",".join(FOURCC_CANDIDATES),
                        help=f"кодеки через запятую (по умолчанию {','.join(FOURCC_CANDIDATES)})")
    parser.add_argument("--progress-every", type=int, default=300,
                        help="строка прогресса каждые N кадров (0 — не печатать)")
    parser.add_argument("--time-offset-ms", type=float, default=None,
                        help="сдвиг времени позы (по умолчанию — config.VIDEO_TIME_OFFSET_MS)")
    parser.add_argument("--stills", type=Path, default=None,
                        help="режим PNG: сохранить кадры --frames в этот каталог и выйти")
    parser.add_argument("--frames", default=",".join(str(k) for k in DEBUG_FRAMES),
                        help="кадры для --stills (по умолчанию 5 кадров 07a)")
    parser.add_argument("--no-trail", action="store_true",
                        help="не рисовать след кистей (кадр «как в 08a»)")
    parser.add_argument("--no-hud", action="store_true",
                        help="не рисовать HUD (панель времени/кадра/состояния рук)")
    parser.add_argument("--trail-seconds", type=float, default=TRAIL_SECONDS,
                        help=f"окно следа в секундах (по умолчанию {TRAIL_SECONDS:g})")
    skeleton_group = parser.add_mutually_exclusive_group()
    skeleton_group.add_argument(
        "--all-joints",
        action="store_true",
        help="рисовать все 26 суставов OpenXR на руку (по умолчанию густой скелет из 17)",
    )
    skeleton_group.add_argument(
        "--five-points",
        action="store_true",
        help="рисовать прежний 5-точечный скелет 08a (для сравнений «до/после»)",
    )
    depth_group = parser.add_mutually_exclusive_group()
    depth_group.add_argument(
        "--depth-color",
        dest="depth_shading",
        action="store_true",
        default=None,
        help="кодировать глубину цветом принудительно (по умолчанию — для густого скелета)",
    )
    depth_group.add_argument(
        "--no-depth-color",
        dest="depth_shading",
        action="store_false",
        default=None,
        help="плоские цвета рук, как в 08f (кадр 5 точек 08a воспроизводится без изменений)",
    )
    alpha_group = parser.add_mutually_exclusive_group()
    alpha_group.add_argument(
        "--depth-alpha",
        dest="depth_alpha",
        action="store_true",
        default=None,
        help="прозрачность по глубине принудительно (по умолчанию — для густого скелета)",
    )
    alpha_group.add_argument(
        "--no-depth-alpha",
        dest="depth_alpha",
        action="store_false",
        default=None,
        help="только яркость, без прозрачности (кадр 08g-fix)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    skeleton = ALL_JOINTS if args.all_joints else FIVE_POINT if args.five_points else DEFAULT_SKELETON
    depth_shading = resolve_depth_shading(args.depth_shading, skeleton)
    if args.depth_alpha is True and args.depth_shading is not False:
        # alpha считается из глубин: явный --depth-alpha включает и яркость (кроме
        # --no-depth-color, который отключает всё разом)
        depth_shading = True
    depth_alpha = depth_shading and resolve_depth_alpha(args.depth_alpha, skeleton)
    try:
        scene = load_scene(args.meta, args.tracking, extra_keys=skeleton.extra_keys)
        if args.time_offset_ms is not None:
            scene = replace(
                scene,
                time_offset_ns=int(round(args.time_offset_ms * 1e6)),
            )
        print(
            f"видео: {scene.frame_count} кадров, {scene.duration_s:.3f} с, "
            f"{scene.fps:.4f} fps; SBS {scene.intrinsics.full_w}×{scene.intrinsics.full_h} "
            f"→ глаз {scene.eye_size[0]}×{scene.eye_size[1]}"
        )
        print(
            f"трекинг: {len(scene.track)} сэмплов; конец трекинга "
            f"+{(scene.tracking_end_ns - scene.header_ns) / 1e9:.3f} с от header; "
            f"поправка времени {scene.time_offset_ns / 1e6:+.0f} мс"
        )
        print(
            f"скелет: {skeleton.name} — {skeleton.title}, "
            f"рёбер {len(skeleton.bones)} на руку (--all-joints — 26 суставов, "
            f"--five-points — 5 точек)"
        )
        print(
            f"глубина → цвет: {'да' if depth_shading else 'нет'}"
            + (
                f" (окно {DEPTH_NEAR_M:g}…{DEPTH_FAR_M:g} м, множитель "
                f"{DEPTH_MIN_FACTOR:g}…1.0)"
                if depth_shading
                else ""
            )
        )
        print(
            f"глубина → alpha: {'да' if depth_alpha else 'нет'}"
            + (f" (пол {DEPTH_MIN_ALPHA:g}, дальние раньше ближних)" if depth_alpha else "")
        )
        trail = not args.no_trail
        hud = not args.no_hud
        if args.stills is not None:
            paths = render_stills(
                args.video,
                scene,
                args.stills,
                frames=parse_frames(args.frames),
                scale=args.scale,
                trail=trail,
                hud=hud,
                trail_seconds=args.trail_seconds,
                skeleton=skeleton,
                depth_shading=depth_shading,
                depth_alpha=depth_alpha,
            )
            print(f"PNG с наложением: {len(paths)} файлов в {args.stills}")
            return 0
        fourcc = tuple(part.strip() for part in args.fourcc.split(",") if part.strip())
        stats = render_video(
            args.video,
            args.out,
            scene,
            start=args.start,
            end=args.end,
            scale=args.scale,
            fourcc_candidates=fourcc,
            progress_every=args.progress_every,
            trail=trail,
            hud=hud,
            trail_seconds=args.trail_seconds,
            skeleton=skeleton,
            depth_shading=depth_shading,
            depth_alpha=depth_alpha,
        )
    except (OSError, ValueError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1
    print(_format_stats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
