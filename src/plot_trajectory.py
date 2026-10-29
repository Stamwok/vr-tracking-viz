"""plot_trajectory.py — интерактивные графики траекторий (Plotly).

Задача 09: из ``output/tracking.csv`` (03d) собираем **один** self-contained HTML
(plotly.js встроен в файл, ни CDN, ни сервера не нужно):

1. 3D-траектории головы и ладоней обеих рук в одной сцене с метками «старт»/«финиш»,
2. x(t)/y(t)/z(t) — три ряда, в каждом голова и обе кисти,
3. скорость (м/с) по времени,
4. высота кистей относительно головы (``y_кисти − y_головы``).

Данные для показа прореживаются равномерным шагом (``--max-points``), но вся статистика
(диапазоны, скорость, путь, NaN) считается по **полным** 21 787 строкам CSV и печатается
CLI + встраивается в HTML блоком «Сводка». Пик скорости каждого канала всегда остаётся
в выборке, поэтому hover на максимуме совпадает со сводкой.

Невалидные точки рук (NaN из 03c) не «затягиваются»: на графиках это разрыв линии,
на 3D-траектории — отдельный отрезок на каждый непрерывный валидный участок.

Запуск:
    python src/plot_trajectory.py --tracking output/tracking.csv -o output/trajectory.html
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

#: Цвета согласованы с наложением на видео (08a/08b): левая кисть голубая, правая красная.
HEAD_COLOR = "#5c6bc0"
LEFT_COLOR = "#00acc1"
RIGHT_COLOR = "#e53935"

#: Сколько точек оставлять на серию при показе (0 — не прореживать). Полные данные — 21 787.
DEFAULT_MAX_POINTS = 5000

#: Порядок и подписи блоков HTML.
FIGURE_CAPTIONS = {
    "trajectories": "3D-траектории: голова и ладони обеих рук, метки «старт»/«финиш»",
    "positions": "x(t), y(t), z(t): голова и ладони обеих рук",
    "speed": "Скорость ладони и головы, м/с",
    "height": "Высота кистей относительно головы, м",
}


@dataclass(frozen=True)
class Series:
    """Канал, который рисуем: подпись, цвет и префикс колонок в tracking.csv."""

    key: str
    label: str
    color: str
    prefix: str


#: Рисуем ровно три канала: поза головы и ладонь каждой руки (5 точек руки — в 08a).
SERIES: tuple[Series, ...] = (
    Series("head", "голова", HEAD_COLOR, "head"),
    Series("left", "левая кисть (ладонь)", LEFT_COLOR, "left_palm"),
    Series("right", "правая кисть (ладонь)", RIGHT_COLOR, "right_palm"),
)

#: Колонки tracking.csv, без которых графики не построить (03d).
REQUIRED_COLUMNS: tuple[str, ...] = ("t_rel_s",) + tuple(
    f"{series.prefix}_{axis}" for series in SERIES for axis in ("x", "y", "z")
)


@dataclass
class SeriesData:
    """Временной ряд одного канала: время, мировые координаты (м), скорость, высота."""

    series: Series
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    speed: np.ndarray
    rel_height: np.ndarray | None = None  # только для кистей: y − y_головы


# ---------------------------------------------------------------- данные


def load_tracking(path: str | Path):
    """Читает tracking.csv (03d) и проверяет, что нужные колонки на месте.

    NaN остаются NaN: пустые поля CSV — это «нет данных» (неактивная рука), а не нули.
    """
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"нет файла: {path} (сначала python src/parse_tracking.py … -o {path})")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path}: нет колонок {', '.join(missing)} — это не tracking.csv из 03d")
    if frame.empty:
        raise ValueError(f"{path}: файл пуст")
    return frame


def _finite_at(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray, index: np.ndarray
) -> np.ndarray:
    """Конечны ли t/x/y/z в точках ``index``."""
    return (
        np.isfinite(t[index])
        & np.isfinite(x[index])
        & np.isfinite(y[index])
        & np.isfinite(z[index])
    )


def speeds(t: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Модуль скорости, м/с: центральная разность, на краях — односторонняя.

    Точка получает NaN, если невалидна сама (NaN) или один из её соседей — через дырку
    «нет данных» скорость не мостится, как и интерполяция в ``sync.pose_at``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if not (t.shape == x.shape == y.shape == z.shape):
        raise ValueError("скорость: массивы t/x/y/z разной длины")
    n = t.size
    out = np.full(n, np.nan)
    if n < 2:
        return out

    index = np.arange(n)
    lo = np.clip(index - 1, 0, n - 1)
    hi = np.clip(index + 1, 0, n - 1)
    valid = (hi > lo) & _finite_at(t, x, y, z, index) & _finite_at(t, x, y, z, lo)
    valid &= _finite_at(t, x, y, z, hi)
    dt = np.where(valid, t[hi] - t[lo], np.nan)
    valid &= dt > 0
    distance = np.hypot(np.hypot(x[hi] - x[lo], y[hi] - y[lo]), z[hi] - z[lo])
    out[valid] = distance[valid] / dt[valid]
    return out


def path_length(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Длина пройденного пути, м: сумма шагов между соседними валидными точками."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if x.size < 2:
        return 0.0
    ok = (
        np.isfinite(x[:-1])
        & np.isfinite(x[1:])
        & np.isfinite(y[:-1])
        & np.isfinite(y[1:])
        & np.isfinite(z[:-1])
        & np.isfinite(z[1:])
    )
    steps = np.hypot(np.hypot(x[1:] - x[:-1], y[1:] - y[:-1]), z[1:] - z[:-1])
    return float(np.sum(steps[ok]))


def series_data(frame) -> list[SeriesData]:
    """Достаёт из DataFrame ряды каналов + скорости + высоту кистей относительно головы."""
    t = frame["t_rel_s"].to_numpy(dtype=float)
    head_y = frame["head_y"].to_numpy(dtype=float)
    rows = []
    for series in SERIES:
        x = frame[f"{series.prefix}_x"].to_numpy(dtype=float)
        y = frame[f"{series.prefix}_y"].to_numpy(dtype=float)
        z = frame[f"{series.prefix}_z"].to_numpy(dtype=float)
        rel_height = None
        if series.key != "head":
            present = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            rel_height = np.where(present, y - head_y, np.nan)
        rows.append(SeriesData(series, t, x, y, z, speeds(t, x, y, z), rel_height))
    return rows


# ---------------------------------------------------------------- сводка


def _range(values: np.ndarray) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(finite.min()), float(finite.max())


def _speed_stats(data: SeriesData) -> dict:
    finite = np.isfinite(data.speed)
    if not finite.any():
        return {"mean": None, "p95": None, "max": None, "max_t_s": None}
    values = data.speed[finite]
    peak = int(np.argmax(np.where(finite, data.speed, -np.inf)))
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "max_t_s": float(data.t[peak]),
    }


def summarize(data: Sequence[SeriesData]) -> dict:
    """Сводка по полным данным: строки, длительность, диапазоны, скорость, путь, NaN."""
    if not data:
        raise ValueError("сводка: нет данных")
    t = data[0].t
    dt_ms = np.diff(t) / 1e-3
    summary: dict = {
        "rows": int(t.size),
        "duration_s": float(t[-1] - t[0]) if t.size > 1 else 0.0,
        "median_dt_ms": float(np.median(dt_ms)) if dt_ms.size else 0.0,
        "series": {},
    }
    for row in data:
        present = np.isfinite(row.x) & np.isfinite(row.y) & np.isfinite(row.z)
        summary["series"][row.series.key] = {
            "label": row.series.label,
            "points": int(present.sum()),
            "nan": int((~present).sum()),
            "x": _range(row.x),
            "y": _range(row.y),
            "z": _range(row.z),
            "speed": _speed_stats(row),
            "path_m": path_length(row.x, row.y, row.z),
            "rel_height": _range(row.rel_height) if row.rel_height is not None else None,
        }
    return summary


def _format_range(span: tuple[float, float] | None) -> str:
    if span is None:
        return "нет данных"
    return f"{span[0]:.3f}…{span[1]:.3f}"


def format_summary(summary: dict) -> list[str]:
    """Сводка в виде строк для CLI и блока «Сводка» в HTML."""
    lines = [
        f"строк: {summary['rows']}, длительность: {summary['duration_s']:.3f} с, "
        f"медиана шага: {summary['median_dt_ms']:.2f} мс",
    ]
    for key, item in summary["series"].items():
        speed = item["speed"]
        speed_text = (
            "нет данных"
            if speed["mean"] is None
            else (
                f"средняя {speed['mean']:.3f}, p95 {speed['p95']:.3f}, "
                f"пик {speed['max']:.3f} м/с (t = {speed['max_t_s']:.2f} с)"
            )
        )
        lines.append(
            f"{item['label']}: точек {item['points']}, NaN {item['nan']}; "
            f"x {_format_range(item['x'])} м, y {_format_range(item['y'])} м, "
            f"z {_format_range(item['z'])} м"
        )
        lines.append(f"  скорость: {speed_text}; путь {item['path_m']:.2f} м")
        if item["rel_height"] is not None:
            lines.append(f"  высота относительно головы: {_format_range(item['rel_height'])} м")
    return lines


# ---------------------------------------------------------------- прореживание


def downsample_indices(
    count: int, max_points: int, must_include: Iterable[int] = ()
) -> np.ndarray:
    """Индексы 0…count−1 с равномерным шагом, не больше ``max_points`` (+1 на хвост).

    Первый и последний индексы сохраняются всегда, ``must_include`` (например, пик
    скорости) добавляются принудительно. При ``max_points <= 0`` — всё без прореживания.
    """
    if count < 0:
        raise ValueError("прореживание: count < 0")
    if count == 0:
        return np.empty(0, dtype=int)
    if max_points <= 0:
        return np.arange(count)
    if max_points < 2:
        raise ValueError("прореживание: max_points должен быть >= 2")
    if count <= max_points:
        return np.arange(count)
    step = math.ceil(count / max_points)
    picked = list(range(0, count, step))
    if picked[-1] != count - 1:
        picked.append(count - 1)
    picked.extend(index for index in must_include if 0 <= index < count)
    return np.unique(np.asarray(picked, dtype=int))


def valid_segments(*arrays: np.ndarray) -> list[slice]:
    """Непрерывные участки, где все переданные массивы конечны (NaN рвёт линию)."""
    if not arrays:
        return []
    ok = np.ones(arrays[0].size, dtype=bool)
    for values in arrays:
        ok &= np.isfinite(np.asarray(values, dtype=float))
    segments: list[slice] = []
    start = None
    for index, flag in enumerate(ok):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            segments.append(slice(start, index))
            start = None
    if start is not None:
        segments.append(slice(start, ok.size))
    return segments


# ---------------------------------------------------------------- фигуры


def _stride_indices(start: int, stop: int, step: int) -> np.ndarray:
    """Индексы участка ``start:stop`` с шагом ``step``; последний элемент участка сохранён."""
    indices = np.arange(start, stop, max(1, step))
    if indices.size and indices[-1] != stop - 1:
        indices = np.append(indices, stop - 1)
    return indices


def _speed_step(count: int, max_points: int) -> int:
    return 1 if max_points <= 0 or count <= max_points else math.ceil(count / max_points)


def _hover(axis_labels: Sequence[str]) -> str:
    parts = " ".join(f"{label}: %{{{name}:.3f}} м" for label, name in zip(axis_labels, "xyz"))
    return parts + "<extra>%{fullData.name}</extra>"


def build_trajectory_figure(data: Sequence[SeriesData], max_points: int = DEFAULT_MAX_POINTS):
    """3D-траектории трёх каналов + метки «старт»/«финиш» (по каждой серии)."""
    import plotly.graph_objects as go

    figure = go.Figure()
    for row in data:
        step = _speed_step(row.t.size, max_points)
        for number, segment in enumerate(valid_segments(row.x, row.y, row.z)):
            indices = _stride_indices(segment.start, segment.stop, step)
            figure.add_trace(
                go.Scatter3d(
                    x=row.x[indices],
                    y=row.y[indices],
                    z=row.z[indices],
                    mode="lines",
                    name=row.series.label,
                    legendgroup=row.series.key,
                    showlegend=number == 0,
                    line={"color": row.series.color, "width": 2},
                    hovertemplate=_hover(("x", "y", "z")),
                )
            )
        # Метки старт/финиш: не в легенде (иначе она вдвое длиннее), но подписаны на сцене.
        finite = np.flatnonzero(np.isfinite(row.x) & np.isfinite(row.y) & np.isfinite(row.z))
        if finite.size:
            for index, label, symbol in (
                (int(finite[0]), "старт", "diamond-open"),
                (int(finite[-1]), "финиш", "x"),
            ):
                figure.add_trace(
                    go.Scatter3d(
                        x=[row.x[index]],
                        y=[row.y[index]],
                        z=[row.z[index]],
                        mode="markers+text",
                        text=[label],
                        textposition="top center",
                        textfont={"size": 13, "color": row.series.color},
                        marker={"size": 8, "color": row.series.color, "symbol": symbol},
                        name=f"{row.series.label}: {label}",
                        legendgroup=row.series.key,
                        showlegend=False,
                        hovertemplate=(
                            f"{row.series.label} — {label}, t = {row.t[index]:.2f} с"
                            "<br>x: %{x:.3f} м y: %{y:.3f} м z: %{z:.3f} м<extra></extra>"
                        ),
                    )
                )
    figure.update_layout(
        height=760,
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        legend={"orientation": "h", "y": 1.0, "x": 0, "bgcolor": "rgba(255,255,255,0.85)"},
        scene={
            "xaxis_title": "x, м",
            "yaxis_title": "y, м",
            "zaxis_title": "z, м",
            "aspectmode": "data",  # честные пропорции: диапазоны в метрах сравнимы
            "camera": {"eye": {"x": 1.4, "y": 1.2, "z": 1.1}},
        },
    )
    return figure


def _downsampled(
    row: SeriesData, values: np.ndarray, max_points: int, preserve_peak: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Индексы и значения; ``preserve_peak`` оставляет точку максимума (для скорости)."""
    must: tuple[int, ...] = ()
    if preserve_peak:
        finite = np.isfinite(row.speed)
        if finite.any():
            must = (int(np.argmax(np.where(finite, row.speed, -np.inf))),)
    indices = downsample_indices(row.t.size, max_points, must)
    return indices, values[indices]


def build_position_figure(data: Sequence[SeriesData], max_points: int = DEFAULT_MAX_POINTS):
    """x(t)/y(t)/z(t): три ряда, в каждом голова и обе кисти (оси в метрах)."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("x(t), м", "y(t), м", "z(t), м"),
    )
    for row_number, axis in enumerate(("x", "y", "z"), start=1):
        for series_row in data:
            values = getattr(series_row, axis)
            indices, picked = _downsampled(series_row, values, max_points)
            figure.add_trace(
                go.Scatter(
                    x=series_row.t[indices],
                    y=picked,
                    mode="lines",
                    name=series_row.series.label,
                    legendgroup=series_row.series.key,
                    showlegend=row_number == 1,
                    line={"color": series_row.series.color, "width": 1.4},
                    connectgaps=False,
                    hovertemplate=(
                        f"{series_row.series.label}<br>t: %{{x:.2f}} с"
                        f"<br>{axis}: %{{y:.3f}} м<extra></extra>"
                    ),
                ),
                row=row_number,
                col=1,
            )
        figure.update_yaxes(title_text=f"{axis}, м", row=row_number, col=1)
    figure.update_xaxes(title_text="t, с", row=3, col=1)
    figure.update_layout(
        height=760,
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        hovermode="x unified",
    )
    return figure


def build_speed_figure(data: Sequence[SeriesData], max_points: int = DEFAULT_MAX_POINTS):
    """Скорость каналов по времени (пик каждой серии остаётся в выборке)."""
    import plotly.graph_objects as go

    figure = go.Figure()
    for row in data:
        indices, picked = _downsampled(row, row.speed, max_points, preserve_peak=True)
        figure.add_trace(
            go.Scatter(
                x=row.t[indices],
                y=picked,
                mode="lines",
                name=row.series.label,
                line={"color": row.series.color, "width": 1.2},
                connectgaps=False,
                hovertemplate=(
                    f"{row.series.label}<br>t: %{{x:.2f}} с"
                    "<br>скорость: %{y:.3f} м/с<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        height=440,
        margin={"l": 60, "r": 20, "t": 30, "b": 40},
        xaxis_title="t, с",
        yaxis_title="скорость, м/с",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    return figure


def build_height_figure(data: Sequence[SeriesData], max_points: int = DEFAULT_MAX_POINTS):
    """Высота кистей относительно головы (y_кисти − y_головы) по времени."""
    import plotly.graph_objects as go

    figure = go.Figure()
    for row in data:
        if row.rel_height is None:
            continue
        indices, picked = _downsampled(row, row.rel_height, max_points)
        figure.add_trace(
            go.Scatter(
                x=row.t[indices],
                y=picked,
                mode="lines",
                name=row.series.label,
                line={"color": row.series.color, "width": 1.2},
                connectgaps=False,
                hovertemplate=(
                    f"{row.series.label}<br>t: %{{x:.2f}} с"
                    "<br>относительно головы: %{y:.3f} м<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        height=400,
        margin={"l": 60, "r": 20, "t": 30, "b": 40},
        xaxis_title="t, с",
        yaxis_title="y кисти − y головы, м",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    return figure


def build_figures(
    data: Sequence[SeriesData], max_points: int = DEFAULT_MAX_POINTS
) -> list[tuple[str, object]]:
    """Все четыре блока отчёта: (ключ из FIGURE_CAPTIONS, фигура)."""
    return [
        ("trajectories", build_trajectory_figure(data, max_points)),
        ("positions", build_position_figure(data, max_points)),
        ("speed", build_speed_figure(data, max_points)),
        ("height", build_height_figure(data, max_points)),
    ]


# ---------------------------------------------------------------- HTML


#: Настройки панели Plotly: без логотипа, адаптивная ширина, кнопка «сохранить PNG».
PLOT_CONFIG = {"responsive": True, "displaylogo": False}

#: Оформление страницы: подписи и текст — HTML, поэтому кириллица здесь не проблема
#: (в отличие от `cv2.putText` в 08b, где HUD пришлось писать латиницей).
PAGE_STYLE = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
     margin:0 auto;max-width:1180px;padding:18px 20px 48px;color:#1f2933;background:#fff}
h1{font-size:23px;margin:0 0 6px} h2{font-size:17px;margin:30px 0 6px}
p.lead{color:#52606d;margin:0 0 14px;font-size:14px}
ul.summary{background:#f5f7fa;border:1px solid #dfe3e8;border-radius:6px;
           padding:10px 14px 10px 30px;font-size:13px;line-height:1.55;margin:0}
ul.summary li{margin:1px 0}
code{background:#eef1f4;border-radius:3px;padding:0 3px;font-size:12px}
"""


def render_html(
    figures: Sequence[tuple[str, object]],
    summary_lines: Sequence[str],
    title: str,
    data_note: str,
    include_plotlyjs: bool = True,
) -> str:
    """Собирает один self-contained HTML: plotly.js один раз + все фигуры подряд."""
    from plotly.offline import get_plotlyjs

    body = [
        "<!DOCTYPE html>",
        '<html lang="ru">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
    ]
    if include_plotlyjs:
        body.append(f'<script type="text/javascript">{get_plotlyjs()}</script>')
    body += [
        f"<style>{PAGE_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>{escape(title)}</h1>",
        f'<p class="lead">{escape(data_note)}</p>',
        '<ul class="summary">',
        *(f"<li>{escape(line)}</li>" for line in summary_lines),
        "</ul>",
    ]
    for key, figure in figures:
        caption = FIGURE_CAPTIONS.get(key, key)
        body.append(f"<h2>{escape(caption)}</h2>")
        body.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=False,
                div_id=f"figure-{key}",
                config=PLOT_CONFIG,
            )
        )
    body += ["</body>", "</html>"]
    return "\n".join(body)


def write_html(
    figures: Sequence[tuple[str, object]],
    out_path: str | Path,
    summary_lines: Sequence[str],
    title: str,
    data_note: str,
) -> dict:
    """Пишет self-contained HTML и возвращает его размер; файл создаётся атомарно."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(figures, summary_lines, title, data_note)
    partial = out_path.with_suffix(out_path.suffix + ".part")
    partial.write_text(html, encoding="utf-8")
    partial.replace(out_path)
    size = out_path.stat().st_size
    return {"path": str(out_path), "bytes": size, "size_mb": size / 2**20, "figures": len(figures)}


# ---------------------------------------------------------------- CLI


def build_report(
    tracking_path: str | Path,
    out_path: str | Path,
    max_points: int = DEFAULT_MAX_POINTS,
    title: str = "Траектории головы и рук (PICO 4 Ultra, XRoboToolkit)",
) -> dict:
    """Полный конвейер: CSV → сводка → 4 фигуры → self-contained HTML."""
    frame = load_tracking(tracking_path)
    data = series_data(frame)
    summary = summarize(data)
    summary_lines = format_summary(summary)
    figures = build_figures(data, max_points)
    note = (
        "Данные: trackingData_*.txt → output/tracking.csv (задача 03d), 5 точек на руку. "
        "Показаны ладони; невалидные точки рук дают разрыв линии. "
        + (
            "Для показа данные прорежены равномерным шагом, вся статистика — по полным строкам CSV."
            if max_points > 0
            else "Показаны все строки CSV без прореживания."
        )
    )
    result = write_html(figures, out_path, summary_lines, title, note)
    result["summary"] = summary
    result["summary_lines"] = summary_lines
    return result


def main() -> None:
    """CLI: строит output/trajectory.html и печатает сводку по данным."""
    parser = argparse.ArgumentParser(
        description="Plotly-графики траекторий из tracking.csv в один self-contained HTML."
    )
    parser.add_argument(
        "--tracking",
        type=Path,
        default=Path("output/tracking.csv"),
        help="CSV из задачи 03d (по умолчанию output/tracking.csv)",
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("output/trajectory.html"), help="куда писать HTML"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help=f"точек на серию при показе, 0 — без прореживания (по умолчанию {DEFAULT_MAX_POINTS})",
    )
    parser.add_argument("--title", default=None, help="заголовок страницы")
    args = parser.parse_args()

    if args.max_points < 0:
        raise SystemExit("--max-points должен быть >= 0 (0 — без прореживания)")

    title = args.title or "Траектории головы и рук (PICO 4 Ultra, XRoboToolkit)"
    result = build_report(args.tracking, args.out, args.max_points, title)
    print(f"{args.tracking} → {result['path']}")
    for line in result["summary_lines"]:
        print(f"  {line}")
    print(
        f"  фигур: {result['figures']}, размер: {result['size_mb']:.2f} МиБ "
        f"({result['bytes']} байт)"
    )


if __name__ == "__main__":
    main()
