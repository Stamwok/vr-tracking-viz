"""sync_viewer.py — видео и 3D-траектории, синхронные по времени (задача 09b).

Из ``output/tracking.csv`` (03d) собираем **один** self-contained HTML
(plotly.js внутри файла, ни CDN, ни интернета): над сценой — видео
``overlay.mp4`` (наложение 08a/08b), под ним — 3D-траектории головы и ладоней
обеих рук, которые **дорисовываются по мере проигрывания**:

* в ``Plotly.newPlot`` один раз уходят полные (прореженные) трассы;
* при проигрывании ``Plotly.restyle`` оставляет видимый префикс до
  ``video.currentTime`` (поиск индекса — бисекцией по времени, не чаще 10 Гц
  через ``requestAnimationFrame``);
* маркер «сейчас» — отдельная ``scatter3d``-трасса, которая переставляется на
  текущий сэмпл (``layout.shapes`` в 3D-сценах Plotly не работает: он рисует
  фигуру в координатах бумаги/2D-осей, а не в координатах сцены; проверено
  скриншотом — линия ложится поверх сцены и не следует за 3D-точкой);
* метки «старт»/«финиш» — как в 09 (по первой/последней валидной точке серии);
* клик по точке сцены перематывает видео на её время, hover показывает время и
  координаты.

Данные для показа прореживаются до ``--max-points`` (по умолчанию 2000 точек на
серию). Концы каждой серии и пик скорости сохраняются принудительно — той же
функцией ``downsample_indices`` из 09, поэтому hover на пике совпадает со сводкой.
Разрывы «нет данных» (NaN у правой руки в первые ~0.8 с) не затягиваются: каждая
непрерывная серия — отдельная трасса.

Видео и трекинг имеют общее начало (``header.timeStampNs`` — первый кадр),
поэтому ``t_rel_s`` из 03d — сразу видео-время. Видео на 119 мс длиннее трекинга:
после последнего сэмпла маркер замирает на последней точке (как «заморозка» позы
в 08a).

Открывать только через сервер: браузеры блокируют загрузку ``<video>`` с
``file://``. Команда: ``python -m http.server -d output 8000`` →
``http://localhost:8000/sync_viewer.html``. ⚠️ ``http.server`` не поддерживает
Range-запросы, поэтому Chrome считает видео неперематываемым (``seekable = [0, 0]``)
даже при полном буфере — клик по сцене тогда перемотать не может. Вьюер это
замечает и один раз загружает файл в память (``blob:``), после чего перемотка
работает; на сервере с поддержкой Range этого не происходит. Проверено в
headless Chrome: на ``http.server`` ``seekable`` пустой, после подмены на blob —
полный, клик и ``?t=`` работают.

Запуск:
    python src/sync_viewer.py --tracking output/tracking.csv -o output/sync_viewer.html
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import plot_trajectory as _pt  # noqa: E402
from src.plot_trajectory import SERIES  # noqa: E402

#: Сколько точек на серию оставлять в сцене (0 — не прореживать). Полные данные — 21 787.
DEFAULT_MAX_POINTS = 2000

#: Видео по умолчанию — рядом с HTML (лежит в ``output/``, как и результат рендера 08a/08b).
DEFAULT_VIDEO = "overlay.mp4"

#: Заголовок страницы по умолчанию.
DEFAULT_TITLE = "Синхронный просмотр: видео + 3D-траектории (PICO 4 Ultra)"

#: Знаков после запятой в JSON: 1e-5 м = 10 мкм (пиксель на дистанции руки ≈0.6 мм),
#: 1e-4 с = 0.1 мс (кадр видео — 33 мс). Точность с большим запасом, размер — вдвое меньше.
VALUE_DECIMALS = 5
TIME_DECIMALS = 4

#: Ограничение на размер встроенного JSON для теста и отчёта.
MAX_PAYLOAD_BYTES = 300 * 1024


# ---------------------------------------------------------------- данные


def load_tracking(path: str | Path):
    """Читает tracking.csv (03d): те же колонки и проверки, что в 09, плюс числовой тип.

    ``plot_trajectory.load_tracking`` уже проверяет наличие файла, нужных колонок и
    непустоту. Здесь добавлена проверка, что значения действительно числа: CSV с
    текстом в числовой колонке иначе упал бы позже, при приведении к float, с
    невнятным сообщением.
    """
    frame = _pt.load_tracking(path)
    import pandas as pd

    bad = [
        column
        for column in _pt.REQUIRED_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if bad:
        raise ValueError(
            f"{path}: нечисловые значения в колонках {', '.join(bad)} — "
            "это не tracking.csv из задачи 03d"
        )
    return frame


def _series_arrays(frame, series):
    """x/y/z канала из DataFrame (float, NaN = «нет данных»)."""
    arrays = tuple(
        frame[f"{series.prefix}_{axis}"].to_numpy(dtype=float) for axis in ("x", "y", "z")
    )
    return arrays


def frame_indices(frame, max_points: int = DEFAULT_MAX_POINTS) -> np.ndarray:
    """Индексы строк CSV для сцены: равномерный шаг + концы и пик скорости каждой серии.

    Шаг — из ``downsample_indices`` (09): ``ceil(rows / max_points)``, первый и
    последний индексы сохраняются. Дополнительно принудительно оставляем первую и
    последнюю **валидную** точку каждой серии (у правой руки начало — NaN, без этого
    её линия начиналась бы на ~12 сэмплов позже) и индекс её пика скорости, чтобы
    hover на максимуме совпадал со сводкой, как в 09.
    """
    t = frame["t_rel_s"].to_numpy(dtype=float)
    count = int(t.size)
    must: set[int] = {0, count - 1} if count else set()
    for series in SERIES:
        x, y, z = _series_arrays(frame, series)
        present = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if present.any():
            finite = np.flatnonzero(present)
            must.add(int(finite[0]))
            must.add(int(finite[-1]))
            speed = _pt.speeds(t, x, y, z)
            if np.isfinite(speed).any():
                must.add(int(np.argmax(np.where(np.isfinite(speed), speed, -np.inf))))
    return _pt.downsample_indices(count, max_points, sorted(index for index in must if 0 <= index < count))


def _rounded(values: Iterable[float], decimals: int) -> list[float | None]:
    """Округлённые значения с ``None`` вместо NaN/inf: JSON не знает NaN без потери валидности."""
    out: list[float | None] = []
    for value in values:
        number = float(value)
        out.append(None if not math.isfinite(number) else round(number, decimals))
    return out


def viewer_payload(frame, max_points: int = DEFAULT_MAX_POINTS) -> dict:
    """Данные сцены в виде JSON-совместимого словаря (встраивается в HTML).

    Схема::

        {"t": [...], "t_rel_s": [...],
         "series": {"head"|"left"|"right":
                    {"label", "color", "x", "y", "z", "segments", "valid", "nan"}},
         "stats": {"rows", "duration_s", "points", "step", "max_points"}}

    ``t`` — видео-время (то же, что ``t_rel_s``; второе имя оставлено явно, чтобы
    JSON был самодокументирован по колонке 03d). ``segments`` — пары
    ``[начало, конец)`` непрерывных валидных участков: так разрыв на «нет данных»
    остаётся разрывом, а не линией через дырку. ``valid``/``nan`` считаются по
    **полным** строкам CSV, а не по прореженной выборке.
    """
    t = frame["t_rel_s"].to_numpy(dtype=float)
    indices = frame_indices(frame, max_points)
    times = _rounded(t[indices], TIME_DECIMALS)
    rows = int(t.size)
    if max_points > 0 and rows > max_points:
        step = math.ceil(rows / max_points)
    else:
        step = 1
    payload: dict = {
        "t": times,
        "t_rel_s": times,
        "series": {},
        "stats": {
            "rows": rows,
            "duration_s": round(float(t[-1] - t[0]), 3) if rows > 1 else 0.0,
            "points": int(indices.size),
            "step": int(step),
            "max_points": int(max_points),
        },
    }
    for series in SERIES:
        x, y, z = _series_arrays(frame, series)
        present = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        segments = [
            [int(segment.start), int(segment.stop)]
            for segment in _pt.valid_segments(x[indices], y[indices], z[indices])
        ]
        payload["series"][series.key] = {
            "label": series.label,
            "color": series.color,
            "x": _rounded(x[indices], VALUE_DECIMALS),
            "y": _rounded(y[indices], VALUE_DECIMALS),
            "z": _rounded(z[indices], VALUE_DECIMALS),
            "segments": segments,
            "valid": int(present.sum()),
            "nan": int(rows - present.sum()),
        }
    return payload


def payload_json(payload: dict) -> str:
    """JSON без NaN (``allow_nan=False``) и с экранированием ``</`` для вставки в ``<script>``."""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return text.replace("</", "<\\/")


def viewer_layout(payload: dict) -> dict:
    """Раскладка Plotly: подписи осей в метрах и **фиксированные** диапазоны.

    Диапазоны считаются по полным данным, поэтому оси не «дёргаются», пока
    траектория дорисовывается. ``aspectmode="data"`` (как в 09) сохраняет честные
    пропорции, потому что диапазоны в метрах.
    """
    spans = {axis: [math.inf, -math.inf] for axis in ("x", "y", "z")}
    for series in payload["series"].values():
        for axis in ("x", "y", "z"):
            values = [value for value in series[axis] if value is not None]
            if values:
                spans[axis][0] = min(spans[axis][0], min(values))
                spans[axis][1] = max(spans[axis][1], max(values))
    scene: dict = {
        "xaxis_title": "x, м",
        "yaxis_title": "y, м",
        "zaxis_title": "z, м",
        "aspectmode": "data",
        "camera": {"eye": {"x": 1.4, "y": 1.2, "z": 1.1}},
    }
    for axis in ("x", "y", "z"):
        low, high = spans[axis]
        if not (math.isfinite(low) and math.isfinite(high)):
            scene[f"{axis}axis"] = {"title": f"{axis}, м"}
            continue
        pad = 0.06 * (high - low) if high > low else 0.05
        scene[f"{axis}axis"] = {
            "title": f"{axis}, м",
            "range": [round(low - pad, VALUE_DECIMALS), round(high + pad, VALUE_DECIMALS)],
        }
    return {
        "margin": {"l": 0, "r": 0, "b": 0, "t": 0},
        "height": 700,
        "legend": {
            "orientation": "h",
            "y": 1.0,
            "x": 0,
            "bgcolor": "rgba(255,255,255,0.85)",
        },
        "scene": scene,
    }


# ---------------------------------------------------------------- HTML


#: Настройки панели Plotly: без логотипа, адаптивная ширина, кнопка «сохранить PNG».
PLOT_CONFIG = {"responsive": True, "displaylogo": False}

#: Оформление страницы (кириллица — это HTML, в отличие от `cv2.putText` в 08b).
PAGE_STYLE = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
     margin:0 auto;max-width:1180px;padding:18px 20px 48px;color:#1f2933;background:#fff}
h1{font-size:23px;margin:0 0 6px} h2{font-size:17px;margin:30px 0 6px}
p.lead{color:#52606d;margin:0 0 12px;font-size:14px}
p.status{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;
         color:#334e68;background:#f5f7fa;border:1px solid #dfe3e8;border-radius:6px;
         padding:6px 10px;margin:8px 0 0}
video{width:100%;max-width:1080px;display:block;margin:0 auto;background:#000;
      border-radius:6px}
#scene{width:100%;height:700px;margin-top:8px}
ul.summary{background:#f5f7fa;border:1px solid #dfe3e8;border-radius:6px;
           padding:10px 14px 10px 30px;font-size:13px;line-height:1.55;margin:0}
ul.summary li{margin:1px 0}
ul.hint{font-size:13px;line-height:1.6;margin:0;padding-left:20px;color:#334e68}
code{background:#eef1f4;border-radius:3px;padding:0 3px;font-size:12px}
"""

#: JS вьюера. Данные и раскладка лежат в отдельных `<script type="application/json">`,
#: поэтому шаблон не требует подстановки (и не ломается от фигурных скобок).
APP_JS = r"""
(function () {
  'use strict';

  var payload = JSON.parse(document.getElementById('viewer-data').textContent);
  var layout = JSON.parse(document.getElementById('viewer-layout').textContent);
  var config = {responsive: true, displaylogo: false};
  var video = document.getElementById('video');
  var statusEl = document.getElementById('status');
  var plotEl = document.getElementById('scene');
  var keys = Object.keys(payload.series);
  var times = payload.t;

  var pathTraces = [];
  var nowIndex = -1;
  var lastIndex = -2;
  var lastPaint = 0;
  var rafId = 0;

  // Первый индекс со временем > target (бисекция по возрастающему t).
  function bisectRight(values, target) {
    var low = 0;
    var high = values.length;
    while (low < high) {
      var middle = (low + high) >> 1;
      if (values[middle] <= target) { low = middle + 1; }
      else { high = middle; }
    }
    return low;
  }

  // Сэмпл, который был последним к текущему времени видео; -1 до первого.
  function currentIndex() {
    if (!isFinite(video.currentTime)) { return -1; }
    return Math.max(-1, bisectRight(times, video.currentTime) - 1);
  }

  function buildTraces() {
    var traces = [];
    pathTraces = [];
    keys.forEach(function (key) {
      var series = payload.series[key];
      series.segments.forEach(function (segment, number) {
        var start = segment[0];
        var stop = segment[1];
        var xs = series.x.slice(start, stop);
        var ys = series.y.slice(start, stop);
        var zs = series.z.slice(start, stop);
        var ts = times.slice(start, stop);
        pathTraces.push({index: traces.length, start: start, stop: stop,
                         x: xs, y: ys, z: zs, times: ts});
        traces.push({
          type: 'scatter3d', mode: 'lines',
          x: xs, y: ys, z: zs, customdata: ts,
          name: series.label, legendgroup: key, showlegend: number === 0,
          line: {color: series.color, width: 3},
          hovertemplate: '<b>' + series.label + '</b><br>t = %{customdata:.2f} с' +
                         '<br>x = %{x:.3f} м, y = %{y:.3f} м, z = %{z:.3f} м<extra></extra>',
          meta: {kind: 'path', series: key}
        });
      });
      if (series.segments.length) {
        var first = series.segments[0];
        var lastSegment = series.segments[series.segments.length - 1];
        [['старт', first[0], 'diamond-open'], ['финиш', lastSegment[1] - 1, 'x']]
          .forEach(function (item) {
            var position = item[1];
            traces.push({
              type: 'scatter3d', mode: 'markers+text',
              x: [series.x[position]], y: [series.y[position]], z: [series.z[position]],
              customdata: [times[position]],
              text: [item[0]], textposition: 'top center',
              textfont: {size: 12, color: series.color},
              marker: {size: 7, color: series.color, symbol: item[2],
                       line: {color: '#ffffff', width: 1}},
              name: series.label + ': ' + item[0],
              legendgroup: key, showlegend: false,
              hovertemplate: series.label + ' — ' + item[0] +
                             ', t = %{customdata:.2f} с' +
                             '<br>x = %{x:.3f} м, y = %{y:.3f} м, z = %{z:.3f} м<extra></extra>'
            });
          });
      }
    });

    // Маркер «сейчас»: одна трасса, по точке на канал (x/y/z = null, пока данных нет).
    var nowX = [];
    var nowY = [];
    var nowZ = [];
    var nowT = [];
    var nowText = [];
    var nowColors = [];
    keys.forEach(function (key) {
      var series = payload.series[key];
      nowX.push(null);
      nowY.push(null);
      nowZ.push(null);
      nowT.push(null);
      nowText.push(series.label);
      nowColors.push(series.color);
    });
    traces.push({
      type: 'scatter3d', mode: 'markers',
      x: nowX, y: nowY, z: nowZ, customdata: nowT, text: nowText,
      marker: {size: 12, color: nowColors, symbol: 'circle',
               line: {color: '#ffffff', width: 2}},
      name: 'сейчас', showlegend: false,
      hovertemplate: '<b>%{text}</b> · сейчас<br>t = %{customdata:.2f} с' +
                     '<br>x = %{x:.3f} м, y = %{y:.3f} м, z = %{z:.3f} м<extra></extra>',
      meta: {kind: 'now'}
    });
    nowIndex = traces.length - 1;
    return traces;
  }

  // Видимый префикс полных трасс: restyle (newPlot делается один раз при загрузке).
  function update(force) {
    if (nowIndex < 0) { return; }
    var index = currentIndex();
    if (!force && index === lastIndex) { return; }
    lastIndex = index;

    var restyle = {x: [], y: [], z: [], customdata: []};
    var indices = [];
    pathTraces.forEach(function (trace) {
      var visible = Math.max(0, Math.min(index - trace.start + 1, trace.stop - trace.start));
      restyle.x.push(trace.x.slice(0, visible));
      restyle.y.push(trace.y.slice(0, visible));
      restyle.z.push(trace.z.slice(0, visible));
      restyle.customdata.push(trace.times.slice(0, visible));
      indices.push(trace.index);
    });

    var nowX = [];
    var nowY = [];
    var nowZ = [];
    var nowT = [];
    keys.forEach(function (key) {
      var series = payload.series[key];
      var present = index >= 0 && index < times.length &&
                    series.x[index] !== null && series.x[index] !== undefined;
      nowX.push(present ? series.x[index] : null);
      nowY.push(present ? series.y[index] : null);
      nowZ.push(present ? series.z[index] : null);
      nowT.push(present ? times[index] : null);
    });
    restyle.x.push(nowX);
    restyle.y.push(nowY);
    restyle.z.push(nowZ);
    restyle.customdata.push(nowT);
    indices.push(nowIndex);

    Plotly.restyle(plotEl, restyle, indices);
    setStatus(index);
  }

  function setStatus(index) {
    var current = video.currentTime;
    if (!isFinite(current)) {
      statusEl.textContent = 'видео загружается…';
      return;
    }
    var text = 't = ' + current.toFixed(2) + ' с';
    if (index < 0) {
      text += ' — трекинг ещё не начался';
    } else {
      text += ' · точка ' + (index + 1) + ' из ' + times.length;
      if (index >= times.length - 1 && current > times[times.length - 1] + 0.05) {
        text += ' · трекинг закончился, поза заморожена';
      }
    }
    statusEl.textContent = text;
  }

  function isSeekable() {
    try {
      var last = video.seekable.length - 1;
      return last >= 0 && video.seekable.end(last) > 0;
    } catch (error) {
      return false;
    }
  }

  // Цель перемотки, которую ещё не удалось применить: по http.server видео неперематываемо
  // (сервер не поддерживает Range), и currentTime в этом состоянии не меняется.
  var pendingSeek = null;
  var deepLinkApplied = false;
  var memoryFallback = false;

  function seekTo(seconds) {
    var limit = isFinite(video.duration) ? video.duration : seconds;
    var target = Math.max(0, Math.min(seconds, limit));
    pendingSeek = target;
    video.currentTime = target;
    if (isSeekable()) { pendingSeek = null; }
    update(true);
  }

  // python -m http.server не отвечает на Range-запросы, поэтому Chrome считает видео
  // неперематываемым (seekable = [0, 0]), хотя файл уже в буфере. Тогда один раз грузим
  // видео в память и играем из blob: — перемотка и клик по сцене начинают работать.
  // На сервере с поддержкой Range (nginx, большинство хостингов) этого не происходит.
  function loadIntoMemory() {
    if (memoryFallback) { return; }
    memoryFallback = true;
    var source = video.currentSrc || video.src;
    if (location.protocol === 'file:') {
      statusEl.textContent = 'по file:// видео не загружается — запустите python -m http.server ' +
        '(см. подсказку ниже)';
      return;
    }
    if (!source || source.indexOf('blob:') === 0) { return; }
    var position = video.currentTime;
    var wasPlaying = !video.paused && !video.ended;
    // Однопоточный http.server обслуживает один запрос за раз: пока медиа-поток держит
    // соединение, fetch ждёт в очереди. Отпускаем поток (позиция запоминается) — сервер
    // освобождается, и видео приезжает одним blob'ом.
    video.pause();
    video.removeAttribute('src');
    video.load();
    statusEl.textContent = 'сервер без Range: загружаю видео в память для перемотки…';
    fetch(source).then(function (response) {
      if (!response.ok) { throw new Error('HTTP ' + response.status); }
      return response.blob();
    }).then(function (blob) {
      var url = URL.createObjectURL(blob);
      video.addEventListener('loadedmetadata', function () {
        if (pendingSeek !== null) {
          video.currentTime = pendingSeek;
          pendingSeek = null;
        } else if (position > 0) {
          video.currentTime = Math.min(position, video.duration || position);
        }
        if (wasPlaying) { video.play(); }
        update(true);
      }, {once: true});
      video.src = url;
      video.load();
    }).catch(function (error) {
      statusEl.textContent = 'не удалось загрузить видео в память (' + error.message +
        ') — перемотка на этом сервере недоступна';
    });
  }

  function maybeMemoryFallback(attempt) {
    if (memoryFallback || isSeekable()) { return; }
    var tries = attempt || 0;
    var buffered = video.buffered.length
      ? video.buffered.end(video.buffered.length - 1)
      : 0;
    // Пока видео ещё качается, у Range-сервера seekable появится позже — не спешим.
    if (video.readyState >= 1 && video.networkState !== 2 && buffered > 0) {
      loadIntoMemory();
      return;
    }
    if (tries < 20) {
      setTimeout(function () { maybeMemoryFallback(tries + 1); }, 500);
    }
  }

  function seekFromPoint(point) {
    if (!point) { return; }
    var value = point.customdata;
    if (Array.isArray(value)) { value = value[0]; }
    var seconds = Number(value);
    if (isFinite(seconds)) { seekTo(seconds); }
  }

  // Троттлинг 10 Гц поверх requestAnimationFrame: обновляем не чаще, чем раз в 100 мс.
  function playbackLoop(stamp) {
    rafId = 0;
    if (video.paused || video.ended) {
      update(true);
      return;
    }
    if (stamp - lastPaint >= 100) {
      lastPaint = stamp;
      update(false);
    }
    rafId = window.requestAnimationFrame(playbackLoop);
  }

  function startLoop() {
    if (!rafId) { rafId = window.requestAnimationFrame(playbackLoop); }
  }

  function deepLinkTime() {
    try {
      var value = parseFloat(new URLSearchParams(window.location.search).get('t'));
      return isFinite(value) ? value : null;
    } catch (error) {
      return null;
    }
  }

  function applyDeepLink() {
    if (!deepLinkApplied) {
      deepLinkApplied = true;
      var value = deepLinkTime();
      if (value !== null) { pendingSeek = value; }
    }
    if (pendingSeek !== null && video.readyState >= 1 && isSeekable()) {
      seekTo(pendingSeek);
    } else {
      update(true);
    }
  }

  video.addEventListener('play', startLoop);
  video.addEventListener('pause', function () { update(true); });
  video.addEventListener('seeked', function () { update(true); });
  video.addEventListener('timeupdate', function () {
    if (video.paused || video.ended) { update(false); }
  });
  video.addEventListener('loadedmetadata', function () {
    applyDeepLink();
    setTimeout(function () { maybeMemoryFallback(0); }, 1500);
  });
  video.addEventListener('canplay', function () { maybeMemoryFallback(0); });
  video.addEventListener('suspend', function () { maybeMemoryFallback(0); });

  // gd.on() появляется только после newPlot — вешаем обработчик на готовый graph div.
  function bindPlot(graphDiv) {
    graphDiv.on('plotly_click', function (event) {
      if (event && event.points && event.points.length) {
        seekFromPoint(event.points[0]);
      }
    });
    applyDeepLink();
    maybeMemoryFallback(0);
    setTimeout(function () { maybeMemoryFallback(0); }, 500);
  }

  var setup = Plotly.newPlot(plotEl, buildTraces(), layout, config);
  if (setup && typeof setup.then === 'function') {
    setup.then(bindPlot, applyDeepLink);
  } else {
    bindPlot(plotEl);
  }
})();
"""


def format_summary(payload: dict) -> list[str]:
    """Строки сводки для страницы и CLI."""
    stats = payload["stats"]
    lines = [
        f"строк CSV: {stats['rows']}, длительность трекинга: {stats['duration_s']:.3f} с; "
        f"в сцене {stats['points']} точек (шаг {stats['step']} по исходным строкам)",
    ]
    for series in payload["series"].values():
        lines.append(
            f"{series['label']}: валидных {series['valid']}, NaN {series['nan']}, "
            f"непрерывных участков {len(series['segments'])}"
        )
    return lines


def build_html(
    payload: dict,
    title: str = DEFAULT_TITLE,
    video: str = DEFAULT_VIDEO,
) -> str:
    """Self-contained HTML: видео + Plotly 3D-сцена + данные и логика вьюера."""
    from plotly.offline import get_plotlyjs

    data_json = payload_json(payload)
    layout_json = payload_json(viewer_layout(payload))
    stats = payload["stats"]
    limit = (
        f"до {stats['max_points']} точек на серию"
        if stats["max_points"]
        else "без прореживания"
    )
    note = (
        f"Видео {video} с наложением скелета и следом кисти (08a/08b) и 3D-сцена по данным "
        f"output/tracking.csv (03d, 5 точек на руку). Для показа траектории прорежены {limit}: "
        "концы и пик скорости сохранены, разрывы «нет данных» не затягиваются."
    )
    body = [
        "<!DOCTYPE html>",
        '<html lang="ru">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
        f"<style>{PAGE_STYLE}</style>",
        "</head>",
        "<body>",
        f"<h1>{escape(title)}</h1>",
        f'<p class="lead">{escape(note)}</p>',
        f'<video id="video" src="{escape(video, quote=True)}" controls preload="metadata" '
        'playsinline></video>',
        '<p class="status" id="status">видео загружается…</p>',
        '<div id="scene"></div>',
        '<h2>Сводка</h2>',
        '<ul class="summary">',
        *(f"<li>{escape(line)}</li>" for line in format_summary(payload)),
        "</ul>",
        "<h2>Как смотреть</h2>",
        '<ul class="hint">',
        "<li>Нажмите ▶ в плеере: траектории головы и ладоней дорисовываются до текущего "
        "времени видео (обновление 10 раз в секунду).</li>",
        "<li>Клик по любой точке 3D-сцены перематывает видео на её время; hover показывает "
        "время и координаты в метрах. Если сервер не поддерживает Range (например, "
        "<code>python -m http.server</code>), видео один раз догружается в память — иначе "
        "браузер запрещает перемотку.</li>",
        "<li>Открывать через сервер: <code>python -m http.server -d output 8000</code> → "
        "<code>http://localhost:8000/sync_viewer.html</code> — по <code>file://</code> "
        "браузер блокирует загрузку видео.</li>",
        "<li>Можно открыть сразу на нужной секунде: <code>sync_viewer.html?t=60</code>.</li>",
        "</ul>",
        f'<script type="application/json" id="viewer-data">{data_json}</script>',
        f'<script type="application/json" id="viewer-layout">{layout_json}</script>',
        f'<script type="text/javascript">{get_plotlyjs()}</script>',
        f"<script type=\"text/javascript\">{APP_JS}</script>",
        "</body>",
        "</html>",
    ]
    return "\n".join(body)


def write_viewer(
    payload: dict,
    out_path: str | Path,
    title: str = DEFAULT_TITLE,
    video: str = DEFAULT_VIDEO,
) -> dict:
    """Пишет HTML атомарно (``.part`` → ``replace``) и возвращает размер и число точек."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(payload, title, video)
    partial = out_path.with_suffix(out_path.suffix + ".part")
    partial.write_text(html, encoding="utf-8")
    partial.replace(out_path)
    size = out_path.stat().st_size
    return {
        "path": str(out_path),
        "bytes": size,
        "size_mb": size / 2**20,
        "points": payload["stats"]["points"],
        "video": video,
    }


def build_viewer(
    tracking_path: str | Path,
    out_path: str | Path,
    max_points: int = DEFAULT_MAX_POINTS,
    video: str = DEFAULT_VIDEO,
    title: str = DEFAULT_TITLE,
) -> dict:
    """Полный конвейер: CSV → прореженный JSON → self-contained HTML."""
    frame = load_tracking(tracking_path)
    payload = viewer_payload(frame, max_points)
    result = write_viewer(payload, out_path, title, video)
    result["rows"] = payload["stats"]["rows"]
    result["summary_lines"] = format_summary(payload)
    return result


# ---------------------------------------------------------------- CLI


def main() -> None:
    """CLI: строит output/sync_viewer.html и печатает сводку."""
    parser = argparse.ArgumentParser(
        description="Видео + Plotly 3D-траектории, синхронные с video.currentTime, в одном HTML."
    )
    parser.add_argument(
        "--tracking",
        type=Path,
        default=Path("output/tracking.csv"),
        help="CSV из задачи 03d (по умолчанию output/tracking.csv)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("output/sync_viewer.html"),
        help="куда писать HTML (по умолчанию output/sync_viewer.html)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=DEFAULT_MAX_POINTS,
        help=f"точек на серию в сцене, 0 — без прореживания (по умолчанию {DEFAULT_MAX_POINTS})",
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO,
        help=f"путь к видео относительно HTML (по умолчанию {DEFAULT_VIDEO})",
    )
    parser.add_argument("--title", default=None, help="заголовок страницы")
    args = parser.parse_args()

    if args.max_points < 0:
        raise SystemExit("--max-points должен быть >= 0 (0 — без прореживания)")
    if args.max_points == 1:
        raise SystemExit("--max-points: 0 (без прореживания) или >= 2")
    if not args.video.strip():
        raise SystemExit("--video не должен быть пустым")

    title = args.title or DEFAULT_TITLE
    result = build_viewer(args.tracking, args.out, args.max_points, args.video, title)
    print(f"{args.tracking} → {result['path']}")
    for line in result["summary_lines"]:
        print(f"  {line}")
    print(
        f"  размер: {result['size_mb']:.2f} МиБ ({result['bytes']} байт), "
        f"видео: {result['video']}"
    )
    print(
        "  открывать через сервер: python -m http.server -d "
        f"{Path(result['path']).parent} 8000 → "
        f"http://localhost:8000/{Path(result['path']).name}"
    )


if __name__ == "__main__":
    main()
