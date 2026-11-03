# CLAUDE.md — архитектура и ограничения

Файл для агентов: только то, что нужно, чтобы безопасно менять код. Наблюдения по данным, история
решений и допущения — в [`docs/OBSERVATIONS.md`](docs/OBSERVATIONS.md), журнал задач — в
[`docs/NOTES.md`](docs/NOTES.md). Историю задач сюда не писать.

## 1. Что это

Тестовое задание: визуализация данных движения VR-устройства (PICO 4 Ultra / XRoboToolkit). Один
подход: 2D-наложение трекинга на видео через OpenCV + интерактивные графики траекторий через Plotly.
Другие подходы не добавлять — решение должно остаться маленьким и законченным.

## 2. Входные данные (факты)

- `data/CameraRecord_*.mp4` — 2160×810, стерео side-by-side (2 × 1080×810), 30 fps, 242.66 с, H.264.
- `data/trackingData_*.txt` — JSON Lines: первая строка header, затем 21 787 сэмплов, ~90 Гц
  (медиана шага 11.12 мс). Все координаты в метрах, система OpenXR: x вправо, y вверх, z назад.
- Ловушка парсинга: десятичная запятая совпадает с разделителем полей (`split(",")` даёт вдвое
  больше токенов). Парсер склеивает пары `<int,frac>` и проверяет `|q| ≈ 1`.
- header содержит `timeStampNs` первого кадра, `cameraIntrinsics = [cx, cy, fx, fy]` полного
  SBS-кадра и `cameraExtrinsics` — две матрицы 4×4 (левый/правый глаз). В статье эти поля не
  описаны; трактовка — вывод из данных.
- Сэмпл: `Head.pose` (`x,y,z,qx,qy,qz,qw`), `Hand.leftHand/rightHand` — 26 джойнтов OpenXR с
  `isActive` и флагами `s`, `Body` — 24 джойнта (берутся только локти 18/19 для указателя на
  локоть). Контроллеров в записи нет.

## 3. Архитектура

```
src/
├── parse_tracking.py  # txt (JSON Lines) → tracking.csv, все 26 джойнтов на руку
├── video_meta.py      # video_meta.json: разрешение, fps, длительность, SBS
├── sync.py            # frame_time(k), pose_at, SLERP, поправка времени кадра
├── camera_model.py    # интринсики, world→camera, проекция 3D→2D
├── config.py          # калибровка: оси, флаги, T_head_cam, VIDEO_TIME_OFFSET_MS
├── calib_frames.py    # эталон калибровки: 5 кадров, рамки кистей
├── calib_bones.py     # костная метрика наложения
├── calib_search.py    # подбор конвенций осей и T_head_cam, проверка --verify-config
├── overlay_video.py   # рендер: скелет, раскраска по глубине, указатель на локоть, HUD
├── drift_diag.py      # диагностика слёта скелета, подбор поправки времени
├── plot_trajectory.py # Plotly: 3D-траектории, x/y/z(t), скорость, высота кистей
└── sync_viewer.py     # видео + 3D-траектории, синхронные с video.currentTime
```

Поток данных: `txt + mp4 → video_meta → parse_tracking → tracking.csv → overlay_video →
overlay.mp4 → plot_trajectory → trajectory.html`. `sync_viewer` — боковая ветвь:
`tracking.csv + overlay.mp4 → sync_viewer.html`.

Ключевые контракты:

- `config.py` — **единственное** место калибровки: оси, `INVERT_QUAT`, `Y_DOWN_FLIP`, `T_head_cam`,
  `VIDEO_TIME_OFFSET_MS`. Код камеры и рендера читает константы оттуда и не хардкодит их;
  `camera_model.check_config()` проверяет согласованность.
- `sync.pose_at_frame()` — **единственный** путь позы для рендера (время кадра + поправка задержки).
  `calib_search` берёт время оттуда же, поэтому `--verify-config` согласован с рендером.
- Рендер читает **сырой** txt, а не `tracking.csv`: нужен `header.timeStampNs`, которого в CSV нет.
  `plot_trajectory` и `sync_viewer` работают по CSV.

## 4. Команды

```bash
python src/video_meta.py data/CameraRecord_20260505_165740.mp4
python src/parse_tracking.py data/trackingData_20260505_165740.txt -o output/tracking.csv
python src/overlay_video.py data/CameraRecord_20260505_165740.mp4 \
                            --tracking data/trackingData_20260505_165740.txt \
                            --out output/overlay.mp4
python src/plot_trajectory.py --tracking output/tracking.csv -o output/trajectory.html
python src/sync_viewer.py --tracking output/tracking.csv -o output/sync_viewer.html
python -m http.server -d output 8000          # sync_viewer.html только через сервер
python -m unittest discover -s tests -t .
python src/calib_search.py data/trackingData_*.txt --verify-config
```

## 5. Ограничения (не нарушать)

- Стек: Python 3.12+, `opencv-python`, `numpy`, `pandas`, `plotly`, stdlib `json`. Новые тяжёлые
  зависимости — только с обоснованием.
- **Никаких** CV-моделей распознавания рук (MediaPipe и т. п.): 3D-координаты уже есть в данных.
- `data/` и `output/` — в `.gitignore`; входные файлы и артефакты не коммитить.
- Коммиты на английском (conventional commits: `feat:`, `fix:`, `docs:`, `chore:`; императив, в теле
  — что и почему).
- Ветки `VTR-<номер>` от `main`. Мерж и пуш делает пользователь; агент не мержит, не пушит и не
  переписывает `main`.
- Калибровка меняется **только** в `config.py` и только с проверкой
  `python src/calib_search.py data/trackingData_*.txt --verify-config`.
- Одна поправка времени — `VIDEO_TIME_OFFSET_MS`. Вторую (отдельно для head и hand) не вводить:
  данные её не подтверждают.
- Точность наложения ограничена кистью: ладонь и запястье ложатся на руку, фаланги сжатых пальцев
  уезжают. Не обещать точность больше измеренной.

## 6. Рабочий процесс

- До `in_review` — запись в `docs/NOTES.md` (формат в конце файла) и execution report; в `done`
  переводит только пользователь.
- Умолчания CLI должны совпадать с README и превью; откат режимов — через `--no-*`; `--help`
  держать актуальным.
- Тесты — `unittest`; ожидания не менять без причины. Упавший тест разбирать, а не подгонять.
- Наблюдения и допущения писать в `docs/OBSERVATIONS.md`, историю задач — в `docs/NOTES.md`.
