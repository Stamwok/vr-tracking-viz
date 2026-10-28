"""overlay_video.py — наложение трекинга (голова + руки) на видео с камеры шлема.

Заглушка каркаса (задача 01). Реализация — задачи 08a–08b (после калибровки 07c):
  * 08a — покадровый рендер скелета: точки + линии, цвета по рукам;
  * 08b — след (trail) + HUD.

Запуск (после реализации):
  python src/overlay_video.py --video data/CameraRecord_*.mp4 \\
                              --tracking output/tracking.csv \\
                              --out output/overlay.mp4
"""
from __future__ import annotations


def main() -> None:
    """Точка входа CLI. Появится в задаче 08a."""
    raise NotImplementedError("overlay_video: реализация — задачи 08a–08b")


if __name__ == "__main__":
    main()
