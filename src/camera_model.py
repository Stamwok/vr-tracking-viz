"""camera_model.py — модель камеры: интринсики, world→camera, проекция 3D → пиксели.

Заглушка каркаса (задача 01). Реализация — задачи 05a–05c:
  * 05a — парсинг интринсиков + обоснование порядка [cx, cy, fx, fy];
  * 05b — world_to_camera(p_world, head_pose, T_head_cam);
  * 05c — project(p_cam, intrinsics) с обрезкой SBS-кадра.

Поза камеры берётся из Head.pose плюс калибруемая константа T_head_cam (задачи 07a–07c).
"""
from __future__ import annotations


def main() -> None:
    """Точка входа (демо/самопроверка). Появится в задаче 05a."""
    raise NotImplementedError("camera_model: реализация — задачи 05a–05c")


if __name__ == "__main__":
    main()
