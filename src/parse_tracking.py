"""parse_tracking.py — файл трекинга (JSON Lines) → pandas.DataFrame → output/tracking.csv.

Заглушка каркаса (задача 01). Реализация — задачи 03a–03d:
  * 03a — стриминговый ридер JSON Lines + парсинг header;
  * 03b — парсер pose с десятичной запятой (проверка |q| ≈ 1.0);
  * 03c — извлечение Head + Hand (5 точек на руку, флаг isActive);
  * 03d — запись tracking.csv + быстрая статистика.

Запуск (после реализации): python src/parse_tracking.py data/trackingData_*.txt -o output/tracking.csv
"""
from __future__ import annotations


def main() -> None:
    """Точка входа CLI. Появится в задаче 03a."""
    raise NotImplementedError("parse_tracking: реализация — задачи 03a–03d")


if __name__ == "__main__":
    main()
