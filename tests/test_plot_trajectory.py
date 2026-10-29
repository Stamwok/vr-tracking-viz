"""Assert-тесты plot_trajectory (задача 09).

Проверяем слой, который можно проверить без браузера: скорость/путь/сегменты, прореживание,
сводку по данным, состав фигур и то, что HTML действительно self-contained (plotly.js внутри,
ни CDN, ни сервера). Тесты на реальном файле пропускаются, если output/tracking.csv отсутствует.

Запуск: python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import plot_trajectory as pt  # noqa: E402

TRACKING_CSV = ROOT / "output" / "tracking.csv"
NO_DATA = "нет output/tracking.csv — артефакт 03d не коммитится (см. README)"


def make_frame(rows: int = 5) -> "object":
    """Синтетический tracking.csv: голова в покое, левая ладонь едет по x со скоростью 1 м/с."""
    import pandas as pd

    t = np.arange(rows, dtype=float) * 0.1
    columns: dict[str, np.ndarray] = {"t_rel_s": t}
    for prefix in ("head", "left_palm", "right_palm"):
        columns[f"{prefix}_x"] = np.zeros(rows)
        columns[f"{prefix}_y"] = np.zeros(rows)
        columns[f"{prefix}_z"] = np.zeros(rows)
    columns["left_palm_x"] = t.copy()  # 1 м/с по x
    columns["left_palm_y"] = -0.2 * np.ones(rows)  # 20 см ниже головы
    return pd.DataFrame(columns)


class SpeedsTest(unittest.TestCase):
    def test_constant_motion_gives_exact_speed(self) -> None:
        t = np.arange(6) * 0.1
        speed = pt.speeds(t, t, np.zeros(6), np.zeros(6))
        self.assertTrue(np.allclose(speed, 1.0))

    def test_own_nan_breaks_the_chain(self) -> None:
        t = np.arange(5) * 0.1
        x = np.array([0.0, 1.0, np.nan, 2.0, 3.0])
        speed = pt.speeds(t, x, np.zeros(5), np.zeros(5))
        self.assertTrue(np.isfinite(speed[0]))
        self.assertTrue(np.isnan(speed[1]))  # правый сосед — дырка
        self.assertTrue(np.isnan(speed[2]))  # сама точка невалидна
        self.assertTrue(np.isnan(speed[3]))  # левый сосед — дырка
        self.assertTrue(np.isfinite(speed[4]))

    def test_zero_dt_is_not_a_division_by_zero(self) -> None:
        t = np.array([0.0, 0.0, 0.1, 0.2])
        speed = pt.speeds(t, np.arange(4.0), np.zeros(4), np.zeros(4))
        self.assertTrue(np.isnan(speed[0]))  # hi = 1, dt = 0
        self.assertTrue(np.isfinite(speed[3]))

    def test_single_sample_and_edges(self) -> None:
        speed = pt.speeds(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
        self.assertEqual(speed.shape, (1,))
        self.assertTrue(np.isnan(speed[0]))

    def test_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            pt.speeds(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(4))


class PathLengthTest(unittest.TestCase):
    def test_straight_line(self) -> None:
        self.assertAlmostEqual(pt.path_length(np.array([0.0, 1.0, 2.0]), np.zeros(3), np.zeros(3)), 2.0)

    def test_nan_gap_is_not_bridged(self) -> None:
        x = np.array([0.0, np.nan, 1.0])
        self.assertEqual(pt.path_length(x, np.zeros(3), np.zeros(3)), 0.0)

    def test_three_axes(self) -> None:
        x, y, z = np.array([0.0, 0.0]), np.array([0.0, 3.0]), np.array([0.0, 4.0])
        self.assertAlmostEqual(pt.path_length(x, y, z), 5.0)


class ValidSegmentsTest(unittest.TestCase):
    def test_gaps_split_the_line(self) -> None:
        values = np.array([np.nan, 1.0, 2.0, np.nan, 3.0])
        segments = pt.valid_segments(values)
        self.assertEqual(segments, [slice(1, 3), slice(4, 5)])

    def test_all_valid_and_all_nan(self) -> None:
        self.assertEqual(pt.valid_segments(np.ones(4)), [slice(0, 4)])
        self.assertEqual(pt.valid_segments(np.full(4, np.nan)), [])

    def test_requires_all_arrays(self) -> None:
        x = np.array([1.0, np.nan, 1.0])
        y = np.array([1.0, 1.0, np.nan])
        self.assertEqual(pt.valid_segments(x, y), [slice(0, 1)])


class DownsampleTest(unittest.TestCase):
    def test_stride_keeps_ends_and_stays_small(self) -> None:
        indices = pt.downsample_indices(100, 10)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)
        self.assertLessEqual(indices.size, 11)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_must_include_is_kept(self) -> None:
        self.assertIn(55, pt.downsample_indices(100, 10, must_include=[55, 1000]))

    def test_small_input_is_returned_whole(self) -> None:
        self.assertEqual(pt.downsample_indices(7, 10).tolist(), list(range(7)))
        self.assertEqual(pt.downsample_indices(7, 0).tolist(), list(range(7)))

    def test_edge_budgets(self) -> None:
        self.assertEqual(pt.downsample_indices(0, 10).size, 0)
        with self.assertRaises(ValueError):
            pt.downsample_indices(10, 1)
        with self.assertRaises(ValueError):
            pt.downsample_indices(-1, 10)


class SeriesDataTest(unittest.TestCase):
    def frame(self) -> "object":
        frame = make_frame(5)
        frame.loc[2, "right_palm_x"] = np.nan  # неактивный сустав — «нет данных»
        return frame

    def test_series_and_columns(self) -> None:
        data = pt.series_data(self.frame())
        self.assertEqual([row.series.key for row in data], ["head", "left", "right"])
        self.assertTrue(np.allclose(data[1].x, np.arange(5) * 0.1))
        self.assertTrue(np.allclose(data[1].y, -0.2))
        self.assertTrue(np.isnan(data[2].x[2]))

    def test_relative_height_is_hand_minus_head(self) -> None:
        data = pt.series_data(self.frame())
        self.assertIsNone(data[0].rel_height)  # у головы «относительно себя» нет смысла
        self.assertTrue(np.allclose(data[1].rel_height, -0.2))
        self.assertTrue(np.isnan(data[2].rel_height[2]))

    def test_summary_ranges_and_speed(self) -> None:
        summary = pt.summarize(pt.series_data(self.frame()))
        self.assertEqual(summary["rows"], 5)
        self.assertAlmostEqual(summary["duration_s"], 0.4)
        self.assertAlmostEqual(summary["median_dt_ms"], 100.0)
        left = summary["series"]["left"]
        self.assertEqual(left["nan"], 0)
        self.assertEqual(left["x"], (0.0, 0.4))
        self.assertAlmostEqual(left["speed"]["max"], 1.0)
        self.assertAlmostEqual(left["path_m"], 0.4)
        right = summary["series"]["right"]
        self.assertEqual(right["nan"], 1)
        self.assertEqual(right["points"], 4)
        self.assertIsNone(summary["series"]["head"]["rel_height"])

    def test_summary_peak_time(self) -> None:
        t = np.arange(5, dtype=float) * 0.1
        speed = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        row = pt.SeriesData(pt.SERIES[0], t, np.zeros(5), np.zeros(5), np.zeros(5), speed)
        summary = pt.summarize([row])
        self.assertAlmostEqual(summary["series"]["head"]["speed"]["max"], 3.0)
        self.assertAlmostEqual(summary["series"]["head"]["speed"]["max_t_s"], 0.2)


class FigureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frame = make_frame(40)
        cls.data = pt.series_data(frame)
        cls.figures = pt.build_figures(cls.data, max_points=20)

    def test_four_blocks_in_order(self) -> None:
        self.assertEqual([key for key, _ in self.figures], list(pt.FIGURE_CAPTIONS))
        self.assertEqual(len(self.figures), 4)

    def test_3d_scene_has_units_and_start_finish(self) -> None:
        figure = dict(self.figures)["trajectories"]
        self.assertTrue(all(trace.type == "scatter3d" for trace in figure.data))
        self.assertEqual(figure.layout.scene.xaxis.title.text, "x, м")
        self.assertEqual(figure.layout.scene.yaxis.title.text, "y, м")
        self.assertEqual(figure.layout.scene.zaxis.title.text, "z, м")
        texts = [t for trace in figure.data if trace.mode == "markers+text" for t in trace.text]
        self.assertEqual(texts.count("старт"), 3)
        self.assertEqual(texts.count("финиш"), 3)
        legend = [trace.name for trace in figure.data if trace.showlegend]
        self.assertEqual(legend, [row.series.label for row in self.data])

    def test_positions_has_three_metric_rows(self) -> None:
        figure = dict(self.figures)["positions"]
        self.assertEqual(len(figure.data), 9)
        titles = [figure.layout[name].title.text for name in ("yaxis", "yaxis2", "yaxis3")]
        self.assertEqual(titles, ["x, м", "y, м", "z, м"])
        self.assertEqual(figure.layout.xaxis3.title.text, "t, с")

    def test_speed_and_height_axes(self) -> None:
        speed = dict(self.figures)["speed"]
        self.assertEqual(speed.layout.yaxis.title.text, "скорость, м/с")
        self.assertEqual(len(speed.data), 3)
        height = dict(self.figures)["height"]
        self.assertEqual(len(height.data), 2)  # у головы относительной высоты нет
        self.assertIn("y кисти", height.layout.yaxis.title.text)

    def test_downsizing_respects_budget(self) -> None:
        figure = dict(self.figures)["positions"]
        for trace in figure.data:
            self.assertLessEqual(len(trace.x), 21)  # 40 точек при бюджете 20 -> шаг 2 + хвост


class HtmlTest(unittest.TestCase):
    def test_self_contained_and_four_figures(self) -> None:
        data = pt.series_data(make_frame(20))
        figures = pt.build_figures(data, max_points=20)
        lines = ["строка сводки с <b>экранированием</b>"]
        html = pt.render_html(figures, lines, "Заголовок", "подпись")
        self.assertTrue(html.startswith("<!DOCTYPE html>"))
        self.assertIn("Plotly.newPlot", html)  # plotly.js встроен и фигуры инициализируются
        self.assertNotIn('src="https://cdn.plot.ly', html)  # внешних CDN нет
        for key in pt.FIGURE_CAPTIONS:
            self.assertIn(f'id="figure-{key}"', html)
        self.assertIn("&lt;b&gt;экранированием&lt;/b&gt;", html)  # текст экранирован

    def test_write_html_is_atomic(self) -> None:
        data = pt.series_data(make_frame(20))
        figures = pt.build_figures(data, max_points=20)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "trajectory.html"
            result = pt.write_html(figures, out, ["сводка"], "t", "n")
            self.assertTrue(out.is_file())
            self.assertFalse(out.with_suffix(".html.part").exists())
            self.assertEqual(result["figures"], 4)
            self.assertEqual(result["bytes"], out.stat().st_size)
            self.assertGreater(result["size_mb"], 1.0)  # встроенный plotly.js


class LoadTrackingTest(unittest.TestCase):
    def test_missing_file(self) -> None:
        with self.assertRaises(ValueError):
            pt.load_tracking("/nonexistent/tracking.csv")

    def test_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("t_rel_s,head_x\n0,0\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                pt.load_tracking(path)
            self.assertIn("right_palm_z", str(ctx.exception))

    def test_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            path.write_text(",".join(pt.REQUIRED_COLUMNS) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                pt.load_tracking(path)


@unittest.skipIf(TRACKING_CSV is None or not TRACKING_CSV.is_file(), NO_DATA)
class RealDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = pt.load_tracking(TRACKING_CSV)
        cls.data = pt.series_data(cls.frame)
        cls.summary = pt.summarize(cls.data)

    def test_counts_and_duration(self) -> None:
        self.assertEqual(self.summary["rows"], 21787)
        self.assertAlmostEqual(self.summary["duration_s"], 242.542, places=3)
        self.assertAlmostEqual(self.summary["median_dt_ms"], 11.12, places=2)

    def test_ranges_match_parser_statistics(self) -> None:
        head_y = self.summary["series"]["head"]["y"]
        self.assertAlmostEqual(head_y[0], -0.1154, places=3)
        self.assertAlmostEqual(head_y[1], -0.0275, places=3)
        right = self.summary["series"]["right"]
        self.assertEqual(right["nan"], 135)  # 03c/03d: невалидная правая рука
        self.assertEqual(self.summary["series"]["left"]["nan"], 0)
        self.assertAlmostEqual(self.summary["series"]["head"]["path_m"], 15.89, places=1)

    def test_hand_speed_matches_record_metrics(self) -> None:
        right = self.summary["series"]["right"]["speed"]
        self.assertAlmostEqual(right["mean"], 0.35, places=2)
        self.assertGreater(right["max"], 4.5)  # пик ≈5.06 м/с на стартовом «прыжке» руки
        self.assertLess(right["max"], 5.5)
        self.assertAlmostEqual(self.summary["series"]["left"]["speed"]["mean"], 0.071, places=2)

    def test_full_report_is_under_ten_megabytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "trajectory.html"
            result = pt.build_report(TRACKING_CSV, out, max_points=5000)
            self.assertEqual(result["figures"], 4)
            self.assertLess(result["bytes"], 10 * 2**20, "HTML должен быть < 10 МБ")
            html = out.read_text(encoding="utf-8")
            self.assertIn('id="figure-height"', html)
            self.assertIn("правая кисть (ладонь)", html)


if __name__ == "__main__":
    unittest.main()
