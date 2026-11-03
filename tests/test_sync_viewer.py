"""Assert-тесты sync_viewer (задача 09b).

Проверяем то, что можно проверить без браузера: прореживание (концы серий и пик
скорости сохраняются), состав JSON и разрывы на «нет данных», сборку HTML (видео +
Plotly, plotly.js внутри, никаких внешних src), атомарную запись и понятные ошибки
на битом CSV. Поведение в браузере (дорисовка префикса, клик → seek) проверяется
отдельно headless Chrome — см. `docs/NOTES.md`, 29.10.

Тесты на реальном `output/tracking.csv` пропускаются, если артефакт 03d отсутствует.

Запуск: python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import plot_trajectory as pt  # noqa: E402
from src import sync_viewer as sv  # noqa: E402

TRACKING_CSV = ROOT / "output" / "tracking.csv"
NO_DATA = "нет output/tracking.csv — артефакт 03d не коммитится (см. README)"


def make_frame(rows: int = 100, right_jump_at: int | None = None) -> "object":
    """Синтетический tracking.csv: голова стоит, левая ладонь едет, правая — по вкусу."""
    import pandas as pd

    t = np.arange(rows, dtype=float) * 0.05
    columns: dict[str, np.ndarray] = {"t_rel_s": t}
    for prefix in ("head", "left_palm", "right_palm"):
        columns[f"{prefix}_x"] = np.zeros(rows)
        columns[f"{prefix}_y"] = np.zeros(rows)
        columns[f"{prefix}_z"] = np.zeros(rows)
    columns["left_palm_x"] = 0.5 * t
    columns["left_palm_y"] = -0.2 * np.ones(rows)
    if right_jump_at is not None:
        columns["right_palm_x"][right_jump_at] = 5.0  # пик скорости — именно здесь
    return pd.DataFrame(columns)


class FrameIndicesTest(unittest.TestCase):
    """Прореживание: бюджет, концы, первая/последняя валидная точка, пик скорости."""

    def test_budget_ends_and_order(self) -> None:
        indices = sv.frame_indices(make_frame(100), 10)
        self.assertLessEqual(indices.size, 11)  # шаг + хвост
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_max_points_zero_keeps_every_row(self) -> None:
        self.assertEqual(sv.frame_indices(make_frame(37), 0).tolist(), list(range(37)))

    def test_keeps_first_and_last_valid_point_per_series(self) -> None:
        # правая рука невалидна в начале и в конце: линия должна начинаться/кончаться
        # на её первой/последней валидной точке, а не на общем краю файла.
        frame = make_frame(100)
        frame.loc[0:19, "right_palm_x"] = np.nan
        frame.loc[0:19, "right_palm_y"] = np.nan
        frame.loc[0:19, "right_palm_z"] = np.nan
        frame.loc[90:99, "right_palm_x"] = np.nan
        frame.loc[90:99, "right_palm_y"] = np.nan
        frame.loc[90:99, "right_palm_z"] = np.nan
        indices = sv.frame_indices(frame, 10)
        self.assertIn(20, indices)
        self.assertIn(89, indices)

    def test_keeps_speed_peak(self) -> None:
        frame = make_frame(100, right_jump_at=63)
        speed = pt.speeds(
            frame["t_rel_s"].to_numpy(float),
            frame["right_palm_x"].to_numpy(float),
            frame["right_palm_y"].to_numpy(float),
            frame["right_palm_z"].to_numpy(float),
        )
        peak = int(np.argmax(np.where(np.isfinite(speed), speed, -np.inf)))
        self.assertNotIn(peak % 10, (0, 9))  # пик не попал бы в равномерную сетку
        self.assertIn(peak, sv.frame_indices(frame, 10))

    def test_too_small_budget_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            sv.frame_indices(make_frame(10), 1)


class PayloadTest(unittest.TestCase):
    def test_schema_and_lengths(self) -> None:
        payload = sv.viewer_payload(make_frame(60), 20)
        self.assertEqual(set(payload["series"]), {"head", "left", "right"})
        self.assertEqual(payload["t"], payload["t_rel_s"])
        self.assertEqual(len(payload["t"]), payload["stats"]["points"])
        self.assertLessEqual(len(payload["t"]), 21)
        for series in payload["series"].values():
            self.assertEqual(len(series["x"]), len(payload["t"]))
            self.assertEqual(len(series["y"]), len(payload["t"]))
            self.assertEqual(len(series["z"]), len(payload["t"]))
            self.assertIn("label", series)
            self.assertIn("color", series)
        self.assertEqual(payload["stats"]["rows"], 60)
        self.assertAlmostEqual(payload["stats"]["duration_s"], 2.95, places=3)

    def test_nan_becomes_none_and_splits_segments(self) -> None:
        frame = make_frame(60)
        for column in ("right_palm_x", "right_palm_y", "right_palm_z"):
            frame.loc[20:29, column] = np.nan
        payload = sv.viewer_payload(frame, 0)
        right = payload["series"]["right"]
        self.assertIsNone(right["x"][25])
        self.assertIsNone(right["y"][25])
        self.assertEqual(right["segments"], [[0, 20], [30, 60]])
        self.assertEqual(right["nan"], 10)
        self.assertEqual(right["valid"], 50)
        self.assertEqual(payload["series"]["left"]["segments"], [[0, 60]])

    def test_all_invalid_series_has_no_segments(self) -> None:
        frame = make_frame(30)
        frame[["right_palm_x", "right_palm_y", "right_palm_z"]] = np.nan
        payload = sv.viewer_payload(frame, 0)
        self.assertEqual(payload["series"]["right"]["segments"], [])
        self.assertEqual(payload["series"]["right"]["valid"], 0)

    def test_values_are_rounded(self) -> None:
        payload = sv.viewer_payload(make_frame(30), 0)
        for value in payload["t"]:
            self.assertEqual(value, round(value, sv.TIME_DECIMALS))
        for series in payload["series"].values():
            for axis in ("x", "y", "z"):
                for value in series[axis]:
                    if value is not None:
                        self.assertEqual(value, round(value, sv.VALUE_DECIMALS))

    def test_json_is_strict_and_escaped(self) -> None:
        payload = sv.viewer_payload(make_frame(30), 0)
        text = sv.payload_json(payload)
        self.assertNotIn("NaN", text)
        self.assertNotIn("</", text)  # нельзя преждевременно закрыть <script>
        restored = json.loads(text)
        self.assertEqual(restored["series"]["head"]["label"], "голова")

    def test_payload_json_escapes_script_terminator(self) -> None:
        payload = {
            "t": [0.0],
            "t_rel_s": [0.0],
            "series": {
                "head": {
                    "label": "</script><b>x</b>",
                    "color": "#fff",
                    "x": [0.0],
                    "y": [0.0],
                    "z": [0.0],
                    "segments": [[0, 1]],
                    "valid": 1,
                    "nan": 0,
                }
            },
            "stats": {"rows": 1, "duration_s": 0.0, "points": 1, "step": 1, "max_points": 0},
        }
        text = sv.payload_json(payload)
        self.assertNotIn("</script>", text)
        self.assertEqual(json.loads(text)["series"]["head"]["label"], "</script><b>x</b>")


class LayoutTest(unittest.TestCase):
    def test_fixed_ranges_and_units(self) -> None:
        payload = sv.viewer_payload(make_frame(50), 0)
        layout = sv.viewer_layout(payload)
        scene = layout["scene"]
        self.assertEqual(scene["aspectmode"], "data")
        self.assertEqual(scene["xaxis_title"], "x, м")
        self.assertEqual(scene["yaxis_title"], "y, м")
        self.assertEqual(scene["zaxis_title"], "z, м")
        for axis in ("x", "y", "z"):
            low, high = scene[f"{axis}axis"]["range"]
            self.assertLess(low, high)
        # левая кисть уходит на 0.5·t по x, поэтому диапазон не нулевой
        self.assertGreater(scene["xaxis"]["range"][1], 0.5)

    def test_all_invalid_series_does_not_crash_layout(self) -> None:
        frame = make_frame(10)
        for prefix in ("head", "left_palm", "right_palm"):
            for axis in ("x", "y", "z"):
                frame[f"{prefix}_{axis}"] = np.nan
        layout = sv.viewer_layout(sv.viewer_payload(frame, 0))
        self.assertNotIn("range", layout["scene"]["xaxis"])


class HtmlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = sv.viewer_payload(make_frame(25), 10)
        cls.html = sv.build_html(cls.payload, "Заголовок <b>тест</b>")

    def test_video_and_3d_scene_are_present(self) -> None:
        self.assertIn('<video id="video"', self.html)
        self.assertIn('src="overlay.mp4"', self.html)
        self.assertIn('id="scene"', self.html)
        self.assertIn('id="status"', self.html)
        self.assertIn("Plotly.newPlot", self.html)
        self.assertIn("Plotly.restyle", self.html)
        self.assertIn("plotly_click", self.html)
        self.assertIn("requestAnimationFrame", self.html)
        self.assertIn("bisectRight", self.html)
        self.assertIn("старт", self.html)
        self.assertIn("финиш", self.html)
        self.assertIn("сейчас", self.html)

    def test_start_finish_and_now_marker_are_traces(self) -> None:
        # layout.shapes в 3D-сценах Plotly не работает; маркер «сейчас» — scatter3d-трасса.
        self.assertIn("mode: 'markers'", self.html)
        self.assertIn("kind: 'now'", self.html)
        self.assertIn("mode: 'markers+text'", self.html)

    def test_self_contained_no_external_sources(self) -> None:
        self.assertNotIn("<script src", self.html)
        self.assertNotIn("<link", self.html)
        self.assertNotIn('src="http', self.html)
        self.assertGreater(len(self.html), 4_000_000)  # plotly.js встроен внутрь

    def test_embedded_json_is_valid(self) -> None:
        match = re.search(
            r'<script type="application/json" id="viewer-data">(.*?)</script>',
            self.html,
            re.S,
        )
        self.assertIsNotNone(match)
        restored = json.loads(match.group(1))
        self.assertEqual(restored["stats"]["rows"], 25)
        self.assertEqual(set(restored["series"]), {"head", "left", "right"})
        layout = re.search(
            r'<script type="application/json" id="viewer-layout">(.*?)</script>',
            self.html,
            re.S,
        )
        self.assertIsNotNone(layout)
        self.assertEqual(json.loads(layout.group(1))["scene"]["aspectmode"], "data")

    def test_title_and_video_are_escaped(self) -> None:
        self.assertIn("&lt;b&gt;тест&lt;/b&gt;", self.html)
        html = sv.build_html(self.payload, "t", video='overlay" onclick="x.mp4')
        self.assertIn('src="overlay&quot; onclick=&quot;x.mp4"', html)

    def test_size_is_dominated_by_plotlyjs(self) -> None:
        self.assertLess(len(self.html), 6 * 2**20)  # не раздуваем JSON/разметкой


class WriteViewerTest(unittest.TestCase):
    def test_write_is_atomic_and_creates_parents(self) -> None:
        payload = sv.viewer_payload(make_frame(20), 10)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "sync_viewer.html"
            result = sv.write_viewer(payload, out, "t")
            self.assertTrue(out.is_file())
            self.assertFalse(out.with_suffix(".html.part").exists())
            self.assertEqual(result["bytes"], out.stat().st_size)
            self.assertEqual(result["points"], payload["stats"]["points"])
            self.assertEqual(result["video"], "overlay.mp4")
            self.assertGreater(result["size_mb"], 4.0)  # plotly.js внутри


class LoadTrackingTest(unittest.TestCase):
    def _write(self, tmp: str, header: list[str], rows: list[str]) -> Path:
        path = Path(tmp) / "tracking.csv"
        path.write_text(",".join(header) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        return path

    def test_missing_file(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            sv.load_tracking("/nonexistent/tracking.csv")
        self.assertIn("нет файла", str(ctx.exception))

    def test_missing_columns_are_named(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, ["t_rel_s", "head_x"], ["0,0"])
            with self.assertRaises(ValueError) as ctx:
                sv.load_tracking(path)
            self.assertIn("right_palm_z", str(ctx.exception))
            self.assertIn("03d", str(ctx.exception))

    def test_non_numeric_column_is_a_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            header = list(pt.REQUIRED_COLUMNS)
            good = ["0"] * len(header)
            good[header.index("left_palm_y")] = "abc"
            path = self._write(tmp, header, [",".join(good)])
            with self.assertRaises(ValueError) as ctx:
                sv.load_tracking(path)
            message = str(ctx.exception)
            self.assertIn("left_palm_y", message)
            self.assertIn("tracking.csv", message)

    def test_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, list(pt.REQUIRED_COLUMNS), [])
            with self.assertRaises(ValueError):
                sv.load_tracking(path)


@unittest.skipIf(TRACKING_CSV is None or not TRACKING_CSV.is_file(), NO_DATA)
class RealDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = sv.load_tracking(TRACKING_CSV)
        cls.payload = sv.viewer_payload(cls.frame, 2000)

    def test_counts_duration_and_step(self) -> None:
        stats = self.payload["stats"]
        self.assertEqual(stats["rows"], 21787)
        self.assertAlmostEqual(stats["duration_s"], 242.542, places=3)
        self.assertEqual(stats["step"], 11)
        self.assertLessEqual(stats["points"], 2005)
        self.assertGreaterEqual(stats["points"], 1980)

    def test_series_gaps_match_parser(self) -> None:
        series = self.payload["series"]
        self.assertEqual(series["head"]["nan"], 0)
        self.assertEqual(series["left"]["nan"], 0)
        self.assertEqual(series["right"]["nan"], 135)
        self.assertEqual(len(series["right"]["segments"]), 1)
        # первые ~0.8 с правой руки нет: линия начинается позже начала файла
        self.assertGreater(series["right"]["segments"][0][0], 0)
        self.assertEqual(len(series["left"]["segments"]), 1)

    def test_times_are_sorted_and_start_near_video_zero(self) -> None:
        times = self.payload["t"]
        self.assertAlmostEqual(times[0], 0.0184, places=4)
        self.assertAlmostEqual(times[-1], 242.5607, places=4)
        self.assertTrue(np.all(np.diff(times) > 0))

    def test_embedded_json_is_under_budget(self) -> None:
        text = sv.payload_json(self.payload)
        self.assertLess(len(text.encode("utf-8")), sv.MAX_PAYLOAD_BYTES)
        json.loads(text)  # валидный JSON без NaN

    def test_small_budget_reduces_points(self) -> None:
        small = sv.viewer_payload(self.frame, 100)
        self.assertLess(small["stats"]["points"], 120)
        self.assertGreater(small["stats"]["step"], 100)

    def test_full_viewer_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sync_viewer.html"
            result = sv.build_viewer(TRACKING_CSV, out, max_points=2000)
            self.assertEqual(result["rows"], 21787)
            self.assertLess(result["bytes"], 6 * 2**20)
            self.assertGreater(result["bytes"], 4 * 2**20)
            html = out.read_text(encoding="utf-8")
            self.assertIn('<video id="video" src="overlay.mp4"', html)
            self.assertIn("Plotly.newPlot", html)
            self.assertNotIn("<script src", html)
            self.assertIn("правая кисть (ладонь)", html)
            self.assertIn("135", html)  # NaN правой руки в сводке
            payload = json.loads(
                re.search(
                    r'<script type="application/json" id="viewer-data">(.*?)</script>',
                    html,
                    re.S,
                ).group(1)
            )
            self.assertEqual(payload["stats"]["rows"], 21787)


if __name__ == "__main__":
    unittest.main()
