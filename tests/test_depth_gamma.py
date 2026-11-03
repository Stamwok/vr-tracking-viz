"""Assert-тесты для нелинейной шкалы глубины (задача 08g-fix4).

Запуск: python -m unittest discover -s tests -t . -v

Проверяется чистая математика ``t``/``t'``/γ (края, монотонность, ValueError), совпадение
γ = 1.0 с прежней линейной формулой, градиент кости по ``t'`` середины отрезка, авторежим и
CLI-флаги ``--depth-gamma``/``--no-depth-gamma``, детерминизм и побайтовое совпадение
``--no-depth-gamma`` с линейной шкалой. Рендер-тесты используют синтетическое видео из
``tests.test_overlay_video`` и пропускаются без OpenCV.
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import camera_model, overlay_video as ov  # noqa: E402

try:  # pragma: no cover — проверка окружения, а не логики
    import cv2
except ImportError:
    cv2 = None

from tests.test_overlay_video import BASE_NS, SyntheticVideoCase, _write_tracking  # noqa: E402


class DepthParameterTest(unittest.TestCase):
    """08g-fix4: t = clamp((z − z_ref − near)/(far − near), 0, 1)."""

    def test_edges_and_clamp(self) -> None:
        near, far = 0.22, 0.48
        self.assertEqual(ov.depth_parameter(near), 0.0)
        self.assertEqual(ov.depth_parameter(far), 1.0)
        self.assertEqual(ov.depth_parameter(near - 0.05), 0.0)
        self.assertEqual(ov.depth_parameter(far + 0.05), 1.0)
        self.assertAlmostEqual(ov.depth_parameter((near + far) / 2), 0.5, places=12)

    def test_relative_reference_shifts_the_parameter(self) -> None:
        # та же относительная глубина на разной абсолютной — тот же t
        for depth, z_ref in ((0.30, 0.30), (0.45, 0.45), (0.90, 0.90)):
            with self.subTest(depth=depth):
                self.assertAlmostEqual(
                    ov.depth_parameter(depth + 0.09, near=0.0, far=0.18, z_ref=z_ref),
                    0.5,
                    places=12,
                )

    def test_unknown_and_behind_camera_are_zero(self) -> None:
        for depth in (math.nan, math.inf, -math.inf, -1.0, 0.0):
            with self.subTest(depth=depth):
                self.assertEqual(ov.depth_parameter(depth), 0.0)
                self.assertEqual(
                    ov.depth_parameter(depth, near=0.0, far=0.18, z_ref=0.3), 0.0
                )

    def test_window_validation(self) -> None:
        with self.assertRaises(ValueError):
            ov.depth_parameter(0.4, near=0.0)  # near = 0 без z_ref
        with self.assertRaises(ValueError):
            ov.depth_parameter(0.4, near=0.5, far=0.4)
        with self.assertRaises(ValueError):
            ov.depth_parameter(0.4, z_ref=math.nan)
        with self.assertRaises(ValueError):
            ov.depth_parameter("0.4")  # type: ignore[arg-type]


class GammaParameterTest(unittest.TestCase):
    """08g-fix4: t' = t**γ, края ровно 0/1, γ = 1.0 — прежняя линейная шкала."""

    def test_power_and_endpoints(self) -> None:
        self.assertEqual(ov.gamma_parameter(0.0, 0.6), 0.0)
        self.assertEqual(ov.gamma_parameter(1.0, 0.6), 1.0)
        for t in (0.1, 0.25, 0.5, 0.75, 0.9):
            with self.subTest(t=t):
                self.assertAlmostEqual(ov.gamma_parameter(t, 0.6), t**0.6, places=15)

    def test_gamma_one_is_identity(self) -> None:
        for t in (0.0, 0.1, 1 / 3, 0.5, 0.9, 1.0):
            with self.subTest(t=t):
                self.assertEqual(ov.gamma_parameter(t, 1.0), float(t))

    def test_monotone_in_t_and_gamma(self) -> None:
        values = [ov.gamma_parameter(t, 0.6) for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
        self.assertEqual(values, sorted(values))
        # меньшая γ при 0 < t < 1 даёт больший t' (сильнее приглушает середину окна)
        mid = [ov.gamma_parameter(0.5, g) for g in (0.5, 0.6, 0.7, 0.8, 1.0)]
        self.assertEqual(mid, sorted(mid, reverse=True))

    def test_validation(self) -> None:
        for gamma in (0.0, -0.5, 1.5, math.nan, math.inf, True, "0.6"):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    ov.gamma_parameter(0.5, gamma)  # type: ignore[arg-type]
        for t in (-0.1, 1.1, math.nan, True, "0.5"):
            with self.subTest(t=t):
                with self.assertRaises(ValueError):
                    ov.gamma_parameter(t, 0.6)  # type: ignore[arg-type]


class DepthFactorGammaTest(unittest.TestCase):
    """08g-fix4: множитель считается по t', края и фолбэки не меняются."""

    def test_gamma_one_matches_the_old_linear_formula(self) -> None:
        near, far, minimum = 0.22, 0.48, 0.35
        for depth in (0.22, 0.26, 0.30, 0.35, 0.40, 0.47, 0.48, 0.60):
            with self.subTest(depth=depth):
                t = ov.depth_parameter(depth, near=near, far=far)
                expected = 1.0 if t <= 0 else minimum if t >= 1 else 1.0 - (1 - minimum) * t
                self.assertEqual(
                    ov.depth_factor(depth, near=near, far=far, min_factor=minimum),
                    expected,
                )
                self.assertAlmostEqual(
                    ov.depth_alpha(depth, near=near, far=far, min_alpha=0.45),
                    1.0 - (1.0 - 0.45) * t if 0 < t < 1 else 1.0 if t <= 0 else 0.45,
                    places=15,
                )

    def test_edges_are_exactly_full_colour_and_floor(self) -> None:
        near, far, z_ref = 0.0, 0.18, 0.30
        self.assertEqual(ov.depth_factor(0.30, near=near, far=far, z_ref=z_ref, gamma=0.6), 1.0)
        self.assertEqual(
            ov.depth_factor(0.48, near=near, far=far, z_ref=z_ref, gamma=0.6),
            ov.DEPTH_MIN_FACTOR,
        )
        self.assertEqual(
            ov.depth_factor(0.60, near=near, far=far, z_ref=z_ref, gamma=0.6),
            ov.DEPTH_MIN_FACTOR,
        )
        self.assertEqual(
            ov.depth_alpha(0.30, near=near, far=far, z_ref=z_ref, gamma=0.6), 1.0
        )
        self.assertEqual(
            ov.depth_alpha(0.48, near=near, far=far, z_ref=z_ref, gamma=0.6),
            ov.DEPTH_MIN_ALPHA,
        )

    def test_unknown_depths_stay_full_colour(self) -> None:
        for depth in (math.nan, math.inf, -math.inf, 0.0, -0.3):
            with self.subTest(depth=depth):
                self.assertEqual(ov.depth_factor(depth, gamma=0.6), 1.0)
                self.assertEqual(ov.depth_alpha(depth, gamma=0.6), 1.0)

    def test_monotone_decreasing_with_depth_for_every_gamma(self) -> None:
        depths = [0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48]
        for gamma in (0.5, 0.6, 0.7, 0.8, 1.0):
            with self.subTest(gamma=gamma):
                factors = [
                    ov.depth_factor(d, near=0.0, far=0.18, z_ref=0.30, gamma=gamma)
                    for d in depths
                ]
                self.assertEqual(factors, sorted(factors, reverse=True))
                self.assertGreater(factors[0], factors[-1])

    def test_smaller_gamma_gives_darker_far_half(self) -> None:
        depth = 0.30 + 0.5 * 0.155  # t = 0.5 в окне 0.155
        factors = [
            ov.depth_factor(depth, near=0.0, far=0.155, z_ref=0.30, gamma=g)
            for g in (0.5, 0.6, 0.7, 0.8, 1.0)
        ]
        self.assertEqual(factors, sorted(factors))  # γ меньше — множитель меньше

    def test_invalid_gamma_is_rejected(self) -> None:
        for gamma in (0.0, -0.5, 1.5, math.nan, True, "0.6"):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    ov.depth_factor(0.3, gamma=gamma)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    ov.depth_alpha(0.3, gamma=gamma)  # type: ignore[arg-type]


class BoneGradientGammaTest(unittest.TestCase):
    """08g-fix4: градиент кости — по t' середины отрезка, γ = 1.0 — прежний расчёт."""

    def test_gamma_one_keeps_linear_interpolation(self) -> None:
        start, end = 0.32, 0.40
        z_ref, near, far = 0.30, 0.0, 0.16
        colors = ov.bone_gradient_colors(
            ov.COLORS["left"], start, end, z_ref=z_ref, near=near, far=far
        )
        first = ov.depth_factor(start, near=near, far=far, z_ref=z_ref)
        last = ov.depth_factor(end, near=near, far=far, z_ref=z_ref)
        step = last - first
        for index, color in enumerate(colors):
            expected = ov.shade_color(
                ov.COLORS["left"], first + step * ((index + 0.5) / len(colors))
            )
            self.assertEqual(color, expected)

    def test_colours_use_midpoint_gamma(self) -> None:
        start, end = 0.32, 0.40
        z_ref, near, far = 0.30, 0.0, 0.18
        colors = ov.bone_gradient_colors(
            ov.COLORS["left"], start, end, z_ref=z_ref, near=near, far=far, gamma=0.6
        )
        span = end - start
        for index, color in enumerate(colors):
            mid = start + span * ((index + 0.5) / len(colors))
            expected = ov.shade_color(
                ov.COLORS["left"],
                ov.depth_factor(mid, near=near, far=far, z_ref=z_ref, gamma=0.6),
            )
            self.assertEqual(color, expected)

    def test_alphas_use_midpoint_gamma(self) -> None:
        start, end = 0.32, 0.40
        z_ref, near, far = 0.30, 0.0, 0.18
        alphas = ov.bone_gradient_alphas(
            start, end, z_ref=z_ref, near=near, far=far, gamma=0.6
        )
        span = (end - start) / len(alphas)
        for index, value in enumerate(alphas):
            mid = start + span * (index + 0.5)
            self.assertAlmostEqual(
                value,
                ov.depth_alpha(mid, near=near, far=far, z_ref=z_ref, gamma=0.6),
                places=15,
            )

    def test_invalid_gamma_is_rejected(self) -> None:
        for gamma in (0.0, 1.5, math.nan):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    ov.bone_gradient_colors(ov.COLORS["left"], 0.3, 0.4, gamma=gamma)
                with self.assertRaises(ValueError):
                    ov.bone_gradient_alphas(0.3, 0.4, gamma=gamma)


class RelativeDepthWindowTest(unittest.TestCase):
    """08g-fix4: γ = 1.0 — окно 08g-fix3 (0.155), γ < 1 — окно 0.18."""

    def test_window_depends_on_gamma(self) -> None:
        self.assertEqual(ov.relative_depth_window(1.0), (0.0, ov.DEPTH_RELATIVE_FAR_M))
        self.assertEqual(ov.relative_depth_window(0.6), (0.0, ov.DEPTH_GAMMA_FAR_M))
        self.assertAlmostEqual(ov.DEPTH_GAMMA_FAR_M, 0.18, places=12)
        self.assertAlmostEqual(ov.DEPTH_RELATIVE_FAR_M, 0.155, places=12)

    def test_validation(self) -> None:
        for gamma in (0.0, 1.5, math.nan, True):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    ov.relative_depth_window(gamma)  # type: ignore[arg-type]


class ResolveDepthGammaTest(unittest.TestCase):
    """08g-fix4: авторежим — 0.6 для густого/26, 1.0 для 5 точек; явное значение главнее."""

    def test_auto_mode_follows_skeleton(self) -> None:
        self.assertEqual(ov.resolve_depth_gamma(None, ov.DENSE), ov.DEPTH_GAMMA)
        self.assertEqual(ov.resolve_depth_gamma(None, ov.ALL_JOINTS), ov.DEPTH_GAMMA)
        self.assertEqual(ov.resolve_depth_gamma(None, ov.FIVE_POINT), 1.0)
        clone = ov.Skeleton("five", "5 точек", ov.JOINTS, ov.HAND_CHAINS)
        self.assertEqual(ov.resolve_depth_gamma(None, clone), 1.0)

    def test_explicit_value_overrides_auto_mode(self) -> None:
        self.assertEqual(ov.resolve_depth_gamma(1.0, ov.DENSE), 1.0)
        self.assertEqual(ov.resolve_depth_gamma(0.5, ov.FIVE_POINT), 0.5)

    def test_validation(self) -> None:
        for gamma in (0.0, 1.1, math.nan, True, "0.6"):
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    ov.resolve_depth_gamma(gamma, ov.DENSE)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ov.resolve_depth_gamma(None, "dense")


@unittest.skipIf(cv2 is None, "OpenCV недоступен")
class GammaDrawTest(unittest.TestCase):
    """08g-fix4: draw_skeleton передаёт γ в точку и кость, γ = 1.0 не меняет кадр."""

    def setUp(self) -> None:
        self.style = ov.style_for(1080)
        self.pixels = {
            "left": {
                "palm": camera_model.ProjectedPoint(100.0, 100.0, True),
                "thumb": camera_model.ProjectedPoint(300.0, 100.0, True),
            },
            "right": {
                "palm": camera_model.ProjectedPoint(600.0, 100.0, True),
                "thumb": camera_model.ProjectedPoint(800.0, 100.0, True),
            },
        }
        self.joints = ("palm", "thumb")
        self.bones = (("palm", "thumb"),)
        self.depths = {
            "left": {"palm": 0.30, "thumb": 0.40},
            "right": {"palm": 0.60, "thumb": 0.70},
        }

    def _draw(self, **kwargs) -> np.ndarray:
        image = np.zeros((810, 1080, 3), np.uint8)
        ov.draw_skeleton(
            image,
            self.pixels,
            self.style,
            joints=self.joints,
            bones=self.bones,
            depths=self.depths,
            relative=True,
            **kwargs,
        )
        return image

    def test_gamma_one_equals_the_default(self) -> None:
        self.assertTrue(np.array_equal(self._draw(), self._draw(gamma=1.0)))

    def test_nonlinear_gamma_darkens_the_far_joint(self) -> None:
        image = self._draw(gamma=0.6)
        self.assertEqual(tuple(int(v) for v in image[100, 100]), ov.COLORS["left"])
        factor = ov.depth_factor(
            0.40, near=0.0, far=ov.DEPTH_GAMMA_FAR_M, z_ref=0.30, gamma=0.6
        )
        self.assertEqual(
            tuple(int(v) for v in image[100, 300]),
            ov.shade_color(ov.COLORS["left"], factor),
        )
        # правая рука нормируется отдельно и получает тот же множитель
        self.assertEqual(
            tuple(int(v) for v in image[100, 800]),
            ov.shade_color(ov.COLORS["right"], factor),
        )

    def test_gamma_one_uses_the_legacy_window(self) -> None:
        image = self._draw(gamma=1.0)
        factor = ov.depth_factor(
            0.40, near=0.0, far=ov.DEPTH_RELATIVE_FAR_M, z_ref=0.30, gamma=1.0
        )
        self.assertEqual(
            tuple(int(v) for v in image[100, 300]),
            ov.shade_color(ov.COLORS["left"], factor),
        )

    def test_invalid_gamma_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._draw(gamma=2.0)


@unittest.skipIf(cv2 is None, "OpenCV недоступен")
class GammaRenderTest(SyntheticVideoCase):
    """08g-fix4: CLI, авторежим, детерминизм и побайтовость ``--no-depth-gamma``."""

    def _dense_scene(self, skeleton: ov.Skeleton = ov.DENSE) -> ov.Scene:
        return ov.load_scene(self.meta_path, self.tracking, extra_keys=skeleton.extra_keys)

    def test_render_reports_depth_gamma(self) -> None:
        dense = ov.render_video(
            self.video, self.out, self._dense_scene(), start=0, end=1,
            progress_every=0, log=lambda _: None, skeleton=ov.DENSE,
        )
        self.assertAlmostEqual(dense.depth_gamma, ov.DEPTH_GAMMA, places=12)
        five = ov.render_video(
            self.video, self.out, self._scene(), start=0, end=1,
            progress_every=0, log=lambda _: None, skeleton=ov.FIVE_POINT,
        )
        self.assertEqual(five.depth_gamma, 1.0)
        linear = ov.render_video(
            self.video, self.out, self._dense_scene(), start=0, end=1,
            progress_every=0, log=lambda _: None, skeleton=ov.DENSE,
            depth_gamma=1.0,
        )
        self.assertEqual(linear.depth_gamma, 1.0)
        no_color = ov.render_video(
            self.video, self.out, self._dense_scene(), start=0, end=1,
            progress_every=0, log=lambda _: None, skeleton=ov.DENSE,
            depth_shading=False, depth_gamma=0.5,
        )
        self.assertEqual(no_color.depth_gamma, 1.0)
        # абсолютная шкала остаётся линейной (08g-fix2 побайтово), если γ не задана явно
        absolute = ov.render_video(
            self.video, self.out, self._dense_scene(), start=0, end=1,
            progress_every=0, log=lambda _: None, skeleton=ov.DENSE,
            depth_relative=False,
        )
        self.assertEqual(absolute.depth_gamma, 1.0)
        explicit_absolute = ov.render_video(
            self.video, self.out, self._dense_scene(), start=0, end=1,
            progress_every=0, log=lambda _: None, skeleton=ov.DENSE,
            depth_relative=False, depth_gamma=0.7,
        )
        self.assertAlmostEqual(explicit_absolute.depth_gamma, 0.7)

    def _stills(self, name: str, *flags: str, frames: int = 5) -> list[np.ndarray]:
        out = self.dir / name
        frame_list = ",".join(str(k) for k in range(frames))
        with contextlib.redirect_stdout(io.StringIO()):
            code = ov.main(
                [
                    str(self.video), "--stills", str(out), "--frames", frame_list,
                    "--tracking", str(self.tracking), "--meta", str(self.meta_path), *flags,
                ]
            )
        self.assertEqual(code, 0)
        return [cv2.imread(str(out / f"frame_{k:05d}.png")) for k in range(frames)]

    def _write_varied_tracking(self) -> None:
        """Синтетический трек с разбросом глубин внутри левой руки (для видимой γ)."""
        duration = self.FRAMES / self.FPS
        step_ns = duration / self.FRAMES
        times = [BASE_NS + int(round(k * step_ns * 1e9)) for k in range(self.FRAMES)]
        series = [
            {
                "palm": (0.10, 0.00, -0.90),
                "wrist": (0.11, 0.00, -0.94),
                "thumb": (0.12, 0.01, -1.08),
                "index": (0.13, -0.01, -1.12),
                "middle": (0.14, 0.00, -1.00),
            }
            for _ in range(self.FRAMES)
        ]
        _write_tracking(self.tracking, times, left_series=series)

    def test_cli_gamma_flags_and_determinism(self) -> None:
        self._write_varied_tracking()
        base = ("--five-points", "--depth-relative")
        nonlinear = self._stills("gamma_nonlinear", *base, "--depth-gamma", "0.6")
        linear = self._stills("gamma_linear", *base, "--no-depth-gamma")
        explicit_one = self._stills("gamma_one", *base, "--depth-gamma", "1.0")
        self.assertFalse(np.array_equal(nonlinear[0], linear[0]))
        for index in range(5):
            self.assertTrue(np.array_equal(linear[index], explicit_one[index]))
        # детерминизм: тот же прогон ещё раз даёт те же 5 кадров
        repeated = self._stills("gamma_nonlinear_again", *base, "--depth-gamma", "0.6")
        for index in range(5):
            self.assertTrue(np.array_equal(nonlinear[index], repeated[index]))

    def test_no_depth_gamma_equals_legacy_linear_render(self) -> None:
        self._write_varied_tracking()
        # 10c: у CLI по умолчанию указатель на локоть включён, а след выключен — здесь
        # сравнивается только шкала глубины, поэтому оба эффекта выключаются явно.
        base = ("--five-points", "--depth-relative", "--no-forearm")
        linear_cli = self._stills("gamma_legacy_cli", *base, "--no-depth-gamma")
        direct = self.dir / "gamma_legacy_direct"
        ov.render_stills(
            self.video, self._scene(), direct, frames=range(5),
            skeleton=ov.FIVE_POINT, depth_shading=True, depth_relative=True,
            depth_gamma=1.0, trail=False, forearm=False, log=lambda _: None,
        )
        for index in range(5):
            expected = cv2.imread(str(direct / f"frame_{index:05d}.png"))
            self.assertTrue(np.array_equal(linear_cli[index], expected))

    def test_invalid_gamma_is_a_cli_error(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            code = ov.main(
                [
                    str(self.video), "--stills", str(self.dir / "bad"),
                    "--tracking", str(self.tracking), "--meta", str(self.meta_path),
                    "--depth-gamma", "1.5",
                ]
            )
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
