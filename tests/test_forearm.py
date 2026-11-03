"""Assert-тесты для предплечья 08h: Body.joints[18/19] → Hand.wrist.

Запуск: python -m unittest discover -s tests -t . -v

Проверяется извлечение локтей (валидность, «все нули», отсутствие Body), то, что
``tracking.csv``-путь не меняется (локти включаются только по запросу), загрузка через
``sync.extra_keys``/``joint_at``, защита от «улетевшего» локтя (0.15…0.40 м), стиль
(тоньше костей кисти, но **ровно базовым цветом руки**, 08h-fix), разрывы ребра, HUD-маркер,
CLI-флаги ``--forearm``/``--no-forearm``, дефолт выключен и детерминизм. Отдельно — цвет
ребра: точное равенство ``LEFT_COLOR``/``RIGHT_COLOR`` и независимость от шести флагов
глубины (``--depth-color``/``--no-depth-color``/``--depth-relative``/``--no-depth-relative``/
``--depth-gamma``/``--no-depth-gamma``). Рендер-тесты используют синтетическое видео из
``tests.test_overlay_video`` и пропускаются без OpenCV.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import camera_model, overlay_video as ov, parse_tracking, sync  # noqa: E402

try:  # pragma: no cover — проверка окружения, а не логики
    import cv2
except ImportError:
    cv2 = None

from tests.test_overlay_video import (  # noqa: E402
    BASE_NS,
    DATA,
    EYE,
    HEADER,
    LEFT_POINTS,
    META_PATH,
    NO_CV2,
    NO_DATA,
    RIGHT_POINTS,
    SyntheticVideoCase,
    VIDEO,
    _hand,
    _pose_string,
    _write_tracking,
)

HEAD_POSE = (
    "-0,08517717,-0,02745685,-0,0058761,0,162562191,-0,2861428,-0,116477557,-0,9370853"
)


def _body_p(x: float, y: float, z: float) -> str:
    """Pose-строка Body-джойнта: позиция + единичный кватернион (десятичная запятая)."""
    return _pose_string([x, y, z, 0.0, 0.0, 0.0, 1.0])


def _sample_with_body(
    left_elbow=(0.0, -0.4, 0.1),
    right_elbow=(0.1, -0.3, -0.1),
    joints: int = 24,
) -> dict:
    body = [
        {"p": _body_p(0.0, 0.0, 0.0), "t": 0, "va": "0,0,0,0,0,0", "wva": "0,0,0,0,0,0"}
        for _ in range(joints)
    ]
    if joints > 18:
        body[18] = {"p": _body_p(*left_elbow), "t": 0, "va": "0,0,0,0,0,0", "wva": "0,0,0,0,0,0"}
    if joints > 19:
        body[19] = {"p": _body_p(*right_elbow), "t": 0, "va": "0,0,0,0,0,0", "wva": "0,0,0,0,0,0"}
    return {
        "timeStampNs": BASE_NS,
        "Head": {"pose": HEAD_POSE, "status": 3},
        "Hand": {
            "leftHand": _hand(LEFT_POINTS),
            "rightHand": _hand(RIGHT_POINTS),
        },
        "Body": {"joints": body},
    }


class ForearmParseTest(unittest.TestCase):
    """08h: извлечение только локтей 18/19, валидность по значениям."""

    def test_indices_and_names(self) -> None:
        self.assertEqual(parse_tracking.ELBOW_KEYS, ("left_elbow", "right_elbow"))
        self.assertEqual(dict(parse_tracking.BODY_ELBOW_JOINTS)["left_elbow"], 18)
        self.assertEqual(dict(parse_tracking.BODY_ELBOW_JOINTS)["right_elbow"], 19)
        self.assertEqual(ov.FOREARM_KEYS, parse_tracking.ELBOW_KEYS)

    def test_extract_forearm_reads_18_and_19(self) -> None:
        record = parse_tracking.extract_forearm(
            _sample_with_body(left_elbow=(0.25, -0.5, 0.125), right_elbow=(-0.3, -0.4, 0.05))
        )
        self.assertEqual(record["left_elbow_x"], 0.25)
        self.assertEqual(record["left_elbow_y"], -0.5)
        self.assertEqual(record["left_elbow_z"], 0.125)
        self.assertEqual(record["right_elbow_x"], -0.3)
        self.assertEqual(record["right_elbow_y"], -0.4)
        self.assertEqual(record["right_elbow_z"], 0.05)
        # Body.wrist (20/21) и Body.hand (22/23) в данные не попадают
        self.assertEqual(len(record), 6)
        self.assertNotIn("left_wrist_x", record)

    def test_default_extract_sample_does_not_change_csv_path(self) -> None:
        sample = _sample_with_body()
        plain = parse_tracking.extract_sample(sample)
        self.assertNotIn("left_elbow_x", plain)
        # ровно 78 координат на руку — старый контракт 08d/08g не тронут
        for side in ("left", "right"):
            keys = [
                key
                for key in plain
                if key.startswith(f"{side}_") and key.rsplit("_", 1)[-1] in ("x", "y", "z")
            ]
            self.assertEqual(len(keys), 78)
        with_forearm = parse_tracking.extract_sample(sample, include_forearm=True)
        self.assertEqual(with_forearm["left_elbow_x"], 0.0)
        self.assertEqual(len(with_forearm), len(plain) + 6)

    def test_missing_body_and_short_joint_list_give_nan(self) -> None:
        sample = _sample_with_body()
        del sample["Body"]
        record = parse_tracking.extract_forearm(sample)
        for key in ("left_elbow_x", "right_elbow_y", "left_elbow_z"):
            self.assertTrue(math.isnan(record[key]))
        short = _sample_with_body(joints=19)
        record = parse_tracking.extract_forearm(short)
        self.assertTrue(math.isnan(record["right_elbow_x"]), "индекса 19 нет — NaN, а не IndexError")
        self.assertFalse(math.isnan(record["left_elbow_x"]))

    def test_all_zero_joint_is_nan(self) -> None:
        record = parse_tracking.extract_forearm(
            _sample_with_body(left_elbow=(0.0, 0.0, 0.0), right_elbow=(0.1, 0.2, 0.3))
        )
        self.assertTrue(math.isnan(record["left_elbow_x"]))
        self.assertTrue(math.isnan(record["left_elbow_y"]))
        self.assertTrue(math.isnan(record["left_elbow_z"]))
        self.assertEqual(record["right_elbow_x"], 0.1)
        sample = parse_tracking.extract_sample(
            _sample_with_body(left_elbow=(0.0, 0.0, 0.0)), include_forearm=True
        )
        self.assertTrue(math.isnan(sample["left_elbow_x"]))


def _write_tracking_series(path: Path, times, left_elbows) -> None:
    """Синтетический файл: Body-локоть левой руки меняется по сэмплам (для interpolate)."""
    lines = [json.dumps(HEADER)]
    for index, ts_ns in enumerate(times):
        body = [
            {"p": _body_p(0.0, 0.0, 0.0), "t": 0, "va": "0,0,0,0,0,0", "wva": "0,0,0,0,0,0"}
            for _ in range(24)
        ]
        body[18] = {
            "p": _body_p(*left_elbows[index]),
            "t": 0,
            "va": "0,0,0,0,0,0",
            "wva": "0,0,0,0,0,0",
        }
        sample = {
            "timeStampNs": int(ts_ns),
            "Head": {"pose": _pose_string([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), "status": 3},
            "Hand": {"leftHand": _hand(LEFT_POINTS), "rightHand": _hand(RIGHT_POINTS)},
            "Body": {"joints": body},
        }
        lines.append(json.dumps(sample))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ForearmSyncTest(unittest.TestCase):
    """08h: локти грузятся через extra_keys и интерполируются как суставы 08f."""

    def test_extra_keys_and_joint_at_interpolate_elbow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trackingData_elbow.txt"
            times = [BASE_NS, BASE_NS + 100_000_000]
            _write_tracking_series(path, times, [(0.0, -0.4, 0.1), (0.2, -0.4, 0.1)])
            track = sync.load_poses(path, extra_keys=("left_elbow",))
        self.assertEqual(track.extra_keys, ("left_elbow",))
        x, y, z = track.joint_at(BASE_NS + 50_000_000, "left_elbow")
        self.assertAlmostEqual(x, 0.1, places=12)
        self.assertAlmostEqual(y, -0.4, places=12)
        self.assertAlmostEqual(z, 0.1, places=12)
        # концы интервала — точные значения без float-арифметики
        self.assertEqual(track.joint_at(times[0], "left_elbow"), (0.0, -0.4, 0.1))
        self.assertEqual(track.joint_at(times[1], "left_elbow"), (0.2, -0.4, 0.1))
        with self.assertRaises(ValueError):
            track.joint_at(times[0], "right_elbow")

    def test_forearm_keys_not_loaded_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trackingData_elbow.txt"
            _write_tracking_series(path, [BASE_NS], [(0.0, -0.4, 0.1)])
            track = sync.load_poses(path)
        self.assertEqual(track.extra_keys, ())
        with self.assertRaises(ValueError):
            track.joint_at(BASE_NS, "left_elbow")

    def test_scene_can_load_forearm_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            meta = directory / "video_meta.json"
            tracking = directory / "trackingData_test.txt"
            meta.write_text(
                json.dumps({"frame_count": 1, "duration": 1 / 30, "width": 2160, "height": 810}),
                encoding="utf-8",
            )
            _write_tracking_series(tracking, [BASE_NS], [(0.0, -0.4, 0.1)])
            scene = ov.load_scene(meta, tracking, extra_keys=ov.FOREARM_KEYS)
        self.assertEqual(scene.track.extra_keys, ov.FOREARM_KEYS)


def _forearm_track(
    left_elbow, right_elbow, *, left_wrist=LEFT_POINTS["wrist"], right_wrist=RIGHT_POINTS["wrist"]
) -> sync.PoseTrack:
    """PoseTrack с двумя руками и локтями (только нужные колонки)."""
    fields = (*sync.POSITION_FIELDS, *sync.QUAT_FIELDS, "left_elbow_x", "left_elbow_y", "left_elbow_z",
              "right_elbow_x", "right_elbow_y", "right_elbow_z")
    values = {name: [0.0] for name in fields}
    values["head_qw"] = [1.0]
    for axis, value in zip("xyz", left_wrist):
        values[f"left_wrist_{axis}"] = [value]
    for axis, value in zip("xyz", right_wrist):
        values[f"right_wrist_{axis}"] = [value]
    for prefix, point in (("left_elbow", left_elbow), ("right_elbow", right_elbow)):
        for axis, value in zip("xyz", point):
            values[f"{prefix}_{axis}"] = [value]
    return sync.PoseTrack(
        [BASE_NS],
        values,
        {"left": [True], "right": [True]},
        extra_keys=("left_elbow", "right_elbow"),
    )


def _forearm_pose(
    track: sync.PoseTrack,
    *,
    left_wrist=LEFT_POINTS["wrist"],
    right_wrist=RIGHT_POINTS["wrist"],
) -> sync.Pose:
    return sync.Pose(
        ts_ns=BASE_NS,
        head_pos=(0.0, 0.0, 0.0),
        head_quat=(0.0, 0.0, 0.0, 1.0),
        hands={
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
        },
        active={"left": True, "right": True},
    )


class ForearmGeometryTest(unittest.TestCase):
    """08h: длина ребра, порог 0.15…0.40 м, видимость и «хвост» HUD-маркера."""

    def test_length_and_validation(self) -> None:
        self.assertAlmostEqual(ov.forearm_length((0.0, 0.0, 0.0), (0.0, 0.0, 0.25)), 0.25)
        self.assertAlmostEqual(ov.forearm_length((0.0, 0.0, 0.0), (0.03, 0.04, 0.0)), 0.05)
        for bad in (math.nan, math.inf, -math.inf):
            self.assertTrue(math.isnan(ov.forearm_length((bad, 0.0, 0.0), (0.0, 0.0, 0.25))))
        with self.assertRaises(ValueError):
            ov.forearm_length((0.0, 0.0), (0.0, 0.0, 0.25))  # type: ignore[arg-type]

    def test_projection_and_out_of_range(self) -> None:
        # перед камерой (OpenXR вперёд — −z), длина 0.22 м → ребро рисуется
        track = _forearm_track((0.11, 0.0, -0.78), (0.21, 0.1, -0.78))
        forearms = ov.project_forearms(_forearm_pose(track), EYE, track)
        self.assertTrue(forearms["left"].drawable)
        self.assertAlmostEqual(forearms["left"].length, 0.22, places=12)
        self.assertEqual(forearms["left"].status, "IK")
        self.assertFalse(forearms["left"].out_of_range)
        # локоть улетел: длина 0.5 м → ребра нет, но это не «нет данных»
        far = _forearm_track((0.11, 0.0, -0.5), (0.21, 0.1, -0.78))
        forearms = ov.project_forearms(_forearm_pose(far), EYE, far)
        self.assertFalse(forearms["left"].drawable)
        self.assertTrue(forearms["left"].out_of_range)
        self.assertEqual(forearms["left"].status, "n/a")

    def test_nan_and_behind_camera_are_not_drawn(self) -> None:
        behind_wrist = (0.21, 0.1, 1.0)  # за камерой: z > 0 в мире
        track = _forearm_track(
            (math.nan, math.nan, math.nan),
            (0.21, 0.1, 0.78),
            right_wrist=behind_wrist,
        )
        forearms = ov.project_forearms(
            _forearm_pose(track, right_wrist=behind_wrist), EYE, track
        )
        self.assertFalse(forearms["left"].drawable)
        self.assertFalse(forearms["left"].out_of_range)
        self.assertFalse(forearms["right"].drawable)
        self.assertFalse(forearms["right"].elbow.visible)
        self.assertFalse(forearms["right"].out_of_range)

    def test_requires_elbow_extra_keys(self) -> None:
        track = _forearm_track((0.11, 0.0, -0.78), (0.21, 0.1, -0.78))
        bare = sync.PoseTrack(
            [BASE_NS],
            {name: getattr(track, "_values")[name] for name in (*sync.POSITION_FIELDS, *sync.QUAT_FIELDS)},
            {"left": [True], "right": [True]},
        )
        with self.assertRaises(ValueError):
            ov.project_forearms(_forearm_pose(track), EYE, bare)

    def test_style_is_thinner_than_hand_bones(self) -> None:
        style = ov.style_for(1080)
        self.assertEqual(style.line_thickness, 2)
        self.assertEqual(ov.forearm_style(style).line_thickness, 2)
        self.assertEqual(ov.forearm_style(ov.style_for(1080, 2.0)).line_thickness, 4)
        # цвет ребра не рисуется множителем: в модуле больше нет «бледного» коэффициента
        self.assertFalse(hasattr(ov, "FOREARM_COLOR_FACTOR"))


@unittest.skipIf(cv2 is None, "OpenCV недоступен")
class ForearmDrawTest(unittest.TestCase):
    """08h: ребро рисуется до кисти, разрывы и счётчики пропусков."""

    def _image(self) -> np.ndarray:
        return np.zeros((810, 1080, 3), np.uint8)

    def _forearm(self, side="left", *, elbow=(100.0, 100.0), wrist=(300.0, 100.0),
                 length=0.22, visible=True) -> ov.Forearm:
        return ov.Forearm(
            side=side,
            elbow=camera_model.ProjectedPoint(elbow[0], elbow[1], visible),
            wrist=camera_model.ProjectedPoint(wrist[0], wrist[1], visible),
            length=length,
        )

    def test_draws_base_colour_line_and_counts(self) -> None:
        image = self._image()
        forearms = {"left": self._forearm(), "right": self._forearm("right", elbow=(600, 100), wrist=(800, 100))}
        counts = ov.draw_forearms(image, forearms, ov.style_for(1080))
        self.assertEqual(counts, (2, 0, 0))
        # 08h-fix: ровно базовый цвет руки, без множителя ×0.60 и без shade_color
        self.assertEqual(tuple(int(v) for v in image[100, 200]), ov.COLORS["left"])
        self.assertEqual(tuple(int(v) for v in image[100, 700]), ov.COLORS["right"])

    def test_out_of_range_and_invisible_are_skipped(self) -> None:
        image = self._image()
        forearms = {
            "left": self._forearm(length=0.50),
            "right": self._forearm("right", elbow=(600.0, 100.0), wrist=(800.0, 100.0), visible=False),
        }
        counts = ov.draw_forearms(image, forearms, ov.style_for(1080))
        self.assertEqual(counts, (0, 2, 1))
        self.assertFalse(image.any(), "пропущенное ребро не должно оставлять пикселей")

    def test_flat_colour_is_base_and_does_not_depend_on_depth(self) -> None:
        # глубина/alpha к ребру не применяются: цвет постоянен вдоль всей линии, а не
        # градиентом по z (Body-локоть — IK-оценка, глубина не измерена), и это ровно
        # базовый цвет руки, а не затемнённый (08h-fix)
        image = self._image()
        forearms = {"left": self._forearm(), "right": self._forearm("right", elbow=(600, 100), wrist=(800, 100))}
        counts = ov.draw_forearms(image, forearms, ov.style_for(1080))
        self.assertEqual(counts, (2, 0, 0))
        expected = ov.COLORS["left"]
        for u in (120, 150, 200, 250, 280):
            self.assertEqual(tuple(int(v) for v in image[100, u]), expected)


class ForearmHudTest(unittest.TestCase):
    """08h: маркер forearm появляется только при включённом предплечье."""

    def _moment(self) -> sync.FrameTime:
        return sync.FrameTime(index=3, ts_ns=BASE_NS, t_rel_s=1.5, in_tracking=True)

    def _pose(self) -> sync.Pose:
        return sync.Pose(
            ts_ns=BASE_NS,
            head_pos=(0.0, 0.0, 0.0),
            head_quat=(0.0, 0.0, 0.0, 1.0),
            hands={},
            active={"left": True, "right": False},
        )

    def test_default_rows_unchanged(self) -> None:
        texts = [text for text, _color in ov.hud_items(self._moment(), self._pose(), 100)]
        self.assertEqual(texts[2], "left: OK")
        self.assertEqual(texts[3], "right: NO DATA")

    def test_marker_ik_and_na(self) -> None:
        rows = ov.hud_items(
            self._moment(), self._pose(), 100, forearm={"left": True, "right": False}
        )
        texts = [text for text, _color in rows]
        self.assertEqual(texts[2], "left: OK | forearm: IK")
        self.assertEqual(texts[3], "right: NO DATA | forearm: n/a")
        self.assertEqual(rows[3][1], ov.HUD_WARNING_COLOR)

    def test_validation(self) -> None:
        for bad in ({"left": True}, {"left": True, "right": "no"}, "left"):
            with self.subTest(forearm=bad), self.assertRaises(ValueError):
                ov.hud_items(self._moment(), self._pose(), 100, forearm=bad)  # type: ignore[arg-type]


@unittest.skipIf(cv2 is None, "OpenCV недоступен")
class ForearmRenderTest(SyntheticVideoCase):
    """08h: CLI-флаги, дефолт выключен, побайтовость дефолта и детерминизм."""

    FRAMES = 5
    FPS = 30.0

    def setUp(self) -> None:
        super().setUp()
        # локоть на 0.2 м позади запястья по −z: ребро видно и не «улетает»
        self._write_video(
            self.FRAMES,
            self.FRAMES,
            left_series=None,
        )
        _write_tracking(
            self.tracking,
            [BASE_NS + int(round(k * (self.FRAMES / self.FPS) / self.FRAMES * 1e9)) for k in range(self.FRAMES)],
            body_elbows={
                "left": (LEFT_POINTS["wrist"][0], LEFT_POINTS["wrist"][1], LEFT_POINTS["wrist"][2] + 0.2),
                "right": (RIGHT_POINTS["wrist"][0], RIGHT_POINTS["wrist"][1], RIGHT_POINTS["wrist"][2] + 0.2),
            },
        )

    def _scene(self) -> ov.Scene:
        return ov.load_scene(self.meta_path, self.tracking, extra_keys=ov.FOREARM_KEYS)

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

    def test_default_is_off_and_byte_identical(self) -> None:
        default_cli = self._stills("default")
        explicit_off = self._stills("off", "--no-forearm")
        direct_scene = ov.load_scene(
            self.meta_path, self.tracking, extra_keys=ov.DENSE.extra_keys
        )
        direct = self.dir / "direct"
        ov.render_stills(
            self.video, direct_scene, direct, frames=range(5),
            skeleton=ov.DENSE, forearm=False, log=lambda _: None,
        )
        for index in range(5):
            self.assertTrue(np.array_equal(default_cli[index], explicit_off[index]))
            self.assertTrue(
                np.array_equal(default_cli[index], cv2.imread(str(direct / f"frame_{index:05d}.png")))
            )

    def test_forearm_changes_frame_and_is_deterministic(self) -> None:
        with_arm = self._stills("on", "--forearm")
        off = self._stills("off", "--no-forearm")
        self.assertFalse(np.array_equal(with_arm[0], off[0]))
        again = self._stills("on_again", "--forearm")
        for index in range(5):
            self.assertTrue(np.array_equal(with_arm[index], again[index]))

    def _write_clear_elbows(self) -> None:
        """Трекинг, где ребро уходит в пустую часть кадра: слева вверх, справа вниз.

        Иначе (как в ``setUp``, локоть позади запястья по −z) ребро проецируется почти
        в линию кисти, и «вне кисти» пиксель для проверки цвета не найти.
        """
        _write_tracking(
            self.tracking,
            [BASE_NS + int(round(k * (self.FRAMES / self.FPS) / self.FRAMES * 1e9)) for k in range(self.FRAMES)],
            body_elbows={
                "left": (0.11, 0.25, -1.0),   # v ≈ 233 против v ≈ 405 у запястья
                "right": (0.21, -0.30, -1.0),  # v ≈ 611 против v ≈ 336 у запястья
            },
        )

    def test_colour_is_exact_base_in_six_depth_modes(self) -> None:
        # 08h-fix: цвет ребра — ровно LEFT_COLOR/RIGHT_COLOR и не зависит ни от одного из
        # шести флагов глубины. Ребро уведено в пустую область, пиксель берётся в середине
        # отрезка — вне кисти и HUD.
        self._write_clear_elbows()
        index = 2
        scene = ov.load_scene(
            self.meta_path, self.tracking, extra_keys=ov.DENSE.extra_keys + ov.FOREARM_KEYS
        )
        moment = sync.frame_time(index, scene.meta, scene.header_ns, scene.tracking_end_ns)
        pose = sync.pose_at_frame(scene.track, moment, scene.time_offset_ns)
        forearms = ov.project_forearms(pose, scene.intrinsics, scene.track)
        expected: dict[str, tuple[int, int]] = {}
        for side in ("left", "right"):
            forearm = forearms[side]
            self.assertTrue(forearm.drawable, f"ребро {side} должно рисоваться")
            u = (forearm.elbow.u + forearm.wrist.u) / 2.0
            v = (forearm.elbow.v + forearm.wrist.v) / 2.0
            x0, y0, x1, y1 = ov.hud_rect(scene.eye_size)
            self.assertFalse(x0 <= u <= x1 and y0 <= v <= y1, "пиксель не должен быть на HUD")
            expected[side] = (int(round(u)), int(round(v)))
        modes = {
            "--depth-color": ["--depth-color"],
            "--no-depth-color": ["--no-depth-color"],
            "--depth-relative": ["--depth-relative"],
            "--no-depth-relative": ["--no-depth-relative"],
            "--depth-gamma 0.7": ["--depth-gamma", "0.7"],
            "--no-depth-gamma": ["--no-depth-gamma"],
        }
        images: dict[str, np.ndarray] = {}
        for name, flags in modes.items():
            out = self.dir / f"modes_{name.lstrip('-').replace(' ', '_').replace('.', '_')}"
            with contextlib.redirect_stdout(io.StringIO()):
                code = ov.main(
                    [
                        str(self.video), "--stills", str(out), "--frames", str(index),
                        "--tracking", str(self.tracking), "--meta", str(self.meta_path),
                        "--forearm", *flags,
                    ]
                )
            self.assertEqual(code, 0)
            images[name] = cv2.imread(str(out / f"frame_{index:05d}.png"))
        for name, image in images.items():
            for side, (u, v) in expected.items():
                window = image[v - 3 : v + 4, u - 3 : u + 4]
                colours = {tuple(int(channel) for channel in pixel) for pixel in window.reshape(-1, 3)}
                self.assertIn(
                    ov.COLORS[side], colours,
                    f"{name}: у ({u}, {v}) нет пикселя ребра ровно базового цвета {ov.COLORS[side]}",
                )
        # флаги глубины действительно применены (кисть в этих режимах разная) — иначе
        # утверждение о независимости цвета ничего бы не проверяло
        self.assertFalse(np.array_equal(images["--depth-color"], images["--no-depth-color"]))

    def test_render_stats_and_hud_marker(self) -> None:
        stats = ov.render_video(
            self.video, self.out, self._scene(), start=0, end=4,
            progress_every=0, log=lambda _: None, forearm=True,
        )
        self.assertTrue(stats.forearm)
        self.assertEqual(stats.forearm_drawn, 10)
        self.assertEqual(stats.forearm_skipped, 0)
        self.assertEqual(stats.forearm_out_of_range, 0)
        self.assertEqual(stats.forearm_total, 10)
        self.assertEqual(stats.forearm_drawn_share, 1.0)
        image = self._stills("marker", "--forearm")[0]
        crop = image[12:148, 12:342]
        self.assertGreater(int((crop.astype(int).sum(axis=2) > 300).sum()), 0)

    def test_out_of_range_elbow_is_skipped(self) -> None:
        _write_tracking(
            self.tracking,
            [BASE_NS + int(round(k * (self.FRAMES / self.FPS) / self.FRAMES * 1e9)) for k in range(self.FRAMES)],
            body_elbows={
                "left": (1.0, 0.0, -1.0),  # длина ≫ 0.40 м
                "right": (RIGHT_POINTS["wrist"][0], RIGHT_POINTS["wrist"][1], RIGHT_POINTS["wrist"][2] + 0.2),
            },
        )
        stats = ov.render_video(
            self.video, self.out, self._scene(), start=0, end=4,
            progress_every=0, log=lambda _: None, forearm=True,
        )
        self.assertEqual(stats.forearm_drawn, 5)
        self.assertEqual(stats.forearm_skipped, 5)
        self.assertEqual(stats.forearm_out_of_range, 5)

    def test_missing_elbow_keys_raise_before_render(self) -> None:
        scene = ov.load_scene(self.meta_path, self.tracking)
        with self.assertRaises(ValueError):
            ov.render_video(
                self.video, self.out, scene, progress_every=0, log=lambda _: None, forearm=True
            )
        self.assertFalse(self.out.exists())


@unittest.skipIf(cv2 is None, "OpenCV недоступен")
class ForearmFlatSkeletonTest(unittest.TestCase):
    """08h: предплечье работает и с 5-точечным плоским кадром (без глубины)."""

    def test_five_points_with_forearm_draws(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            image = np.zeros((810, 1080, 3), np.uint8)
            forearms = {
                "left": ov.Forearm(
                    side="left",
                    elbow=camera_model.ProjectedPoint(100.0, 100.0, True),
                    wrist=camera_model.ProjectedPoint(300.0, 100.0, True),
                    length=0.22,
                ),
                "right": ov.Forearm(
                    side="right",
                    elbow=camera_model.ProjectedPoint(600.0, 100.0, True),
                    wrist=camera_model.ProjectedPoint(800.0, 100.0, True),
                    length=0.22,
                ),
            }
            counts = ov.draw_forearms(image, forearms, ov.style_for(1080))
        self.assertEqual(counts, (2, 0, 0))
        self.assertTrue(image.any())


@unittest.skipUnless(DATA is not None and META_PATH.is_file(), NO_DATA)
@unittest.skipIf(cv2 is None, NO_CV2)
@unittest.skipIf(VIDEO is None, "нет data/CameraRecord_*.mp4 (видео не коммитится)")
class ForearmColourRealDataTest(unittest.TestCase):
    """08h-fix: на 5 кадрах 07a цвет ребра точно равен базовому во всех 6 режимах глубины.

    Путь тот же, что у ``--stills`` (``sync.frame_time`` → ``sync.pose_at_frame`` →
    ``project_forearms`` → ``_draw_pose``), а PNG пишется без потерь — поэтому цвет пикселя
    массива и есть цвет пикселя PNG. Пиксель выбирается на видимой части ребра: не ближе
    20 px к любому суставу густого скелета (обе руки) и вне панели HUD, то есть вне обводки
    и пересечений с кистью. Где видимого куска ребра нет (локоть за кадром, на 07a это
    левый на k = 1800/3600/5400), пара пропускается; видимых пар 6 из 10.
    """

    #: Шесть режимов = шесть значений флагов глубины; значения уже разрешены так же, как в
    #: ``render_stills`` (авто: alpha и относительность включены для густого скелета).
    MODES = {
        "--depth-color": dict(
            depth_shading=True, depth_alpha=True, depth_relative=True, depth_gamma=0.7
        ),
        "--no-depth-color": dict(
            depth_shading=False, depth_alpha=False, depth_relative=False, depth_gamma=1.0
        ),
        "--depth-relative": dict(
            depth_shading=True, depth_alpha=True, depth_relative=True, depth_gamma=0.7
        ),
        "--no-depth-relative": dict(
            depth_shading=True, depth_alpha=True, depth_relative=False, depth_gamma=1.0
        ),
        "--depth-gamma 0.7": dict(
            depth_shading=True, depth_alpha=True, depth_relative=True, depth_gamma=0.7
        ),
        "--no-depth-gamma": dict(
            depth_shading=True, depth_alpha=True, depth_relative=True, depth_gamma=1.0
        ),
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.scene = ov.load_scene(
            META_PATH, DATA, extra_keys=ov.DENSE.extra_keys + ov.FOREARM_KEYS
        )
        cls.style = ov.style_for(cls.scene.eye_size[0])

    @staticmethod
    def _blocked_joints(pose: sync.Pose, scene: ov.Scene) -> list[tuple[float, float]]:
        """Пиксели всех суставов густого скелета (обе руки) — «не пересекаться с кистью»."""
        pixels = ov.project_skeleton(pose, scene.intrinsics, ov.DENSE, track=scene.track)
        blocked: list[tuple[float, float]] = []
        for hand in pixels.values():
            for point in hand.values():
                if point.visible and math.isfinite(point.u) and math.isfinite(point.v):
                    blocked.append((point.u, point.v))
        return blocked

    def _find_pixel(
        self,
        image: np.ndarray,
        reference: np.ndarray,
        forearm: ov.Forearm,
        blocked: list[tuple[float, float]],
        colour: tuple[int, int, int],
    ) -> tuple[int, int] | None:
        """Пиксель на ребре ровно цвета ``colour``, которого нет на кадре без ребра."""
        height, width = image.shape[:2]
        x0, y0, x1, y1 = ov.hud_rect((width, height))
        for step in range(15, 86, 2):
            t = step / 100.0
            u = forearm.elbow.u + (forearm.wrist.u - forearm.elbow.u) * t
            v = forearm.elbow.v + (forearm.wrist.v - forearm.elbow.v) * t
            if not (10.0 <= u <= width - 11.0 and 10.0 <= v <= height - 11.0):
                continue
            if x0 <= u <= x1 and y0 <= v <= y1:
                continue
            if any(math.hypot(u - joint_u, v - joint_v) < 20.0 for joint_u, joint_v in blocked):
                continue
            for du in range(-2, 3):
                for dv in range(-2, 3):
                    x, y = int(round(u)) + du, int(round(v)) + dv
                    if tuple(int(channel) for channel in image[y, x]) == colour and tuple(
                        int(channel) for channel in reference[y, x]
                    ) != colour:
                        return x, y
        return None

    def test_forearm_pixel_is_base_colour_in_all_six_modes(self) -> None:
        capture = cv2.VideoCapture(str(VIDEO))
        self.assertTrue(capture.isOpened())
        checked = 0
        try:
            for frame_index in ov.DEBUG_FRAMES:
                self.assertTrue(capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index))
                ok, raw = capture.read()
                self.assertTrue(ok and raw is not None, f"не читается кадр {frame_index}")
                view = ov._eye_view(raw, self.scene)
                moment = sync.frame_time(
                    frame_index, self.scene.meta, self.scene.header_ns, self.scene.tracking_end_ns
                )
                pose = sync.pose_at_frame(self.scene.track, moment, self.scene.time_offset_ns)
                forearms = ov.project_forearms(pose, self.scene.intrinsics, self.scene.track)
                blocked = self._blocked_joints(pose, self.scene)
                images: dict[str, np.ndarray] = {}
                for name, kwargs in self.MODES.items():
                    image = view.copy()
                    ov._draw_pose(
                        image, pose, self.scene, self.style, ov.DENSE,
                        forearms=forearms, **kwargs,
                    )
                    images[name] = image
                reference = view.copy()
                ov._draw_pose(
                    reference, pose, self.scene, self.style, ov.DENSE, forearms=None
                )
                for side in ("left", "right"):
                    forearm = forearms[side]
                    if not forearm.drawable:
                        continue
                    colour = ov.COLORS[side]
                    found = self._find_pixel(
                        images["--depth-color"], reference, forearm, blocked, colour
                    )
                    if found is None:
                        continue
                    checked += 1
                    x, y = found
                    for name, image in images.items():
                        self.assertEqual(
                            tuple(int(channel) for channel in image[y, x]),
                            colour,
                            f"кадр {frame_index}, {side}, {name}: ребро не базового цвета "
                            f"в ({x}, {y})",
                        )
        finally:
            capture.release()
        self.assertGreaterEqual(
            checked, 5, "на 5 кадрах 07a должно найтись не меньше 5 видимых рёбер"
        )


if __name__ == "__main__":
    unittest.main()
