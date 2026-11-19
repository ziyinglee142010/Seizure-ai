"""Facial emotion recognition inside MDS01's standalone video analysis.

Covers the bundled module, the isolated worker step (including failure), and
the public API projection. Synthetic inputs only; no face images are stored.
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.app.api.video_detection import _public_detection_result, _public_emotion_result
from backend.app.emotion_recognition.backends import EMOTIONS, OpenCVZooFER
from backend.app.emotion_recognition.face import TEMPLATE_112, Detection, _unrotate, align_face, select_patient
from backend.app.emotion_recognition.pipeline import analyse_video
from backend.app.services import video_detection_service as service
from backend.app.video_detection.contract import validate_predictions


def blank_video(path: Path, seconds: int = 3, fps: int = 25) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for _ in range(seconds * fps):
        writer.write(np.full((240, 320, 3), 40, np.uint8))
    writer.release()
    return path


class FlatWorkStorage:
    """Minimal stand-in for VideoStorage's work folder."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.deleted: list[Path] = []

    def work_path(self, job_id: str, name: str) -> Path:
        folder = self.root / job_id / "work"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / name

    def delete_work_file(self, path: Path) -> None:
        self.deleted.append(path)
        path.unlink(missing_ok=True)


class EmotionModuleTests(unittest.TestCase):
    def test_unrotate_maps_points_back_for_every_rotation(self) -> None:
        h, w = 60, 90
        pts = np.array([[5, 7], [80, 50], [44, 30]], np.float32)
        img = np.zeros((h, w), np.uint8)
        for x, y in pts.astype(int):
            img[y, x] = 255
        for rot in (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            r = img if rot is None else cv2.rotate(img, rot)
            ys, xs = np.nonzero(r)
            back = _unrotate(np.stack([xs, ys], 1).astype(np.float32), rot, w, h)
            self.assertEqual(sorted(map(tuple, back.astype(int))), sorted(map(tuple, pts.astype(int))))

    def test_patient_is_the_persistent_face_not_one_strong_hit(self) -> None:
        det = lambda x, y, s: Detection(s, np.array([x, y], np.float32), 80.0, np.zeros((5, 2), np.float32))
        frames = [[det(500, 200, 0.4)] for _ in range(10)]
        frames[3].append(det(800, 900, 0.95))                      # e.g. a foot, once
        self.assertTrue(all(c is not None and c.center[0] == 500 for c in select_patient(frames, 1000)))

    def test_sideways_face_is_aligned_to_the_template(self) -> None:
        frame = np.zeros((400, 400, 3), np.uint8)
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])            # lying on its side
        landmarks = (TEMPLATE_112 - 56) @ rotation.T * 1.5 + 200
        for x, y in landmarks:
            cv2.circle(frame, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
        out = align_face(frame, landmarks.astype(np.float32))
        for tx, ty in TEMPLATE_112:
            self.assertGreater(out[int(ty) - 2:int(ty) + 3, int(tx) - 2:int(tx) + 3, 2].max(), 100)

    def test_classifier_returns_seven_probabilities(self) -> None:
        faces = [np.random.default_rng(i).integers(0, 255, (112, 112, 3), dtype=np.uint8) for i in range(3)]
        p = OpenCVZooFER().predict(faces)
        self.assertEqual(p.shape, (3, 7))
        self.assertTrue(np.allclose(p.sum(1), 1, atol=1e-4))

    def test_video_without_faces_gives_no_face_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = analyse_video(blank_video(root / "clip.mp4"), root / "out", OpenCVZooFER(),
                                    sample_fps=5, save_preview=False)
            self.assertEqual((summary["face_coverage"], summary["windows"]), (0, 3))
            self.assertFalse((root / "out" / "face_samples.jpg").exists())


class EmotionWorkerTests(unittest.TestCase):
    def test_worker_runs_in_a_subprocess_and_leaves_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            storage = FlatWorkStorage(root)
            result = service._run_emotion_recognition(blank_video(root / "clip.mp4"), "job1", storage, 120)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["timeline"]["dominant_emotion"], ["no_face"] * 3)
            self.assertEqual(result["labels"], list(EMOTIONS))
            self.assertEqual(list((root / "job1" / "work").iterdir()), [])   # outputs deleted
            self.assertNotIn("video", result)          # no file name or other source metadata

    def test_failure_is_reported_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = FlatWorkStorage(Path(directory))
            failing = subprocess.CalledProcessError(1, ["emotion"])
            with patch.object(service, "execute", side_effect=failing):
                result = service._run_emotion_recognition(Path(directory) / "missing.mp4", "job2", storage, 30)
            self.assertEqual(result, {"status": "failed", "error_code": "emotion_failed"})
            self.assertEqual(len(storage.deleted), 2)

    def test_disabled_setting_skips_the_worker(self) -> None:
        with patch.object(service, "EMOTION_RECOGNITION_ENABLED", False), \
                patch.object(service, "execute") as execute:
            self.assertEqual(service._run_emotion_recognition(Path("x"), "job3", None, 30), {"status": "disabled"})
            execute.assert_not_called()

    def test_command_passes_settings_and_ends_with_the_output_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            storage = FlatWorkStorage(Path(directory))
            with patch.object(service, "EMOTION_FACE_REGION", "0.35,0,0.65,0.55"), \
                    patch.object(service, "execute", side_effect=OSError) as execute:
                service._run_emotion_recognition(Path("clip.mp4"), "job4", storage, 30)
            command = execute.call_args.args[0]
            self.assertIn("--no-preview", command)
            self.assertEqual(command[command.index("--roi") + 1], "0.35,0,0.65,0.55")
            self.assertEqual(command[-2:], ["--out", str(Path(directory) / "job4" / "work")])

    def test_out_of_range_scores_from_the_worker_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary, timeline = root / "s.json", root / "t.csv"
            summary.write_text(json.dumps({"labels": list(EMOTIONS)}))
            with timeline.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["start_sec", "end_sec", "frames_sampled", "frames_with_face", "face_coverage",
                                 "dominant_emotion", "dominant_probability", *[f"p_{e}" for e in EMOTIONS]])
                writer.writerow([0, 1, 2, 2, 1, "happy", 7.0, 0, 0, 0, 1, 0, 0, 0])
            with self.assertRaises(ValueError):
                service._read_emotion_outputs(summary, timeline)


def complete_emotion() -> dict:
    probs = {e: [0.1, None] for e in EMOTIONS}
    return {
        "status": "complete", "labels": list(EMOTIONS), "face_coverage": 0.5, "windows": 2,
        "windows_with_face": 1, "dominant_emotion_share": {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS},
        "mean_probabilities": {e: 0.1 for e in EMOTIONS},
        "model": {"model": "m", "trained_on": "RAF-DB", "weights_sha256": "abc"},
        "settings": {"roi": None, "sample_fps": 2.0, "window_sec": 1.0, "upscale": 1.0},
        "validation_note": "not validated", "video": {"file": "patient-name.mp4"},
        "timeline": {"start_sec": [0.0, 1.0], "end_sec": [1.0, 2.0], "face_coverage": [1.0, 0.0],
                     "dominant_emotion": ["neutral", "no_face"], "dominant_probability": [0.4, None],
                     "probabilities": probs},
    }


class EmotionApiTests(unittest.TestCase):
    def test_public_result_keeps_only_reviewed_fields(self) -> None:
        public = _public_emotion_result(complete_emotion())
        self.assertNotIn("video", public)                       # e.g. a source file name
        self.assertNotIn("weights_sha256", public["model"])
        self.assertNotIn("upscale", public["settings"])
        self.assertEqual(public["timeline"]["dominant_emotion"], ["neutral", "no_face"])

    def test_invalid_values_are_rejected(self) -> None:
        bad = complete_emotion()
        bad["timeline"]["dominant_emotion"] = ["neutral", "<script>"]
        with self.assertRaises(ValueError):
            _public_emotion_result(bad)
        short = complete_emotion()
        short["timeline"]["end_sec"] = [1.0]
        with self.assertRaises(ValueError):
            _public_emotion_result(short)

    def test_failed_emotion_does_not_affect_the_detection_result(self) -> None:
        result = validate_predictions([{"start_time": 0, "end_time": 2, "raw_score": 0.8}], 10, {"threshold": 0.5})
        result["emotion"] = {"status": "failed", "error_code": "emotion_failed", "detail": "x"}
        public = _public_detection_result(result)
        self.assertEqual(public["emotion"], {"status": "failed", "error_code": "emotion_failed"})
        self.assertEqual(public["predictions"][0]["score"], 0.8)

    def test_older_results_without_emotion_are_unchanged(self) -> None:
        result = validate_predictions([{"start_time": 0, "end_time": 2, "raw_score": 0.8}], 10, {"threshold": 0.5})
        self.assertNotIn("emotion", _public_detection_result(result))


if __name__ == "__main__":
    unittest.main()
