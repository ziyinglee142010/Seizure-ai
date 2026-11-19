"""Video -> per-window emotion timeline.

Steps for one video:
  1. Sample frames at ``sample_fps`` (default 2 per second).
  2. Find face candidates in the region of interest at four rotations.
  3. Every ``segment_sec`` seconds, pick the patient's face among the
     candidates (persistence + tracking, see ``face.select_patient``).
  4. Align the chosen face to 112x112 with its 5 landmarks and classify it.
  5. Average the probabilities in windows of ``window_sec`` seconds.

Outputs, in ``out_dir``:
  emotion_timeline.csv   one row per window
  emotion_summary.json   settings, model provenance, overall shares
  face_samples.jpg       a strip of aligned faces, for checking the right face was tracked
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from . import __version__
from .backends import EMOTIONS, YUNET_PATH, EmotionBackend, sha256
from .face import FaceFinder, align_face, select_patient

VALIDATION_NOTE = ("Emotion predictions on seizure video are not validated: there is no "
                   "emotion ground truth for this data. They describe the visible facial "
                   "expression and are not a seizure or diagnostic output.")


@dataclass
class Sample:
    t: float
    face: bool
    score: float = float("nan")
    probs: Optional[np.ndarray] = None


def _progress(done: int, total: int, started: float, last: list) -> None:
    """Print 'progress: NN.N%' lines (at most every 2 s) for the job runner's log."""
    now = time.time()
    if total <= 0 or (now - last[0] < 2 and done < total):
        return
    last[0] = now
    pct = 100.0 * done / total
    eta = (now - started) / max(done, 1) * (total - done)
    print(f"progress: {int(pct)}% ~{eta / 60:.1f} min left", flush=True)


def analyse_video(video: str | Path, out_dir: str | Path, backend: EmotionBackend, *,
                  roi: Optional[Sequence[float]] = None, sample_fps: float = 2.0,
                  window_sec: float = 1.0, segment_sec: float = 10.0, upscale: float = 1.5,
                  score_threshold: float = 0.3, min_face_px: float = 40.0,
                  recording_id: Optional[str] = None, save_preview: bool = True) -> dict:
    video, out_dir = Path(video), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / sample_fps))
    seg_len = max(1, round(segment_sec * fps / step))     # sampled frames per segment

    finder = FaceFinder(YUNET_PATH, score_threshold=score_threshold, upscale=upscale,
                        min_face_px=min_face_px)
    samples: list[Sample] = []
    preview: list[np.ndarray] = []
    seg_t: list[float] = []
    seg_dets: list = []
    seg_crops: list = []
    started, last = time.time(), [0.0]

    def flush_segment():
        if not seg_t:
            return
        chosen = select_patient(seg_dets, W)
        crops = [seg_crops[i][next(j for j, d in enumerate(seg_dets[i]) if d is c)]
                 for i, c in enumerate(chosen) if c is not None]
        if hasattr(backend, "reset"):
            backend.reset()                      # motion features restart at each segment
        probs = backend.predict(crops) if crops else np.zeros((0, len(EMOTIONS)))
        k = 0
        for t, c in zip(seg_t, chosen):
            if c is None:
                samples.append(Sample(t, False))
            else:
                samples.append(Sample(t, True, c.score, probs[k]))
                if save_preview and len(preview) < 400:
                    preview.append(crops[k])
                k += 1
        seg_t.clear(); seg_dets.clear(); seg_crops.clear()

    idx = 0
    while True:
        if idx % step:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        dets = finder.detect(frame, roi)
        seg_t.append(idx / fps)
        seg_dets.append(dets)
        seg_crops.append([align_face(frame, d.landmarks, backend.input_size) for d in dets])
        if len(seg_t) >= seg_len:
            flush_segment()
        idx += 1
        _progress(idx, n_frames, started, last)
    flush_segment()
    cap.release()
    duration = idx / fps
    print("progress: 100%", flush=True)

    # ---- windows ---------------------------------------------------------
    n_win = max(1, math.ceil(duration / window_sec))
    rows = []
    for w in range(n_win):
        a, b = w * window_sec, min((w + 1) * window_sec, duration)
        inside = [s for s in samples if a <= s.t < b]
        with_face = [s for s in inside if s.face]
        row = {"start_sec": round(a, 3), "end_sec": round(b, 3), "frames_sampled": len(inside),
               "frames_with_face": len(with_face),
               "face_coverage": round(len(with_face) / len(inside), 3) if inside else 0.0}
        if with_face:
            p = np.mean([s.probs for s in with_face], axis=0)
            top = int(np.argmax(p))
            row.update(dominant_emotion=EMOTIONS[top], dominant_probability=round(float(p[top]), 4),
                       **{f"p_{e}": round(float(v), 4) for e, v in zip(EMOTIONS, p)})
        else:
            row.update(dominant_emotion="no_face", dominant_probability="",
                       **{f"p_{e}": "" for e in EMOTIONS})
        rows.append(row)

    cols = ["start_sec", "end_sec", "frames_sampled", "frames_with_face", "face_coverage",
            "dominant_emotion", "dominant_probability"] + [f"p_{e}" for e in EMOTIONS]
    with open(out_dir / "emotion_timeline.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader(); wr.writerows(rows)

    face_samples = [s for s in samples if s.face]
    face_windows = [r for r in rows if r["dominant_emotion"] != "no_face"]
    summary = {
        "module": "emotion_recognition", "module_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "recording_id": recording_id or video.stem,
        "video": {"file": video.name, "fps": round(fps, 3), "frame_count": idx,
                  "duration_sec": round(duration, 3), "width": W, "height": H},
        "settings": {"roi": list(roi) if roi else None, "sample_fps": sample_fps,
                     "window_sec": window_sec, "segment_sec": segment_sec, "upscale": upscale,
                     "score_threshold": score_threshold, "min_face_px": min_face_px,
                     "rotations": [0, 90, 180, 270]},
        "detector": {"model": "YuNet (OpenCV Zoo 2023mar)", "weights": YUNET_PATH.name,
                     "weights_sha256": sha256(YUNET_PATH)},
        "emotion_model": backend.describe(),
        "labels": list(EMOTIONS),
        "frames_sampled": len(samples),
        "face_coverage": round(len(face_samples) / len(samples), 4) if samples else 0.0,
        "windows": len(rows), "windows_with_face": len(face_windows),
        "dominant_emotion_share": {e: round(sum(r["dominant_emotion"] == e for r in face_windows)
                                             / len(face_windows), 4) if face_windows else 0.0
                                   for e in EMOTIONS},
        "mean_probabilities": ({e: round(float(v), 4) for e, v in
                                zip(EMOTIONS, np.mean([s.probs for s in face_samples], axis=0))}
                               if face_samples else None),
        "processing_sec": round(time.time() - started, 1),
        "validation_note": VALIDATION_NOTE,
    }
    (out_dir / "emotion_summary.json").write_text(json.dumps(summary, indent=2))

    if save_preview and preview:
        pick = [preview[i] for i in np.linspace(0, len(preview) - 1, min(8, len(preview))).astype(int)]
        cv2.imwrite(str(out_dir / "face_samples.jpg"), np.hstack(pick))
    return summary
