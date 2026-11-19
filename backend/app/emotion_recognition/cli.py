"""python -m emotion_recognition run --video clip.mp4 --out results/emotion/clip"""
from __future__ import annotations

import argparse
import json


def _roi(text: str | None):
    if not text:
        return None
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 4 or not (0 <= vals[0] < vals[2] <= 1 and 0 <= vals[1] < vals[3] <= 1):
        raise argparse.ArgumentTypeError("--roi must be x1,y1,x2,y2 fractions with x1<x2, y1<y2")
    return vals


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="emotion_recognition", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="analyse one video")
    run.add_argument("--video", required=True)
    run.add_argument("--out", required=True, help="output folder")
    run.add_argument("--recording-id", default=None)
    run.add_argument("--roi", type=_roi, default=None,
                     help="region to search for the face, x1,y1,x2,y2 fractions "
                          "(e.g. 0.35,0,0.65,0.55 for the head of the bed); default whole frame")
    run.add_argument("--backend", default="opencv-zoo", choices=["opencv-zoo", "faceemotion"])
    run.add_argument("--checkpoint", default=None, help="weights for the faceemotion backend")
    run.add_argument("--sample-fps", type=float, default=2.0, help="frames analysed per second")
    run.add_argument("--window-sec", type=float, default=1.0, help="timeline resolution")
    run.add_argument("--segment-sec", type=float, default=10.0, help="patient re-selection interval")
    run.add_argument("--upscale", type=float, default=1.5)
    run.add_argument("--score-threshold", type=float, default=0.3)
    run.add_argument("--min-face-px", type=float, default=40.0)
    run.add_argument("--no-preview", action="store_true")
    a = ap.parse_args(argv)

    from .backends import make_backend
    from .pipeline import analyse_video
    backend = make_backend(a.backend, a.checkpoint)
    summary = analyse_video(a.video, a.out, backend, roi=a.roi, sample_fps=a.sample_fps,
                            window_sec=a.window_sec, segment_sec=a.segment_sec, upscale=a.upscale,
                            score_threshold=a.score_threshold, min_face_px=a.min_face_px,
                            recording_id=a.recording_id, save_preview=not a.no_preview)
    print(json.dumps({k: summary[k] for k in ("recording_id", "face_coverage", "windows",
                                              "windows_with_face", "dominant_emotion_share")}, indent=2))
