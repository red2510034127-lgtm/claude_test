#!/usr/bin/env python3
"""Check a rendered film and make a contact sheet of it.

Fails (exit 1) if the file is not 1920×1080 (or the 960×540 preview),
30 fps H.264 with an AAC track, of the expected length; or if any second
of it is unexpectedly black or frozen where the film should be moving.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from film import timeline as T  # noqa: E402


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def frames_at(path, times, w, h):
    imgs = []
    for t in times:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
        imgs.append(np.frombuffer(raw, np.uint8)[: w * h * 3].reshape(h, w, 3))
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--sheet", default=None)
    args = ap.parse_args()

    info = probe(args.video)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = [s for s in info["streams"] if s["codec_type"] == "audio"]
    w, h = int(v["width"]), int(v["height"])
    fps = eval(v["r_frame_rate"])  # e.g. "30/1"
    dur = float(info["format"]["duration"])
    problems = []
    if (w, h) not in ((1920, 1080), (960, 540)):
        problems.append(f"resolution {w}x{h}")
    if abs(fps - T.FPS) > 1e-3:
        problems.append(f"fps {fps}")
    if v["codec_name"] != "h264":
        problems.append(f"codec {v['codec_name']}")
    if not a or a[0]["codec_name"] != "aac":
        problems.append("no AAC audio")
    if abs(dur - T.END) > 0.25:
        problems.append(f"duration {dur:.2f}s, expected {T.END:.2f}s")

    # sample the film once per second
    times = np.arange(0.5, T.END - 0.2, 1.0)
    imgs = frames_at(args.video, times, w, h)
    means = [float(im.mean()) for im in imgs]
    for t, m in zip(times, means):
        # the film is dark at the very beginning, during silence it is still lit, and it ends in black
        expect_light = T.IGNITE + 0.5 < t < T.FADE_OUT[1] - 0.5
        if expect_light and m < 1.5:
            problems.append(f"t={t:.1f}s is black (mean {m:.2f})")
    print(f"{args.video}: {w}x{h} @ {fps:g} fps, {dur:.2f}s, audio={a[0]['codec_name'] if a else None}, "
          f"{os.path.getsize(args.video) / 1e6:.1f} MB")
    print("mean luminance per second:", " ".join(f"{m:.0f}" for m in means))

    if args.sheet:
        import cv2
        pick = np.linspace(1, T.END - 1.5, 24)
        thumbs = frames_at(args.video, pick, w, h)
        tw, th = 480, 270
        thumbs = [cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA) for im in thumbs]
        for im, t in zip(thumbs, pick):
            cv2.putText(im, f"{t:4.1f}s", (8, th - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        rows = [np.hstack(thumbs[i:i + 4]) for i in range(0, len(thumbs), 4)]
        cv2.imwrite(args.sheet, cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("contact sheet:", args.sheet)

    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
