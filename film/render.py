"""Render 残像 / AFTERIMAGE to an MP4.

    python -m film.render                       # full film, 1920×1080 @ 30 fps
    python -m film.render --preview             # half resolution, quick look
    python -m film.render --stills 5,16,28.6    # single frames as PNG
    python -m film.render --frames 300:420      # a slice

Frames are rendered in parallel chunks; each worker streams raw RGB into its
own ffmpeg process, the segments are concatenated without re-encoding, and
the score is muxed in at the end.
"""
import argparse
import hashlib
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

from . import timeline as T
from .lights import R_RING, build as build_lights

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── shared preparation ────────────────────────────────────────────────

def lights_digest(lights):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(lights.pos[:, ::50]).astype(np.float32).tobytes())
    return h.hexdigest()[:12]


def prepare(cache_dir, width, height, log=print):
    """Everything that must exist before workers start: dust checkpoints and the afterimage."""
    from .camera import camera_at
    from .dust import checkpoint_path, precompute_checkpoints
    from .light_act import LightAct

    lights = build_lights()
    key = lights_digest(lights)
    dust_dir = os.path.join(cache_dir, f"dust-{key}")
    if not os.path.exists(checkpoint_path(dust_dir, (T.LIGHT_ACT_END_FRAME // 40) * 40)):
        log("· simulating dust …")
        t0 = time.time()
        precompute_checkpoints(lights, dust_dir, T.LIGHT_ACT_END_FRAME)
        log(f"  done in {time.time() - t0:.0f}s")

    after = os.path.join(cache_dir, f"afterimage-{key}-{width}.npy")
    if not os.path.exists(after):
        log("· exposing the afterimage …")
        act = LightAct(lights, width, height, dust_cache=dust_dir)
        hdr, _ = act.render_hdr(T.LIGHT_ACT_END_FRAME - 3)
        np.save(after, hdr)

    # where the circle lands on screen at the instant of the flash
    cam = camera_at(T.FLASH - 1.0 / T.FPS, lights, width, height)
    ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    ring = np.stack([R_RING * np.cos(ang), R_RING * np.sin(ang), np.zeros_like(ang)], -1)
    sx, sy, _, _ = cam.project(ring)
    center = (float(sx.mean()), float(sy.mean()))
    radius = float(np.mean(np.hypot(sx - center[0], sy - center[1])))
    return dict(dust_dir=dust_dir, afterimage=after, center=center, radius=radius)


class Renderer:
    """Per-process renderer; builds what it needs lazily."""

    def __init__(self, width, height, prep, cache_dir):
        self.W, self.H = width, height
        self.prep = prep
        self.cache_dir = cache_dir
        self.lights = build_lights()
        self.light_act = None
        self.ink_act = None

    def frame(self, f):
        if f < T.LIGHT_ACT_END_FRAME:
            if self.light_act is None:
                from .light_act import LightAct
                self.light_act = LightAct(self.lights, self.W, self.H, dust_cache=self.prep["dust_dir"])
            return self.light_act.render(f)
        if self.ink_act is None:
            from .ink_act import InkAct
            hdr = np.load(self.prep["afterimage"])
            self.ink_act = InkAct(self.W, self.H, self.prep["center"], self.prep["radius"], hdr,
                                  cache_dir=self.cache_dir)
        return self.ink_act.render(f)


# ── workers ───────────────────────────────────────────────────────────

_G = {}


def _init_worker(width, height, prep, cache_dir):
    cv2.setNumThreads(1)
    _G["r"] = Renderer(width, height, prep, cache_dir)


def _ffmpeg_writer(path, width, height, crf):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(T.FPS), "-i", "-",
        "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-tune", "film",
        "-x264-params", "keyint=60:min-keyint=60:scenecut=0",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _render_chunk(job):
    a, b, path, crf = job
    r = _G["r"]
    t0 = time.time()
    proc = _ffmpeg_writer(path, r.W, r.H, crf)
    for f in range(a, b):
        img = r.frame(f)
        proc.stdin.write(np.ascontiguousarray(img).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on segment {path}")
    return a, b, time.time() - t0


def chunk_plan(first, last, size_light=36, size_ink=90):
    jobs = []
    f = first
    while f < last:
        size = size_light if f < T.LIGHT_ACT_END_FRAME else size_ink
        b = min(f + size, last)
        if f < T.LIGHT_ACT_END_FRAME < b:
            b = T.LIGHT_ACT_END_FRAME
        jobs.append((f, b))
        f = b
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "afterimage.mp4"))
    ap.add_argument("--preview", action="store_true", help="960×540")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--frames", default=None, help="A:B frame slice")
    ap.add_argument("--stills", default=None, help="comma separated times (s) to write as PNG")
    ap.add_argument("--crf", type=int, default=19)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--cache", default=os.path.join(ROOT, "out", "cache"))
    args = ap.parse_args()

    width, height = (960, 540) if args.preview else (T.WIDTH, T.HEIGHT)
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t_start = time.time()
    prep = prepare(args.cache, width, height)

    if args.stills:
        cv2.setNumThreads(max(1, os.cpu_count() or 1))
        r = Renderer(width, height, prep, args.cache)
        outdir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "stills")
        os.makedirs(outdir, exist_ok=True)
        for s in args.stills.split(","):
            f = int(round(float(s) * T.FPS))
            t0 = time.time()
            img = r.frame(f)
            p = os.path.join(outdir, f"frame_{f:05d}.png")
            cv2.imwrite(p, img[..., ::-1])
            print(f"{p}  ({time.time() - t0:.1f}s)")
        return

    first, last = 0, T.N_FRAMES
    if args.frames:
        a, b = args.frames.split(":")
        first, last = int(a or 0), int(b or T.N_FRAMES)
    seg_dir = os.path.join(args.cache, "segments")
    shutil.rmtree(seg_dir, ignore_errors=True)
    os.makedirs(seg_dir)
    plan = chunk_plan(first, last)
    jobs = [(a, b, os.path.join(seg_dir, f"seg_{a:05d}.mp4"), args.crf) for a, b in plan]
    # heaviest first keeps the pool busy until the end
    order = sorted(jobs, key=lambda j: -(j[1] - j[0]) * (3.0 if j[0] < T.LIGHT_ACT_END_FRAME else 1.0))
    total = last - first
    done = 0
    print(f"· rendering {total} frames at {width}×{height} with {args.workers} workers")
    ctx = mp.get_context("spawn")   # fork after OpenCV threads have started can deadlock
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(width, height, prep, args.cache)) as pool:
        for a, b, dt in pool.imap_unordered(_render_chunk, order):
            done += b - a
            el = time.time() - t_start
            print(f"  frames {a:5d}–{b:5d} in {dt:5.1f}s   [{done}/{total}]  elapsed {el / 60:.1f} min", flush=True)

    concat = os.path.join(seg_dir, "list.txt")
    with open(concat, "w") as fh:
        for a, b, p, _ in jobs:
            fh.write(f"file '{p}'\n")
    video = os.path.join(seg_dir, "video.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", concat,
                    "-c", "copy", video], check=True)

    if args.no_audio:
        shutil.move(video, args.out)
    else:
        from .score import render_score
        wav = os.path.join(args.cache, "score.wav")
        render_score(wav, build_lights())
        start = first / T.FPS
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-ss", f"{start:.4f}",
                        "-t", f"{total / T.FPS:.4f}", "-i", wav,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                        "-movflags", "+faststart", "-shortest", args.out], check=True)
    print(f"✓ {args.out}  ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    sys.exit(main())
