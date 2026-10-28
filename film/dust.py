"""Dust: the dark is not empty, it is only unlit.

A quarter of a million motes drift through an Arnold–Beltrami–Childress
flow (a divergence-free, gently chaotic current). Passing lights drag them
into wakes. During the convergence a vortex gathers them around the circle.
"""
import os

import numpy as np

from . import timeline as T
from .lights import R_RING

SUBSTEPS = 2


class Dust:
    def __init__(self, n=220_000, seed=7):
        rng = np.random.default_rng(seed)
        n_core = int(n * 0.72)
        core = rng.normal(size=(n_core, 3)) * np.array([1.05, 1.05, 0.75])
        n_far = n - n_core
        u = rng.normal(size=(n_far, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        far = u * (1.5 + 5.0 * rng.random(n_far) ** 0.6)[:, None]
        self.x = np.concatenate([core, far]).astype(np.float32)
        self.v = np.zeros_like(self.x)
        self.x_prev = self.x.copy()
        self.albedo = (rng.uniform(0.15, 1.0, n) ** 2.2).astype(np.float32)
        self.twinkle = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        self.frame = 0          # positions correspond to the end of this frame

    # ── dynamics ──────────────────────────────────────────────────────
    def _abc(self, x, t):
        k = 1.15
        A, B, C = 1.0, 0.72, 0.48
        ph = 0.05 * t
        X, Y, Z = x[:, 0] * k, x[:, 1] * k, x[:, 2] * k
        v = np.empty_like(x)
        v[:, 0] = A * np.sin(Z + ph) + C * np.cos(Y)
        v[:, 1] = B * np.sin(X) + A * np.cos(Z + ph)
        v[:, 2] = C * np.sin(Y - ph) + B * np.cos(X)
        return v * np.float32(0.030)

    def _step(self, t, dt, lights):
        x, v = self.x, self.v
        target = self._abc(x, t)
        relax = np.float32(0.9 * dt)
        v += (target - v) * relax

        p, u, L = lights.head(T.sim_time(t))
        for i in range(len(L)):
            if L[i] <= 1e-4:
                continue
            d = x - p[i].astype(np.float32)
            r2 = np.einsum("ij,ij->i", d, d)
            m = np.nonzero(r2 < 0.36)[0]
            if len(m) == 0:
                continue
            g = np.exp(-r2[m] / np.float32(0.05))[:, None] * np.float32(min(L[i], 2.0))
            ui = u[i].astype(np.float32)
            sp = float(np.linalg.norm(ui)) + 1e-6
            swirl = np.cross(ui / sp, d[m]) * np.float32(0.9 * sp)
            v[m] += (np.float32(0.85) * (ui - v[m]) + swirl) * g * np.float32(dt)

        c = float(lights.conv[lights.index(T.sim_time(t))])
        if c > 1e-3:
            r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) + np.float32(1e-4)
            tx, ty = x[:, 1] / r, -x[:, 0] / r            # clockwise seen from +z
            band = 1.0 / (1.0 + ((r - R_RING) / 0.7) ** 2)
            om = np.float32(2 * np.pi * 0.55) * band * r
            vt = np.empty_like(x)
            vt[:, 0] = om * tx + (R_RING - r) * 0.45 * x[:, 0] / r
            vt[:, 1] = om * ty + (R_RING - r) * 0.45 * x[:, 1] / r
            vt[:, 2] = -x[:, 2] * 0.6 * band
            v += (vt - v) * np.float32(1.6 * c * dt)

        x += v * np.float32(dt)

    def advance_to(self, frame, lights):
        """Step forward until positions correspond to the end of `frame`."""
        while self.frame < frame:
            f = self.frame + 1
            t0 = (f - 1) / T.FPS
            self.x_prev = self.x.copy()
            if t0 < T.FREEZE:
                dt = 1.0 / T.FPS / SUBSTEPS
                for s in range(SUBSTEPS):
                    tt = t0 + s * dt
                    if tt < T.FREEZE:
                        self._step(tt, dt, lights)
            self.frame = f

    # ── checkpoints, so parallel workers need not replay everything ──
    def save(self, path):
        np.savez(path, x=self.x, v=self.v, x_prev=self.x_prev, frame=self.frame)

    def load(self, path):
        d = np.load(path)
        self.x, self.v, self.x_prev = d["x"], d["v"], d["x_prev"]
        self.frame = int(d["frame"])


def checkpoint_path(cache_dir, frame):
    return os.path.join(cache_dir, f"dust_{frame:05d}.npz")


def precompute_checkpoints(lights, cache_dir, last_frame, every=40, n=220_000):
    os.makedirs(cache_dir, exist_ok=True)
    dust = Dust(n=n)
    dust.save(checkpoint_path(cache_dir, 0))
    for f in range(every, last_frame + 1, every):
        dust.advance_to(f, lights)
        dust.save(checkpoint_path(cache_dir, f))


def dust_for_frame(lights, cache_dir, frame, every=40, n=220_000, current=None):
    """Return a Dust whose state is at `frame`, reusing `current` if it is behind."""
    if current is not None and current.frame <= frame and frame - current.frame <= every:
        current.advance_to(frame, lights)
        return current
    base = (frame // every) * every
    dust = Dust(n=n)
    p = checkpoint_path(cache_dir, base) if cache_dir else None
    if p and os.path.exists(p):
        dust.load(p)
    dust.advance_to(frame, lights)
    return dust
