"""One continuous take.

There are no cuts in this film. The camera begins a hand's breadth from a
newborn light, drifts back as the light learns to move, threads through the
woven structure, and finally rises above it to look straight down the axis,
where every orbit has become one circle.
"""
from dataclasses import dataclass

import numpy as np

from . import timeline as T

# (time, position, target)
KEYS = [
    (0.0,  (0.000, -0.640, 0.090), (0.0, 0.0, 0.0)),
    (4.0,  (0.020, -0.600, 0.095), (0.0, 0.0, 0.0)),
    (7.0,  (0.060, -0.640, 0.110), (0.0, 0.0, 0.0)),
    (10.2, (0.300, -1.050, 0.300), (0.0, 0.0, 0.0)),
    (13.5, (0.700, -1.750, 0.600), (0.0, 0.0, 0.0)),
    (17.0, (1.500, -2.500, 1.000), (0.0, 0.0, 0.0)),
    (20.0, (2.550, -2.050, 1.250), (0.0, 0.0, 0.0)),
    (23.2, (3.000,  0.250, 0.650), (0.0, 0.0, 0.0)),
    (26.2, (1.700,  1.550, 0.200), (0.0, 0.0, 0.0)),
    (28.6, (0.420,  0.950, 0.060), (-0.3, -0.2, 0.0)),
    (31.0, (-1.300, -0.350, 0.300), (0.0, 0.0, 0.0)),
    (33.6, (-3.100, -1.900, 1.300), (0.0, 0.0, 0.0)),
    (36.6, (-1.900, -1.800, 3.700), (0.0, 0.0, 0.0)),
    (39.2, (-0.300, -0.450, 5.600), (-0.15, 0.0, 0.0)),
    (41.2, (0.000,  0.000, 6.050), (-0.20, 0.0, 0.0)),
    (42.0, (0.000,  0.000, 5.880), (-0.20, 0.0, 0.0)),
]

# (time, vertical fov deg, aperture radius in world units, focus-on-head weight)
LENS = [
    (0.0, 31.0, 0.034, 1.0),
    (7.0, 31.0, 0.032, 1.0),
    (13.5, 34.0, 0.040, 0.8),
    (20.0, 38.0, 0.050, 0.3),
    (26.2, 40.0, 0.040, 0.2),
    (28.6, 42.0, 0.030, 0.6),
    (31.0, 40.0, 0.045, 0.2),
    (34.0, 38.0, 0.060, 0.0),
    (39.2, 31.0, 0.050, 0.0),
    (42.0, 30.0, 0.050, 0.0),
]


@dataclass
class Camera:
    pos: np.ndarray
    fwd: np.ndarray
    right: np.ndarray
    up: np.ndarray
    f_px: float
    focus: float
    aperture: float
    width: int
    height: int

    def project(self, P):
        """World points (N,3) → screen x, y, depth z, circle of confusion (px)."""
        d = P - self.pos
        z = d @ self.fwd
        x = d @ self.right
        y = d @ self.up
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        sx = self.width * 0.5 + self.f_px * x / zs
        sy = self.height * 0.5 - self.f_px * y / zs
        coc = self.aperture * self.f_px * np.abs(1.0 / self.focus - 1.0 / zs)
        return sx, sy, z, coc


def _hermite(keys_t, values, t):
    """C1 Catmull–Rom style interpolation with time-aware tangents."""
    keys_t = np.asarray(keys_t, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(keys_t)
    t = float(np.clip(t, keys_t[0], keys_t[-1]))
    k = int(np.clip(np.searchsorted(keys_t, t, side="right") - 1, 0, n - 2))

    def tangent(i):
        if i == 0 or i == n - 1:
            j0, j1 = max(i - 1, 0), min(i + 1, n - 1)
        else:
            j0, j1 = i - 1, i + 1
        return (values[j1] - values[j0]) / (keys_t[j1] - keys_t[j0])

    t0, t1 = keys_t[k], keys_t[k + 1]
    h = t1 - t0
    s = (t - t0) / h
    m0, m1 = tangent(k) * h, tangent(k + 1) * h
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return h00 * values[k] + h10 * m0 + h01 * values[k + 1] + h11 * m1


def _smooth_lens(t):
    kt = np.array([k[0] for k in LENS])
    vals = np.array([k[1:] for k in LENS])
    t = float(np.clip(t, kt[0], kt[-1]))
    k = int(np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 2))
    s = T.smootherstep(kt[k], kt[k + 1], t)
    return vals[k] * (1 - s) + vals[k + 1] * s


def _shake(t):
    """A breathing, hand-held drift; stillness once time has stopped."""
    amp = 0.0035 * (1.0 - T.smoothstep(T.CONVERGE + 3.0, T.FREEZE, t))
    return amp * np.array([
        np.sin(2 * np.pi * 0.31 * t + 0.2) + 0.5 * np.sin(2 * np.pi * 0.83 * t + 1.7),
        np.sin(2 * np.pi * 0.27 * t + 2.2) + 0.5 * np.sin(2 * np.pi * 0.71 * t + 0.4),
        np.sin(2 * np.pi * 0.37 * t + 1.1) + 0.5 * np.sin(2 * np.pi * 0.91 * t + 2.9),
    ])


def _raw_pose(t):
    kt = [k[0] for k in KEYS]
    pos = _hermite(kt, [k[1] for k in KEYS], t)
    tgt = _hermite(kt, [k[2] for k in KEYS], t)
    dist = np.linalg.norm(pos - tgt)
    pos = pos + _shake(t) * max(dist, 0.5)
    return pos, tgt


def camera_at(t, lights, width=T.WIDTH, height=T.HEIGHT):
    pos, tgt = _raw_pose(t)
    fwd = tgt - pos
    fwd /= np.linalg.norm(fwd)
    # world-up turns from +z (side views) to +y (looking down the axis)
    w = T.smootherstep(T.CONVERGE + 1.0, 39.0, t)
    up_hint = np.array([0.0, 0.0, 1.0]) * (1 - w) + np.array([0.0, 1.0, 0.0]) * w
    right = np.cross(fwd, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    fov, aperture, head_w = _smooth_lens(t)
    f_px = 0.5 * height / np.tan(np.radians(fov) * 0.5)

    focus = _focus(t, lights)
    return Camera(pos, fwd, right, up, f_px, focus, aperture, width, height)


def _focus_raw(t, lights):
    pos, tgt = _raw_pose(t)
    fwd = (tgt - pos) / np.linalg.norm(tgt - pos)
    _, _, head_w = _smooth_lens(t)
    p, _, _ = lights.head(T.sim_time(t))
    d_head = max(float((p[0] - pos) @ fwd), 0.08)
    d_tgt = float((tgt - pos) @ fwd)
    if t > T.CONVERGE:
        d_tgt = float((np.zeros(3) - pos) @ fwd)   # the ring's plane
    return head_w * d_head + (1 - head_w) * d_tgt


def _focus(t, lights):
    """A focus puller follows, smoothly and a little late."""
    offs = np.linspace(-0.45, 0.15, 9)
    wts = np.exp(-((offs + 0.15) / 0.22) ** 2)
    vals = [_focus_raw(max(t + o, 0.0), lights) for o in offs]
    return float(np.sum(wts * np.array(vals)) / np.sum(wts))
