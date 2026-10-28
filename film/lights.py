"""The lights: knots that learn harmony.

Each light travels on a torus knot. It winds around the central axis and
around its own tube at two rates whose ratio is a just-intonation interval:
3:2 (fifth), 5:4 (third), 4:3 (fourth) … The first light begins out of tune.
Its ratio wanders and its knot never closes, until the ratio settles on a
perfect fifth and the figure locks.

A torus knot whose tube radius shrinks to zero is a circle. In the
convergence every tube collapses, every axis aligns, and every ratio stops
mattering: all the knots were winding around the same circle the whole time.
"""
from dataclasses import dataclass, field

import numpy as np

from . import timeline as T

SIM_DT = 1.0 / 600.0
F0 = 0.27          # base frequency of the shared clock (Hz)
R_RING = 1.0


def _cumint(rate, dt):
    out = np.zeros_like(rate)
    out[1:] = np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt)
    return out


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(np.shape(a) + (3, 3))
    R[..., 0, 0] = c; R[..., 0, 1] = -s
    R[..., 1, 0] = s; R[..., 1, 1] = c
    R[..., 2, 2] = 1.0
    return R


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(np.shape(a) + (3, 3))
    R[..., 0, 0] = 1.0
    R[..., 1, 1] = c; R[..., 1, 2] = -s
    R[..., 2, 1] = s; R[..., 2, 2] = c
    return R


@dataclass
class Voice:
    name: str
    p: int                # windings around the axis
    q: int                # windings around the tube   (q/p = the interval)
    color: tuple          # linear RGB
    pitch: float          # Hz — the note this light rings
    born: float
    parent: int
    R: float              # major radius
    r: float              # tube radius
    lon: float            # longitude speed multiplier
    tilt: float           # tilt of the torus axis (rad)
    spin: float           # precession of the tilt (rad/s)
    az0: float            # initial precession angle
    phase: tuple          # (longitude, meridian) offsets
    detune: float = 0.0


VOICES = [
    Voice("fifth", 2, 3, (1.00, 0.64, 0.32), 880.00, 0.0, -1,
          0.92, 0.42, 1.00, 0.22, 0.045, 0.0, (0.0, 0.0), 0.0035),
    Voice("third", 4, 5, (1.00, 0.36, 0.42), 739.99, T.SPLIT1, 0,
          0.70, 0.30, 0.85, 0.38, -0.060, 1.9, (0.0, 1.3), -0.0042),
    Voice("fourth", 3, 4, (0.38, 0.62, 1.00), 783.99, T.SPLIT1, 0,
          1.18, 0.26, 0.80, 0.30, 0.070, 4.1, (0.0, 2.6), 0.0030),
    Voice("sixth", 3, 5, (1.00, 0.78, 0.48), 987.77, T.SPLIT2, 1,
          0.52, 0.24, 1.25, 0.55, 0.085, 0.7, (0.0, 0.4), -0.0028),
    Voice("octave", 1, 2, (0.70, 0.82, 1.00), 1174.66, T.SPLIT2, 2,
          1.42, 0.36, 0.62, 0.18, -0.050, 2.8, (0.0, 3.0), 0.0050),
    Voice("second", 8, 9, (0.66, 0.46, 1.00), 1318.51, T.SPLIT2, 0,
          0.98, 0.14, 0.95, 0.46, 0.062, 5.2, (0.0, 1.9), -0.0020),
]


@dataclass
class Lights:
    t: np.ndarray                    # (Nt,)
    pos: np.ndarray                  # (L, Nt, 3)
    lum: np.ndarray                  # (L, Nt)
    voices: list
    conv: np.ndarray = None          # (Nt,) convergence weight
    chimes: list = field(default_factory=list)   # (time, light, pitch, strength)

    @property
    def colors(self):
        return np.array([v.color for v in self.voices], dtype=np.float64)

    def index(self, time):
        return int(np.clip(np.floor(time / SIM_DT), 0, len(self.t) - 1))

    def head(self, time):
        """Positions (L,3), velocities (L,3), luminosities (L,) at `time`."""
        frozen = time >= T.FREEZE - 1e-9
        time = min(time, self.t[-1])
        f = time / SIM_DT
        i0 = int(np.clip(np.floor(f), 0, len(self.t) - 2))
        a = f - i0
        p = self.pos[:, i0] * (1 - a) + self.pos[:, i0 + 1] * a
        v = (self.pos[:, i0 + 1] - self.pos[:, i0]) / SIM_DT
        if frozen:
            v = np.zeros_like(v)
        L = self.lum[:, i0] * (1 - a) + self.lum[:, i0 + 1] * a
        return p, v, L


def _wander(t):
    """The out-of-tune search: a ratio drifting around 3:2."""
    return (0.24 * np.sin(2 * np.pi * 0.13 * t + 1.1)
            + 0.15 * np.sin(2 * np.pi * 0.31 * t + 2.3)
            + 0.06 * np.sin(2 * np.pi * 0.71 * t + 0.4))


def _hesitation(t):
    """The first movements stop and start."""
    h = np.ones_like(t)
    for (c, width, depth) in [(8.2, 0.50, 0.85), (9.7, 0.45, 0.75), (11.0, 0.45, 0.6), (12.3, 0.35, 0.45)]:
        h -= depth * np.exp(-((t - c) / width) ** 2)
    return np.clip(h, 0.06, 1.0)


def build():
    t = np.arange(0.0, T.FREEZE + SIM_DT * 0.5, SIM_DT)
    n = len(t)
    L = len(VOICES)

    # ── the shared clock ──────────────────────────────────────────────
    search = T.smootherstep(T.MOVE, T.LOCK, t)
    speed = (0.45 + 0.55 * search ** 0.8) * np.where(t < T.LOCK, _hesitation(t), 1.0)
    speed *= 1.0 + 0.12 * T.smoothstep(T.SPLIT2, T.CONVERGE, t)
    speed = np.where(t < T.MOVE, 0.0, speed)

    conv = T.smootherstep(T.CONVERGE, T.FREEZE - 0.55, t)
    # in the convergence the clock accelerates toward a blur
    ring_hz = 0.20 + 1.45 * T.smoothstep(T.CONVERGE, T.FREEZE, t) ** 2.0
    omega = 2 * np.pi * (F0 * speed * (1 - conv) + ring_hz * conv)
    Phi = _cumint(omega, SIM_DT)

    lock = T.smoothstep(T.LOCK - 0.8, T.LOCK, t)
    ratio0 = 1.5 + _wander(t) * (1.0 - lock)

    pos = np.zeros((L, n, 3))
    lum = np.zeros((L, n))
    dPhi = np.diff(Phi, prepend=0.0)

    # longitudes are integrated first so the beads can be spaced evenly on the ring
    lons = []
    for i, v in enumerate(VOICES):
        lon_rate = v.lon * (1 - conv) + 1.0 * conv
        lons.append(-np.cumsum(lon_rate * dPhi) + v.phase[0])
    lons = np.array(lons)
    end = lons[:, -1]
    target = end[0] - 2 * np.pi * np.arange(L) / L
    corr = np.angle(np.exp(1j * (target - end)))
    lons += corr[:, None] * conv[None, :]

    for i, v in enumerate(VOICES):
        if i == 0:
            grow = (0.45 * T.smootherstep(T.MOVE, T.LOCK + 0.4, t) ** 0.75
                    + 0.55 * T.smootherstep(T.LOCK, T.SPLIT1, t))
            Rm = v.R * grow
            rm = v.r * grow * (0.6 + 0.4 * lock)
            mer_rate = v.lon * ratio0 * (1.0 + v.detune * lock)
        else:
            g = T.smootherstep(v.born, v.born + 4.0, t)
            Rm = v.R * (0.55 + 0.45 * g)
            rm = v.r * g
            mer_rate = v.lon * (v.q / v.p) * (1.0 + v.detune)
        mer = np.cumsum(mer_rate * dPhi) + v.phase[1]

        Rm = Rm * (1 - conv) + R_RING * conv
        rm = rm * (1 - conv) ** 1.5
        lon = lons[i]
        rho = Rm + rm * np.cos(mer)
        local = np.stack([rho * np.cos(lon), rho * np.sin(lon), rm * np.sin(mer)], axis=-1)

        if i == 0:
            # before it moves, it only breathes
            breath = 0.004 * np.stack([np.sin(2 * np.pi * t / 3.1), np.sin(2 * np.pi * t / 4.3 + 1),
                                       np.sin(2 * np.pi * t / 5.3 + 2)], axis=-1)
            local += breath * (1.0 - T.smoothstep(T.MOVE, T.MOVE + 2, t))[:, None]

        # gyroscopic tilt of each torus, settling flat for the ring
        tilt = v.tilt * (1 - conv) * (0.85 + 0.15 * np.sin(0.37 * t + i))
        az = v.az0 + v.spin * t
        rot = np.einsum("nij,njk,nkl->nil", _rz(az), _rx(tilt), _rz(-az))
        world = np.einsum("nij,nj->ni", rot, local)

        if v.parent >= 0:
            e = T.smootherstep(v.born, v.born + 3.2, t)[:, None]
            world = pos[v.parent] * (1.0 - e) + world * e
        pos[i] = world

        # ── luminosity ────────────────────────────────────────────────
        if i == 0:
            Lm = T.flicker_envelope(t)
            Lm = Lm * (1.0 + 0.14 * np.sin(2 * np.pi * (t - T.IGNITE) / 3.1)
                       * T.smoothstep(T.IGNITE, T.IGNITE + 1, t) * (1.0 - T.smoothstep(T.MOVE, T.MOVE + 1.5, t)))
            Lm = Lm * (1.0 + 1.2 * np.exp(-((t - T.LOCK - 0.35) / 0.55) ** 2))
        else:
            Lm = T.smoothstep(v.born, v.born + 1.0, t) * (0.9 + 0.1 * np.sin(2 * np.pi * 0.21 * t + i))
            Lm = Lm * (1.0 + 2.5 * np.exp(-((t - v.born - 0.25) / 0.35) ** 2) * (t >= v.born))
        Lm = Lm * (1.0 + 1.4 * conv ** 2)
        lum[i] = np.where(t >= v.born, Lm, 0.0)

    lights = Lights(t=t, pos=pos, lum=lum, voices=VOICES, conv=conv)
    lights.chimes = _chimes(lights, lons, ratio0)
    return lights


def _chimes(lights, lons, ratio0):
    """Each light rings its note once per turn around the axis.

    The melody is not composed: it is read off the motion.
    """
    t = lights.t
    events = []
    for i, v in enumerate(lights.voices):
        turns = np.floor(-(lons[i] - v.phase[0]) / (2 * np.pi) * 2.0)   # twice per revolution
        idx = np.nonzero(np.diff(turns) > 0)[0]
        for j in idx:
            tt = float(t[j])
            if tt < max(v.born + 0.5, T.MOVE + 0.8) or tt > T.CONVERGE + 3.0:
                continue
            pitch = v.pitch
            if i == 0 and tt < T.LOCK:
                pitch = 587.33 * float(ratio0[j])     # the out-of-tune fifth above D5
            conv = float(lights.conv[j])
            strength = float(lights.lum[i, j]) / (1.0 + 1.4 * conv ** 2)
            events.append((tt, i, pitch, strength))
    events.sort()
    return events
