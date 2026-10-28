"""The score — synthesised from nothing, sample by sample.

Nothing here is recorded or borrowed. Every sound is built from sine
waves and noise, and wherever possible it is driven by the picture:

  · the first light's flickers tick; when it stays, a glass tone rings
  · while it searches, a voice glides out of tune, following its ratio,
    and settles on a perfect fifth at the instant the knot locks
  · each light rings its own note every half-turn — the melody is read
    off the motion, and panned to where the light is on screen
  · a heartbeat enters with the first division and quickens as the orbits
    collapse; every voice of the harmony glides toward unison
  · silence, when time stops
  · the flash; then a high faint ringing — the ear's own afterimage
  · a brush on paper; a felt piano under the inscription; a seal
  · and the last sound is the first: a tiny tick, as a light tries again
"""
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, istft, sosfilt, stft

from . import timeline as T
from .camera import camera_at

SR = 48_000
D2, A2, D3 = 73.42, 110.0, 146.83


def note(name):
    names = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}
    n, o = name[:-1], int(name[-1])
    midi = 12 * (o + 1) + names[n]
    return 440.0 * 2 ** ((midi - 69) / 12)


class Mix:
    def __init__(self, dur):
        self.n = int(dur * SR)
        self.dry = np.zeros((self.n, 2))
        self.send = np.zeros((self.n, 2))

    def add(self, t0, mono, gain=1.0, pan=0.0, rev=0.3):
        i0 = int(round(t0 * SR))
        if i0 >= self.n:
            return
        if i0 < 0:
            mono = mono[-i0:]
            i0 = 0
        mono = mono[: self.n - i0]
        pan = np.clip(pan, -1, 1)
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        seg = np.stack([mono * l, mono * r], -1) * gain
        self.dry[i0:i0 + len(mono)] += seg
        self.send[i0:i0 + len(mono)] += seg * rev

    def add_stereo(self, t0, st, gain=1.0, rev=0.3):
        i0 = int(round(t0 * SR))
        st = st[: self.n - i0]
        self.dry[i0:i0 + len(st)] += st * gain
        self.send[i0:i0 + len(st)] += st * gain * rev


# ── instruments ───────────────────────────────────────────────────────

def env_ad(n, attack, decay, sr=SR):
    t = np.arange(n) / sr
    return np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / decay)


def glass(f, dur=4.0, decay=2.2, index=1.6, ratio=3.5, bright=1.0):
    """FM glass bell."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    I = index * bright * np.exp(-t / 0.35)
    y = np.sin(2 * np.pi * f * t + I * np.sin(2 * np.pi * f * ratio * t))
    y += 0.18 * np.sin(2 * np.pi * f * 2.756 * t) * np.exp(-t / (decay * 0.35))
    y += 0.08 * np.sin(2 * np.pi * f * 5.404 * t) * np.exp(-t / (decay * 0.15))
    return y * env_ad(n, 0.0015, decay)


def tick(f=3520.0, dur=0.25):
    n = int(dur * SR)
    t = np.arange(n) / SR
    y = np.sin(2 * np.pi * f * t + 2.0 * np.exp(-t / 0.01) * np.sin(2 * np.pi * f * 1.41 * t))
    return y * env_ad(n, 0.0004, 0.035)


def felt_piano(f, dur=6.0, vel=0.6, rng=None):
    """Additive piano with inharmonic, two-stage decaying partials, muffled by felt."""
    rng = rng or np.random.default_rng(int(f * 10))
    n = int(dur * SR)
    t = np.arange(n) / SR
    B = 0.00035
    y = np.zeros(n)
    tau0 = 5.5 * (220.0 / f) ** 0.45
    nmax = int(min(24, 9000 / f))
    soft = 2.6 + 4.5 * vel
    for k in range(1, nmax + 1):
        fk = k * f * np.sqrt(1 + B * k * k)
        a = (1.0 / k ** 0.9) * np.exp(-(k - 1) / soft)
        tf, ts = tau0 * 0.18 / (1 + 0.3 * k), tau0 / (1 + 0.18 * k)
        e = 0.55 * np.exp(-t / tf) + 0.45 * np.exp(-t / ts)
        for det in (-0.25, 0.3):           # two strings, slightly apart: a slow beat
            fd = fk * 2 ** (det / 1200)
            y += 0.5 * a * e * np.sin(2 * np.pi * fd * t + rng.uniform(0, 6.28))
    y *= np.minimum(t / 0.004, 1.0)
    thump = rng.standard_normal(int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / (0.006 * SR))
    thump = sosfilt(butter(2, 900, "lp", fs=SR, output="sos"), thump)
    y[: len(thump)] += thump * 0.25 * vel
    y *= np.minimum(1.0, (n - np.arange(n)) / (0.3 * SR))        # release at the end of the buffer
    return y * vel


def pad_voice(freq_fn, t0, t1, attack=1.6, release=2.6, harmonics=10, seed=0, bright=1.0):
    """Warm additive pad: three detuned copies, soft harmonics. freq_fn(t_abs) → Hz."""
    rng = np.random.default_rng(seed)
    n = int((t1 - t0 + release) * SR)
    ta = t0 + np.arange(n) / SR
    f = freq_fn(ta)
    y = np.zeros(n)
    for det in (-6.0, 0.0, 5.0):
        ph = 2 * np.pi * np.cumsum(f * 2 ** (det / 1200)) / SR
        for k in range(1, harmonics + 1):
            a = 1.0 / k ** 1.7 * np.exp(-(k - 1) / (2.5 * bright))
            y += a * np.sin(k * ph + rng.uniform(0, 6.28))
    tt = ta - t0
    env = np.minimum(tt / attack, 1.0) ** 2 * np.clip(1.0 - (ta - t1) / release, 0, 1) ** 1.5
    env *= 1.0 + 0.08 * np.sin(2 * np.pi * 0.17 * tt + seed)
    return y * env / 3.0


def kick(f0=95.0, f1=42.0, dur=0.6, decay=0.22):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = f1 + (f0 - f1) * np.exp(-t / 0.035)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return y * env_ad(n, 0.002, decay)


def shaped_noise(dur, gain_fn, rng, nperseg=2048):
    """Noise whose spectrum over time is gain_fn(times, freqs) → (F, T) amplitudes."""
    n = int(dur * SR)
    x = rng.standard_normal(n)
    f, tt, Z = stft(x, fs=SR, nperseg=nperseg)
    G = gain_fn(tt[None, :], f[:, None])
    _, y = istft(Z * G, fs=SR, nperseg=nperseg)
    return y[:n]


def reverb_ir(seconds=4.8, seed=1):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = np.zeros((n, 2))
    bands = [((None, 400), 1.45), ((400, 2500), 1.05), ((2500, None), 0.55)]
    for ch in range(2):
        for (lo, hi), tau in bands:
            x = rng.standard_normal(n)
            if lo is None:
                sos = butter(2, hi, "lp", fs=SR, output="sos")
            elif hi is None:
                sos = butter(2, lo, "hp", fs=SR, output="sos")
            else:
                sos = butter(2, [lo, hi], "bp", fs=SR, output="sos")
            ir[:, ch] += sosfilt(sos, x) * np.exp(-t / (tau / 3.0)) * 0.6
    pre = int(0.024 * SR)
    ir = np.concatenate([np.zeros((pre, 2)), ir])[:n]
    ir[:, :] *= np.minimum(np.arange(n) / (0.01 * SR), 1.0)[:, None]
    return ir / np.sqrt((ir ** 2).sum(axis=0, keepdims=True))


# ── the score ─────────────────────────────────────────────────────────

def screen_pan(lights, time, idx):
    tt = min(time, T.FREEZE - 1e-3)
    cam = camera_at(tt, lights)
    p, _, _ = lights.head(tt)
    sx, _, z, _ = cam.project(p[idx:idx + 1])
    if z[0] <= 0:
        return 0.0, 1.0
    pan = float(np.clip((sx[0] - T.WIDTH / 2) / (T.WIDTH / 2), -1, 1)) * 0.85
    dist = float(np.linalg.norm(p[idx] - cam.pos))
    return pan, float(np.clip(1.3 / max(dist, 0.3), 0.35, 1.6))


CHORDS = {
    "D5": ["D2", "A2", "D3", "A3"],
    "Dadd9": ["D2", "A2", "F#3", "A3", "E4"],
    "Bm11": ["B1", "F#2", "D3", "A3", "E4", "F#4"],
    "Gmaj9#11": ["G1", "D3", "F#3", "A3", "B3", "C#4"],
    "Em9": ["E2", "B2", "D3", "F#3", "G3"],
}
UNISON = ["D2", "D3", "A3", "D4", "A4"]


def render_score(path, lights):
    dur = T.END
    rng = np.random.default_rng(2026)
    mix = Mix(dur)
    t_all = np.arange(mix.n) / SR

    # ── the void: a low drone and moving air ──────────────────────────
    drone_t = t_all
    swell = T.smoothstep(T.DRONE_IN, T.LOCK, drone_t) * 0.55 + T.smoothstep(T.LOCK, T.CONVERGE, drone_t) * 0.45
    swell = swell * (1 + 0.6 * T.smoothstep(T.CONVERGE, T.FREEZE, drone_t))
    swell *= 1 - T.smoothstep(T.FREEZE, T.FREEZE + 0.01, drone_t)
    drone = (np.sin(2 * np.pi * 36.71 * drone_t) + 0.55 * np.sin(2 * np.pi * 73.42 * drone_t + 1)
             + 0.25 * np.sin(2 * np.pi * 110.0 * drone_t + 2) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * drone_t)))
    mix.add(0, drone * swell, gain=0.055, rev=0.15)

    def air_gain(tt, ff):
        a = 0.6 * T.smoothstep(0.3, 4.0, tt) + 0.4 * T.smoothstep(T.LOCK, T.CONVERGE, tt)
        a = a * (1 + 2.0 * T.smoothstep(T.CONVERGE, T.FREEZE, tt) ** 2)
        a = a * (1 - T.smoothstep(T.FREEZE, T.FREEZE + 0.01, tt))
        centre = 500 + 2500 * T.smoothstep(T.CONVERGE, T.FREEZE, tt) ** 1.5
        return a * np.exp(-((np.log((ff + 30) / centre)) / 0.9) ** 2) / np.sqrt(ff + 60)
    air = shaped_noise(T.FREEZE + 0.2, air_gain, rng)
    mix.add_stereo(0, np.stack([air, np.roll(air, 2400)], -1), gain=0.10, rev=0.3)

    # ── ignition ──────────────────────────────────────────────────────
    for k, (t0, d, peak) in enumerate(T.FLICKERS):
        mix.add(t0, tick(3520.0 * (1 + 0.03 * k)), gain=0.05 + 0.05 * peak, pan=-0.05, rev=0.6)
    mix.add(T.IGNITE, glass(note("D6"), dur=7, decay=3.2, index=0.9), gain=0.10, pan=0.0, rev=0.7)
    mix.add(T.IGNITE + 0.02, glass(note("D5"), dur=7, decay=3.8, index=0.5), gain=0.06, pan=0.0, rev=0.7)

    # the light breathes
    tb = np.arange(int((T.MOVE + 1.5 - T.IGNITE) * SR)) / SR
    breath = 0.5 + 0.5 * np.sin(2 * np.pi * tb / 3.1 - np.pi / 2)
    hum = (np.sin(2 * np.pi * note("D4") * tb) + 0.4 * np.sin(2 * np.pi * note("A4") * tb)) * breath
    hum *= T.smoothstep(0, 1.0, tb) * (1 - T.smoothstep(T.MOVE - T.IGNITE, T.MOVE + 1.5 - T.IGNITE, tb))
    mix.add(T.IGNITE, hum, gain=0.018, rev=0.5)

    # ── the searching voice ───────────────────────────────────────────
    t0, t1 = T.MOVE - 0.3, T.SPLIT1
    tv = np.arange(int((t1 - t0) * SR)) / SR + t0
    lt = lights.t
    from .lights import _wander
    lock = T.smoothstep(T.LOCK - 0.8, T.LOCK, tv)
    ratio = 1.5 + _wander(tv) * (1 - lock)
    f = 587.33 * ratio * (1 + 0.004 * np.sin(2 * np.pi * 5.2 * tv) * (1 - 0.6 * lock))
    spd = np.interp(tv, lt, np.linalg.norm(np.gradient(lights.pos[0], axis=0), axis=-1) / (lt[1] - lt[0]))
    spd_env = np.clip(spd / 0.5, 0, 1) ** 0.7
    amp = spd_env * T.smoothstep(T.MOVE - 0.3, T.MOVE + 1.2, tv) * (1 - T.smoothstep(T.LOCK + 2.0, T.SPLIT1, tv))
    amp = amp * (1 + 0.8 * np.exp(-((tv - T.LOCK - 0.4) / 0.7) ** 2))
    ph = 2 * np.pi * np.cumsum(f) / SR
    voice = (np.sin(ph) + 0.22 * np.sin(2 * ph + 0.4) + 0.06 * np.sin(3 * ph)) * amp
    # pan with the light
    pans = np.array([screen_pan(lights, x, 0)[0] for x in np.arange(t0, t1, 0.1)])
    pan_curve = np.interp(tv, np.arange(t0, t1, 0.1), pans)
    l, r = np.cos((pan_curve + 1) * np.pi / 4), np.sin((pan_curve + 1) * np.pi / 4)
    mix.add_stereo(t0, np.stack([voice * l, voice * r], -1), gain=0.045, rev=0.55)

    # ── chimes, read off the motion ───────────────────────────────────
    for (tc, i, pitch, strength) in lights.chimes:
        pan, near = screen_pan(lights, tc, i)
        g = 0.05 * min(strength, 1.6) * near
        if i == 0 and tc < T.LOCK:
            g *= 0.8
        late = T.smoothstep(T.CONVERGE, T.CONVERGE + 3.0, tc)
        mix.add(tc, glass(pitch, dur=4.5, decay=2.0, index=1.3), gain=g * (1 - 0.7 * late), pan=pan, rev=0.55)

    # ── harmony ───────────────────────────────────────────────────────
    seq = T.CHORDS + [(T.CONVERGE, "converge")]
    for k, (tc, name) in enumerate(T.CHORDS):
        tend = seq[k + 1][0]
        notes = CHORDS[name]
        for j, nm in enumerate(notes):
            fr = note(nm)
            if name == "Em9":
                # the last chord does not end: its voices glide toward unison
                target = note(UNISON[min(j, len(UNISON) - 1)])
                def fn(ta, fr=fr, target=target):
                    g = T.smootherstep(T.CONVERGE, T.FREEZE - 0.8, ta)
                    return fr * (target / fr) ** g
                v = pad_voice(fn, tc, T.FREEZE, attack=1.8, release=0.01, seed=k * 10 + j)
            else:
                v = pad_voice(lambda ta, fr=fr: np.full_like(ta, fr), tc, tend + 0.3, attack=1.8, release=2.4,
                              seed=k * 10 + j)
            lvl = 0.030 * (1.0 if fr > 120 else 1.3)
            build = 0.7 + 0.3 * T.smoothstep(T.LOCK, T.CONVERGE, tc)
            pan = ((j / max(len(notes) - 1, 1)) - 0.5) * 0.7
            mix.add(tc, v, gain=lvl * build, pan=pan, rev=0.45)
    # extra unison voices arriving during the convergence, with a quickening tremolo
    for j, nm in enumerate(UNISON):
        fr = note(nm)
        ta0, ta1 = T.CONVERGE + 1.0, T.FREEZE
        v = pad_voice(lambda ta, fr=fr: np.full_like(ta, fr), ta0, ta1, attack=4.5, release=0.01,
                      seed=90 + j, bright=1.4)
        tt = np.arange(len(v)) / SR + ta0
        hz = 0.5 + 9.0 * T.smoothstep(T.CONVERGE, T.FREEZE, tt) ** 2
        trem = 0.75 + 0.25 * np.sin(2 * np.pi * np.cumsum(hz) / SR)
        mix.add(ta0, v * trem * T.smoothstep(ta0, T.FREEZE, tt) ** 1.5, gain=0.045, pan=(j - 2) * 0.2, rev=0.4)

    # ── heartbeat ─────────────────────────────────────────────────────
    tb = T.SPLIT1
    while tb < T.FREEZE - 0.15:
        prog = float(T.smoothstep(T.SPLIT1, T.FREEZE, tb))
        bpm = 64 + 34 * prog ** 1.8
        lvl = 0.11 + 0.21 * prog
        mix.add(tb, kick(), gain=lvl, rev=0.12)
        mix.add(tb + 0.26 * 60 / bpm * 1.1, kick(80, 40, decay=0.16), gain=lvl * 0.55, rev=0.12)
        tb += 60.0 / bpm

    # ── riser: an endless ascent (Shepard), and breath drawn in ───────
    t0, t1 = T.CONVERGE, T.FREEZE
    tr = np.arange(int((t1 - t0) * SR)) / SR
    prog = tr / (t1 - t0)
    riser = np.zeros_like(tr)
    for k in range(7):
        oc = (k + prog * 2.2) % 7
        fr = 55.0 * 2 ** oc
        w = np.exp(-((oc - 3.5) / 1.6) ** 2)
        riser += w * np.sin(2 * np.pi * np.cumsum(fr) / SR + k)
    riser *= prog ** 2.2
    mix.add(t0, riser, gain=0.05, rev=0.3)

    # ── silence (master gate below) ───────────────────────────────────

    # ── the flash ─────────────────────────────────────────────────────
    tf = T.FLASH
    nb = int(4.0 * SR)
    tt = np.arange(nb) / SR
    fboom = 30 + 45 * np.exp(-tt / 0.12)
    boom = np.sin(2 * np.pi * np.cumsum(fboom) / SR) * env_ad(nb, 0.003, 1.3)
    mix.add(tf, boom, gain=0.55, rev=0.15)
    burst = rng.standard_normal(int(2.5 * SR)) * env_ad(int(2.5 * SR), 0.002, 0.35)
    burst = sosfilt(butter(2, 2200, "lp", fs=SR, output="sos"), burst)
    mix.add_stereo(tf, np.stack([burst, np.roll(burst, 300)], -1), gain=0.16, rev=1.0)
    for j, nm in enumerate(["D2", "A2", "D3", "F#3", "A3", "E4", "F#4", "A4", "C#5", "D5", "A5", "E6"]):
        fr = note(nm)
        if fr < 300:
            v = pad_voice(lambda ta, fr=fr: np.full_like(ta, fr), tf, tf + 0.1, attack=0.01, release=7.0, seed=200 + j)
            mix.add(tf, v, gain=0.06, pan=(j / 11 - 0.5) * 0.8, rev=0.6)
        else:
            mix.add(tf + 0.004 * j, glass(fr, dur=9, decay=3.6, index=0.8), gain=0.045, pan=(j / 11 - 0.5) * 1.2,
                    rev=0.9)

    # the ear's own afterimage: a faint high ringing that fades as the white does
    nr = int(6.5 * SR)
    tr = np.arange(nr) / SR
    ring = np.sin(2 * np.pi * 6272.0 * tr) * T.smoothstep(0.05, 0.8, tr) * np.exp(-tr / 2.2)
    mix.add(tf, ring, gain=0.0035, pan=0.15, rev=0.05)

    # ── the brush ─────────────────────────────────────────────────────
    from .ink_act import stroke_progress
    ts0, ts1 = T.STROKE_START - 0.1, T.STROKE_END + 0.3
    times = np.linspace(ts0, ts1, 400)
    prog = np.array([stroke_progress(x) for x in times])
    speed = np.gradient(prog, times)
    def brush_gain(tt, ff):
        ta = tt + ts0
        sp = np.interp(ta, times, speed)
        pr = np.interp(ta, times, prog)
        press = np.exp(-((ta - T.STROKE_START - 0.05) / 0.08) ** 2)
        a = np.clip(sp / speed.max(), 0, 1) ** 0.8 + 0.6 * press
        centre = 1400 + 2600 * pr
        return a * np.exp(-((np.log((ff + 50) / centre)) / 0.75) ** 2)
    swish = shaped_noise(ts1 - ts0, brush_gain, rng, nperseg=1024)
    # dry bristles catching on fibre: sparse crackle, thicker toward the lift
    crack = np.zeros_like(swish)
    tc = np.arange(len(swish)) / SR + ts0
    dens = np.interp(tc, times, np.array([stroke_progress(x) for x in times])) ** 2 * (tc < T.STROKE_END + 0.05)
    hits = rng.random(len(swish)) < dens * 0.004
    crack[hits] = rng.standard_normal(hits.sum())
    crack = sosfilt(butter(2, 2500, "hp", fs=SR, output="sos"), crack)
    pan_b = np.interp(tc, times, [np.cos(np.radians(122) + np.radians(322) * stroke_progress(x)) * 0.5 for x in times])
    sig = swish * 1.0 + crack * 0.6
    l, r = np.cos((pan_b + 1) * np.pi / 4), np.sin((pan_b + 1) * np.pi / 4)
    mix.add_stereo(ts0, np.stack([sig * l, sig * r], -1), gain=0.12, rev=0.25)

    # ── felt piano ────────────────────────────────────────────────────
    c1, c2, c3 = (c[0] for c in T.COLUMNS)
    phrase = [
        (47.30, ["F#4"], 0.42), (48.55, ["A4"], 0.38),
        (c1 + 0.05, ["D3", "D5"], 0.50), (c1 + 1.45, ["C#5"], 0.36),
        (c2 + 0.05, ["G2", "B4"], 0.48), (c2 + 1.50, ["A4"], 0.36), (c2 + 2.35, ["F#4"], 0.30),
        (c3 + 0.05, ["D2", "A3", "F#4"], 0.48), (c3 + 1.40, ["E4"], 0.34), (c3 + 2.30, ["D4"], 0.32),
        (T.SEAL + 0.02, ["D2", "A2", "F#3", "D4"], 0.42),
        (T.SEAL + 1.60, ["A5"], 0.30),
    ]
    for (tn, names, vel) in phrase:
        for k, nm in enumerate(names):
            fr = note(nm)
            pan = np.clip((np.log2(fr / 261.6)) * 0.25, -0.6, 0.6)
            mix.add(tn + 0.012 * k, felt_piano(fr, dur=7.0, vel=vel), gain=0.16, pan=pan, rev=0.5)

    # ── the seal ──────────────────────────────────────────────────────
    ns = int(0.5 * SR)
    th = np.arange(ns) / SR
    thump = np.sin(2 * np.pi * (70 + 60 * np.exp(-th / 0.01)) * th) * env_ad(ns, 0.001, 0.07)
    knock = sosfilt(butter(2, [300, 1800], "bp", fs=SR, output="sos"), rng.standard_normal(ns)) * env_ad(ns, 0.0005, 0.02)
    mix.add(T.SEAL, thump * 0.8 + knock * 0.5, gain=0.35, pan=-0.35, rev=0.35)

    # ── coda: the birth tone again, then the first tick ───────────────
    mix.add(T.TITLE[0] + 0.3, glass(note("D6"), dur=6, decay=3.0, index=0.6), gain=0.05, rev=0.8)
    mix.add(T.END - 0.85, tick(3520.0), gain=0.07, rev=0.6)

    # ── reverb, silence, master ───────────────────────────────────────
    ir = reverb_ir()
    wet = np.stack([fftconvolve(mix.send[:, ch], ir[:, ch])[: mix.n] for ch in range(2)], -1)
    out = mix.dry + wet * 0.55
    gate = np.ones(mix.n)
    a = int(T.FREEZE * SR)
    b = int(T.FLASH * SR)
    fade = int(0.012 * SR)
    gate[a:a + fade] = np.linspace(1, 0, fade)
    gate[a + fade:b] = 0.0
    out *= gate[:, None]
    # a breath of room tone so the silence is a room, not a dropout
    room = sosfilt(butter(2, 900, "lp", fs=SR, output="sos"), rng.standard_normal((mix.n, 2)), axis=0)
    out[a + fade:b] += room[a + fade:b] * 0.0006

    out = sosfilt(butter(2, 24, "hp", fs=SR, output="sos"), out, axis=0)
    out[-int(0.08 * SR):] *= np.linspace(1, 0, int(0.08 * SR))[:, None]
    peak = np.abs(out).max()
    out = out / peak * 0.89          # −1 dBFS
    out = np.tanh(out * 1.05) / np.tanh(1.05)
    wavfile.write(path, SR, (np.clip(out, -1, 1) * 32767).astype(np.int16))
    return out
