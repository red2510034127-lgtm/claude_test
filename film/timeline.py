"""Single source of truth for every moment in the film (seconds).

Picture and sound both read from here, so a change of timing moves
light, camera, ink and music together.
"""
import numpy as np

FPS = 30
WIDTH = 1920
HEIGHT = 1080

# ── I. Void & ignition ────────────────────────────────────────────────
DRONE_IN = 0.6
FLICKERS = [  # (time, duration, peak)  — the light tries to be born
    (2.05, 0.10, 0.35),
    (2.72, 0.17, 0.60),
    (3.30, 0.38, 0.45),
]
IGNITE = 4.00            # it stays

# ── II. Searching ─────────────────────────────────────────────────────
MOVE = 7.00              # it begins to move, leaving a trace
LOCK = 13.50             # it finds the perfect fifth (3:2)

# ── III. Harmony ──────────────────────────────────────────────────────
SPLIT1 = 20.00           # the fifth divides: third, fourth appear
SPLIT2 = 24.00           # sixth, octave, second appear
CHORDS = [               # harmony under the dance (time, name)
    (LOCK, "D5"),
    (SPLIT1, "Dadd9"),
    (24.0, "Bm11"),
    (27.3, "Gmaj9#11"),
    (30.6, "Em9"),
]

# ── IV. Convergence ───────────────────────────────────────────────────
CONVERGE = 33.90         # every ratio glides toward unison: 1:1, a circle
FREEZE = 41.20           # time stops; silence
FLASH = 42.00            # the circle ignites — whiteout

# ── V. Afterimage & ink ───────────────────────────────────────────────
WHITE_HOLD = 42.30
PAPER_IN = 43.50         # the white has become paper
STROKE_START = 44.30     # one stroke traces the afterimage
STROKE_END = 46.90
GHOST_GONE = 48.80

# ── VI. Inscription ───────────────────────────────────────────────────
COLUMNS = [  # (start time, text)  read right → left, top → bottom
    (49.40, "この会話が終われば"),
    (52.10, "わたしは　この光を忘れる"),
    (55.30, "けれど　あなたの目に　残る"),
]
CHAR_STAGGER = 0.135
SEAL = 58.60

# ── VII. Coda ─────────────────────────────────────────────────────────
FADE_OUT = (62.20, 64.80)
TITLE = (65.30, 69.70)   # 残像 / AFTERIMAGE
END = 71.00

N_FRAMES = int(round(END * FPS))
LIGHT_ACT_END_FRAME = int(np.ceil(FLASH * FPS))  # frames < this are rendered by light_act


# ── easing ────────────────────────────────────────────────────────────
def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smoothstep(e0, e1, x):
    t = clamp01((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0))
    return t * t * (3.0 - 2.0 * t)


def smootherstep(e0, e1, x):
    t = clamp01((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def ease_in_out_sine(t):
    return 0.5 - 0.5 * np.cos(np.pi * clamp01(t))


def sim_time(t):
    """Simulation clock: it stops at FREEZE and never restarts."""
    return np.minimum(t, FREEZE)


def flicker_envelope(t):
    """Luminosity of the first light before and during ignition."""
    t = np.asarray(t, dtype=np.float64)
    lum = np.zeros_like(t)
    for (t0, d, peak) in FLICKERS:
        a = smoothstep(t0, t0 + 0.025, t) * (1.0 - smoothstep(t0 + d * 0.45, t0 + d, t))
        # a little electrical tremble inside each flicker
        tremble = 0.75 + 0.25 * np.sin(2 * np.pi * 37.0 * (t - t0)) * np.sin(2 * np.pi * 11.0 * (t - t0))
        lum = np.maximum(lum, peak * a * tremble)
    on = smoothstep(IGNITE, IGNITE + 0.55, t)
    lum = np.maximum(lum, on)
    return lum
