"""Acts V–VII: what remains.

Stare at a bright light and look away at a white wall: you will see its
shape again, dark and in the complementary colour. That negative
afterimage is the hinge of the film. The circle of light burns out into
white; the white becomes washi paper; the afterimage floats on it; and a
single brush stroke traces the afterimage before it fades — the ensō, the
Zen circle drawn in one breath, that cannot be corrected or repeated.
Then an inscription, as on a Zen painting, and a seal.
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import timeline as T
from .optics import grain, quantize

SS = 2                                     # supersampling of the paper canvas
PAPER_RGB = np.array([0.930, 0.905, 0.848], np.float32)      # display-referred
SUMI = np.array([3.05, 3.15, 3.30], np.float32)               # absorbance of dense ink
WASH = np.array([0.80, 0.86, 0.95], np.float32)               # absorbance of diluted ink
SEAL_RGB = np.array([0.74, 0.17, 0.13], np.float32)
WHITE = np.array([1.0, 0.988, 0.962], np.float32)

ENSO_START = np.radians(122.0)             # screen angle (y down): lower left
ENSO_SWEEP = np.radians(322.0)             # clockwise, leaving a gap at the bottom


def _font_path(name):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "assets", "fonts", name)


# ── paper ─────────────────────────────────────────────────────────────

def make_paper(W, H, seed=11):
    """Washi: cloudy formation, long kozo fibres, fine tooth. Returns (H,W,3) float32."""
    rng = np.random.default_rng(seed)

    def noise(scale, sigma):
        h, w = max(H // scale, 4), max(W // scale, 4)
        n = rng.standard_normal((h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        n /= n.std() + 1e-6
        return cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)

    cloud = 0.016 * noise(32, 3.0) + 0.012 * noise(8, 2.0)
    tooth = 0.010 * noise(1, 0.7)

    fib = np.zeros((H, W), np.uint8)
    dark = np.zeros((H, W), np.uint8)
    n_fib = int(9000 * (W * H) / (3840 * 2160))
    for k in range(n_fib):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        ang = rng.uniform(0, np.pi)
        L = rng.uniform(30, 420) * (W / 3840)
        steps = max(int(L / 6), 3)
        curv = rng.normal(0, 0.035)
        pts = []
        for s in range(steps):
            pts.append((x, y))
            ang += curv + rng.normal(0, 0.03)
            x += np.cos(ang) * 6 * (W / 3840)
            y += np.sin(ang) * 6 * (W / 3840)
        pts = np.array(pts, np.int32).reshape(-1, 1, 2)
        target = fib if rng.random() < 0.72 else dark
        cv2.polylines(target, [pts], False, int(rng.uniform(40, 150)), 1, cv2.LINE_AA)
    fibres = (fib.astype(np.float32) * 0.00028 - dark.astype(np.float32) * 0.00018)
    fibres = cv2.GaussianBlur(fibres, (0, 0), 0.6)

    lum = 1.0 + cloud + tooth + fibres
    paper = PAPER_RGB[None, None, :] * lum[..., None]
    # a warm, slightly uneven light falling across the sheet
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    light = 1.0 + 0.035 * (1.0 - x / W) * (1.0 - 0.6 * y / H) - 0.02 * (y / H)
    paper *= light[..., None]
    fibre_map = cv2.GaussianBlur(fib.astype(np.float32) / 255.0, (0, 0), 1.2)
    return paper.astype(np.float32), fibre_map.astype(np.float32), (cloud * 30).astype(np.float32)


# ── the ensō ──────────────────────────────────────────────────────────

class Enso:
    """One brush stroke, described in its own coordinates.

    Every pixel near the circle is given stroke coordinates: s, how far
    along the stroke (0 = the press, 1 = the lift) and u, where across the
    brush (-1 inner edge … +1 outer edge). Ink is then a function of (s, u):
    a load that runs out along s, and a field of dryness that is long along
    the stroke and narrow across it, so the paper shows through in streaks
    as the brush goes dry (kasure).
    """

    def __init__(self, cx, cy, r0, W, H, paper_cloud, fibre_map, seed=3):
        rng = np.random.default_rng(seed)
        self.cx, self.cy, self.r0 = cx, cy, r0
        self.W, self.H = W, H
        self.s = np.linspace(-0.2, 1.2, 2801)
        s = self.s
        self.rpath = r0 * (1.0 + 0.020 * np.sin(2 * np.pi * 1.15 * s + 0.6) + 0.011 * np.sin(2 * np.pi * 2.6 * s + 2.0)
                           - 0.030 * np.clip(s, 0, None) ** 2.4)
        w0 = 0.150 * r0
        self.wpath = w0 * (1.0 + 0.20 * np.exp(-((s - 0.012) / 0.035) ** 2)
                           + 0.10 * np.sin(2 * np.pi * 0.85 * s + 1.4)
                           - 0.50 * T.smoothstep(0.66, 1.0, s) ** 1.2)
        self.th = ENSO_START + ENSO_SWEEP * s

        # region of interest: the annulus around the circle
        R = r0 + w0 * 1.6
        x0, x1 = int(max(cx - R, 0)), int(min(cx + R, W))
        y0, y1 = int(max(cy - R, 0)), int(min(cy + R, H))
        self.roi = (y0, y1, x0, x1)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = (xx - cx) / 1.012, yy - cy
        rpix = np.sqrt(dx * dx + dy * dy)
        ang = np.arctan2(dy, dx)
        a = np.mod(ang - ENSO_START, 2 * np.pi)
        gap = 2 * np.pi - ENSO_SWEEP
        a = np.where(a > ENSO_SWEEP + 0.55 * gap, a - 2 * np.pi, a)
        sraw = a / ENSO_SWEEP
        sc = np.clip(sraw, 0.0, 1.0)
        rc = np.interp(sc, s, self.rpath).astype(np.float32)
        wc = np.interp(sc, s, self.wpath).astype(np.float32)
        u = (rpix - rc) / (0.5 * wc)
        # the press: a rounded pool of ink before s = 0
        start = np.array([cx + self.rpath[np.searchsorted(s, 0)] * np.cos(ENSO_START) * 1.012,
                          cy + self.rpath[np.searchsorted(s, 0)] * np.sin(ENSO_START)])
        dstart = np.sqrt((xx - start[0]) ** 2 + (yy - start[1]) ** 2) / (0.5 * self.wpath[np.searchsorted(s, 0)])
        blob = cv2.GaussianBlur(rng.standard_normal(dstart.shape).astype(np.float32), (0, 0), 9.0 * W / 3840)
        blob /= blob.std() + 1e-6
        dstart = dstart * (1.0 + 0.035 * blob)
        self.start = start
        self.cap_r = 0.5 * self.wpath[np.searchsorted(s, 0)]
        self.dstart_px = np.sqrt((xx - start[0]) ** 2 + (yy - start[1]) ** 2).astype(np.float32)
        self.near_start = (sraw < 0.03).astype(np.float32)
        before = sraw < 0
        u = np.where(before, np.sign(u) * np.maximum(np.abs(u), dstart), u)

        # ── anisotropic fields in (s, u) space ─────────────────────────
        S_RES, U_RES = 2400, 640
        def field(sig_s, sig_u):
            n = rng.standard_normal((U_RES, S_RES)).astype(np.float32)
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=sig_s, sigmaY=sig_u)
            n = (n - n.mean()) / (n.std() + 1e-6)
            return n
        streak = 0.75 * field(70, 4.5) + 0.45 * field(14, 2.2) + 0.25 * field(4, 1.4)
        streak = 0.5 + 0.5 * np.tanh(streak / 1.2)
        tone = 0.5 + 0.5 * np.tanh(field(120, 18) / 1.5)
        edge1 = field(25, 400)[0]
        edge2 = field(9, 400)[0]
        mx = ((sraw + 0.2) / 1.4 * (S_RES - 1)).astype(np.float32)
        my = ((np.clip(u, -1.6, 1.6) + 1.6) / 3.2 * (U_RES - 1)).astype(np.float32)
        N = cv2.remap(streak, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        Tn = cv2.remap(tone, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        ex = np.clip(mx, 0, S_RES - 1).astype(np.int32)
        rough = 0.055 * edge1[ex] + 0.03 * edge2[ex]
        rough = np.where(u > 0, rough, -0.8 * rough + 0.02 * edge2[np.clip(S_RES - 1 - ex, 0, S_RES - 1)])
        grain_ = cv2.GaussianBlur(rng.standard_normal(u.shape).astype(np.float32), (0, 0), 0.9)
        grain_ /= grain_.std() + 1e-6
        cloud = paper_cloud[y0:y1, x0:x1]

        # ── ink ───────────────────────────────────────────────────────
        load = (1.05 + 0.45 * np.exp(-(np.clip(sraw, -1, None) / 0.06) ** 2)
                - 1.02 * T.smoothstep(0.24, 1.04, sraw) ** 1.45).astype(np.float32)
        dry = 0.20 + 0.62 * N + 0.42 * np.abs(u) ** 5
        catch = load - dry + 0.10 * grain_ + 0.025 * cloud
        cov = T.smoothstep(-0.06, 0.06, catch).astype(np.float32)
        shape = T.smoothstep(1.0 + rough + 0.05, 1.0 + rough - 0.05, np.abs(u)).astype(np.float32)
        # the lift: hairlines run on past the end of the stroke
        tail_len = 0.035 + 0.05 * (1 - N)
        tail = T.smoothstep(1.0 + tail_len, 1.0, sraw) * (N < 0.33) * (np.abs(u) < 0.75)
        shape = np.where(sraw > 1.0, shape * tail, shape)
        shape = np.where(sraw < -0.12, 0.0, shape)
        wet = np.interp(sraw, [-0.1, 0.0, 0.1, 0.45, 0.8, 1.0], [1.0, 1.0, 0.85, 0.5, 0.2, 0.05]).astype(np.float32)
        dens = cov * shape * (0.90 + 0.16 * (Tn - 0.5)) * (0.93 + 0.12 * N)
        # pigment gathers at the edges while the ink is wet
        dens *= 1.0 + 0.22 * wet * T.smoothstep(0.62, 0.95, np.abs(u))
        core = np.clip(dens, 0, 1.25).astype(np.float32)

        # nijimi — the wet beginning bleeds into the fibres
        fib = fibre_map[y0:y1, x0:x1]
        soft = cv2.GaussianBlur(core, (0, 0), 6.0 * W / 3840)
        halo = np.clip(soft - core * 0.5, 0, 1) * wet * (0.30 + 1.1 * fib) * 1.25
        halo += 0.30 * np.clip(cv2.GaussianBlur(core, (0, 0), 2.0 * W / 3840) - core, 0, 1) * (0.4 + wet)

        self.core = core
        self.wash = np.clip(halo * 0.75, 0, 1).astype(np.float32)
        self.s_pix = np.clip(sraw, 0.0, 1.0).astype(np.float32)

    def head(self, s_now):
        r = float(np.interp(s_now, self.s, self.rpath))
        th = ENSO_START + ENSO_SWEEP * s_now
        return (self.cx + r * np.cos(th) * 1.012, self.cy + r * np.sin(th), float(np.interp(s_now, self.s, self.wpath)))


def stroke_progress(t):
    u = float(T.clamp01((t - T.STROKE_START) / (T.STROKE_END - T.STROKE_START)))
    # a slow press, a sweep, a slowing lift
    return float(np.clip(u - 0.72 * np.sin(2 * np.pi * u) / (2 * np.pi), 0, 1))


def stroke_time_of(s):
    us = np.linspace(0, 1, 2001)
    ss = np.array([np.clip(u - 0.72 * np.sin(2 * np.pi * u) / (2 * np.pi), 0, 1) for u in us])
    return T.STROKE_START + np.interp(s, ss, us) * (T.STROKE_END - T.STROKE_START)


# ── words and seal ────────────────────────────────────────────────────

class Inscription:
    def __init__(self, W, H, scale, fibre_map):
        self.W, self.H = W, H
        size = int(round(42 * scale))
        adv = 48 * scale
        font = ImageFont.truetype(_font_path("YujiSyuku-Regular.ttf"), size)
        xs = [568 * scale, 506 * scale, 444 * scale]
        tops = [188 * scale, 188 * scale, 262 * scale]
        self.chars = []   # (t_start, x0, y0, mask)
        for (t0, text), cx, top in zip(T.COLUMNS, xs, tops):
            y = top
            k = 0
            for ch in text:
                if ch == "　":
                    y += adv * 0.55
                    k += 2
                    continue
                pad = int(size * 0.35)
                im = Image.new("L", (size + 2 * pad, size + 2 * pad), 0)
                ImageDraw.Draw(im).text((pad + size / 2, pad + size / 2), ch, fill=255, font=font, anchor="mm")
                m = np.asarray(im, np.float32) / 255.0
                x0 = int(round(cx - m.shape[1] / 2))
                y0 = int(round(y + size / 2 - m.shape[0] / 2))
                self.chars.append((t0 + k * T.CHAR_STAGGER, x0, y0, m))
                y += adv
                k += 1
        self.fibre = fibre_map
        self.end_y = y

    def density(self, t, out):
        """Add ink density of every character written by time t into `out` (H,W)."""
        for (t0, x0, y0, m) in self.chars:
            if t < t0:
                continue
            u = (t - t0) / 0.42
            h = m.shape[0]
            yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
            wipe = np.clip(u * 1.35 - yy, 0, 1) ** 0.7 if u < 1.4 else 1.0
            soak = float(T.smoothstep(0, 1.6, t - t0))
            sharp = m * wipe
            bled = cv2.GaussianBlur(sharp, (0, 0), 1.0 + 1.2 * (1 - soak)) if u < 3 else cv2.GaussianBlur(sharp, (0, 0), 1.0)
            dens = 0.88 * sharp + 0.30 * bled * (0.6 + 0.6 * self.fibre[y0:y0 + h, x0:x0 + m.shape[1]])
            out[y0:y0 + h, x0:x0 + m.shape[1]] += dens
        return out


class Seal:
    """一期一会 — once, only once — carved white into vermilion stone."""

    def __init__(self, scale, x, y, rng_seed=5):
        size = int(round(74 * scale))
        rng = np.random.default_rng(rng_seed)
        font = ImageFont.truetype(_font_path("ShipporiMincho-Bold.ttf"), int(size * 0.43))
        b = int(size * 0.065)
        # right column read first: 一 期 ; left column: 一 会
        cells = [("一", 0.73, 0.28), ("期", 0.73, 0.71), ("一", 0.27, 0.28), ("会", 0.27, 0.71)]
        m = np.ones((size, size), np.float32)
        cell = int(size * 0.47)
        for ch, fx, fy in cells:
            ci = Image.new("L", (cell, cell), 255)
            ImageDraw.Draw(ci).text((cell / 2, cell / 2), ch, fill=0, font=font, anchor="mm")
            c = np.asarray(ci, np.float32) / 255.0
            if ch == "一":
                c = cv2.resize(c, (int(cell * 0.72), cell), interpolation=cv2.INTER_AREA)
                c = np.pad(c, ((0, 0), ((cell - c.shape[1]) // 2, cell - c.shape[1] - (cell - c.shape[1]) // 2)),
                           constant_values=1.0)
            x0 = int(fx * size - cell / 2); y0 = int(fy * size - cell / 2)
            m[y0:y0 + cell, x0:x0 + cell] = np.minimum(m[y0:y0 + cell, x0:x0 + cell], c)
        # stone: chipped border, uneven paste
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        border = np.minimum.reduce([xx, yy, size - 1 - xx, size - 1 - yy])
        chip = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), 2.0 * scale)
        chip /= chip.std() + 1e-6
        edge_ok = border + chip * 1.4 * scale > b * 0.15
        paste = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), 1.2 * scale)
        paste = 0.9 + 0.08 * paste / (paste.std() + 1e-6)
        speck = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 0.8 * scale)
        speck = (speck > np.percentile(speck, 4)).astype(np.float32)
        a = m * edge_ok * paste * speck
        a = cv2.GaussianBlur(a, (0, 0), 0.5 * scale)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), -1.6, 1.0)
        self.alpha = cv2.warpAffine(np.clip(a, 0, 1), M, (size, size), flags=cv2.INTER_LINEAR)
        self.x, self.y = int(x - size / 2), int(y)
        self.size = size


# ── the act ───────────────────────────────────────────────────────────

class InkAct:
    def __init__(self, width, height, ring_center, ring_radius, afterimage_hdr, cache_dir=None):
        self.W, self.H = width, height
        W2, H2 = width * SS, height * SS
        self.W2, self.H2 = W2, H2
        self.scale = width / T.WIDTH
        paper, fibre, cloud = self._cached(cache_dir, "paper.npz", lambda: make_paper(W2, H2))
        self.paper_hi = paper
        self.paper = cv2.resize(paper, (width, height), interpolation=cv2.INTER_AREA)
        self.fibre = fibre
        cx, cy = ring_center
        self.enso = Enso(cx * SS, cy * SS, ring_radius * SS, W2, H2, cloud, fibre)
        self.words = Inscription(W2, H2, self.scale * SS, fibre)
        self.seal = Seal(self.scale * SS, 444 * self.scale * SS, self.words.end_y + 30 * self.scale * SS)
        self.ghost = self._ghost(afterimage_hdr)
        self.arrival = stroke_time_of(self.enso.s_pix)

    @staticmethod
    def _cached(cache_dir, name, fn):
        if cache_dir:
            p = os.path.join(cache_dir, name)
            if os.path.exists(p):
                d = np.load(p)
                return d["a"], d["b"], d["c"]
            a, b, c = fn()
            np.savez(p, a=a, b=b, c=c)
            return a, b, c
        return fn()

    def _ghost(self, hdr):
        lum = hdr[..., 0] * 0.35 + hdr[..., 1] * 0.5 + hdr[..., 2] * 0.15
        g = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 3.0 * self.scale)
        g = g / (np.percentile(g, 99.7) + 1e-6)
        g = 1.0 - np.exp(-2.2 * np.clip(g, 0, 4))
        return g.astype(np.float32)

    # ── frame ────────────────────────────────────────────────────────
    def _ink_density(self, t):
        e = self.enso
        core = np.zeros((self.H2, self.W2), np.float32)
        wash = np.zeros((self.H2, self.W2), np.float32)
        if t >= T.STROKE_START:
            y0, y1, x0, x1 = e.roi
            s_now = stroke_progress(t)
            hx, hy, hw = e.head(s_now)
            reveal = T.smoothstep(-0.003, 0.0, s_now - e.s_pix).astype(np.float32)
            # the brush touches down: the pool of ink spreads from a point
            press = float(T.smoothstep(T.STROKE_START - 0.03, T.STROKE_START + 0.24, t))
            if press < 1.0:
                rho = e.cap_r * (0.15 + 1.1 * press)
                grow = T.smoothstep(rho + 2.0, rho - 2.0, e.dstart_px).astype(np.float32)
                reveal = reveal * (1 - e.near_start) + reveal * e.near_start * grow
            hw = hw * (0.3 + 0.7 * press)
            if s_now < 1.0:
                # the round belly of the brush at the leading edge
                yy, xx = np.ogrid[y0:y1, x0:x1]
                dist = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2).astype(np.float32)
                belly = T.smoothstep(hw * 0.62, hw * 0.48, dist).astype(np.float32) * (e.s_pix < s_now + 0.04)
                reveal = np.maximum(reveal, belly)
            age = np.clip((t - self.arrival) / 1.1, 0, 1).astype(np.float32)
            core[y0:y1, x0:x1] = e.core * reveal
            wash[y0:y1, x0:x1] = e.wash * reveal * age
        words = np.zeros((self.H2, self.W2), np.float32)
        self.words.density(t, words)
        return core, wash, words

    def render(self, frame):
        t = frame / T.FPS
        sc = self.scale
        if t >= T.FADE_OUT[1]:
            return self._title(frame, t)

        core, wash, words = self._ink_density(t)
        core1 = cv2.resize(core, (self.W, self.H), interpolation=cv2.INTER_AREA)
        wash1 = cv2.resize(wash, (self.W, self.H), interpolation=cv2.INTER_AREA)
        words1 = cv2.resize(words, (self.W, self.H), interpolation=cv2.INTER_AREA)
        dens = core1[..., None] * SUMI + wash1[..., None] * WASH + np.clip(words1, 0, 1.3)[..., None] * SUMI * 0.92
        img = self.paper * np.exp(-dens)

        # the seal
        if t >= T.SEAL:
            k = float(T.smoothstep(T.SEAL, T.SEAL + 0.09, t))
            sl = self.seal
            a = cv2.resize(sl.alpha, (sl.size // SS, sl.size // SS), interpolation=cv2.INTER_AREA)
            sz = a.shape[0]
            zoom = 1.0 + 0.05 * (1 - k)
            if zoom != 1.0:
                M = cv2.getRotationMatrix2D((sz / 2, sz / 2), 0, zoom)
                a = cv2.warpAffine(a, M, (sz, sz))
            x0, y0 = sl.x // SS, sl.y // SS
            region = img[y0:y0 + sz, x0:x0 + sz]
            paper_l = region.mean(axis=-1, keepdims=True) / PAPER_RGB.mean()
            seal = SEAL_RGB * paper_l
            region[:] = region * (1 - a[..., None] * k) + seal * (a[..., None] * k)

        # the flash: white, then paper
        wv = 1.0 - float(T.smoothstep(T.WHITE_HOLD, T.PAPER_IN, t))
        if wv > 0:
            wv = wv ** 0.8
            img = img * (1 - wv) + WHITE * wv

        # the afterimage: dark, cool, drifting as the eye drifts
        a = self._ghost_amount(t)
        if a > 1e-3:
            dx = 3.0 * sc * np.sin(0.9 * (t - T.FLASH)) + 1.5 * sc * np.sin(2.3 * (t - T.FLASH) + 1)
            dy = 2.0 * sc * np.sin(0.7 * (t - T.FLASH) + 2) + 1.0 * sc * np.sin(1.9 * (t - T.FLASH))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            g = cv2.warpAffine(self.ghost, M, (self.W, self.H), borderMode=cv2.BORDER_CONSTANT)
            blur = 0.5 + 3.5 * float(T.smoothstep(T.FLASH, T.GHOST_GONE, t))
            g = cv2.GaussianBlur(g, (0, 0), blur * sc)
            tint = np.array([0.60, 0.46, 0.24], np.float32)
            img = img * (1.0 - a * g[..., None] * tint)

        # the slowest push-in, then the light in the room goes out
        z = 1.0 + 0.045 * float(T.smoothstep(T.FLASH, T.FADE_OUT[1], t))
        if abs(z - 1) > 1e-4:
            M = cv2.getRotationMatrix2D((self.W / 2, self.H / 2), 0, z)
            img = cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        y, x = np.ogrid[0:self.H, 0:self.W]
        vig = 1.0 - 0.11 * np.clip((((x - self.W / 2) / (self.W / 2)) ** 2 + ((y - self.H / 2) / (self.W / 2)) ** 2) / 0.6, 0, 1.6) ** 1.5
        img = img * vig[..., None].astype(np.float32) ** (1 - wv)
        fade = 1.0 - float(T.smoothstep(*T.FADE_OUT, t))
        img = img * fade ** 1.4
        img = grain(img, frame, amount=0.016)
        return quantize(img, frame)

    def _ghost_amount(self, t):
        rise = float(T.smoothstep(T.FLASH + 0.15, T.PAPER_IN - 0.4, t))
        fall = 1.0 - float(T.smoothstep(T.STROKE_START + 0.6, T.GHOST_GONE, t))
        return 0.78 * rise * fall

    # ── coda ─────────────────────────────────────────────────────────
    def _title(self, frame, t):
        if not hasattr(self, "_title_img"):
            self._title_img = self._make_title()
        a0, a1 = T.TITLE
        k = float(T.smoothstep(a0, a0 + 1.4, t) * (1 - T.smoothstep(a1 - 1.4, a1, t)))
        img = self._title_img * k
        # and in the dark, somewhere, a light tries to be born
        flick = 0.0
        for (dt0, d, peak) in ((0.0, 0.08, 0.55),):
            tt = t - (T.END - 0.85) - dt0
            if 0 <= tt <= d:
                flick = peak * np.sin(np.pi * tt / d)
        if flick > 0:
            y, x = np.ogrid[0:self.H, 0:self.W]
            r2 = ((x - self.W / 2) ** 2 + (y - self.H / 2) ** 2) / (self.scale ** 2)
            glow = flick * (np.exp(-r2 / (2 * 2.5 ** 2)) + 0.25 * np.exp(-r2 / (2 * 14 ** 2)) + 0.05 * np.exp(-r2 / (2 * 50 ** 2)))
            img = img + glow[..., None].astype(np.float32) * np.array([1.0, 0.85, 0.62], np.float32)
        img = np.clip(img, 0, 1) + np.array([0.004, 0.005, 0.008], np.float32)
        img = grain(img, frame, amount=0.018)
        return quantize(img, frame)

    def _make_title(self):
        sc = self.scale
        W, H = self.W * SS, self.H * SS
        im = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(im)
        f1 = ImageFont.truetype(_font_path("ShipporiMincho-Regular.ttf"), int(64 * sc * SS))
        f2 = ImageFont.truetype(_font_path("CormorantGaramond-Regular.ttf"), int(21 * sc * SS))
        cx, cy = W / 2, H / 2 - 18 * sc * SS
        # 残 像, generously spaced
        d.text((cx - 44 * sc * SS, cy), "残", fill=255, font=f1, anchor="mm")
        d.text((cx + 44 * sc * SS, cy), "像", fill=255, font=f1, anchor="mm")
        label = "A F T E R I M A G E"
        d.text((cx, cy + 82 * sc * SS), label, fill=200, font=f2, anchor="mm")
        m = np.asarray(im, np.float32) / 255.0
        m = cv2.resize(m, (self.W, self.H), interpolation=cv2.INTER_AREA)
        glow = cv2.GaussianBlur(m, (0, 0), 6 * sc) * 0.35 + cv2.GaussianBlur(m, (0, 0), 22 * sc) * 0.2
        col = np.array([0.93, 0.90, 0.84], np.float32)
        return (np.clip(m * 0.92 + glow, 0, 1))[..., None] * col
