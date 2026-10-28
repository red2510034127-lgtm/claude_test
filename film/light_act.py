"""Acts I–IV: the light.

Every frame is a long exposure of everything the lights have done so far.
A trail's brightness falls with age — recent motion burns, older motion
lingers as a faint memory — but within the film nothing is ever completely
forgotten until the flash.
"""
import cv2
import numpy as np

from . import timeline as T
from .camera import camera_at
from .dust import dust_for_frame
from .lights import SIM_DT
from .optics import (Film, aces, bloom, chromatic_aberration, grain, halation,
                     quantize, to_srgb, vignette)

K_TRAIL = 900.0
K_HEAD = 40.0
GLOW = [(4.0, 1.3), (14.0, 0.32), (45.0, 0.085), (130.0, 0.022)]   # (sigma px, peak per unit lum)
K_DUST = 0.55
V_REF = 0.35          # below this speed a light deposits trace ∝ distance, not time
BACKGROUND = np.array([0.0011, 0.0014, 0.0024], np.float32)

# (time, exposure)
EXPOSURE = [
    (0.0, 1.30), (4.0, 1.30), (7.0, 1.25), (13.5, 1.25), (20.0, 1.35),
    (26.0, 1.30), (29.0, 1.15), (34.0, 1.25), (38.0, 1.05), (41.2, 1.00),
]


def exposure_at(t):
    kt = np.array([e[0] for e in EXPOSURE])
    kv = np.array([e[1] for e in EXPOSURE])
    k = int(np.clip(np.searchsorted(kt, t, side="right") - 1, 0, len(kt) - 2))
    s = T.smootherstep(kt[k], kt[k + 1], t)
    return float(kv[k] * (1 - s) + kv[k + 1] * s)


def trail_weight(age, ts, conv_now):
    hot = 2.4 * np.exp(-age / 0.085)
    warm = 0.90 * np.exp(-age / (4.2 * (1.0 - 0.72 * conv_now)))
    floor = 0.10
    w = hot + warm + floor
    # in the convergence older memories recede so the circle can stand alone
    w = np.where(ts < T.CONVERGE, w * (1.0 - 0.93 * conv_now), w)
    return w


def _hg(cos, g=0.55):
    return (1 - g * g) / (4 * np.pi * np.power(1 + g * g - 2 * g * cos, 1.5))


class LightAct:
    def __init__(self, lights, width=T.WIDTH, height=T.HEIGHT, dust_cache=None, dust_n=220_000):
        self.lights = lights
        self.W, self.H = width, height
        self.film = Film(width, height)
        self.dust_cache = dust_cache
        self.dust_n = dust_n
        self.dust = None
        self.scale = width / T.WIDTH        # resolution scale for preview renders
        cols = lights.colors.astype(np.float32)
        self.colors = cols
        # per-sample speed for every light (for the distance-weighted deposit)
        d = np.linalg.norm(np.diff(lights.pos, axis=1), axis=-1) / SIM_DT
        self.speed = np.concatenate([d[:, :1], d], axis=1)

    # ── emitters ──────────────────────────────────────────────────────
    def _trails(self, cam, tau):
        L = self.lights
        j_end = L.index(tau)
        conv_now = float(L.conv[j_end])
        for i, v in enumerate(L.voices):
            j0 = L.index(v.born)
            if j_end - j0 < 2:
                continue
            P = L.pos[i, j0:j_end + 1]
            ts = L.t[j0:j_end + 1]
            lum = L.lum[i, j0:j_end + 1]
            spd = self.speed[i, j0:j_end + 1]
            E = lum * SIM_DT * np.minimum(1.0, spd / V_REF) * trail_weight(tau - ts, ts, conv_now) * K_TRAIL
            self._polyline(cam, P, E, self.colors[i])

    def _polyline(self, cam, P, E, color):
        sx, sy, z, coc = cam.project(P)
        coc = coc * self.scale
        valid = z > 0.03
        a, b = slice(0, -1), slice(1, None)
        seg_ok = valid[a] & valid[b]
        # cull segments wholly outside the frame on one side
        mx = 120.0 * self.scale
        out = (((sx[a] < -mx) & (sx[b] < -mx)) | ((sx[a] > self.W + mx) & (sx[b] > self.W + mx)) |
               ((sy[a] < -mx) & (sy[b] < -mx)) | ((sy[a] > self.H + mx) & (sy[b] > self.H + mx)))
        seg_ok &= ~out
        Eseg = 0.5 * (E[a] + E[b])
        seg_ok &= Eseg > 1e-7
        idx = np.nonzero(seg_ok)[0]
        if len(idx) == 0:
            return
        x0, y0, c0 = sx[idx], sy[idx], coc[idx]
        x1, y1, c1 = sx[idx + 1], sy[idx + 1], coc[idx + 1]
        Es = Eseg[idx]
        length = np.hypot(x1 - x0, y1 - y0)
        spacing = np.maximum(0.7, 0.4 * np.maximum(c0, c1))
        n = np.clip(np.ceil(length / spacing), 1, 3000).astype(np.int64)
        total = int(n.sum())
        seg = np.repeat(np.arange(len(idx)), n)
        start = np.repeat(np.cumsum(n) - n, n)
        fr = ((np.arange(total) - start) + 0.5) / np.repeat(n, n)
        X = x0[seg] + (x1 - x0)[seg] * fr
        Y = y0[seg] + (y1 - y0)[seg] * fr
        C = c0[seg] + (c1 - c0)[seg] * fr
        e = (Es / n)[seg]
        self.film.splat(X, Y, C, e[:, None] * color[None, :])

    def _heads(self, cam, tau):
        p, _, Lm = self.lights.head(tau)
        sx, sy, z, coc = cam.project(p)
        ok = (z > 0.03) & (Lm > 1e-4)
        self._glows = []
        if ok.any():
            E = (Lm[ok] * K_HEAD)[:, None] * (self.colors[ok] * 0.45 + 0.55)
            c = coc[ok] * self.scale
            self.film.splat(sx[ok], sy[ok], c, E)
            # light scattering in the air around each source
            near = np.sqrt(np.clip(1.6 / np.maximum(z[ok], 0.05), 0.35, 4.0))
            self._glows.append((sx[ok], sy[ok], Lm[ok][:, None] * self.colors[ok], near))

    def _render_glows(self, img):
        if not self._glows:
            return img
        s = 4
        h, w = int(np.ceil(self.H / s)), int(np.ceil(self.W / s))
        acc = np.zeros((h, w, 3), np.float32)
        for sx, sy, col, near in self._glows:
            for (sigma, peak) in GLOW:
                buf = np.zeros((h, w, 3), np.float32)
                sg = sigma * self.scale / s
                for k in range(len(sx)):
                    x = int(round((sx[k] + 0.5) / s - 0.5))
                    y = int(round((sy[k] + 0.5) / s - 0.5))
                    if 0 <= x < w and 0 <= y < h:
                        sgk = sg * near[k]
                        buf[y, x] += col[k] * peak * 2 * np.pi * sgk * sgk / (near[k] ** 2)
                acc += cv2.GaussianBlur(buf, (0, 0), max(sg * float(np.mean(near)), 0.5))
        return img + cv2.resize(acc, (w * s, h * s), interpolation=cv2.INTER_LINEAR)[:self.H, :self.W]

    def _dust(self, cam, frame, tau):
        self.dust = dust_for_frame(self.lights, self.dust_cache, frame, n=self.dust_n, current=self.dust)
        x = self.dust.x
        p, _, Lm = self.lights.head(tau)
        E = np.zeros((len(x), 3), np.float32)
        to_cam = cam.pos.astype(np.float32)[None, :] - x
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-6
        for i in range(len(Lm)):
            if Lm[i] <= 1e-4:
                continue
            d = x - p[i].astype(np.float32)
            r2 = np.einsum("ij,ij->i", d, d)
            cos = np.einsum("ij,ij->i", d, to_cam) / (np.sqrt(r2) + 1e-6)
            irr = (Lm[i] * _hg(cos) / (r2 + 0.010)).astype(np.float32)
            E += irr[:, None] * self.colors[i][None, :]
        E *= (self.dust.albedo * K_DUST)[:, None]
        lum = E.max(axis=1)
        keep = np.nonzero(lum > 2e-4)[0]
        if len(keep) == 0:
            return
        xs = x[keep]
        xp = self.dust.x_prev[keep]
        Ek = E[keep] * 0.5
        for P in (xs, 0.5 * (xs + xp)):
            sx, sy, z, coc = cam.project(P.astype(np.float64))
            ok = z > 0.03
            self.film.splat(sx[ok], sy[ok], coc[ok] * self.scale, Ek[ok])

    # ── frame ─────────────────────────────────────────────────────────
    def render_hdr(self, frame, with_dust=True):
        t = frame / T.FPS
        tau = float(T.sim_time(t))
        cam = camera_at(t, self.lights, self.W, self.H)
        self._trails(cam, tau)
        self._heads(cam, tau)
        if with_dust:
            self._dust(cam, frame, tau)
        img = self.film.develop()
        img = self._render_glows(img)
        img += BACKGROUND
        return img, cam

    def render(self, frame):
        t = frame / T.FPS
        img, cam = self.render_hdr(frame)
        exp = exposure_at(t)
        # the flash: the circle ignites and the white swallows everything
        s = float(T.smoothstep(T.FLASH - 0.16, T.FLASH, t))
        exp *= 1.0 + 90.0 * s ** 2.2
        img = img * exp
        img = bloom(img, strength=0.11 + 0.5 * s)
        img = halation(img, 0.07)
        img = aces(img * 0.9)
        img = img + (1.0 - img) * (s ** 3)
        img = to_srgb(img)
        img = chromatic_aberration(img, 0.0007)
        img = vignette(img, 0.30 * (1 - s))
        img = grain(img, frame, amount=0.018)
        return quantize(img, frame)
