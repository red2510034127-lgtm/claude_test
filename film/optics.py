"""A virtual lens and film stock.

Everything luminous in the film is a point of energy. Points are splatted
into a stack of depth-of-field layers, one per circle-of-confusion radius;
each layer is convolved with a disc-shaped bokeh kernel (slightly brighter
at the rim, as real lenses are), then the stack is summed. Bloom, halation,
a filmic tone curve, lateral chromatic aberration, vignetting and grain
follow, in that order.
"""
import cv2
import numpy as np

LAYER_RADII = np.array([0.0, 1.0, 2.0, 3.5, 5.5, 8.0, 12.0, 17.0, 24.0, 34.0, 48.0, 68.0, 96.0])
LAYER_SCALE = np.array([1, 1, 1, 1, 2, 2, 2, 4, 4, 4, 4, 8, 8])


def bokeh_kernel(radius):
    """Disc kernel (radius in layer pixels) with soft edge and bright rim."""
    if radius < 0.35:
        return None
    R = int(np.ceil(radius + 1.0))
    y, x = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float64)
    r = np.sqrt(x * x + y * y)
    disc = np.clip(radius + 0.5 - r, 0.0, 1.0)
    rim = 0.82 + 0.38 * np.clip(r / max(radius, 1e-3), 0, 1) ** 6
    k = disc * rim
    return (k / k.sum()).astype(np.float32)


class Film:
    def __init__(self, width, height):
        self.W, self.H = width, height
        self.nl = len(LAYER_RADII)
        self.sizes = []
        self.offsets = []
        self.pads = []
        off = 0
        for r, s in zip(LAYER_RADII, LAYER_SCALE):
            pad = int(np.ceil(r / s)) + 2          # guard band so off-frame bokeh still spills in
            w = int(np.ceil(width / s)) + 2 * pad
            h = int(np.ceil(height / s)) + 2 * pad
            self.sizes.append((w, h))
            self.offsets.append(off)
            self.pads.append(pad)
            off += w * h
        self.total = off
        self.kernels = [bokeh_kernel(r / s) for r, s in zip(LAYER_RADII, LAYER_SCALE)]
        self.sizes_arr = np.array(self.sizes)
        self.offsets_arr = np.array(self.offsets)
        self.pads_arr = np.array(self.pads, np.float32)
        self.reset()

    def reset(self):
        self._idx = []
        self._w = []

    def splat(self, x, y, coc, rgb):
        """Queue points: screen x, y (px), CoC radius (px), linear energy (N,3)."""
        if len(x) == 0:
            return
        x = np.asarray(x, np.float32)
        y = np.asarray(y, np.float32)
        rgb = np.asarray(rgb, np.float32)
        c = np.clip(np.asarray(coc, np.float32), 0.0, LAYER_RADII[-1])
        margin = c + 2.0
        ok = (x > -margin) & (x < self.W + margin) & (y > -margin) & (y < self.H + margin)
        if not ok.all():
            x, y, c, rgb = x[ok], y[ok], c[ok], rgb[ok]
        k = np.clip(np.searchsorted(LAYER_RADII, c, side="right") - 1, 0, self.nl - 2)
        f = np.clip((c - LAYER_RADII[k]) / (LAYER_RADII[k + 1] - LAYER_RADII[k]), 0.0, 1.0)
        for kk, ww in ((k, 1.0 - f), (k + 1, f)):
            keep = ww > 1e-4
            kk, ww2 = kk[keep], ww[keep]
            s = LAYER_SCALE[kk].astype(np.float32)
            pad = self.pads_arr[kk]
            lx = (x[keep] + 0.5) / s - 0.5 + pad
            ly = (y[keep] + 0.5) / s - 0.5 + pad
            w = self.sizes_arr[kk, 0]
            h = self.sizes_arr[kk, 1]
            x0 = np.floor(lx).astype(np.int64)
            y0 = np.floor(ly).astype(np.int64)
            fx = (lx - x0).astype(np.float32)
            fy = (ly - y0).astype(np.float32)
            inb = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
            if not inb.all():
                x0, y0, fx, fy, w, kk, ww2 = x0[inb], y0[inb], fx[inb], fy[inb], w[inb], kk[inb], ww2[inb]
                e = rgb[keep][inb]
            else:
                e = rgb[keep]
            base = self.offsets_arr[kk] + y0 * w + x0
            e = e * ww2[:, None].astype(np.float32)
            for (dx, dy, wt) in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                                 (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                self._idx.append(base + dx + dy * w)
                self._w.append(e * wt[:, None])

    def develop(self):
        """Resolve all queued points into a linear HDR image (H, W, 3)."""
        out = np.zeros((self.H, self.W, 3), np.float32)
        if not self._idx:
            return out
        idx = np.concatenate(self._idx)
        wts = np.concatenate(self._w)
        flat = np.stack([np.bincount(idx, weights=wts[:, ch], minlength=self.total) for ch in range(3)], axis=-1)
        flat = flat.astype(np.float32)
        self.reset()
        groups = {}
        for li in range(self.nl):
            w, h = self.sizes[li]
            o = self.offsets[li]
            layer = flat[o:o + w * h].reshape(h, w, 3)
            if not layer.any():
                continue
            k = self.kernels[li]
            if k is not None:
                layer = cv2.filter2D(layer, -1, k, borderType=cv2.BORDER_CONSTANT)
            s = int(LAYER_SCALE[li])
            p = self.pads[li]
            layer = layer[p:p + int(np.ceil(self.H / s)), p:p + int(np.ceil(self.W / s))]
            if s in groups:
                groups[s] += layer
            else:
                groups[s] = layer.copy()
        for s, img in groups.items():
            if s == 1:
                out += img[:self.H, :self.W]
            else:
                up = cv2.resize(img, (img.shape[1] * s, img.shape[0] * s), interpolation=cv2.INTER_LINEAR)
                # the resize treats pixel centres consistently with the splat mapping
                out += up[:self.H, :self.W]
        return out


# ── post ──────────────────────────────────────────────────────────────

def bloom(img, strength=0.08, levels=7):
    H, W = img.shape[:2]
    acc = np.zeros_like(img)
    cur = img
    weights = [0.55, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]
    tot = 0.0
    for l in range(levels):
        nh, nw = max(cur.shape[0] // 2, 2), max(cur.shape[1] // 2, 2)
        cur = cv2.resize(cur, (nw, nh), interpolation=cv2.INTER_AREA)
        b = cv2.GaussianBlur(cur, (0, 0), 1.3)
        acc += cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR) * weights[l]
        tot += weights[l]
    return img + acc * (strength / tot)


def halation(img, amount=0.06):
    lum = img[..., 0] * 0.3 + img[..., 1] * 0.6 + img[..., 2] * 0.1
    hi = np.maximum(lum - 0.9, 0.0)
    small = cv2.resize(hi, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3.0)
    glow = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    tint = np.array([1.0, 0.32, 0.10], np.float32)
    return img + glow[..., None] * tint * amount


def aces(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def from_srgb(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def chromatic_aberration(img, amount=0.0007):
    H, W = img.shape[:2]
    out = img.copy()
    for ch, s in ((0, 1.0 + amount), (2, 1.0 - amount)):
        M = np.array([[s, 0, (1 - s) * W * 0.5], [0, s, (1 - s) * H * 0.5]], np.float32)
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                      borderMode=cv2.BORDER_REFLECT)
    return out


_VIGNETTE = {}


def vignette(img, strength=0.32):
    H, W = img.shape[:2]
    key = (H, W, strength)
    if key not in _VIGNETTE:
        y, x = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (W / 2)) ** 2)
        _VIGNETTE[key] = (1.0 - strength * np.clip(r / 0.82, 0, 1.5) ** 2.4)[..., None].astype(np.float32)
    return img * _VIGNETTE[key]


def grain(img, frame, amount=0.022, size=1.35, seed=1234):
    """Luminance-weighted film grain in display space (img in 0..1 sRGB)."""
    H, W = img.shape[:2]
    rng = np.random.default_rng(seed + frame * 7919)
    gh, gw = int(H / size), int(W / size)
    n = rng.standard_normal((gh, gw, 3), dtype=np.float32)
    n[..., 1] = n[..., 1] * 0.35 + n[..., 0] * 0.65         # mostly luminance, a little chroma
    n[..., 2] = n[..., 2] * 0.35 + n[..., 0] * 0.65
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    lum = img[..., 0] * 0.3 + img[..., 1] * 0.6 + img[..., 2] * 0.1
    w = 0.22 + 2.6 * lum * (1.0 - lum)
    return img + n * (amount * w)[..., None]


def quantize(img, frame, seed=99):
    rng = np.random.default_rng(seed + frame)
    d = rng.random(img.shape[:2], dtype=np.float32)[..., None] - 0.5
    return np.clip(img * 255.0 + d, 0, 255).astype(np.uint8)
