#!/usr/bin/env python3
"""Download the (SIL Open Font License) typefaces used in the film.

Fonts are fetched from Google Fonts at build time rather than committed:
  Yuji Syuku        — brush calligraphy, for the inscription
  Shippori Mincho   — for the title and the carved seal
  Cormorant Garamond — for the latin subtitle
"""
import os
import re
import sys
import time
import urllib.request

FONTS = {
    "YujiSyuku-Regular.ttf": ("Yuji Syuku", 400),
    "ShipporiMincho-Regular.ttf": ("Shippori Mincho", 400),
    "ShipporiMincho-Bold.ttf": ("Shippori Mincho", 700),
    "CormorantGaramond-Regular.ttf": ("Cormorant Garamond", 400),
}

DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts")


def fetch(url, tries=4):
    for k in range(tries):
        try:
            # a plain user agent makes the CSS API answer with full TrueType files
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # network hiccup: back off and retry
            if k == tries - 1:
                raise
            print(f"  retry {url}: {e}", file=sys.stderr)
            time.sleep(2 ** (k + 1))


def main():
    os.makedirs(DEST, exist_ok=True)
    for name, (family, weight) in FONTS.items():
        path = os.path.join(DEST, name)
        if os.path.exists(path) and os.path.getsize(path) > 10_000:
            print(f"✓ {name}")
            continue
        css_url = "https://fonts.googleapis.com/css2?family={}:wght@{}".format(family.replace(" ", "+"), weight)
        css = fetch(css_url).decode("utf-8")
        m = re.search(r"url\((https://[^)]+\.ttf)\)", css)
        if not m:
            sys.exit(f"no TrueType URL for {family} {weight}")
        data = fetch(m.group(1))
        with open(path, "wb") as f:
            f.write(data)
        print(f"↓ {name} ({len(data) // 1024} KiB)")


if __name__ == "__main__":
    main()
