#!/usr/bin/env python3
"""Draw a waveform PNG for a wav file using Pillow (no matplotlib needed).

    python scripts/plot_waveform.py demo/yua-chorus-sung.wav demo/waveform.png
"""

from __future__ import annotations

import argparse
import os

from PIL import Image, ImageDraw, ImageFont

# Fonts that can draw CJK titles; the first existing one wins.
CJK_FONTS = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
LATIN_FONTS = ["arial.ttf", "DejaVuSans.ttf", "/System/Library/Fonts/Helvetica.ttc"]


def load_fonts():
    """Return (label font, small font, supports_cjk)."""
    for path in CJK_FONTS:
        try:
            return (ImageFont.truetype(path, 16), ImageFont.truetype(path, 12), True)
        except OSError:
            continue
    for path in LATIN_FONTS:
        try:
            return (ImageFont.truetype(path, 16), ImageFont.truetype(path, 12), False)
        except OSError:
            continue
    font = ImageFont.load_default()
    return font, font, False


def ascii_only(text):
    cleaned = "".join(ch if ord(ch) < 128 else "?" for ch in text)
    return cleaned.strip() or "waveform"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render a waveform PNG")
    parser.add_argument("wav")
    parser.add_argument("out")
    parser.add_argument("--width", type=int, default=1100)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    peak = float(np.max(np.abs(mono))) or 1.0
    mono = mono / peak

    width, height = args.width, args.height
    pad_l, pad_r, pad_t, pad_b = 60, 30, 50, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    mid = pad_t + plot_h / 2.0

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font, small, cjk = load_fonts()

    title = args.title or os.path.basename(args.wav)
    if not cjk:
        title = ascii_only(title)
    draw.text((pad_l, 18), title, fill="black", font=font)
    duration = len(mono) / sr
    draw.text((width - pad_r - 150, 20), "%.2f s @ %d Hz" % (duration, sr),
              fill=(90, 90, 90), font=small)

    draw.rectangle([pad_l, pad_t, pad_l + plot_w, pad_t + plot_h], outline=(200, 200, 200))
    draw.line([pad_l, mid, pad_l + plot_w, mid], fill=(225, 225, 225), width=1)

    step = max(1, len(mono) // plot_w)
    bars = []
    for x in range(plot_w):
        chunk = mono[x * step:(x + 1) * step]
        if len(chunk) == 0:
            bars.append((0.0, 0.0))
            continue
        bars.append((float(chunk.min()), float(chunk.max())))

    for x, (lo, hi) in enumerate(bars):
        y0 = mid - hi * (plot_h / 2.0)
        y1 = mid - lo * (plot_h / 2.0)
        draw.line([pad_l + x, y0, pad_l + x, max(y1, y0 + 1)], fill=(31, 110, 190), width=1)

    for frac in (0.25, 0.5, 0.75):
        t = duration * frac
        x = pad_l + int(plot_w * frac)
        draw.line([x, pad_t + plot_h, x, pad_t + plot_h + 6], fill=(160, 160, 160))
        draw.text((x - 14, pad_t + plot_h + 10), "%.1fs" % t, fill=(120, 120, 120), font=small)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    image.save(args.out)
    print("wrote %s (%dx%d)" % (args.out, width, height))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
