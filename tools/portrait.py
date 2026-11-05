"""
Make the About block's portrait from John's original photo (ME-28 part 4).

    python -m tools.portrait <original image> [static]

The whole edit, so it can be checked and repeated: a 4:5 head-and-shoulders
crop at the photo's full width, from the top of the frame, centred on the
face (x = 450 of 941 in the 941x1672 original); resized to 480x600 (twice
the 240x300 it is drawn at); then one global grade - red x1.035, blue x0.955
(a slight warm white balance, so the grey backdrop sits on the beige and
stone) and contrast x1.05. Nothing local: the face is not retouched or
reshaped. Written as AVIF (q60), WebP (q80) and progressive JPEG (q82), each
under 30 KB, to static/john-nguyen.*.
"""

import pathlib
import sys

import numpy as np
from PIL import Image, ImageEnhance

SIZE = (480, 600)
FACE_X = 450          # the face's centre in the original, in its pixels
WARM = (1.035, 1.0, 0.955)
CONTRAST = 1.05


def make(src: str, out: str = "static") -> list[pathlib.Path]:
    im = Image.open(src).convert("RGB")
    w, h = im.size
    cw = w                                   # full width: the most shoulders, the face centred
    ch = min(h, cw * 5 // 4)
    left = max(0, min(w - cw, FACE_X - cw // 2))
    im = im.crop((left, 0, left + cw, ch)).resize(SIZE, Image.LANCZOS)
    a = np.asarray(im).astype(np.float32) / 255.0 * np.array(WARM, dtype=np.float32)
    im = Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
    im = ImageEnhance.Contrast(im).enhance(CONTRAST)
    d = pathlib.Path(out)
    files = [d / "john-nguyen.avif", d / "john-nguyen.webp", d / "john-nguyen.jpg"]
    im.save(files[0], quality=60)
    im.save(files[1], quality=80, method=6)
    im.save(files[2], quality=82, optimize=True, progressive=True)
    return files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for f in make(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "static"):
        print(f, f.stat().st_size, "bytes")
