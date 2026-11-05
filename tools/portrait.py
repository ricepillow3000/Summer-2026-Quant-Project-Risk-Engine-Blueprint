"""
Make John's two photographs from his originals (ME-28 part 4).

    python -m tools.portrait <original image> [static]          # the About portrait
    python -m tools.portrait --story <original image> [static]  # the story's photo

The whole edit, so it can be checked and repeated: a 4:5 head-and-shoulders
crop at the photo's full width, from the top of the frame, centred on the
face (x = 450 of 941 in the 941x1672 original); resized to 480x600 (twice
the 240x300 it is drawn at); then one global grade - red x1.035, blue x0.955
(a slight warm white balance, so the grey backdrop sits on the beige and
stone) and contrast x1.05. Nothing local: the face is not retouched or
reshaped. Written as AVIF (q60), WebP (q80) and progressive JPEG (q82), each
under 30 KB, to static/john-nguyen.*.

The story's photo (2026-11-05, John: "another perspective when I'm not in my
suit"): the 1149x2000 selfie, cropped 4:5 at full width from y = 150 (the
ceiling tiles mostly out, the sweater in), resized to 480x600, then one
global grade - saturation x0.88 (the orange booth calmer beside the beige)
and contrast x1.03. Nothing local. Written to static/john-nguyen-story.*.
"""

import pathlib
import sys

import numpy as np
from PIL import Image, ImageEnhance

SIZE = (480, 600)
FACE_X = 450          # the face's centre in the original, in its pixels
WARM = (1.035, 1.0, 0.955)
CONTRAST = 1.05


STORY_TOP = 150       # the story crop's top edge in the original
STORY_SATURATION = 0.88
STORY_CONTRAST = 1.03


def _save(im: Image.Image, d: pathlib.Path, name: str) -> list[pathlib.Path]:
    files = [d / f"{name}.avif", d / f"{name}.webp", d / f"{name}.jpg"]
    im.save(files[0], quality=60)
    im.save(files[1], quality=80, method=6)
    im.save(files[2], quality=82, optimize=True, progressive=True)
    return files


def make_story(src: str, out: str = "static") -> list[pathlib.Path]:
    im = Image.open(src).convert("RGB")
    w, h = im.size
    ch = min(h - STORY_TOP, w * 5 // 4)
    im = im.crop((0, STORY_TOP, w, STORY_TOP + ch)).resize(SIZE, Image.LANCZOS)
    im = ImageEnhance.Color(im).enhance(STORY_SATURATION)
    im = ImageEnhance.Contrast(im).enhance(STORY_CONTRAST)
    return _save(im, pathlib.Path(out), "john-nguyen-story")


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
    return _save(im, pathlib.Path(out), "john-nguyen")


if __name__ == "__main__":
    args = sys.argv[1:]
    story = bool(args) and args[0] == "--story"
    args = args[1:] if story else args
    if not args:
        sys.exit(__doc__)
    for f in (make_story if story else make)(args[0], args[1] if len(args) > 1 else "static"):
        print(f, f.stat().st_size, "bytes")
