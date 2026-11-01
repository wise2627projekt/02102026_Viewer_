"""Show files (`.photovecshow`): what the Show Maker prerenders and the Viewer plays.

A show is a photo grid. Every photo has two sliders that were prepared in advance:

    slider 1 (default name "Threshold")   `steps` prerendered vector pictures, from the lowest to the highest Threshold
    slider 2 (default name "Background")  the background (the photo around the mask) fades between `bg_min` and `bg_max` percent

Nothing is traced in the Viewer. For slider 2 it lays the original photo behind the prerendered vector with the same layer maths the
other windows use (layers.composite), so both apps show exactly the same picture.

The file is a zip: `manifest.json` (names of the sliders and one entry per photo) and, for photo i, step k:
    p{i}/thumb.jpg   small picture for the grid           p{i}/photo.jpg   the colour photo at the render size
    p{i}/mask.png    the finished mask (if any)           p{i}/v{k}.png    the vector picture (BGR) of step k
    p{i}/i{k}.png    how much of every pixel is drawing (8 bit), only for black-and-white drawings with a mask
Names inside the file are made from numbers only; nothing in the file is ever used as a path.
"""
from __future__ import annotations

import json
import os
import time
import zipfile
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional, Tuple

import cv2
import numpy as np

FORMAT = 1
APP = "photovec-show"
EXT = ".photovecshow"
MAX_JSON = 4 * 1024 * 1024
MAX_SIDE = 16384
MAX_PHOTOS = 400
MAX_STEPS = 240
THUMB_SIDE = 480
DEFAULT_LABELS = ("Threshold", "Background")


class ShowError(Exception):
    pass


@dataclass
class PhotoInfo:
    """One photo of a show (everything the Viewer needs to know about it)."""
    w: int = 0                         # size of the stored photo and vector pictures (pixels)
    h: int = 0
    steps: int = 2                     # number of prerendered vector pictures (slider 1 has this many positions)
    thr_min: float = 0.0               # the Threshold of the first and of the last picture (information only)
    thr_max: float = 255.0
    bg_min: float = 0.0                # slider 2 at its left end / right end: how much of the background photo is visible (percent)
    bg_max: float = 100.0
    blur: float = 0.0                  # blur of the background when slider 2 is at its left end (percent of 3 % of the long side)
    start1: float = 0.5                # where the sliders rest when the photo is opened (0..1 along the line)
    start2: float = 1.0
    masked: bool = False               # True: the vector is only inside the mask and slider 2 fades the photo around it
                                       # False: no mask - slider 2 fades the vector into the original photo
    has_ink: bool = False              # the pictures have an ink map (i{k}.png)
    paper: Tuple[int, int, int] = (255, 255, 255)   # colour of the paper (BGR): what a faded-out background turns into
    name: str = ""

    def clamp(self) -> "PhotoInfo":
        self.steps = int(min(max(self.steps, 2), MAX_STEPS))
        for k in ("start1", "start2"):
            setattr(self, k, float(min(max(getattr(self, k), 0.0), 1.0)))
        self.bg_min, self.bg_max = (float(min(max(v, 0.0), 100.0)) for v in (self.bg_min, self.bg_max))
        self.blur = float(min(max(self.blur, 0.0), 100.0))
        self.paper = tuple(int(min(max(v, 0), 255)) for v in self.paper)[:3]
        return self


# --------------------------------------------------------------------------- what the sliders mean (used by BOTH apps)
def step_at(info: PhotoInfo, pos: float) -> int:
    """Slider 1 position (0..1) -> number of the prerendered picture."""
    return int(round(min(max(pos, 0.0), 1.0) * (info.steps - 1)))


def threshold_of(info: PhotoInfo, k: int) -> float:
    return info.thr_min + (info.thr_max - info.thr_min) * k / max(1, info.steps - 1)


def look_values(info: PhotoInfo, pos2: float) -> dict:
    """Layer values (see layers.py) for slider 2 at `pos2` (0..1)."""
    pos2 = min(max(pos2, 0.0), 1.0)
    shown = (info.bg_min + (info.bg_max - info.bg_min) * pos2) / 100.0          # how much of the background photo is visible
    if info.masked:
        return {"layer_f": round(shown, 4), "layer_g": 1.0, "blur_f": round(info.blur * (1.0 - pos2), 3), "layer_p": 1.0}
    return {"layer_f": 1.0, "layer_g": round(1.0 - shown, 4), "blur_f": 0.0}


def compose(info: PhotoInfo, photo: np.ndarray, mask: Optional[np.ndarray], vec: np.ndarray, ink: Optional[np.ndarray], pos2: float) -> np.ndarray:
    """The picture the sliders show: prerendered vector `vec` (+ ink map) over the photo. All arrays have the same size (BGR uint8)."""
    from .layers import _lerp, _to_u8, composite
    v = look_values(info, pos2)
    if info.masked and mask is not None:
        return composite(vec, mask, info.paper, v, photo=photo, ink=ink)
    g = v["layer_g"]
    if g >= 1.0:
        return vec
    if g <= 0.0:
        return photo
    return _to_u8(_lerp(photo.astype(np.float32), vec, g))


def facade_look(info: PhotoInfo, photo: np.ndarray, mask: Optional[np.ndarray], pos2: float) -> Optional[np.ndarray]:
    """The background as it looks with slider 2 at `pos2` (the photo, blurred and faded to paper), as a float picture - or None when it is
    just the photo itself (then `compose` takes its fast path). It does not depend on slider 1, so the Viewer keeps it while slider 1 moves."""
    from .layers import BLUR_MAX, _blur_facade, _fade, _lerp, mask_f32
    if not (info.masked and mask is not None):
        return None
    v = look_values(info, pos2)
    fv, bl = v["layer_f"], v["blur_f"]
    h, w = photo.shape[:2]
    sigma = bl / 100.0 * BLUR_MAX * max(h, w) if (bl > 0 and fv > 0) else 0.0
    if sigma < 0.5 and fv >= 1.0:
        return None
    ph = photo.astype(np.float32)
    if sigma >= 0.5:
        m = mask_f32(mask, w, h)
        ph = _lerp(_blur_facade(ph, m, sigma), ph, m)
    return _fade(ph, np.array(info.paper, np.float32).reshape(1, 1, 3), fv) if fv < 1.0 else ph


def compose_over(look: np.ndarray, mask: np.ndarray, vec: np.ndarray, ink: Optional[np.ndarray]) -> np.ndarray:
    """The same picture as `compose` for a masked photo, from a prepared `look`: look + mask * ink * (vec - look)."""
    from .layers import _alpha, _lerp, _to_u8, mask_f32
    h, w = vec.shape[:2]
    m = mask_f32(mask, w, h)
    a = _alpha(ink, w, h)
    return _to_u8(_lerp(look, vec, m if a is None else m * a))


# --------------------------------------------------------------------------- writing
def _png(a: np.ndarray, level: int = 3) -> bytes:
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(a), [cv2.IMWRITE_PNG_COMPRESSION, level])
    if not ok:
        raise ShowError("could not encode a picture")
    return buf.tobytes()


def _jpg(a: np.ndarray, quality: int) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, 0x111111]                    # 4:4:4: no colour smearing at the edges
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(a), params)
    if not ok:
        raise ShowError("could not encode a picture")
    return buf.tobytes()


def make_thumb(bgr: np.ndarray, side: int = THUMB_SIDE) -> np.ndarray:
    h, w = bgr.shape[:2]
    f = side / float(max(h, w))
    if f >= 1.0:
        return bgr
    return cv2.resize(bgr, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA)


class ShowWriter:
    """Writes a show step by step (a vector picture at a time, so nothing big is kept in memory). The file is finished under a
    temporary name and moved into place, so a cancelled or failed render never damages an older file."""

    def __init__(self, path: str):
        self.path = path
        self.tmp = path + ".part"
        self._z = zipfile.ZipFile(self.tmp, "w", zipfile.ZIP_STORED)              # (the pictures are compressed already)
        self._infos: List[PhotoInfo] = []

    def add_photo(self, i: int, photo_bgr: np.ndarray, mask: Optional[np.ndarray]) -> None:
        self._z.writestr(f"p{i}/photo.jpg", _jpg(photo_bgr, 95))
        self._z.writestr(f"p{i}/thumb.jpg", _jpg(make_thumb(photo_bgr), 90))
        if mask is not None:
            self._z.writestr(f"p{i}/mask.png", _png(mask))

    def add_step(self, i: int, k: int, vec: np.ndarray, ink: Optional[np.ndarray]) -> None:
        self._z.writestr(f"p{i}/v{k}.png", _png(vec))
        if ink is not None:
            a = np.clip(np.rint(ink.astype(np.float32) * 255.0), 0, 255).astype(np.uint8) if ink.dtype != np.uint8 else ink
            self._z.writestr(f"p{i}/i{k}.png", _png(a))

    def finish(self, infos: List[PhotoInfo], title: str = "", labels: Tuple[str, str] = DEFAULT_LABELS) -> None:
        man = {"app": APP, "format": FORMAT, "title": str(title)[:120], "made": time.strftime("%Y-%m-%d %H:%M:%S"),
               "labels": [str(labels[0])[:40], str(labels[1])[:40]], "photos": [asdict(p.clamp()) for p in infos]}
        self._z.writestr("manifest.json", json.dumps(man, indent=1), compress_type=zipfile.ZIP_DEFLATED)
        self._z.close()
        os.replace(self.tmp, self.path)

    def abort(self) -> None:
        try:
            self._z.close()
        except Exception:
            pass
        if os.path.exists(self.tmp):
            try:
                os.remove(self.tmp)
            except OSError:
                pass


# --------------------------------------------------------------------------- reading
class ShowReader:
    """Opens a show. Pictures are decoded on demand (thread-safe)."""

    def __init__(self, path: str):
        import threading
        self.path = path
        self._lock = threading.Lock()
        try:
            self._z = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as exc:
            raise ShowError(f"This is not a PhotoVec show file: {exc}") from exc
        try:
            info = self._z.getinfo("manifest.json")
            if info.file_size > MAX_JSON:
                raise ShowError("The show file is damaged (settings too large).")
            man = json.loads(self._z.read(info))
        except KeyError:
            raise ShowError("This is not a PhotoVec show file (manifest.json is missing).")
        except ValueError as exc:
            raise ShowError(f"The show file is damaged: {exc}") from exc
        if not isinstance(man, dict) or man.get("app") != APP:
            raise ShowError("This is not a PhotoVec show file.")
        if int(man.get("format", 0)) > FORMAT:
            raise ShowError("This show was made by a newer version of the program.")
        self.title = str(man.get("title", ""))
        lab = man.get("labels") or list(DEFAULT_LABELS)
        self.labels = (str(lab[0]) if len(lab) > 0 else DEFAULT_LABELS[0], str(lab[1]) if len(lab) > 1 else DEFAULT_LABELS[1])
        names = set(self._z.namelist())
        self.photos: List[PhotoInfo] = []
        known = {f.name for f in fields(PhotoInfo)}
        for n, d in enumerate(man.get("photos", [])[:MAX_PHOTOS]):
            try:
                kw = {k: v for k, v in d.items() if k in known}
                if "paper" in kw:
                    kw["paper"] = tuple(int(x) for x in kw["paper"])[:3]
                pi = PhotoInfo(**kw).clamp()
            except (TypeError, ValueError, AttributeError) as exc:
                raise ShowError(f"Photo {n + 1} has damaged settings: {exc}") from exc
            if not (1 <= pi.w <= MAX_SIDE and 1 <= pi.h <= MAX_SIDE):
                raise ShowError(f"Photo {n + 1} has a damaged size.")
            need = [f"p{n}/photo.jpg", f"p{n}/thumb.jpg"] + [f"p{n}/v{k}.png" for k in range(pi.steps)]
            missing = [m for m in need if m not in names]
            if missing:
                raise ShowError(f"Photo {n + 1} is incomplete in the show file ({missing[0]} is missing).")
            self.photos.append(pi)
        if not self.photos:
            raise ShowError("This show has no photos.")
        self._names = names

    def _bytes(self, member: str) -> Optional[bytes]:
        if member not in self._names:
            return None
        with self._lock:
            return self._z.read(member)

    def _decode(self, member: str, flags: int) -> Optional[np.ndarray]:
        raw = self._bytes(member)
        if raw is None:
            return None
        a = cv2.imdecode(np.frombuffer(raw, np.uint8), flags)
        if a is None or max(a.shape[:2]) > MAX_SIDE:
            raise ShowError(f"{member} is damaged.")
        return a

    def thumb(self, i: int) -> np.ndarray:
        return self._decode(f"p{i}/thumb.jpg", cv2.IMREAD_COLOR)

    def photo(self, i: int) -> np.ndarray:
        return self._decode(f"p{i}/photo.jpg", cv2.IMREAD_COLOR)

    def mask(self, i: int) -> Optional[np.ndarray]:
        return self._decode(f"p{i}/mask.png", cv2.IMREAD_GRAYSCALE)

    def vec(self, i: int, k: int) -> np.ndarray:
        return self._decode(f"p{i}/v{k}.png", cv2.IMREAD_COLOR)

    def ink(self, i: int, k: int) -> Optional[np.ndarray]:
        return self._decode(f"p{i}/i{k}.png", cv2.IMREAD_GRAYSCALE)

    def close(self) -> None:
        with self._lock:
            self._z.close()
