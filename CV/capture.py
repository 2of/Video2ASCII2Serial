import os
import sys
import time

import cv2
import mss
import numpy as np




os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REGION = {"left": 100, "top": 100, "width": 640, "height": 360}
COLS = 100
COLOR = True
COLOR_MODE = "256"
INVERT = False
FPS = 45
DEPTH_DEVICE = "auto"

PASSES = {
    "luminance": {"on": True, "show": False, "weight": 0.2},
    "depth": {"on": True, "show": False, "weight": 1.0, "every": 2},
    "edges": {"on": False, "show": False, "weight": 0.5},
    "motion": {"on": False, "show": False, "weight": 0.5},
    "yolo": {"on": False, "show": False, "weight": 1.0},
}

LAYERED_CHARS = True
EDGE_CUTOFF = 0.3
NEAR_CUTOFF = 0.66
FAR_CUTOFF = 0.33

CHARS = " .·'`^,:;~!*+xXcCoO0#@░▒▓█"
BGCHARS = " .·'`,:;"
FARCHARS = " .·:-~"
NEARCHARS = "xX0#@▒▓█"
EDGECHARS = "-~=+/|#"

CHAR_ASPECT = 2.0

RESET = "\033[0m"

BASE_RAMP = np.array(list(CHARS))
RAMPS = [np.array(list(cs or CHARS)) for cs in (BGCHARS, FARCHARS, NEARCHARS, EDGECHARS)]

_depth = {}
_motion = {}
_cache = {}


def get_frame(sct, region):
    shot = sct.grab(region)
    return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)


def get_luminance(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def pick_device(torch):
    if DEPTH_DEVICE != "auto":
        return DEPTH_DEVICE
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_depth_map(img):
    import torch

    if not _depth:
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        device = pick_device(torch)
        _depth["model"] = model.to(device).eval()
        _depth["transform"] = transform
        _depth["device"] = device

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    batch = _depth["transform"](rgb).to(_depth["device"])
    with torch.no_grad():
        pred = _depth["model"](batch)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
        ).squeeze()

    lo, hi = pred.min(), pred.max()
    depth = ((pred - lo) / (hi - lo + 1e-6) * 255).to(torch.uint8).cpu().numpy()
    return depth


def get_prominent_edges(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    m = float(np.median(gray))
    low = int(max(0, 0.66 * m))
    high = int(min(255, 1.33 * m))
    return cv2.Canny(gray, low, high)


def get_motion_map(img):
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    prev = _motion.get("gray")
    _motion["gray"] = gray
    if prev is None:
        return np.zeros_like(gray)
    diff = cv2.absdiff(gray, prev)
    mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    return cv2.dilate(mask, None, iterations=2)


def img_yolo_pass(img):
    pass


PASS_FUNCS = {
    "luminance": get_luminance,
    "depth": get_depth_map,
    "edges": get_prominent_edges,
    "motion": get_motion_map,
    "yolo": img_yolo_pass,
}


def run_passes(frame, n):
    maps = {}
    for name, cfg in PASSES.items():
        if not (cfg["on"] or cfg["show"]):
            continue
        if name not in _cache or n % cfg.get("every", 1) == 0:
            _cache[name] = PASS_FUNCS[name](frame)
        maps[name] = _cache[name]
    return maps


def normalize(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else a / 255


def cell_maps(maps, cols, rows):
    cells = {}
    for name, m in maps.items():
        if not PASSES[name]["on"] or m is None:
            continue
        small = cv2.resize(m, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32)
        cells[name] = normalize(small)
    return cells


def build_buffer(cells, cols, rows):
    acc = np.zeros((rows, cols), np.float32)
    total = 0.0
    for name, c in cells.items():
        w = PASSES[name]["weight"]
        acc += w * c
        total += w
    return normalize(acc / total) if total else acc


def pick_layers(cells, cols, rows):
    layer = np.zeros((rows, cols), np.int8)
    depth = cells.get("depth")
    if depth is not None:
        layer[depth < FAR_CUTOFF] = 1
        layer[depth > NEAR_CUTOFF] = 2
    edges = cells.get("edges")
    if edges is not None:
        layer[edges > EDGE_CUTOFF] = 3
    return layer


def render_chars(norm, layer):
    out = np.empty(norm.shape, dtype="<U1")
    for i, ramp in enumerate(RAMPS):
        idx = (norm * (len(ramp) - 1)).round().astype(np.int32)
        mask = layer == i
        out[mask] = ramp[idx][mask]
    return out


def to_256(rgb):
    levels = np.round(rgb.astype(np.float32) / 255 * 5).astype(np.int32)
    return 16 + 36 * levels[..., 0] + 6 * levels[..., 1] + levels[..., 2]


def frame_to_ascii(frame, maps, cols=COLS, invert=INVERT, color=COLOR):
    h, w = frame.shape[:2]
    rows = max(1, int(h / (w / cols) / CHAR_ASPECT))

    cells = cell_maps(maps, cols, rows)
    norm = build_buffer(cells, cols, rows)
    if invert:
        norm = 1 - norm

    if LAYERED_CHARS and ("depth" in cells or "edges" in cells):
        chars = render_chars(norm, pick_layers(cells, cols, rows))
    else:
        idx = (norm * (len(BASE_RAMP) - 1)).round().astype(np.int32)
        chars = BASE_RAMP[idx]

    if not color:
        return ["".join(row) for row in chars]

    small = cv2.resize(frame, (cols, rows), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    if COLOR_MODE == "truecolor":
        keys = (rgb // 8 * 8).tolist()
        code = lambda c: f"\033[38;2;{c[0]};{c[1]};{c[2]}m"
    else:
        keys = to_256(rgb).tolist()
        code = lambda c: f"\033[38;5;{c}m"

    lines = []
    for y in range(rows):
        out, prev = [], None
        for x in range(cols):
            c = keys[y][x]
            if c != prev:
                out.append(code(c))
                prev = c
            out.append(chars[y][x])
        out.append(RESET)
        lines.append("".join(out))
    return lines


def show_views(maps):
    shown = False
    for name, m in maps.items():
        if PASSES[name]["show"] and m is not None:
            cv2.imshow(name, m)
            shown = True
    if shown:
        cv2.waitKey(1)


def run_img_cap():
    with mss.mss() as sct:
        print("\033[2J", end="")
        n = 0
        try:
            while True:
                start = time.time()
                frame = get_frame(sct, REGION)
                maps = run_passes(frame, n)
                lines = frame_to_ascii(frame, maps)
                sys.stdout.write("\033[H" + "\n".join(lines) + "\n")
                sys.stdout.flush()
                show_views(maps)
                n += 1
                time.sleep(max(0, 1 / FPS - (time.time() - start)))
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            print(RESET)


if __name__ == "__main__":
    run_img_cap()