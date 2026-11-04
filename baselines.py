import numpy as np
from scipy import ndimage as ndi
from utils.metrics import METRICS, sample_stats

def persistence(current, **_):
    return np.zeros_like(current, dtype=bool)
    
def _dist_outside(current):
    """Distance from every pixel to the nearest fire pixel."""
    out = np.empty(current.shape, np.float32)
    for i in range(len(current)):
        if current[i].any():
            out[i] = ndi.distance_transform_edt(~current[i])
        else:
            out[i] = np.inf
    return out

def dilation(current, radius, dist=None, **_):
    dist = _dist_outside(current) if dist is None else dist
    return (dist > 0) & (dist <= radius)

def _edge_speed(hist, k):
    area_now = hist[:, -1].sum(axis=(1, 2)).astype(np.float64)
    area_old = hist[:, 0].sum(axis=(1, 2)).astype(np.float64)
    per = np.zeros(len(hist))
    for i in range(len(hist)):
        m = hist[i, -1]
        per[i] = (m & ~ndi.binary_erosion(m, border_value=0)).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (area_now - area_old) / (max(k, 1) * per)
    v[~np.isfinite(v)] = 0.0
    return np.clip(v, 0.0, None)


def speed_dilation(current, hist, horizon, scale, input_span,dist=None, **_):
    dist = _dist_outside(current) if dist is None else dist
    r = scale * horizon * _edge_speed(hist, input_span)     
    return (dist > 0) & (dist <= r[:, None, None])

def _gray(frame):
    g = frame.astype(np.float32).mean(axis=-1)
    return np.clip(g, 0, 255).astype(np.uint8)


def flow_warp(current, frames, horizon, scale, **_):
    import cv2
    n = len(current)
    out = np.zeros_like(current, dtype=bool)
    H, W = current.shape[1:]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    for i in range(n):
        m = current[i]
        if not m.any():
            continue
        a, b = _gray(frames[i, -2]), _gray(frames[i, -1])
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        edge = m & ~ndi.binary_erosion(m, border_value=0)
        if not edge.any():
            continue
        dx = float(flow[..., 0][edge].mean()) * scale * horizon
        dy = float(flow[..., 1][edge].mean()) * scale * horizon
        warped = cv2.remap(
            m.astype(np.float32), xs - dx, ys - dy,
            interpolation=cv2.INTER_LINEAR, borderValue=0) > 0.5
        out[i] = warped & ~m
    return out

RADII = (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32)
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0)
FLOW_SCALES = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)

def _score(current, target, pred, objective):
    st = sample_stats(current, target, pred)
    v = METRICS[objective](st, np.arange(len(current)))
    return -1.0 if np.isnan(v) else v

def tune(name, val, horizon, input_span, objective="growth_dice"):
    """Return the best parameter dict for baseline `name` on val."""
    cur, tgt = val["current"], val["target"]
    if name == "persistence":
        return {}
    if name == "dilation":
        dist = _dist_outside(cur)
        scores = {r: _score(cur, tgt, dilation(cur, r, dist), objective)
                  for r in RADII}
        return {"radius": max(scores, key=scores.get)}
    if name == "speed_dilation":
        dist = _dist_outside(cur)
        scores = {
            s: _score(cur, tgt, speed_dilation(
                cur, val["hist"], horizon, s, input_span, dist), objective)
            for s in SCALES}
        return {"scale": max(scores, key=scores.get)}

    if name == "flow_warp":
        scores = {
            s: _score(cur, tgt, flow_warp(
                cur, val["frames"], horizon, s), objective)
            for s in FLOW_SCALES}
        return {"scale": max(scores, key=scores.get)}
    raise ValueError(name)

def predict(name, data, params, horizon, input_span):
    cur = data["current"]
    if name == "persistence":
        return persistence(cur)
    if name == "dilation":
        return dilation(cur, params["radius"])
    if name == "speed_dilation":
        return speed_dilation(
            cur, data["hist"], horizon, params["scale"], input_span)
    if name == "flow_warp":
        return flow_warp(cur, data["frames"], horizon, params["scale"])
    raise ValueError(name)

BASELINES = ("persistence", "dilation", "speed_dilation", "flow_warp")
