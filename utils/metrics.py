import numpy as np
from scipy import ndimage as ndi

def _counts(pred, gt):
    tp = (pred & gt).sum(axis=(1, 2)).astype(np.int64)
    fp = (pred & ~gt).sum(axis=(1, 2)).astype(np.int64)
    fn = (~pred & gt).sum(axis=(1, 2)).astype(np.int64)
    return tp, fp, fn

def _boundary(mask):
    """Boundary pixels of each mask in [N, H, W]."""
    out = np.zeros_like(mask)
    for i in range(len(mask)):
        er = ndi.binary_erosion(mask[i], border_value=0)
        out[i] = mask[i] & ~er
    return out

def _boundary_counts(pred, gt, tol):
    pbm, gbm = _boundary(pred), _boundary(gt)
    n = len(pred)
    pb = pbm.sum(axis=(1, 2)).astype(np.int64)
    gb = gbm.sum(axis=(1, 2)).astype(np.int64)
    pm = np.zeros(n, np.int64)
    gm = np.zeros(n, np.int64)

    for i in range(n):
        if gb[i]:
            d_to_g = ndi.distance_transform_edt(~gbm[i])
            pm[i] = int((pbm[i] & (d_to_g <= tol)).sum())
        if pb[i]:
            d_to_p = ndi.distance_transform_edt(~pbm[i])
            gm[i] = int((gbm[i] & (d_to_p <= tol)).sum())
    return pb, gb, pm, gm


def _centroids(mask):
    n = len(mask)
    c = np.full((n, 2), np.nan)
    ys, xs = np.mgrid[0:mask.shape[1], 0:mask.shape[2]]
    for i in range(n):
        a = mask[i].sum()
        if a:
            c[i, 0] = (ys * mask[i]).sum() / a
            c[i, 1] = (xs * mask[i]).sum() / a
    return c


def sample_stats(current, target, pred_growth, boundary_tol=3):
    current = current.astype(bool)
    target = target.astype(bool)
    pred_growth = pred_growth.astype(bool) & ~current

    true_growth = target & ~current
    pred_future = current | pred_growth

    tp_g, fp_g, fn_g = _counts(pred_growth, true_growth)
    tp_f, fp_f, fn_f = _counts(pred_future, target)
    pb, gb, pm, gm = _boundary_counts(pred_future, target, boundary_tol)
    cur_c = _centroids(current)
    gt_gc = _centroids(true_growth)
    pr_gc = _centroids(pred_growth)

    return {
        "tp_g": tp_g, "fp_g": fp_g, "fn_g": fn_g,
        "tp_f": tp_f, "fp_f": fp_f, "fn_f": fn_f,
        "pb": pb, "gb": gb, "pm": pm, "gm": gm,
        "area_pred": pred_future.sum(axis=(1, 2)).astype(np.int64),
        "area_true": target.sum(axis=(1, 2)).astype(np.int64),
        "cur_c": cur_c, "gt_gc": gt_gc, "pr_gc": pr_gc,
        "fut_c_pred": _centroids(pred_future),
        "fut_c_true": _centroids(target),
        "true_growth_px": true_growth.sum(axis=(1, 2)).astype(np.int64),
    }

def _ratio(num, den):
    return float(num) / float(den) if den > 0 else float("nan")

def _iou(tp, fp, fn, idx):
    return _ratio(tp[idx].sum(), (tp + fp + fn)[idx].sum())

def _dice(tp, fp, fn, idx):
    return _ratio(2 * tp[idx].sum(), (2 * tp + fp + fn)[idx].sum())

def _sector(vec_gt, vec_pr, n_sectors=8):
    ang_gt = np.arctan2(vec_gt[:, 0], vec_gt[:, 1])
    ang_pr = np.arctan2(vec_pr[:, 0], vec_pr[:, 1])
    w = 2 * np.pi / n_sectors
    s_gt = np.floor(((ang_gt + np.pi) / w)).astype(int) % n_sectors
    s_pr = np.floor(((ang_pr + np.pi) / w)).astype(int) % n_sectors
    return s_gt, s_pr


METRICS = {
    "growth_iou": lambda s, i: _iou(s["tp_g"], s["fp_g"], s["fn_g"], i),
    "growth_dice": lambda s, i: _dice(s["tp_g"], s["fp_g"], s["fn_g"], i),
    "growth_precision": lambda s, i: _ratio(
        s["tp_g"][i].sum(), (s["tp_g"] + s["fp_g"])[i].sum()),
    "growth_recall": lambda s, i: _ratio(
        s["tp_g"][i].sum(), (s["tp_g"] + s["fn_g"])[i].sum()),
    "future_iou": lambda s, i: _iou(s["tp_f"], s["fp_f"], s["fn_f"], i),
    "future_dice": lambda s, i: _dice(s["tp_f"], s["fp_f"], s["fn_f"], i),
    "boundary_f1": lambda s, i: _boundary_f1(s, i),
    "area_mae_px": lambda s, i: float(
        np.abs(s["area_pred"][i] - s["area_true"][i]).mean()),
    "area_bias_px": lambda s, i: float(
        (s["area_pred"][i] - s["area_true"][i]).mean()),
    "centroid_err_px": lambda s, i: _centroid_err(s, i),
    "growth_direction_acc8": lambda s, i: _direction_acc(s, i),
    "growth_direction_err_deg": lambda s, i: _direction_err(s, i),
}


def _boundary_f1(s, i):
    p = _ratio(s["pm"][i].sum(), s["pb"][i].sum())
    r = _ratio(s["gm"][i].sum(), s["gb"][i].sum())
    if np.isnan(p) or np.isnan(r) or (p + r) == 0:
        return float("nan")
    return 2 * p * r / (p + r)


def _centroid_err(s, i):
    d = np.linalg.norm(s["fut_c_pred"][i] - s["fut_c_true"][i], axis=1)
    d = d[~np.isnan(d)]
    return float(d.mean()) if len(d) else float("nan")


def _growth_vectors(s, i):
    gt = s["gt_gc"][i] - s["cur_c"][i]
    pr = s["pr_gc"][i] - s["cur_c"][i]
    has_gt = ~np.isnan(gt).any(axis=1)
    return gt, pr, has_gt


def _direction_acc(s, i):
    gt, pr, has_gt = _growth_vectors(s, i)
    if has_gt.sum() == 0:
        return float("nan")
    gt, pr = gt[has_gt], pr[has_gt]
    has_pr = ~np.isnan(pr).any(axis=1)
    hit = np.zeros(len(gt), bool)
    if has_pr.any():
        s_gt, s_pr = _sector(gt[has_pr], pr[has_pr])
        hit[has_pr] = s_gt == s_pr
    return float(hit.mean())            

def _direction_err(s, i):
    gt, pr, has_gt = _growth_vectors(s, i)
    ok = has_gt & ~np.isnan(pr).any(axis=1)
    if ok.sum() == 0:
        return float("nan")
    a = np.arctan2(gt[ok, 0], gt[ok, 1])
    b = np.arctan2(pr[ok, 0], pr[ok, 1])
    d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return float(np.degrees(d).mean())


def compute_metrics(stats, idx=None):
    n = len(stats["tp_g"])
    idx = np.arange(n) if idx is None else idx
    return {name: fn(stats, idx) for name, fn in METRICS.items()}
    

def _block_indices(n, block, rng):
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
    return idx[:n]


def bootstrap_ci(stats, metric_names=None, block=20, n_boot=1000,seed=0, alpha=0.05):
    names = metric_names or list(METRICS)
    n = len(stats["tp_g"])
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in names}
    for _ in range(n_boot):
        idx = _block_indices(n, block, rng)
        for k in names:
            draws[k].append(METRICS[k](stats, idx))
    out = {}
    for k in names:
        v = np.asarray(draws[k], float)
        v = v[~np.isnan(v)]
        out[k] = (
            (float(np.quantile(v, alpha / 2)),
             float(np.quantile(v, 1 - alpha / 2)))
            if len(v) else (float("nan"), float("nan"))
        )
    return out


def paired_bootstrap_diff(stats_a, stats_b, metric, block=20,n_boot=1000, seed=0, alpha=0.05):
    n = len(stats_a["tp_g"])
    assert n == len(stats_b["tp_g"]), 
    rng = np.random.default_rng(seed)
    fn = METRICS[metric]
    diffs = []
    for _ in range(n_boot):
        idx = _block_indices(n, block, rng)
        a, b = fn(stats_a, idx), fn(stats_b, idx)
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(a - b)
    full = fn(stats_a, np.arange(n)) - fn(stats_b, np.arange(n))
    if not diffs:
        return float(full), float("nan"), float("nan")
    d = np.asarray(diffs)
    return (float(full),
            float(np.quantile(d, alpha / 2)),
            float(np.quantile(d, 1 - alpha / 2)))

def best_threshold(prob, current, target, thresholds,objective="growth_dice"):
    current = current.astype(bool)
    true_growth = target.astype(bool) & ~current
    best_t, best_v = float(thresholds[0]), -1.0
    for t in thresholds:
        pred = (prob >= t) & ~current
        tp, fp, fn = _counts(pred, true_growth)
        if objective == "growth_iou":
            v = _iou(tp, fp, fn, np.arange(len(tp)))
        else:
            v = _dice(tp, fp, fn, np.arange(len(tp)))
        if not np.isnan(v) and v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v
