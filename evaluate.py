
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import baselines as B                              # noqa: E402
from utils.data import FrameStore, build_arrays, load_sequences  # noqa: E402
from utils.metrics import (                                   # noqa: E402
    best_threshold, bootstrap_ci, compute_metrics, paired_bootstrap_diff,
    sample_stats)
from utils.splits import split_sequences                      # noqa: E402

CI_METRICS = ["growth_iou", "growth_dice", "future_iou", "boundary_f1"]
PAIRED_METRICS = ["growth_dice", "future_iou", "boundary_f1"]


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def evaluate(horizon, tags, csv, size, thresholds, preds_dir, out_dir,
             tol, block, n_boot, input_span, fractions):
    df = load_sequences(csv)
    _, val_df, test_df = split_sequences(df, fractions)
    store = FrameStore((size, size))
    val = build_arrays(val_df, store)
    test = build_arrays(test_df, store)
    n = len(test["current"])
    print(f"horizon {horizon}: val {len(val['current'])}  test {n}")

    stats, meta = {}, {}

    # ---- baselines (tuned on val, frozen for test) --------------
    val_scores = {}
    for name in B.BASELINES:
        params = B.tune(name, val, horizon, input_span)
        pv = B.predict(name, val, params, horizon, input_span)
        val_scores[name] = compute_metrics(
            sample_stats(val["current"], val["target"], pv, tol)
        )["growth_dice"]
        pt = B.predict(name, test, params, horizon, input_span)
        stats[name] = sample_stats(test["current"], test["target"], pt, tol)
        meta[name] = {"kind": "baseline", "params": params}

    nonpersist = [b for b in B.BASELINES if b != "persistence"]
    best_baseline = max(
        nonpersist,
        key=lambda b: -1 if np.isnan(val_scores[b]) else val_scores[b])

    # ---- learned models (probability maps from training) --------
    for tag in tags:
        z = np.load(Path(preds_dir) / f"{tag}.npz")
        if not np.array_equal(z["test_frames"], test["frame_no"]):
            raise SystemExit(
                f"{tag}: test frames differ from the current split. "
                "Re-train or fix SPLIT_FRACTIONS / csv.")
        thr, _ = best_threshold(
            z["val_prob"].astype(np.float32), val["current"], val["target"],
            thresholds)
        pred = z["test_prob"].astype(np.float32) >= thr
        stats[tag] = sample_stats(test["current"], test["target"], pred, tol)
        meta[tag] = {"kind": "model", "threshold": thr}

    # ---- metrics, CIs, paired differences ------------------------
    results = {}
    for name, st in stats.items():
        m = compute_metrics(st)
        ci = bootstrap_ci(st, CI_METRICS, block=block, n_boot=n_boot)
        entry = {**meta[name], "metrics": m,
                 "ci95": {k: list(v) for k, v in ci.items()}}
        if meta[name]["kind"] == "model":
            entry["vs_persistence"], entry["vs_best_baseline"] = {}, {}
            for met in PAIRED_METRICS:
                entry["vs_persistence"][met] = list(paired_bootstrap_diff(
                    st, stats["persistence"], met, block, n_boot))
                entry["vs_best_baseline"][met] = list(paired_bootstrap_diff(
                    st, stats[best_baseline], met, block, n_boot))
            entry["best_baseline"] = best_baseline
        results[name] = entry

    payload = {"horizon": horizon, "n_test": n,
               "best_baseline": best_baseline, "results": results}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_h{horizon}.json").write_text(
        json.dumps(_jsonable(payload), indent=2))

    rows = []
    for name, e in results.items():
        r = {"method": name, "kind": e["kind"]}
        for k, v in e["metrics"].items():
            r[k] = v
        for k, (lo, hi) in e["ci95"].items():
            r[f"{k}_lo"], r[f"{k}_hi"] = lo, hi
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"eval_h{horizon}.csv", index=False)
    return payload, table


def main():
    import config

    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--size", type=int, default=config.TEMPORAL_SIZE[0])
    p.add_argument("--tol", type=int, default=config.BOUNDARY_TOLERANCE_PX)
    p.add_argument("--n-boot", type=int, default=config.BOOTSTRAP_SAMPLES)
    p.add_argument("--block", type=int, default=config.BOOTSTRAP_BLOCK)
    p.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    a = p.parse_args()

    csv = a.csv or (config.PROCESSED_DIR / f"sequences_h{a.horizon}.csv")
    span = (config.INPUT_LEN - 1) * config.INPUT_STRIDE
    # neighbouring windows share frames for about horizon+span steps,
    # so a bootstrap block shorter than that makes the CI too narrow
    block = max(a.block, a.horizon + span + 1)
    print(f"bootstrap block = {block} sequences")
    _, table = evaluate(
        a.horizon, a.tags, csv, a.size, config.THRESHOLD_VALUES,
        config.PREDS_DIR, a.out_dir, a.tol, block, a.n_boot,
        span, config.SPLIT_FRACTIONS)

    cols = ["method", "growth_iou", "growth_dice", "growth_precision",
            "growth_recall", "future_iou", "boundary_f1", "centroid_err_px"]
    print(table[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()