import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_small(path, size):
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(im) > 0


def build(pairs_csv, horizon, input_len, stride, mask_source, stat_size=128):
    df = pd.read_csv(pairs_csv)
    df = df[df["status"].str.lower() == "matched"].copy()
    df["frame_number"] = df["frame_number"].astype(int)
    df = df.sort_values("frame_number").reset_index(drop=True)

    if mask_source == "predicted":
        if "pred_mask" not in df.columns:
            raise SystemExit(
                "pairs csv has no 'pred_mask' column; run "
                "preprocessing/predict_masks.py first"
            )
        input_mask_col = "pred_mask"
    else:
        input_mask_col = "mask"

    rec = {int(r.frame_number): r for r in df.itertuples(index=False)}
    frames = sorted(rec)

    cache = {}

    def small(path):
        if path not in cache:
            cache[path] = load_small(path, stat_size)
        return cache[path]

    rows = []
    span_back = (input_len - 1) * stride

    for t in frames:
        needed = [t - span_back + i * stride for i in range(input_len)]
        target_f = t + horizon

        if target_f not in rec or any(f not in rec for f in needed):
            continue

        in_recs = [rec[f] for f in needed]
        cur_mask = getattr(rec[t], input_mask_col)
        tgt_mask = rec[target_f].mask

        growth = int(
            (small(tgt_mask) & ~small(rec[t].mask)).sum()
        )

        rows.append({
            "input_frames": "|".join(r.image for r in in_recs),
            "input_masks": "|".join(
                getattr(r, input_mask_col) for r in in_recs
            ),
            "current_mask": cur_mask,
            "target_mask": tgt_mask,
            "start_frame": needed[0],
            "current_frame_number": t,
            "target_frame": target_f,
            "horizon": horizon,
            "growth_px": growth,
        })

    return pd.DataFrame(rows), len(df)


def main():
    import config

    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=config.PREDICTION_HORIZON)
    p.add_argument("--input-len", type=int, default=config.INPUT_LEN)
    p.add_argument("--stride", type=int, default=config.INPUT_STRIDE)
    p.add_argument("--mask-source", choices=["gt", "predicted"], default="gt")
    p.add_argument("--pairs", type=Path, default=None,
                   help="default: segmentation_pairs.csv, or "
                        "segmentation_pairs_pred.csv with "
                        "--mask-source predicted")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    out = args.out or (
        config.PROCESSED_DIR
        / f"sequences_h{args.horizon}"
        f"{'_pred' if args.mask_source == 'predicted' else ''}.csv"
    )

    pairs = args.pairs or (
        config.PROCESSED_DIR / "segmentation_pairs_pred.csv"
        if args.mask_source == "predicted"
        else config.SEGMENTATION_PAIRS_CSV
    )

    result, n_matched = build(
        pairs, args.horizon, args.input_len, args.stride,
        args.mask_source,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)

    n = len(result)
    nz = int((result["growth_px"] > 0).sum()) if n else 0
    print("=" * 70)
    print(f"SEQUENCES  horizon={args.horizon}  "
          f"input_len={args.input_len}  stride={args.stride}")
    print("=" * 70)
    print(f"Matched frames:    {n_matched}")
    print(f"Sequences built:   {n}")
    if n:
        g = result["growth_px"]
        print(f"With any growth:   {nz} ({100 * nz / n:.1f}%)")
        print(f"Growth px  mean/median/max: "
              f"{g.mean():.1f} / {g.median():.0f} / {g.max()}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
