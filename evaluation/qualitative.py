import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data import FrameStore, build_arrays, load_sequences  # noqa: E402
from utils.metrics import best_threshold                       # noqa: E402
from utils.splits import split_sequences                       # noqa: E402


def per_sample_growth_dice(current, target, pred):
    gt = target & ~current
    pr = pred & ~current
    tp = (gt & pr).sum(axis=(1, 2))
    den = gt.sum(axis=(1, 2)) + pr.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(den > 0, 2 * tp / den, np.nan)
    return d


def make_figure(frames, current, target, pred, picks, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi

    fig, axes = plt.subplots(len(picks), 3, figsize=(9, 3 * len(picks)))
    axes = np.atleast_2d(axes)
    for r, (i, lab) in enumerate(zip(picks, labels)):
        gt = target[i] & ~current[i]
        pr = pred[i] & ~current[i]
        edge = current[i] & ~ndi.binary_erosion(current[i])

        a = frames[i, -1].copy()
        a[edge] = [255, 255, 0]
        axes[r, 0].imshow(a)
        axes[r, 0].set_title(f"{lab}: last frame + current fire", fontsize=8)

        axes[r, 1].imshow(gt, cmap="gray")
        axes[r, 1].set_title("true new growth", fontsize=8)

        rgb = np.zeros(gt.shape + (3,), np.uint8)
        rgb[gt & pr] = [0, 200, 0]
        rgb[pr & ~gt] = [220, 0, 0]
        rgb[gt & ~pr] = [0, 100, 255]
        axes[r, 2].imshow(rgb)
        axes[r, 2].set_title("pred: green hit, red false, blue missed",
                             fontsize=8)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    import config

    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--size", type=int, default=config.TEMPORAL_SIZE[0])
    a = p.parse_args()

    csv = a.csv or (config.PROCESSED_DIR / f"sequences_h{a.horizon}.csv")
    _, val_df, test_df = split_sequences(
        load_sequences(csv), config.SPLIT_FRACTIONS)
    store = FrameStore((a.size, a.size))
    val = build_arrays(val_df, store)
    test = build_arrays(test_df, store)

    z = np.load(config.PREDS_DIR / f"{a.tag}.npz")
    thr, _ = best_threshold(
        z["val_prob"].astype(np.float32), val["current"], val["target"],
        config.THRESHOLD_VALUES)
    pred = z["test_prob"].astype(np.float32) >= thr

    d = per_sample_growth_dice(test["current"], test["target"], pred)
    ok = np.where(~np.isnan(d))[0]          # samples that have growth
    order = ok[np.argsort(d[ok])]
    picks = [order[0], order[len(order) // 2], order[-1]]
    labels = ["worst", "median", "best"]
    out = config.RESULTS_DIR / f"qualitative_{a.tag}.png"
    make_figure(test["frames"], test["current"], test["target"], pred,
                picks, labels, out)
    print("saved", out, "| frames:",
          [int(test["frame_no"][i]) for i in picks])


if __name__ == "__main__":
    main()
