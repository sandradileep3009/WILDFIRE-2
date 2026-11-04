import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
METRICS = ["growth_dice", "growth_iou", "future_iou", "boundary_f1","growth_recall", "growth_precision"]

def group_name(tag):
    return re.sub(r"_s\d+|_h\d+", "", tag)

def load(results_dir):
    rows = []
    for f in sorted(Path(results_dir).glob("eval_h*.json")):
        d = json.loads(f.read_text())
        for name, e in d["results"].items():
            row = {"horizon": d["horizon"], "method": group_name(name),
                   "kind": e["kind"], "run": name}
            for m in METRICS:
                v = e["metrics"].get(m)
                row[m] = np.nan if v is None else v
            for m in ("growth_dice", "future_iou", "boundary_f1"):
                lo, hi = e["ci95"][m]
                row[f"{m}_lo"] = np.nan if lo is None else lo
                row[f"{m}_hi"] = np.nan if hi is None else hi
            if e["kind"] == "model":
                for m in ("growth_dice", "future_iou", "boundary_f1"):
                    diff, lo, hi = e["vs_persistence"][m]
                    row[f"d_{m}_vs_persist"] = diff
                    row[f"d_{m}_vs_persist_lo"] = lo
                    row[f"d_{m}_vs_best"] = e["vs_best_baseline"][m][0]
                    row[f"d_{m}_vs_best_lo"] = e["vs_best_baseline"][m][1]
                    row[f"d_{m}_vs_best_hi"] = e["vs_best_baseline"][m][2]
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(df):
    out = []
    for (h, method), g in df.groupby(["horizon", "method"]):
        r = {"horizon": h, "method": method, "kind": g["kind"].iloc[0],
             "n_seeds": len(g)}
        for m in METRICS:
            r[m] = g[m].mean()
            r[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else np.nan
        for m in ("growth_dice", "future_iou", "boundary_f1"):
            r[f"{m}_ci_lo"] = g[f"{m}_lo"].mean()
            r[f"{m}_ci_hi"] = g[f"{m}_hi"].mean()
        if g["kind"].iloc[0] == "model":
            r["seeds_beating_best_baseline_growth_dice"] = int(
                (g["d_growth_dice_vs_best_lo"] > 0).sum())
        out.append(r)
    return pd.DataFrame(out).sort_values(["horizon", "kind", "method"])


def to_markdown(s):
    lines = ["| H | method | growth Dice | growth IoU | future IoU | "
             "boundary F1 | seeds |",
             "|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        def fmt(m):
            v, sd = r[m], r.get(f"{m}_std", np.nan)
            if np.isnan(v):
                return "n/a"
            return f"{v:.3f}" + ("" if np.isnan(sd) else f" ± {sd:.3f}")
        lines.append(
            f"| {int(r.horizon)} | {r.method} | {fmt('growth_dice')} | "
            f"{fmt('growth_iou')} | {fmt('future_iou')} | "
            f"{fmt('boundary_f1')} | {int(r.n_seeds)} |")
    return "\n".join(lines)


def plot(s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, m, title in zip(
            axes, ["growth_dice", "boundary_f1"],
            ["Growth Dice (new fire pixels)",
             "Boundary F1 (future mask, 3 px tol.)"]):
        for method, g in s.groupby("method"):
            g = g.sort_values("horizon")
            style = "--" if g["kind"].iloc[0] == "baseline" else "-"
            ax.errorbar(
                g["horizon"], g[m], yerr=g.get(f"{m}_std"),
                marker="o", ms=4, lw=1.5, ls=style, label=method,
                capsize=2)
        ax.set_xlabel("prediction horizon (frames)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("score")
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    import config
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else config.RESULTS_DIR
    df = load(d)
    if df.empty:
        raise SystemExit(f"no eval_h*.json found in {d}; run evaluation first")
    s = summarise(df)
    s.to_csv(d / "summary_table.csv", index=False)
    (d / "summary_table.md").write_text(to_markdown(s))
    plot(s, d / "fig_horizon.png")
    print(to_markdown(s))


if __name__ == "__main__":
    main()
