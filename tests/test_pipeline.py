import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic import make_synthetic                       
from preprocessing.build_sequences import build            
from utils import baselines as B                           
from utils.data import FrameStore, build_arrays            
from utils.metrics import (                              
    sample_stats, compute_metrics, bootstrap_ci,
    paired_bootstrap_diff, best_threshold)
from utils.splits import (                                 
    split_sequences, blocked_kfold, assert_no_frame_overlap)


def _pairs_csv(tmp):
    img_dir, msk_dir = make_synthetic(tmp, n_frames=260, size=64)
    rows = []
    for f in range(1, 261):
        rows.append({
            "frame_number": f,
            "image": str(img_dir / f"image_{f}.jpg"),
            "mask": str(msk_dir / f"image_{f}.png"),
            "status": "matched",
        })
    p = Path(tmp) / "pairs.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _seq(tmp, horizon):
    pairs = _pairs_csv(tmp)
    df, n = build(pairs, horizon, 3, 1, "gt", stat_size=64)
    return df


def test_build_sequences_horizon_and_columns():
    with tempfile.TemporaryDirectory() as tmp:
        for h in (1, 5, 20):
            df = _seq(tmp, h)
            assert (df["target_frame"] - df["current_frame_number"] == h).all()
            assert (df["current_frame_number"] - df["start_frame"] == 2).all()
            # frames 1..260, need t>=3 and t+h<=260
            assert len(df) == 260 - h - 2
            assert (df["growth_px"] >= 0).all()


def test_split_has_no_frame_overlap_for_any_horizon():
    with tempfile.TemporaryDirectory() as tmp:
        for h in (1, 10, 30):
            df = _seq(tmp, h)
            tr, va, te = split_sequences(df)
            assert len(tr) and len(va) and len(te)
            assert_no_frame_overlap(tr, va, te)
            assert tr["target_frame"].max() < va["start_frame"].min()
            assert va["target_frame"].max() < te["start_frame"].min()


def test_overlap_check_catches_leak():
    df = pd.DataFrame({
        "start_frame": [1, 5], "target_frame": [10, 14],
        "current_frame_number": [3, 7]})
    try:
        assert_no_frame_overlap(df.iloc[[0]], df.iloc[[1]])
    except AssertionError:
        return
    raise RuntimeError("leak not detected")


def test_blocked_kfold():
    with tempfile.TemporaryDirectory() as tmp:
        df = _seq(tmp, 5)
        folds = blocked_kfold(df, k=4, gap=5)
        assert len(folds) == 4
        for tr, va, te in folds:
            assert len(tr) and len(va) and len(te)


def _arrays(tmp, h, which="test"):
    df = _seq(tmp, h)
    parts = dict(zip(("train", "val", "test"), split_sequences(df)))
    store = FrameStore((64, 64))
    return build_arrays(parts[which], store), parts[which]


def test_metrics_no_eps_artifact():
    cur = np.zeros((4, 8, 8), bool)
    cur[:, 2:5, 2:5] = True
    tgt = cur.copy()
    tgt[:, 2:5, 5] = True
    st = sample_stats(cur, tgt, np.zeros_like(cur))        
    m = compute_metrics(st)
    assert m["growth_iou"] == 0.0
    assert np.isnan(m["growth_precision"])                 
    assert m["growth_recall"] == 0.0


def test_perfect_prediction():
    cur = np.zeros((3, 16, 16), bool)
    cur[:, 4:8, 4:8] = True
    tgt = cur.copy()
    tgt[:, 4:8, 8:10] = True
    m = compute_metrics(sample_stats(cur, tgt, tgt & ~cur))
    assert m["growth_iou"] == 1.0 and m["future_iou"] == 1.0
    assert m["boundary_f1"] == 1.0 and m["area_mae_px"] == 0.0
    assert m["growth_direction_acc8"] == 1.0


def test_baselines_beat_persistence_on_growing_fire():
    with tempfile.TemporaryDirectory() as tmp:
        h = 10
        val, _ = _arrays(tmp, h, "val")
        test, _ = _arrays(tmp, h, "test")
        res = {}
        for name in B.BASELINES:
            params = B.tune(name, val, h, input_span=2)
            pred = B.predict(name, test, params, h, 2)
            st = sample_stats(test["current"], test["target"], pred)
            res[name] = compute_metrics(st)
        assert res["persistence"]["growth_dice"] == 0.0
        for name in ("dilation", "speed_dilation", "flow_warp"):
            assert res[name]["growth_dice"] > 0.2, (name, res[name])


def test_bootstrap_and_paired():
    with tempfile.TemporaryDirectory() as tmp:
        h = 10
        val, _ = _arrays(tmp, h, "val")
        test, _ = _arrays(tmp, h, "test")
        p = B.tune("dilation", val, h, 2)
        st_d = sample_stats(test["current"], test["target"],
                            B.predict("dilation", test, p, h, 2))
        st_p = sample_stats(test["current"], test["target"],
                            B.persistence(test["current"]))
        ci = bootstrap_ci(st_d, ["growth_dice"], block=10, n_boot=100)
        lo, hi = ci["growth_dice"]
        assert lo <= hi
        d, dlo, dhi = paired_bootstrap_diff(
            st_d, st_p, "growth_dice", block=10, n_boot=100)
        assert d > 0 and dlo > 0          


def test_best_threshold_uses_val_only():
    rng = np.random.default_rng(0)
    cur = np.zeros((6, 16, 16), bool)
    cur[:, 4:8, 4:8] = True
    tgt = cur.copy()
    tgt[:, 4:8, 8:10] = True
    prob = np.where(tgt & ~cur, 0.35, 0.02) + rng.random(cur.shape) * 0.01
    t, v = best_threshold(prob, cur, tgt, (0.1, 0.2, 0.3, 0.5))
    assert t in (0.1, 0.2, 0.3) and v > 0.9


def test_evaluate_end_to_end_with_fake_model():
    """A noisy oracle must beat persistence; a useless model must not."""
    from evaluation.evaluate import evaluate
    from utils.splits import DEFAULT_FRACTIONS
    with tempfile.TemporaryDirectory() as tmp:
        h = 10
        pairs = _pairs_csv(tmp)
        df, _ = build(pairs, h, 3, 1, "gt", stat_size=64)
        csv = Path(tmp) / "seq.csv"
        df.to_csv(csv, index=False)
        preds = Path(tmp) / "preds"
        preds.mkdir()

        _, val_df, test_df = split_sequences(df)
        store = FrameStore((64, 64))
        val = build_arrays(val_df, store)
        test = build_arrays(test_df, store)
        rng = np.random.default_rng(0)

        def probs(a, signal):
            base = rng.random(a["growth"].shape) * 0.2
            return np.where(a["growth"], 0.6 * signal + base, base)

        for tag, sig in (("growth_h10_s1", 1.0), ("growth_h10_s2", 0.0)):
            np.savez(preds / f"{tag}.npz",
                     val_prob=probs(val, sig).astype(np.float16),
                     test_prob=probs(test, sig).astype(np.float16),
                     val_frames=val["frame_no"],
                     test_frames=test["frame_no"])

        payload, table = evaluate(
            h, ["growth_h10_s1", "growth_h10_s2"], csv, 64,
            (0.1, 0.2, 0.3, 0.4, 0.5), preds, Path(tmp) / "out",
            tol=3, block=10, n_boot=50, input_span=2,
            fractions=DEFAULT_FRACTIONS)
        r = payload["results"]
        assert r["growth_h10_s1"]["metrics"]["growth_dice"] > \
            r["persistence"]["metrics"]["growth_dice"]
        lo = r["growth_h10_s1"]["vs_persistence"]["growth_dice"][1]
        assert lo > 0
        assert r["growth_h10_s2"]["metrics"]["growth_dice"] < \
            r["growth_h10_s1"]["metrics"]["growth_dice"]
        assert (Path(tmp) / "out" / "eval_h10.json").exists()


def test_aggregate_results():
    import aggregate_results as agg
    from evaluation.evaluate import evaluate
    from utils.splits import DEFAULT_FRACTIONS
    with tempfile.TemporaryDirectory() as tmp:
        pairs = _pairs_csv(tmp)
        out = Path(tmp) / "out"
        preds = Path(tmp) / "preds"
        preds.mkdir()
        rng = np.random.default_rng(1)
        for h in (5, 10):
            df, _ = build(pairs, h, 3, 1, "gt", stat_size=64)
            csv = Path(tmp) / f"seq{h}.csv"
            df.to_csv(csv, index=False)
            _, vdf, tdf = split_sequences(df)
            store = FrameStore((64, 64))
            val, test = build_arrays(vdf, store), build_arrays(tdf, store)
            tags = []
            for seed in (1, 2):
                tag = f"growth_h{h}_s{seed}"
                tags.append(tag)
                def pr(a):
                    b = rng.random(a["growth"].shape) * 0.2
                    return np.where(a["growth"], 0.5 + b, b)
                np.savez(preds / f"{tag}.npz",
                         val_prob=pr(val).astype(np.float16),
                         test_prob=pr(test).astype(np.float16),
                         val_frames=val["frame_no"],
                         test_frames=test["frame_no"])
            evaluate(h, tags, csv, 64, (0.1, 0.3, 0.5), preds, out, 3, 10,
                     30, 2, DEFAULT_FRACTIONS)
        df = agg.load(out)
        summ = agg.summarise(df)
        assert set(summ["horizon"]) == {5, 10}
        row = summ[(summ.horizon == 10) & (summ.method == "growth")]
        assert int(row["n_seeds"].iloc[0]) == 2
        assert "growth" in agg.to_markdown(summ)
        agg.plot(summ, Path(tmp) / "fig.png")
        assert (Path(tmp) / "fig.png").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
