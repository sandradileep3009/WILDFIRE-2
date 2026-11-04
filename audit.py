"""
Dataset / split audit for the publication experiments.

Run from the project root (after preprocessing/match_segmentation.py and
preprocessing/build_sequences.py for each horizon):

    python audit.py                 # all horizons in config.HORIZONS
    python audit.py --horizons 10 20

Checks (exit code 1 if any HARD check fails):
  1. matched frames: count, range, gaps in numbering
  2. for every horizon: train/val/test never share a frame, and the gap
     between splits is at least as long as one window (horizon + span)
  3. the U-Net (which makes the predicted masks) never trained on, or
     was selected on, a frame used by the test split
  4. byte-identical images across train and test (exact duplicates)
Writes results/dataset_audit/audit.json and audit.txt.

NOT checked (say so in the paper): near-duplicate frames, and the fact
that all frames come from ONE continuous event.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config                                              # noqa: E402
from utils.splits import (                                 # noqa: E402
    frames_used, split_dataframe, split_sequences)

OUT = config.RESULTS_DIR / "dataset_audit"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=list(config.HORIZONS))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    lines, report, failed = [], {}, False

    def say(s=""):
        print(s)
        lines.append(s)

    # ---- 1. matched frames -------------------------------------
    pairs = pd.read_csv(config.SEGMENTATION_PAIRS_CSV)
    matched = pairs[pairs["status"].str.lower() == "matched"].copy()
    matched["frame_number"] = matched["frame_number"].astype(int)
    fr = sorted(matched["frame_number"])
    span_all = fr[-1] - fr[0] + 1
    missing = span_all - len(fr)
    say(f"matched frames: {len(fr)}  range {fr[0]}..{fr[-1]}  "
        f"missing numbers inside range: {missing}")
    if missing:
        say("  WARNING: numbering is not contiguous. 'Horizon H' then means "
            "H frame NUMBERS, and sequences with holes are dropped. Report "
            "the real time step between frames (video fps / sampling).")
    report["matched_frames"] = len(fr)
    report["frame_range"] = [fr[0], fr[-1]]
    report["missing_numbers_in_range"] = int(missing)

    # ---- U-Net split (row fractions) ---------------------------
    u_tr, u_va, _ = split_dataframe(
        matched, sort_cols=["frame_number"], fractions=config.SPLIT_FRACTIONS)
    unet_seen = set(u_tr["frame_number"]) | set(u_va["frame_number"])

    span = (config.INPUT_LEN - 1) * config.INPUT_STRIDE
    report["horizons"] = {}

    for h in args.horizons:
        csv = config.PROCESSED_DIR / f"sequences_h{h}.csv"
        say(f"\n=== horizon {h} ===")
        if not csv.exists():
            say(f"  missing {csv.name}: run preprocessing/build_sequences.py "
                f"--horizon {h}")
            failed = True
            continue
        df = pd.read_csv(csv)
        tr, va, te = split_sequences(df, config.SPLIT_FRACTIONS)  # asserts
        entry = {"sequences": int(len(df))}

        names = {"train": tr, "val": va, "test": te}
        used = {k: frames_used(v) for k, v in names.items()}
        for k, v in names.items():
            if len(v):
                say(f"  {k:5s}: {len(v):5d} sequences, frames "
                    f"{int(v['start_frame'].min())}..{int(v['target_frame'].max())}")
            else:
                say(f"  {k:5s}: EMPTY")
                failed = True
            entry[k] = int(len(v))

        window = h + span
        for a, b in (("train", "val"), ("val", "test")):
            if used[a] and used[b]:
                gap = min(used[b]) - max(used[a]) - 1
                ok = gap >= window
                say(f"  gap {a}->{b}: {gap} frames (window = {window}) "
                    f"{'OK' if ok else 'TOO SMALL (not a leak, but note it)'}")
                entry[f"gap_{a}_{b}"] = int(gap)

        # U-Net must not have seen any ConvLSTM-test frame
        test_frames = used["test"]
        leak = sorted(test_frames & unet_seen)
        say(f"  U-Net train/val frames inside this test split: {len(leak)}")
        if leak:
            say(f"    FAIL (e.g. {leak[:5]}). The predicted-mask experiment "
                "would be leaky.")
            failed = True
        entry["unet_frames_in_test"] = len(leak)
        report["horizons"][h] = entry

    # ---- 4. exact duplicate images across splits ---------------
    h0 = min(args.horizons)
    csv = config.PROCESSED_DIR / f"sequences_h{h0}.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        tr, _, te = split_sequences(df, config.SPLIT_FRACTIONS)
        by_frame = dict(zip(matched["frame_number"], matched["image"]))

        def hashes(frame_set):
            return {sha256(by_frame[f]) for f in frame_set if f in by_frame
                    and Path(by_frame[f]).is_file()}

        dup = hashes(frames_used(tr)) & hashes(frames_used(te))
        say(f"\nbyte-identical images shared by train and test: {len(dup)}")
        report["duplicate_images_train_test"] = len(dup)
        if dup:
            failed = True

    say("\nNOT VERIFIED: near-duplicate frames (consecutive frames are highly "
        "similar by nature); independence of physical fire events (all "
        "frames come from one continuous recording).")
    say("AUDIT STATUS: " + ("FAIL" if failed else "PASSED (hard checks)"))

    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    (OUT / "audit.txt").write_text("\n".join(lines))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
