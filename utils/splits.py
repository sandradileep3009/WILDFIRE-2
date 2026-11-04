import numpy as np

DEFAULT_FRACTIONS = (0.70, 0.72, 0.85, 0.87)

def frame_intervals(min_frame, max_frame, fractions=DEFAULT_FRACTIONS):
    a, b, c, d = fractions
    n = max_frame - min_frame + 1
    def at(frac):
        return min_frame + int(n * frac)
    return {"train": (min_frame, at(a)),"val": (at(b), at(c)), "test": (at(d), max_frame + 1),}

def _inside(df, lo, hi):
    return (df["start_frame"] >= lo) & (df["target_frame"] < hi)

def split_sequences(df, fractions=DEFAULT_FRACTIONS):
    for col in ("start_frame", "target_frame"):
        if col not in df.columns:
            raise ValueError(f"sequences table needs a '{col}' column; ""rebuild it with preprocessing/build_sequences.py")
    lo = int(df["start_frame"].min())
    hi = int(df["target_frame"].max())
    iv = frame_intervals(lo, hi, fractions)
    out = []
    for name in ("train", "val", "test"):
        a, b = iv[name]
        part = df[_inside(df, a, b)]
        part = part.sort_values("current_frame_number")
        out.append(part.reset_index(drop=True))

    train, val, test = out
    assert_no_frame_overlap(train, val, test)
    return train, val, test


def blocked_kfold(df, k=5, gap=20):
    lo = int(df["start_frame"].min())
    hi = int(df["target_frame"].max())
    edges = np.linspace(lo, hi + 1, k + 1).round().astype(int)
    blocks = [(int(edges[i]) + gap, int(edges[i + 1]) - gap) for i in range(k)]
    
    def take(block_ids):
        mask = np.zeros(len(df), dtype=bool)
        for j in block_ids:
            a, b = blocks[j]
            mask |= _inside(df, a, b).to_numpy()
        return (df[mask].sort_values("current_frame_number") .reset_index(drop=True))

    folds = []
    for i in range(k):
        test_ids = [i]
        val_ids = [(i + 1) % k]
        train_ids = [j for j in range(k) if j not in (i, (i + 1) % k)]
        tr, va, te = take(train_ids), take(val_ids), take(test_ids)
        assert_no_frame_overlap(tr, va, te)
        folds.append((tr, va, te))
    return folds


def frames_used(df):
    used = set()
    for s, t in zip(df["start_frame"], df["target_frame"]):
        used.update(range(int(s), int(t) + 1))
    return used


def assert_no_frame_overlap(*parts):
    sets = [frames_used(p) for p in parts if len(p)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared = sets[i] & sets[j]
            if shared:
                raise AssertionError(f"Temporal leakage: {len(shared)} frames shared "f"between split {i} and split {j} " f"(e.g. {sorted(shared)[:5]})" )

def split_dataframe(df, sort_cols, fractions=DEFAULT_FRACTIONS):
    train_end_frac, val_start_frac, val_end_frac, test_start_frac = fractions
    df = df.sort_values(sort_cols).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_end_frac)
    val_start = int(n * val_start_frac)
    val_end = int(n * val_end_frac)
    test_start = int(n * test_start_frac)

    return (
        df.iloc[:train_end],
        df.iloc[val_start:val_end],
        df.iloc[test_start:],
    )
