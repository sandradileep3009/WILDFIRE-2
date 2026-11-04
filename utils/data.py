
"""
Numpy-only data loading shared by training and evaluation.

Every unique image / mask is read from disk ONCE, resized, and kept in
memory (about 100 MB for 2,003 frames at 128x128), so training does no
disk I/O per step.
"""

import numpy as np
import pandas as pd
from PIL import Image


class FrameStore:
    """Cache of resized frames (uint8 RGB) and masks (bool)."""

    def __init__(self, size):
        self.size = tuple(size)          # (H, W)
        self._img = {}
        self._msk = {}

    def image(self, path):
        path = str(path)
        if path not in self._img:
            with Image.open(path) as im:
                im = im.convert("RGB").resize(
                    (self.size[1], self.size[0]), Image.Resampling.BILINEAR
                )
                self._img[path] = np.asarray(im, dtype=np.uint8)
        return self._img[path]

    def mask(self, path):
        path = str(path)
        if path not in self._msk:
            with Image.open(path) as im:
                im = im.convert("L").resize(
                    (self.size[1], self.size[0]), Image.Resampling.NEAREST
                )
                self._msk[path] = np.asarray(im) > 0
        return self._msk[path]


def build_arrays(df, store):
    """Turn a sequences table into dense numpy arrays.

    Returns a dict:
      frames   uint8 [N, T, H, W, 3]   input RGB frames
      hist     bool  [N, T, H, W]      input masks (same time steps)
      current  bool  [N, H, W]         mask at the last input frame
      target   bool  [N, H, W]         ground-truth mask at t + horizon
      growth   bool  [N, H, W]         target & ~current
      frame_no int   [N]               current_frame_number
    """
    frames, hist, cur, tgt = [], [], [], []

    for _, row in df.iterrows():
        imgs = str(row["input_frames"]).split("|")
        msks = str(row["input_masks"]).split("|")
        frames.append(np.stack([store.image(p) for p in imgs]))
        hist.append(np.stack([store.mask(p) for p in msks]))
        cur.append(store.mask(row["current_mask"]))
        tgt.append(store.mask(row["target_mask"]))

    if not frames:
        H, W = store.size
        return {
            "frames": np.zeros((0, 0, H, W, 3), np.uint8),
            "hist": np.zeros((0, 0, H, W), bool),
            "current": np.zeros((0, H, W), bool),
            "target": np.zeros((0, H, W), bool),
            "growth": np.zeros((0, H, W), bool),
            "frame_no": np.zeros((0,), int),
        }

    current = np.stack(cur)
    target = np.stack(tgt)
    return {
        "frames": np.stack(frames),
        "hist": np.stack(hist),
        "current": current,
        "target": target,
        "growth": target & ~current,
        "frame_no": df["current_frame_number"].to_numpy(int),
    }


def load_sequences(csv_path):
    df = pd.read_csv(csv_path)
    needed = {
        "input_frames", "input_masks", "current_mask", "target_mask",
        "start_frame", "current_frame_number", "target_frame",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns {sorted(missing)}; rebuild "
            "with: python preprocessing/build_sequences.py --horizon H"
        )
    return df
