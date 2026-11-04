
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                   # noqa: E402
from models.segmentation import UNet            # noqa: E402


@torch.no_grad()
def main(threshold=0.5, batch_size=8):
    device = config.DEVICE
    ckpt = torch.load(config.UNET_CHECKPOINT, map_location="cpu")
    model = UNet()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    df = pd.read_csv(config.SEGMENTATION_PAIRS_CSV)
    df = df[df["status"].str.lower() == "matched"].copy()
    out_dir = config.PROCESSED_DIR / "pred_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    H, W = config.SEGMENTATION_SIZE
    paths = []

    rows = df.to_dict("records")
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        sizes, xs = [], []
        for r in chunk:
            with Image.open(r["image"]) as im:
                im = im.convert("RGB")
                sizes.append(im.size)                       # (W0, H0)
                im = im.resize((W, H), Image.Resampling.BILINEAR)
                xs.append(torch.from_numpy(
                    np.asarray(im, np.float32) / 255.0).permute(2, 0, 1))
        prob = torch.sigmoid(model(torch.stack(xs).to(device)))
        prob = prob.squeeze(1).cpu().numpy()

        for r, p, (w0, h0) in zip(chunk, prob, sizes):
            m = Image.fromarray(((p >= threshold) * 255).astype(np.uint8))
            m = m.resize((w0, h0), Image.Resampling.NEAREST)
            path = out_dir / f"image_{int(r['frame_number'])}.png"
            m.save(path)
            paths.append(str(path))

    df["pred_mask"] = paths
    out = config.PROCESSED_DIR / "segmentation_pairs_pred.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} predicted masks -> {out}")


if __name__ == "__main__":
    main()