from pathlib import Path

import numpy as np
from PIL import Image


def make_synthetic(root, n_frames=260, size=64, seed=0):
    root = Path(root)
    img_dir = root / "data" / "Segmentation" / "Images"
    msk_dir = root / "data" / "Masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size]

    for f in range(1, n_frames + 1):
        cy = 22 + 0.04 * f
        cx = 20 + 0.10 * f
        ry = 4 + 0.05 * f
        rx = 5 + 0.09 * f
        d = ((ys - cy) / ry) ** 2 + ((xs - cx) / rx) ** 2
        d = d + rng.normal(0, 0.03, d.shape)       # noisy boundary
        mask = d <= 1.0

        img = rng.integers(20, 60, (size, size, 3)).astype(np.uint8)
        img[mask] = [230, 110, 30]
        img = np.clip(
            img.astype(int) + rng.integers(-10, 10, img.shape), 0, 255
        ).astype(np.uint8)

        Image.fromarray(img).save(img_dir / f"image_{f}.jpg", quality=95)
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            msk_dir / f"image_{f}.png"
        )
    return img_dir, msk_dir
