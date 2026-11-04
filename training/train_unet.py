import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from models.segmentation import UNet
from training.losses import bce_dice_loss, segmentation_metrics
from utils.seed import set_seed, seeded_generator, worker_init_fn
from utils.splits import split_dataframe


torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class SegmentationDataset(Dataset):
    def __init__(self, df, augment=False):
        self.augment = augment
        self.images, self.masks = [], []

        print(f"Preloading {len(df)} images...")

        for i, (_, row) in enumerate(df.reset_index(drop=True).iterrows()):
            with Image.open(row["image"]) as img:
                img = img.convert("RGB").resize(
                    config.SEGMENTATION_SIZE,
                    Image.Resampling.BILINEAR
                )
                x = torch.from_numpy(
                    np.asarray(img, dtype=np.float32)
                ).permute(2, 0, 1).div_(255.0)

            with Image.open(row["mask"]) as mask:
                mask = mask.convert("L").resize(
                    config.SEGMENTATION_SIZE,
                    Image.Resampling.NEAREST
                )
                y = (
                    torch.from_numpy(
                        np.asarray(mask, dtype=np.float32)
                    ) > 0
                ).float().unsqueeze(0)

            self.images.append(x)
            self.masks.append(y)

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(df)}")

        self.images = torch.stack(self.images).contiguous()
        self.masks = torch.stack(self.masks).contiguous()

        print(
            f"Loaded: {self.images.shape} | "
            f"{self.masks.shape}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.masks[idx]

        if self.augment and torch.rand(()) < 0.5:
            x = torch.flip(x, [-1])
            y = torch.flip(y, [-1])

        return x, y


def load_data():
    df = pd.read_csv(config.SEGMENTATION_PAIRS_CSV)
    df = df[df["status"].str.lower() == "matched"].copy()
    df["frame_number"] = df["frame_number"].astype(int)
    return split_dataframe(df, sort_cols=["frame_number"])


@torch.inference_mode()
def validate(model, loader, amp):
    model.eval()
    loss = iou = dice = 0.0

    for x, y in loader:
        x = x.to(config.DEVICE, non_blocking=True)
        y = y.to(config.DEVICE, non_blocking=True)

        with autocast(
            device_type=config.DEVICE.type,
            enabled=amp
        ):
            logits = model(x)
            l = bce_dice_loss(
                logits, y,
                pos_weight=config.UNET_POS_WEIGHT
            )

        m = segmentation_metrics(logits, y)
        loss += l.item()
        iou += m["iou"]
        dice += m["dice"]

    n = max(len(loader), 1)
    return loss / n, iou / n, dice / n


def suffixed(path, suffix):
    return path if not suffix else path.with_name(
        path.stem + suffix + path.suffix
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--suffix", default="")
    args = p.parse_args()

    set_seed(args.seed)

    checkpoint = suffixed(config.UNET_CHECKPOINT, args.suffix)
    history_file = suffixed(config.UNET_HISTORY_CSV, args.suffix)

    print("=" * 70)
    print("U-NET FIRE SEGMENTATION")
    print("=" * 70)
    print("Device:", config.DEVICE)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("Seed:", args.seed)

    train_df, val_df, _ = load_data()

    train_ds = SegmentationDataset(train_df, augment=True)
    val_ds = SegmentationDataset(val_df)

    amp = config.DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE_UNET,
        shuffle=True,
        num_workers=0,
        pin_memory=amp,
        generator=seeded_generator(args.seed),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE_UNET,
        shuffle=False,
        num_workers=0,
        pin_memory=amp,
    )

    model = UNet().to(config.DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE_UNET,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    scaler = GradScaler("cuda", enabled=amp)

    best_dice = -1
    history = []

    for epoch in range(1, config.UNET_EPOCHS + 1):
        model.train()
        running = 0.0

        for x, y in train_loader:
            x = x.to(config.DEVICE, non_blocking=True)
            y = y.to(config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(
                device_type=config.DEVICE.type,
                enabled=amp
            ):
                logits = model(x)
                loss = bce_dice_loss(
                    logits, y,
                    pos_weight=config.UNET_POS_WEIGHT
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

        train_loss = running / max(len(train_loader), 1)
        val_loss, val_iou, val_dice = validate(
            model, val_loader, amp
        )

        scheduler.step(val_dice)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_dice": val_dice,
        })

        print(
            f"Epoch {epoch:02d}/{config.UNET_EPOCHS} | "
            f"train={train_loss:.4f} | "
            f"val={val_loss:.4f} | "
            f"IoU={val_iou:.4f} | "
            f"Dice={val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_iou": val_iou,
                "val_dice": val_dice,
                "seed": args.seed,
            }, checkpoint)

            print("  Saved best U-Net checkpoint.")

    pd.DataFrame(history).to_csv(history_file, index=False)

    print("=" * 70)
    print(f"BEST VAL DICE: {best_dice:.4f}")
    print("Checkpoint:", checkpoint)
    print("History:", history_file)


if __name__ == "__main__":
    main()
