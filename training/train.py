import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                              # noqa: E402
from models.convlstm import SpreadPredictionModel          # noqa: E402
from training.losses import band_mask, masked_bce_dice_loss  # noqa: E402
from utils.data import FrameStore, build_arrays, load_sequences  # noqa: E402
from utils.metrics import best_threshold                   # noqa: E402
from utils.seed import seeded_generator, set_seed          # noqa: E402
from utils.splits import split_sequences                   # noqa: E402


def make_tag(task, horizon, seed, use_history, suffix=""):
    hist = "" if use_history else "_nohist"
    return f"{task}_h{horizon}_s{seed}{hist}{suffix}"


class SeqDataset(Dataset):
    """In-memory dataset. Items: x [T,C,H,W], current [1,H,W], y [1,H,W].

    task == 'growth'   : y = target & ~current   (new fire pixels)
    task == 'fullmask' : y = target              (whole future mask)
    """

    def __init__(self, arrays, task, use_history, augment):
        self.task = task
        self.use_history = use_history
        self.augment = augment

        # [N,T,H,W,3] uint8 -> [N,T,3,H,W] uint8
        self.rgb = torch.from_numpy(arrays["frames"]).permute(0, 1, 4, 2, 3)
        self.rgb = self.rgb.contiguous()
        self.hist = torch.from_numpy(arrays["hist"]).unsqueeze(2).to(torch.uint8)
        self.current = torch.from_numpy(arrays["current"]).unsqueeze(1)
        self.target = torch.from_numpy(arrays["target"]).unsqueeze(1)

    def __len__(self):
        return len(self.rgb)

    def __getitem__(self, i):
        x = self.rgb[i].float() / 255.0                       # [T,3,H,W]
        cur = self.current[i].float()                         # [1,H,W]
        tgt = self.target[i].float()

        if self.use_history:
            x = torch.cat([x, self.hist[i].float()], dim=1)   # [T,4,H,W]

        y = tgt * (1.0 - cur) if self.task == "growth" else tgt

        if self.augment:
            if torch.rand(()) < 0.5:
                x, cur, y = (t.flip(-1) for t in (x, cur, y))
            if torch.rand(()) < 0.5:
                x, cur, y = (t.flip(-2) for t in (x, cur, y))
            if x.shape[-1] == x.shape[-2]:
                k = int(torch.randint(0, 4, ()).item())
                if k:
                    x, cur, y = (
                        torch.rot90(t, k, dims=(-2, -1)) for t in (x, cur, y)
                    )
            gain = 0.8 + 0.4 * torch.rand(())
            x = x.clone()
            x[:, :3] = (x[:, :3] * gain).clamp(0, 1)

        return x, cur, y


def make_loader(ds, batch_size, shuffle, seed, workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=seeded_generator(seed) if shuffle else None,
    )


@torch.no_grad()
def predict_probs(model, ds, device, batch_size=16):
    """Probability map for every sample, float32 [N,H,W]."""
    model.eval()
    out = []
    for x, cur, _ in make_loader(ds, batch_size, False, 0, 0):
        logits = model(x.to(device), cur.to(device))
        out.append(torch.sigmoid(logits).squeeze(1).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,))


def run(args):
    set_seed(args.seed)
    device = config.DEVICE
    use_history = not args.no_mask_history
    tag = make_tag(args.task, args.horizon, args.seed, use_history,
                   args.tag_suffix)

    csv = args.csv or (config.PROCESSED_DIR / f"sequences_h{args.horizon}.csv")
    df = load_sequences(csv)
    train_df, val_df, test_df = split_sequences(df, config.SPLIT_FRACTIONS)
    if args.max_train:
        train_df = train_df.iloc[: args.max_train]

    print("=" * 70)
    print(f"{tag} | device={device} | train/val/test = "
          f"{len(train_df)}/{len(val_df)}/{len(test_df)}")
    print("=" * 70)
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise SystemExit("A split is empty; horizon too large for the data?")

    store = FrameStore((args.size, args.size))
    train_a = build_arrays(train_df, store)
    val_a = build_arrays(val_df, store)
    test_a = build_arrays(test_df, store)

    train_ds = SeqDataset(train_a, args.task, use_history, augment=True)
    val_ds = SeqDataset(val_a, args.task, use_history, augment=False)
    test_ds = SeqDataset(test_a, args.task, use_history, augment=False)
    train_loader = make_loader(
        train_ds, args.batch_size, True, args.seed, config.NUM_WORKERS)

    model = SpreadPredictionModel(
        in_channels=3 + int(use_history),
        hidden_dims=config.CONVLSTM_HIDDEN_DIMS,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    pos_weight = (
        config.GROWTH_LOSS_POS_WEIGHT if args.task == "growth"
        else config.FULLMASK_POS_WEIGHT
    )

    ckpt_path = config.CHECKPOINT_DIR / f"{tag}.pt"
    hist_rows, best_score, best_state, bad = [], -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, nb = time.time(), 0.0, 0

        for x, cur, y in train_loader:
            x, cur, y = x.to(device), cur.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)

            if args.task == "growth":
                valid = cur < 0.5
                if config.GROWTH_BAND_PX:
                    valid = band_mask(cur, config.GROWTH_BAND_PX)
            else:
                valid = torch.ones_like(cur, dtype=torch.bool)

            with torch.autocast(device_type=device.type,
                                dtype=torch.float16, enabled=use_amp):
                logits = model(x, cur)
                loss = masked_bce_dice_loss(
                    logits.float(), y, valid, pos_weight=pos_weight)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            nb += 1

        sched.step()

        val_prob = predict_probs(model, val_ds, device)
        thr, score = best_threshold(
            val_prob, val_a["current"], val_a["target"],
            config.THRESHOLD_VALUES, objective="growth_dice")

        hist_rows.append({
            "epoch": epoch, "train_loss": total / max(nb, 1),
            "val_growth_dice": score, "val_threshold": thr,
            "seconds": time.time() - t0,
        })
        print(f"epoch {epoch:02d}/{args.epochs} loss {total/max(nb,1):.4f} "
              f"val growth-Dice {score:.4f} @thr {thr:.2f}")

        if score > best_score or best_state is None:
            best_score, bad = score, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= config.EARLY_STOP_PATIENCE:
                print("early stop")
                break

    model.load_state_dict(best_state)
    model.to(device)

    val_prob = predict_probs(model, val_ds, device)
    test_prob = predict_probs(model, test_ds, device)
    thr, _ = best_threshold(
        val_prob, val_a["current"], val_a["target"],
        config.THRESHOLD_VALUES, objective="growth_dice")

    torch.save({
        "model_state_dict": best_state,
        "args": {k: str(v) for k, v in vars(args).items()}, "tag": tag,
        "threshold": thr, "best_val_growth_dice": best_score,
        "in_channels": 3 + int(use_history),
        "hidden_dims": tuple(config.CONVLSTM_HIDDEN_DIMS),
    }, ckpt_path)

    np.savez_compressed(
        config.PREDS_DIR / f"{tag}.npz",
        val_prob=val_prob.astype(np.float16),
        test_prob=test_prob.astype(np.float16),
        val_frames=val_a["frame_no"], test_frames=test_a["frame_no"],
        threshold=thr, task=args.task, horizon=args.horizon,
    )
    pd.DataFrame(hist_rows).to_csv(
        config.RESULTS_DIR / f"{tag}_history.csv", index=False)
    print(f"saved {ckpt_path.name}, preds/{tag}.npz  (thr={thr:.2f})")
    return tag


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["growth", "fullmask"], default="growth")
    p.add_argument("--horizon", type=int, default=config.PREDICTION_HORIZON)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--no-mask-history", action="store_true")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--tag-suffix", type=str, default="")
    p.add_argument("--size", type=int, default=config.TEMPORAL_SIZE[0])
    p.add_argument("--epochs", type=int, default=config.CONVLSTM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE_CONVLSTM)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE_CONVLSTM)
    p.add_argument("--max-train", type=int, default=0,
                   help="debug: use only the first N training sequences")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())
