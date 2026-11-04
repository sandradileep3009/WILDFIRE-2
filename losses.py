import torch
import torch.nn.functional as F


def dice_loss(
    logits,
    targets,
    probs=None,
    eps=1e-6,
):
    if probs is None:
        probs = torch.sigmoid(logits)

    probs = probs.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)

    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return 1.0 - dice.mean()


def bce_dice_loss(
    logits,
    targets,
    pos_weight=10.0,
):
    weight = torch.tensor(
        pos_weight,
        dtype=logits.dtype,
        device=logits.device,
    )

    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=weight,
    )

    dice = dice_loss(
        logits,
        targets,
    )

    return 0.5 * (bce + dice)


@torch.no_grad()
def segmentation_metrics(
    logits,
    targets,
    threshold=0.5,
    eps=1e-6,
):
    probs = torch.sigmoid(logits)
    preds = probs >= threshold
    targets = targets.bool()

    tp = (preds & targets).flatten(1).sum(dim=1).float()
    fp = (preds & ~targets).flatten(1).sum(dim=1).float()
    fn = (~preds & targets).flatten(1).sum(dim=1).float()

    target_area = targets.flatten(1).sum(dim=1)

    union = tp + fp + fn

    iou = (tp + eps) / (union + eps)

    dice = (
        2.0 * tp + eps
    ) / (
        2.0 * tp + fp + fn + eps
    )

    precision = (
        tp + eps
    ) / (
        tp + fp + eps
    )

    recall = (
        tp + eps
    ) / (
        tp + fn + eps
    )

    active = target_area > 0
    active_count = int(active.sum().item())

    if active_count:
        active_iou = iou[active].mean().item()
        active_dice = dice[active].mean().item()
    else:
        active_iou = 0.0
        active_dice = 0.0

    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "active_iou": active_iou,
        "active_dice": active_dice,
        "active_count": active_count,
        "empty_rate": (~active).float().mean().item(),
    }


@torch.no_grad()
def nonzero_target_fraction(targets):
    return (
        targets.flatten(start_dim=1)
        .sum(dim=1)
        .gt(0)
        .float()
        .mean()
        .item()
    )

def band_mask(current, band_px):
    """Pixels within band_px of the current fire, excluding the fire.

    current: float tensor [B,1,H,W] in {0,1}. Uses max-pooling as a
    (square) dilation.
    """
    k = 2 * int(band_px) + 1
    dil = F.max_pool2d(current, kernel_size=k, stride=1, padding=band_px)
    return (dil > 0.5) & (current < 0.5)


def masked_bce_dice_loss(
    logits,
    targets,
    valid,
    pos_weight=5.0,
    smooth=1.0,
):
    """BCE + Dice computed ONLY on `valid` pixels.

    For growth prediction `valid` = pixels outside the current fire
    (optionally inside a band around its edge), because pixels already
    on fire cannot be NEW growth. This removes the trivial negatives
    that dominated the old loss.

    Dice is POOLED over the batch (not averaged per sample), so
    samples with no growth cannot contribute a loss of 1.0 and swamp
    the gradient.

    logits, targets, valid: [B,1,H,W]; valid is bool or {0,1}.
    """
    valid = valid.float()
    targets = targets.float() * valid

    weight = torch.tensor(
        pos_weight, dtype=logits.dtype, device=logits.device
    )

    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=weight, reduction="none"
    )
    bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)

    probs = torch.sigmoid(logits) * valid
    inter = (probs * targets).sum()
    denom = probs.sum() + targets.sum()
    dice = 1.0 - (2.0 * inter + smooth) / (denom + smooth)

    return bce + dice
def band_mask(current, band_px):
    """Pixels within band_px of the current fire, excluding the fire.

    current: float tensor [B,1,H,W] in {0,1}. Uses max-pooling as a
    (square) dilation.
    """
    k = 2 * int(band_px) + 1
    dil = F.max_pool2d(current, kernel_size=k, stride=1, padding=band_px)
    return (dil > 0.5) & (current < 0.5)


def masked_bce_dice_loss(
    logits,
    targets,
    valid,
    pos_weight=5.0,
    smooth=1.0,
):
    """BCE + Dice computed ONLY on `valid` pixels.

    For growth prediction `valid` = pixels outside the current fire
    (optionally inside a band around its edge), because pixels already
    on fire cannot be NEW growth. This removes the trivial negatives
    that dominated the old loss.

    Dice is POOLED over the batch (not averaged per sample), so
    samples with no growth cannot contribute a loss of 1.0 and swamp
    the gradient.

    logits, targets, valid: [B,1,H,W]; valid is bool or {0,1}.
    """
    valid = valid.float()
    targets = targets.float() * valid

    weight = torch.tensor(
        pos_weight, dtype=logits.dtype, device=logits.device
    )

    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=weight, reduction="none"
    )
    bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)

    probs = torch.sigmoid(logits) * valid
    inter = (probs * targets).sum()
    denom = probs.sum() + targets.sum()
    dice = 1.0 - (2.0 * inter + smooth) / (denom + smooth)

    return bce + dice