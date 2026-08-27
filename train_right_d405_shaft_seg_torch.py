#!/usr/bin/env python3
from __future__ import annotations
"""Small reviewed-mask segmentation trainer with no external YOLO dependency.

It is deliberately a baseline for the fixed right-D405 view.  With a
positive-only dataset it measures held-out mask overlap only; it cannot measure
false-positive behaviour because no no-target images are present.
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class ModelConfig:
    width: int = 24
    image_width: int = 320
    image_height: int = 240


class ShaftSegDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: tuple[int, int], augment: str = "none") -> None:
        # P4-focused data keeps its provenance in the filename
        # (e.g. old_frame_*.jpg / auto_frame_*.jpg).  Do not restrict this
        # to the legacy frame_ prefix, otherwise a valid focused dataset is
        # incorrectly treated as empty.
        self.images = sorted((root / "images" / split).glob("*.jpg"))
        self.masks = root / "masks" / split
        self.width, self.height = image_size
        self.augment = augment
        if not self.images:
            raise RuntimeError(f"no images in split {split}: {root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.masks / f"{image_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"failed to read image/mask for {image_path.stem}")
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST) > 127
        if self.augment != "none":
            if random.random() < 0.7:
                image = cv2.convertScaleAbs(image, alpha=random.uniform(0.80, 1.20), beta=random.uniform(-20, 20))
            if random.random() < 0.5:
                shift_x, shift_y = random.randint(-18, 18), random.randint(-18, 18)
                transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                image = cv2.warpAffine(image, transform, (self.width, self.height), borderMode=cv2.BORDER_REFLECT_101)
                mask = cv2.warpAffine(mask.astype(np.uint8), transform, (self.width, self.height), flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
        if self.augment == "strong":
            angle, scale = random.uniform(-12.0, 12.0), random.uniform(0.88, 1.12)
            transform = cv2.getRotationMatrix2D((self.width / 2.0, self.height / 2.0), angle, scale)
            image = cv2.warpAffine(image, transform, (self.width, self.height), borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask.astype(np.uint8), transform, (self.width, self.height), flags=cv2.INTER_NEAREST, borderValue=0).astype(bool)
            if random.random() < 0.35:
                image = cv2.GaussianBlur(image, (3, 3), 0)
            if random.random() < 0.35:
                noise = np.random.normal(0.0, 5.0, image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).float() / 255.0
        mask_tensor = torch.from_numpy(mask.astype(np.float32)[None])
        return image_tensor, mask_tensor, image_path.name


class DoubleConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True),
        )

    def forward(self, value):
        return self.block(value)


class TinyUNet(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.first = DoubleConv(3, width)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(width, width * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(width * 2, width * 4))
        self.up1 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec1 = DoubleConv(width * 4, width * 2)
        self.up2 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec2 = DoubleConv(width * 2, width)
        self.output = nn.Conv2d(width, 1, 1)

    def forward(self, value):
        first = self.first(value)
        second = self.down1(first)
        third = self.down2(second)
        decoded = self.dec1(torch.cat((self.up1(third), second), dim=1))
        decoded = self.dec2(torch.cat((self.up2(decoded), first), dim=1))
        return self.output(decoded)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    total = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (total + 1.0)).mean()


def evaluate(model, loader, device, threshold: float = 0.5, save_dir: Path | None = None) -> dict:
    model.eval()
    intersection = union = pred_total = target_total = 0
    with torch.no_grad():
        for images, targets, names in loader:
            logits = model(images.to(device))
            predictions = logits.sigmoid().cpu() >= threshold
            truth = targets >= 0.5
            intersection += int((predictions & truth).sum())
            union += int((predictions | truth).sum())
            pred_total += int(predictions.sum())
            target_total += int(truth.sum())
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                for image, pred, truth_mask, name in zip(images, predictions, truth, names):
                    rgb = image.permute(1, 2, 0).numpy()[:, :, ::-1] * 255.0
                    view = rgb.astype(np.uint8)
                    view[truth_mask[0].numpy()] = (0, 180, 0)
                    view[pred[0].numpy()] = (0, 0, 255)
                    cv2.imwrite(str(save_dir / name), view)
    precision = intersection / max(pred_total, 1)
    recall = intersection / max(target_total, 1)
    iou = intersection / max(union, 1)
    dice = 2.0 * intersection / max(pred_total + target_total, 1)
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "threshold": threshold}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/right_d405_slotted_shaft_seg_yolo")
    parser.add_argument("--out-dir", default="runs/right_d405_shaft_seg_torch/positive_only_v1")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="optional CPU thread cap; use a modest value on this workstation to avoid oversubscription",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--augmentation",
        choices=("baseline", "strong"),
        default="baseline",
        help="strong adds limited rotation/scale/blur/noise for the P4-specific fine-tune",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="optional reviewed checkpoint to fine-tune instead of random initialization",
    )
    args = parser.parse_args()
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            parser.error("--torch-threads must be at least 1")
        torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    root, out_dir = Path(args.dataset).resolve(), Path(args.out_dir).resolve()
    if not (root / "masks" / "train").is_dir():
        raise SystemExit(f"prepared masks missing: {root}; run prepare_right_d405_shaft_seg_dataset.py first")
    config = ModelConfig()
    train_set = ShaftSegDataset(root, "train", (config.image_width, config.image_height), augment=args.augmentation)
    val_set = ShaftSegDataset(root, "val", (config.image_width, config.image_height), augment="none")
    test_set = ShaftSegDataset(root, "test", (config.image_width, config.image_height), augment="none")
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False, num_workers=0)
    model = TinyUNet(config.width).to(device)
    init_checkpoint = None if args.init_checkpoint is None else Path(args.init_checkpoint).resolve()
    if init_checkpoint is not None:
        if not init_checkpoint.is_file():
            raise SystemExit(f"initial checkpoint not found: {init_checkpoint}")
        initial = torch.load(init_checkpoint, map_location=device, weights_only=True)
        initial_config = ModelConfig(**initial["model_config"])
        if initial_config != config:
            raise SystemExit(
                "initial checkpoint model configuration differs from the current trainer: "
                f"{initial_config} != {config}"
            )
        model.load_state_dict(initial["model_state"])
        print(f"fine-tuning initial checkpoint={init_checkpoint}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([12.0], device=device))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_iou, history = -1.0, []
    print(
        f"training device={device}, torch_threads={torch.get_num_threads()}, "
        f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}, augmentation={args.augmentation}",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, targets, _names in train_loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets) + dice_loss(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history.append(record)
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            torch.save({"model_state": model.state_dict(), "model_config": asdict(config), "metrics": metrics, "epoch": epoch}, out_dir / "best.pt")
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps(record), flush=True)
    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metrics = evaluate(model, val_loader, device, save_dir=out_dir / "val_predictions")
    test_metrics = evaluate(model, test_loader, device, save_dir=out_dir / "test_predictions")
    summary = {
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "best_epoch": checkpoint["epoch"],
        "best_validation": metrics,
        "held_out_test": test_metrics,
        "initial_checkpoint": None if init_checkpoint is None else str(init_checkpoint),
        "negative_images_included": any(
            not path.read_text(encoding="utf-8").strip() for path in (root / "labels" / "train").glob("*.txt")
        ),
        "checkpoint": str(out_dir / "best.pt"),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
