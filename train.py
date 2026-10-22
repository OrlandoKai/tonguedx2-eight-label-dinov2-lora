from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import Dinov2Model


HERE = Path(__file__).resolve().parent
LABELS = [
    "TonguePale", "TipSideRed", "Spot", "Ecchymosis",
    "Crack", "Toothmark", "FurThick", "FurYellow",
]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TongueDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, release_root: Path, size: int, train: bool) -> None:
        self.frame = frame.reset_index(drop=True)
        self.root = release_root
        self.train = train
        aug: list[nn.Module] = [
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        ]
        if train:
            aug.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(5, interpolation=InterpolationMode.BILINEAR, fill=(0, 0, 0)),
                transforms.ColorJitter(brightness=0.05, saturation=0.05, hue=0.01),
            ])
        aug.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        self.transform = transforms.Compose(aug)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        relative = Path(str(row["image_path"]).replace("\\", "/"))
        image_path = self.root / "seg" / relative
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            pixel_values = self.transform(image)
        labels = torch.tensor(row[LABELS].to_numpy(dtype=np.float32), dtype=torch.float32)
        return pixel_values, labels


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def add_lora(model: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for full_name, module in list(model.named_modules()):
        is_target = (
            full_name.endswith((
                "attention.attention.query",
                "attention.attention.key",
                "attention.attention.value",
                "mlp.fc1",
                "mlp.fc2",
            ))
        )
        if isinstance(module, nn.Linear) and is_target:
            parent_name, _, child_name = full_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((parent, child_name, module))
    if not replacements:
        raise RuntimeError("No DINOv2 Linear modules matched the configured LoRA targets.")
    for parent, child_name, base in replacements:
        parent._modules[child_name] = LoRALinear(base, rank, alpha, dropout)
    return len(replacements)


class TongueClassifier(nn.Module):
    def __init__(self, base: Dinov2Model, lora_rank: int, lora_alpha: float, lora_dropout: float) -> None:
        super().__init__()
        self.backbone = base
        self.num_lora_layers = add_lora(self.backbone, lora_rank, lora_alpha, lora_dropout)
        hidden = int(self.backbone.config.hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(hidden, 512), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(512, len(LABELS))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(pixel_values=pixel_values, interpolate_pos_encoding=True).last_hidden_state
        pooled = hidden[:, 1:, :].mean(dim=1)  # GAP over 24x24 patch tokens; exclude CLS.
        return self.classifier(self.projection(pooled))

    def trainable_state(self) -> dict[str, torch.Tensor]:
        trainable = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in trainable}


def load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in ["id", "image_path", *LABELS] if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame[LABELS] = frame[LABELS].astype(np.float32)
    if not np.isin(frame[LABELS].to_numpy(), [0.0, 1.0]).all():
        raise ValueError(f"Non-binary labels found in {path}")
    return frame


def audit_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    for key in ("id", "image_path"):
        sets = [set(frame[key].astype(str)) for frame in (train, val, test)]
        for left, right, name in ((0, 1, "train/val"), (0, 2, "train/test"), (1, 2, "val/test")):
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(f"Split leakage in {key} ({name}): {len(overlap)} overlapping values")


def build_model(model_dir: Path, device: torch.device, rank: int, alpha: float, lora_dropout: float):
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Local DINOv2 checkpoint not found at {model_dir}. Run download_model.py first.")
    base = Dinov2Model.from_pretrained(str(model_dir), local_files_only=True)
    model = TongueClassifier(base, rank, alpha, lora_dropout).to(device)
    return model


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for pixels, labels in loader:
        pixels = pixels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(pixels)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def choose_thresholds(logits: np.ndarray, y: np.ndarray) -> list[float]:
    probabilities = sigmoid(logits)
    thresholds = []
    for column in range(y.shape[1]):
        best_score, best_threshold = -1.0, 0.5
        for threshold in np.arange(0.05, 0.951, 0.01):
            score = f1_score(y[:, column], probabilities[:, column] >= threshold, zero_division=0)
            if score > best_score:
                best_score, best_threshold = float(score), float(threshold)
        thresholds.append(best_threshold)
    return thresholds


def metrics(logits: np.ndarray, y: np.ndarray, thresholds: Iterable[float]) -> dict:
    probabilities = sigmoid(logits)
    threshold_array = np.asarray(list(thresholds), dtype=np.float32)
    predictions = probabilities >= threshold_array[None, :]
    per_label = {}
    for idx, label in enumerate(LABELS):
        per_label[label] = {
            "support_pos": int(y[:, idx].sum()),
            "support_neg": int(len(y) - y[:, idx].sum()),
            "threshold": float(threshold_array[idx]),
            "f1": float(f1_score(y[:, idx], predictions[:, idx], zero_division=0)),
            "precision": float(precision_score(y[:, idx], predictions[:, idx], zero_division=0)),
            "recall": float(recall_score(y[:, idx], predictions[:, idx], zero_division=0)),
            "auprc": float(average_precision_score(y[:, idx], probabilities[:, idx])),
        }
    return {
        "macro_f1": float(np.mean([v["f1"] for v in per_label.values()])),
        "macro_auprc": float(np.mean([v["auprc"] for v in per_label.values()])),
        "per_label": per_label,
    }


@torch.no_grad()
def update_ema(model: nn.Module, ema_state: dict[str, torch.Tensor], decay: float) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            ema_state[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)


class EMAWeights:
    def __init__(self, model: nn.Module, ema_state: dict[str, torch.Tensor]) -> None:
        self.model = model
        self.ema_state = ema_state
        self.backup: dict[str, torch.Tensor] = {}

    def __enter__(self):
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    self.backup[name] = parameter.detach().clone()
                    parameter.copy_(self.ema_state[name])
        return self.model

    def __exit__(self, exc_type, exc, traceback):
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name in self.backup:
                    parameter.copy_(self.backup[name])
        self.backup.clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--release-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--model-dir", type=Path, default=HERE / "models" / "dinov2-small")
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment. Install CUDA-enabled PyTorch on the RTX 4060 machine; CPU training is intentionally disabled.")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {torch.cuda.get_device_name(device)} | VRAM: {props.total_memory / 1024**3:.2f} GiB")
    if props.total_memory < 7 * 1024**3:
        print("Warning: less than 7 GiB visible; use --batch-size 2 --grad-accum 8 if memory pressure occurs.")

    root = args.release_root.resolve()
    train = load_csv(root / "list" / f"train_fold{args.fold}.csv")
    val = load_csv(root / "list" / f"val_fold{args.fold}.csv")
    test = load_csv(root / "list" / "test.csv")
    audit_splits(train, val, test)
    for frame, split in ((train, "train"), (val, "val"), (test, "test")):
        missing = [p for p in frame.image_path if not (root / "seg" / str(p)).is_file()]
        if missing:
            raise FileNotFoundError(f"{split} has {len(missing)} missing images; example: {missing[0]}")

    cfg_path = HERE / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    size = int(cfg["model"]["input_size"])
    train_ds = TongueDataset(train, root, size, train=True)
    val_ds = TongueDataset(val, root, size, train=False)
    test_ds = TongueDataset(test, root, size, train=False)
    common = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": True,
              "persistent_workers": args.workers > 0}
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    model = build_model(args.model_dir, device, int(cfg["model"]["lora_rank"]),
                        float(cfg["model"]["lora_alpha"]), float(cfg["model"]["lora_dropout"]))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    print(f"LoRA-wrapped linear modules: {model.num_lora_layers}; trainable params: {sum(p.numel() for p in trainable_params):,}")
    ema_state = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    positives = train[LABELS].sum(axis=0).to_numpy(dtype=np.float32)
    negatives = len(train) - positives
    if np.any(positives <= 0):
        raise ValueError("A training fold contains a label with zero positives; BCE pos_weight cannot be estimated.")
    pos_weight = np.minimum(negatives / positives, args.max_pos_weight)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lora_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    run_dir = args.output_dir / f"fold{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ap, best_epoch, epochs_without_improvement = -float("inf"), 0, 0
    best_path = run_dir / "best_trainables.pt"
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen = 0.0, 0
        started = time.time()
        progress = tqdm(train_loader, desc=f"fold {args.fold} epoch {epoch}/{args.epochs}", leave=False)
        for step, (pixels, labels) in enumerate(progress, start=1):
            pixels = pixels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(pixels)
                raw_loss = loss_fn(logits.float(), labels)
                group_start = ((step - 1) // args.grad_accum) * args.grad_accum
                group_size = min(args.grad_accum, len(train_loader) - group_start)
                loss = raw_loss / group_size
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_ema(model, ema_state, args.ema_decay)
            total_loss += float(raw_loss.detach()) * len(labels)
            seen += len(labels)
            progress.set_postfix(loss=f"{float(loss.detach()) * args.grad_accum:.4f}")
        scheduler.step(epoch - 1)
        with EMAWeights(model, ema_state):
            val_logits, val_y = collect_logits(model, val_loader, device, amp)
        thresholds = choose_thresholds(val_logits, val_y)
        val_metrics = metrics(val_logits, val_y, thresholds)
        row = {"epoch": epoch, "train_loss": total_loss / max(seen, 1),
               "val_macro_auprc": val_metrics["macro_auprc"], "val_macro_f1": val_metrics["macro_f1"],
               "seconds": time.time() - started}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics["macro_auprc"] > best_ap:
            best_ap, best_epoch, epochs_without_improvement = val_metrics["macro_auprc"], epoch, 0
            torch.save({
                "trainable_state": {name: value.detach().cpu() for name, value in ema_state.items()},
                "label_order": LABELS,
                "fold": args.fold,
                "best_epoch": epoch,
                "best_val_metrics": val_metrics,
                "val_thresholds": thresholds,
                "config": cfg,
                "cli": {key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()},
                "pos_weight": pos_weight.tolist(),
            }, best_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, val macro-AUPRC={best_ap:.4f}")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
    unexpected_trainable = [key for key in unexpected]
    if unexpected_trainable:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_trainable[:5]}")
    thresholds = checkpoint["val_thresholds"]
    test_logits, test_y = collect_logits(model, test_loader, device, amp)
    result = {
        "fold": args.fold,
        "best_epoch": best_epoch,
        "best_val_macro_auprc": best_ap,
        "validation": checkpoint.get("best_val_metrics"),
        "threshold_source": "validation only",
        "test": metrics(test_logits, test_y, thresholds),
        "config": checkpoint["config"],
        "cli": checkpoint["cli"],
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps({"fold": args.fold, "best_epoch": best_epoch, "test": result["test"]}, ensure_ascii=False, indent=2))
    print(f"Saved run to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
