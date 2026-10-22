from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import train as pipeline


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved DINOv2 LoRA checkpoint on the held-out test split.")
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--release-root", type=Path, default=HERE.parents[1])
    parser.add_argument("--model-dir", type=Path, default=HERE / "models" / "dinov2-small")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this evaluation is configured for the local RTX GPU.")

    root = args.release_root.resolve()
    checkpoint_path = args.checkpoint or args.output_dir / f"fold{args.fold}" / "best_trainables.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("fold") != args.fold:
        raise ValueError(f"Checkpoint fold {checkpoint.get('fold')} does not match requested fold {args.fold}.")
    if checkpoint.get("label_order") != pipeline.LABELS:
        raise ValueError("Checkpoint label order does not match the current eight-label task.")

    val = pipeline.load_csv(root / "list" / f"val_fold{args.fold}.csv")
    test = pipeline.load_csv(root / "list" / "test.csv")
    for frame, split in ((val, "validation"), (test, "test")):
        missing = [p for p in frame.image_path if not (root / "seg" / str(p)).is_file()]
        if missing:
            raise FileNotFoundError(f"{split} split has {len(missing)} missing images; example: {missing[0]}")
    size = int(checkpoint["config"]["model"]["input_size"])
    val_ds = pipeline.TongueDataset(val, root, size, train=False)
    test_ds = pipeline.TongueDataset(test, root, size, train=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    cfg = checkpoint["config"]
    device = torch.device("cuda")
    model = pipeline.build_model(
        args.model_dir,
        device,
        int(cfg["model"]["lora_rank"]),
        float(cfg["model"]["lora_alpha"]),
        float(cfg["model"]["lora_dropout"]),
    )
    _, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:5]}")

    thresholds = checkpoint["val_thresholds"]
    val_logits, val_labels = pipeline.collect_logits(model, val_loader, device, amp=True)
    validation_metrics = pipeline.metrics(val_logits, val_labels, thresholds)
    test_logits, test_labels = pipeline.collect_logits(model, test_loader, device, amp=True)
    result = {
        "fold": args.fold,
        "best_epoch": checkpoint["best_epoch"],
        "best_val_macro_auprc": validation_metrics["macro_auprc"],
        "validation": validation_metrics,
        "threshold_source": "validation only",
        "test": pipeline.metrics(test_logits, test_labels, thresholds),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    out_path = args.output_dir / f"fold{args.fold}" / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved metrics to {out_path.resolve()}")


if __name__ == "__main__":
    main()
