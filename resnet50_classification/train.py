from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from classification_utils import (
    ClassificationCsvDataset,
    atomic_torch_save,
    balanced_class_weights,
    build_model,
    build_transforms,
    evaluate_model,
    load_classes,
    model_architecture_version,
    resolve_device,
    save_json,
    set_seed,
    set_parameters_trainable,
    split_backbone_and_task_parameters,
)


LOG_COLUMNS = [
    "epoch",
    "backbone_learning_rate",
    "head_learning_rate",
    "backbone_frozen",
    "train_loss",
    "train_accuracy",
    "eval_loss",
    "eval_accuracy",
    "eval_balanced_accuracy",
    "eval_macro_precision",
    "eval_macro_recall",
    "eval_macro_f1",
    "eval_weighted_f1",
    "best_metric_so_far",
    "early_stopping_counter",
    "epoch_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ResNet-50 classifier from CSV annotations."
    )
    parser.add_argument(
        "--train-data-root",
        "--data-root",
        dest="data_root",
        required=True,
        help="Root directory used to resolve training image_path values.",
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument(
        "--eval-csv",
        required=True,
        help=(
            "CSV evaluated every --eval-every epochs. A header-only CSV "
            "disables evaluation, best-checkpoint selection, and early stopping."
        ),
    )
    parser.add_argument("--classes", required=True, help="Path to classes.txt.")
    parser.add_argument("--output-dir", default="runs/resnet50")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the classification head and optional attention.",
    )
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Stop after this many consecutive evaluations without metric improvement.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=50,
        help="Do not trigger early stopping before this epoch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument(
        "--attention",
        choices=("none", "eca", "eca_mma", "eca_official", "cbam"),
        default="none",
        help=(
            "Stage-level attention. eca_mma adds ECA after all stages and "
            "one Medical Modality Attention module after layer4 ECA. "
            "eca_official uses the 16-Bottleneck k3557 ECA-ResNet50."
        ),
    )
    parser.add_argument(
        "--official-eca-weights",
        default=None,
        help=(
            "Path to the official ImageNet eca_resnet50_k3557 .pth.tar. "
            "Required for pretrained eca_official training."
        ),
    )
    parser.add_argument(
        "--class-weights",
        choices=("balanced", "none"),
        default="balanced",
        help="Balanced loss weighting is useful for this imbalanced dataset.",
    )
    parser.add_argument(
        "--metric-for-best",
        choices=(
            "macro_f1",
            "accuracy",
            "balanced_accuracy",
            "weighted_f1",
        ),
        default="macro_f1",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def resolve_output_dir(requested: str | Path, resume: str | None) -> Path:
    """Keep fresh runs isolated while allowing an interrupted run to resume."""
    requested_path = Path(requested)
    if resume or not requested_path.exists():
        return requested_path

    if not any(requested_path.iterdir()):
        return requested_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = requested_path.with_name(f"{requested_path.name}_{timestamp}")
    collision_index = 2
    while candidate.exists():
        candidate = requested_path.with_name(
            f"{requested_path.name}_{timestamp}_{collision_index}"
        )
        collision_index += 1
    return candidate


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    backbone_frozen: bool,
) -> dict[str, float]:
    model.train()
    if backbone_frozen:
        # Keep pretrained BatchNorm statistics fixed while the backbone is frozen.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def checkpoint_payload(
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.cuda.amp.GradScaler,
    class_names: list[str],
    args: argparse.Namespace,
    eval_metrics: dict[str, Any] | None,
    best_metric: float,
    best_loss: float,
    best_metric_epoch: int,
    early_stopping_counter: int,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_name": f"resnet50_{args.attention}",
        "attention": args.attention,
        "architecture_version": model_architecture_version(args.attention),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "class_names": class_names,
        "args": vars(args),
        "eval_metrics": eval_metrics,
        "best_metric": best_metric,
        "best_loss": best_loss,
        "best_metric_epoch": best_metric_epoch,
        "early_stopping_counter": early_stopping_counter,
    }


def append_log_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in LOG_COLUMNS})


def main() -> None:
    args = parse_args()
    if (
        args.attention == "eca_official"
        and not args.no_pretrained
        and not args.resume
        and not args.official_eca_weights
    ):
        raise ValueError(
            "--official-eca-weights is required when --attention "
            "eca_official uses pretrained weights."
        )
    if args.attention != "eca_official" and args.official_eca_weights:
        raise ValueError(
            "--official-eca-weights can only be used with "
            "--attention eca_official."
        )
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.eval_every < 1:
        raise ValueError("--eval-every must be at least 1.")
    if args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be at least 1.")
    if not 1 <= args.min_epochs <= args.epochs:
        raise ValueError("--min-epochs must be between 1 and --epochs.")
    if not 0 <= args.freeze_backbone_epochs <= args.epochs:
        raise ValueError("--freeze-backbone-epochs must be between 0 and --epochs.")
    if args.learning_rate <= 0 or args.backbone_learning_rate <= 0:
        raise ValueError("Learning rates must be positive.")

    requested_output_dir = Path(args.output_dir)
    output_dir = resolve_output_dir(requested_output_dir, args.resume)
    if output_dir != requested_output_dir:
        print(
            f"Output directory already contains a run; using {output_dir} instead."
        )
    args.output_dir = str(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    metrics_dir = output_dir / "eval_metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)

    set_seed(args.seed, args.deterministic)
    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp
    class_names = load_classes(args.classes)
    train_transform, eval_transform = build_transforms(args.image_size)

    train_dataset = ClassificationCsvDataset(
        args.train_csv, args.data_root, class_names, train_transform
    )
    eval_dataset = ClassificationCsvDataset(
        args.eval_csv,
        args.data_root,
        class_names,
        eval_transform,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    eval_loader = (
        DataLoader(eval_dataset, shuffle=False, **loader_kwargs)
        if len(eval_dataset) > 0
        else None
    )

    model = build_model(
        num_classes=len(class_names),
        pretrained=not args.no_pretrained and not args.resume,
        attention=args.attention,
        dropout=args.dropout,
        official_eca_weights=args.official_eca_weights,
    ).to(device)
    if args.attention == "eca_official" and not args.resume:
        if args.no_pretrained:
            logger.info("official ECA-ResNet50 initialized without pretrained weights")
        else:
            logger.info(
                "loaded official ImageNet ECA-ResNet50 k3557 weights from %s; "
                "classification head initialized randomly",
                args.official_eca_weights,
            )
    backbone_parameters, task_parameters = split_backbone_and_task_parameters(model)

    loss_weights = None
    if args.class_weights == "balanced":
        loss_weights = balanced_class_weights(
            train_dataset.targets, len(class_names)
        ).to(device)
    train_criterion = nn.CrossEntropyLoss(weight=loss_weights)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": args.backbone_learning_rate,
                "name": "backbone",
            },
            {
                "params": task_parameters,
                "lr": args.learning_rate,
                "name": "head_attention",
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.freeze_backbone_epochs, 1),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_metric = -math.inf
    best_loss = math.inf
    best_metric_epoch = 0
    early_stopping_counter = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        if checkpoint["class_names"] != class_names:
            raise ValueError("Checkpoint class_names do not match classes.txt.")
        checkpoint_attention = checkpoint.get(
            "attention", checkpoint.get("args", {}).get("attention", "none")
        )
        if checkpoint_attention != args.attention:
            raise ValueError(
                f"Checkpoint attention={checkpoint_attention!r}, but "
                f"--attention={args.attention!r}."
            )
        expected_architecture = model_architecture_version(args.attention)
        checkpoint_architecture = checkpoint.get("architecture_version")
        if (
            checkpoint_architecture is not None
            and checkpoint_architecture != expected_architecture
        ):
            raise ValueError(
                f"Checkpoint architecture={checkpoint_architecture!r}, but "
                f"this package expects {expected_architecture!r}."
            )
        checkpoint_dropout = float(checkpoint.get("args", {}).get("dropout", 0.0))
        if not math.isclose(checkpoint_dropout, args.dropout):
            raise ValueError(
                f"Checkpoint dropout={checkpoint_dropout}, but "
                f"--dropout={args.dropout}."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint.get("best_metric", -math.inf))
        best_loss = float(checkpoint.get("best_loss", math.inf))
        best_metric_epoch = int(
            checkpoint.get("best_metric_epoch", checkpoint.get("epoch", 0))
        )
        early_stopping_counter = int(
            checkpoint.get("early_stopping_counter", 0)
        )
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    config = {
        **vars(args),
        "output_dir_requested": str(requested_output_dir),
        "output_dir_resolved": str(output_dir),
        "architecture_version": model_architecture_version(args.attention),
        "class_names": class_names,
        "device_resolved": str(device),
        "amp_enabled": amp_enabled,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(config, output_dir / "config.json")
    save_json(
        {str(index): name for index, name in enumerate(class_names)},
        output_dir / "class_mapping.json",
    )

    logger.info(
        "model=resnet50_%s | device=%s | train=%d | eval=%d | classes=%d | amp=%s",
        args.attention,
        device,
        len(train_dataset),
        len(eval_dataset),
        len(class_names),
        amp_enabled,
    )
    logger.info(
        "freeze backbone=%d epochs | backbone LR=%.2e | head/attention LR=%.2e | "
        "dropout=%.2f | weight decay=%.2e",
        args.freeze_backbone_epochs,
        args.backbone_learning_rate,
        args.learning_rate,
        args.dropout,
        args.weight_decay,
    )
    if eval_loader is None:
        logger.info(
            "evaluation disabled because %s contains no samples; "
            "best-checkpoint selection and early stopping are disabled",
            args.eval_csv,
        )
    else:
        logger.info(
            "early stopping=%d evaluations | minimum epochs=%d | monitored metric=%s",
            args.early_stopping_patience,
            args.min_epochs,
            args.metric_for_best,
        )
    if loss_weights is not None:
        logger.info(
            "balanced class weights: %s",
            ", ".join(f"{value:.3f}" for value in loss_weights.cpu().tolist()),
        )

    last_eval_metrics: dict[str, Any] | None = None
    last_completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.perf_counter()
        backbone_frozen = epoch <= args.freeze_backbone_epochs
        set_parameters_trainable(backbone_parameters, not backbone_frozen)
        set_parameters_trainable(task_parameters, True)
        backbone_learning_rate = optimizer.param_groups[0]["lr"]
        head_learning_rate = optimizer.param_groups[1]["lr"]
        if epoch == start_epoch or epoch == args.freeze_backbone_epochs + 1:
            logger.info(
                "epoch %03d | backbone %s",
                epoch,
                "frozen" if backbone_frozen else "unfrozen",
            )
        train_metrics = train_one_epoch(
            model,
            train_loader,
            train_criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
            backbone_frozen,
        )

        should_evaluate = eval_loader is not None and (
            epoch % args.eval_every == 0 or epoch == args.epochs
        )
        eval_metrics = None
        stop_training = False
        if should_evaluate:
            eval_metrics, _ = evaluate_model(
                model,
                eval_loader,
                eval_criterion,
                device,
                class_names,
                amp_enabled,
            )
            eval_metrics["epoch"] = epoch
            last_eval_metrics = eval_metrics
            save_json(eval_metrics, metrics_dir / f"epoch_{epoch:04d}.json")

            metric_value = float(eval_metrics[args.metric_for_best])
            improved_metric = metric_value > best_metric
            improved_loss = float(eval_metrics["loss"]) < best_loss
            if improved_metric:
                best_metric = metric_value
                best_metric_epoch = epoch
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            if improved_loss:
                best_loss = float(eval_metrics["loss"])

            payload = checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                class_names=class_names,
                args=args,
                eval_metrics=eval_metrics,
                best_metric=best_metric,
                best_loss=best_loss,
                best_metric_epoch=best_metric_epoch,
                early_stopping_counter=early_stopping_counter,
            )
            if improved_metric:
                atomic_torch_save(payload, checkpoint_dir / "best_metric.pt")
            if improved_loss:
                atomic_torch_save(payload, checkpoint_dir / "best_loss.pt")
            stop_training = (
                epoch >= args.min_epochs
                and early_stopping_counter >= args.early_stopping_patience
            )

        if epoch > args.freeze_backbone_epochs:
            scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "backbone_learning_rate": backbone_learning_rate,
            "head_learning_rate": head_learning_rate,
            "backbone_frozen": int(backbone_frozen),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "best_metric_so_far": (
                best_metric if math.isfinite(best_metric) else ""
            ),
            "early_stopping_counter": early_stopping_counter,
            "epoch_seconds": epoch_seconds,
        }

        if eval_metrics is None:
            logger.info(
                "epoch %03d/%03d | train loss %.4f acc %.4f | %.1fs",
                epoch,
                args.epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                epoch_seconds,
            )
        else:
            row.update(
                {
                    "eval_loss": eval_metrics["loss"],
                    "eval_accuracy": eval_metrics["accuracy"],
                    "eval_balanced_accuracy": eval_metrics["balanced_accuracy"],
                    "eval_macro_precision": eval_metrics["macro_precision"],
                    "eval_macro_recall": eval_metrics["macro_recall"],
                    "eval_macro_f1": eval_metrics["macro_f1"],
                    "eval_weighted_f1": eval_metrics["weighted_f1"],
                }
            )
            logger.info(
                "epoch %03d/%03d | train loss %.4f acc %.4f | "
                "eval loss %.4f acc %.4f macro-F1 %.4f | early-stop %d/%d | %.1fs",
                epoch,
                args.epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                eval_metrics["loss"],
                eval_metrics["accuracy"],
                eval_metrics["macro_f1"],
                early_stopping_counter,
                args.early_stopping_patience,
                epoch_seconds,
            )
        append_log_row(output_dir / "training_metrics.csv", row)
        last_completed_epoch = epoch
        if eval_loader is None:
            latest_payload = checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                class_names=class_names,
                args=args,
                eval_metrics=None,
                best_metric=best_metric,
                best_loss=best_loss,
                best_metric_epoch=best_metric_epoch,
                early_stopping_counter=early_stopping_counter,
            )
            atomic_torch_save(latest_payload, checkpoint_dir / "latest.pt")
        if stop_training:
            logger.info(
                "early stopping at epoch %d | no %s improvement for %d "
                "evaluations | best %.4f at epoch %d",
                epoch,
                args.metric_for_best,
                early_stopping_counter,
                best_metric,
                best_metric_epoch,
            )
            break

    final_payload = checkpoint_payload(
        epoch=last_completed_epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        class_names=class_names,
        args=args,
        eval_metrics=last_eval_metrics,
        best_metric=best_metric,
        best_loss=best_loss,
        best_metric_epoch=best_metric_epoch,
        early_stopping_counter=early_stopping_counter,
    )
    atomic_torch_save(final_payload, checkpoint_dir / "final.pt")
    if last_eval_metrics is None:
        logger.info(
            "finished at epoch %d without evaluation | final checkpoint=%s",
            last_completed_epoch,
            checkpoint_dir / "final.pt",
        )
    else:
        logger.info(
            "finished at epoch %d | best %s %.4f (epoch %d) | "
            "best eval loss %.4f | checkpoints=%s",
            last_completed_epoch,
            args.metric_for_best,
            best_metric,
            best_metric_epoch,
            best_loss,
            checkpoint_dir,
        )


if __name__ == "__main__":
    main()
