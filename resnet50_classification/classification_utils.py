from __future__ import annotations

import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def load_classes(path: str | Path) -> list[str]:
    rows: list[tuple[int, str]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid classes.txt line {line_number}: {line!r}")
            rows.append((int(parts[0]), parts[1]))

    rows.sort(key=lambda item: item[0])
    expected_ids = list(range(len(rows)))
    actual_ids = [item[0] for item in rows]
    if actual_ids != expected_ids:
        raise ValueError(f"Class IDs must be consecutive from 0; got {actual_ids}.")
    return [item[1] for item in rows]


class ClassificationCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        class_names: list[str],
        transform: Any = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.class_names = class_names
        self.transform = transform
        self.samples: list[tuple[Path, int, str]] = []

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"image_path", "label", "class_id"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"{self.csv_path} must contain columns: image_path,label,class_id"
                )
            for line_number, row in enumerate(reader, start=2):
                class_id = int(row["class_id"])
                if not 0 <= class_id < len(class_names):
                    raise ValueError(
                        f"{self.csv_path}:{line_number}: invalid class_id={class_id}"
                    )
                label = row["label"].strip()
                expected_label = class_names[class_id]
                if label != expected_label:
                    raise ValueError(
                        f"{self.csv_path}:{line_number}: label={label!r} does not match "
                        f"class_id={class_id} ({expected_label!r})"
                    )
                relative_path = row["image_path"].replace("\\", "/")
                image_path = self.data_root / Path(relative_path)
                self.samples.append((image_path, class_id, relative_path))

        if not self.samples:
            raise ValueError(f"No samples found in {self.csv_path}.")

        missing = [str(path) for path, _, _ in self.samples if not path.is_file()]
        if missing:
            examples = "\n".join(missing[:10])
            nested_root = self.data_root / "dataset"
            nested_matches = sum(
                (nested_root / relative_path).is_file()
                for _, _, relative_path in self.samples
            )
            nested_hint = ""
            if nested_matches:
                nested_hint = (
                    f"\n{nested_matches} referenced images were found under "
                    f"{nested_root}. Use --data-root {nested_root} instead."
                )
            raise FileNotFoundError(
                f"{len(missing)} images referenced by {self.csv_path} are missing. "
                f"First examples:\n{examples}{nested_hint}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path, class_id, relative_path = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, class_id, relative_path

    @property
    def targets(self) -> list[int]:
        return [class_id for _, class_id, _ in self.samples]


class PadToSquare:
    """Pad an image to a centered square without cropping or distortion."""

    def __init__(self, fill: tuple[int, int, int]) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        if width == height:
            return image

        padded = Image.new("RGB", (side, side), self.fill)
        left = (side - width) // 2
        top = (side - height) // 2
        padded.paste(image, (left, top))
        return padded


def build_transforms(image_size: int) -> tuple[Any, Any]:
    weights = ResNet50_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std
    fill = tuple(int(round(value * 255)) for value in mean)

    # Preserve the complete field of view for both training and evaluation.
    # Spatial augmentations below do not crop the image.
    train_transform = transforms.Compose(
        [
            PadToSquare(fill),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            PadToSquare(fill),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


class ECAAttention(nn.Module):
    """Efficient Channel Attention with adaptive 1D kernel size."""

    def __init__(self, channels: int, gamma: float = 2.0, bias: float = 1.0) -> None:
        super().__init__()
        kernel_size = int(abs((math.log2(channels) + bias) / gamma))
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.activation = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.pool(inputs).squeeze(-1).transpose(-1, -2)
        weights = self.conv(weights).transpose(-1, -2).unsqueeze(-1)
        return inputs * self.activation(weights)


class MedicalModalityAttention(nn.Module):
    """MMA following Fig. 2 and the equations in Section 2.1."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)

        self.anatomy_attention = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.texture_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.multiscale_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        hidden_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.multiscale_adjust = nn.Sequential(
            nn.Conv2d(
                hidden_channels * 3,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        anatomy_attention = self.anatomy_attention(inputs)
        texture_attention = self.texture_attention(inputs)
        intrinsic_attention = anatomy_attention * texture_attention
        multiscale_features = self.multiscale_adjust(
            torch.cat(
                [branch(inputs) for branch in self.multiscale_branches],
                dim=1,
            )
        )
        return inputs * intrinsic_attention + multiscale_features


class OfficialECALayer(nn.Module):
    """ECA layer with names matching the official ECANet checkpoint."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(inputs)
        weights = self.conv(
            weights.squeeze(-1).transpose(-1, -2)
        ).transpose(-1, -2).unsqueeze(-1)
        return inputs * self.sigmoid(weights).expand_as(inputs)


class OfficialECABottleneck(nn.Module):
    """Official post-BN3 ECA Bottleneck used by ECA-ResNet50."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.eca = OfficialECALayer(
            planes * self.expansion,
            kernel_size,
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs

        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.relu(self.bn2(self.conv2(outputs)))
        outputs = self.bn3(self.conv3(outputs))
        outputs = self.eca(outputs)

        if self.downsample is not None:
            identity = self.downsample(inputs)

        outputs += identity
        return self.relu(outputs)


class OfficialECAResNet50(nn.Module):
    """Official ECA-ResNet50 architecture with stage kernels 3/5/5/7."""

    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, kernel_size=3)
        self.layer2 = self._make_layer(128, 4, kernel_size=5, stride=2)
        self.layer3 = self._make_layer(256, 6, kernel_size=5, stride=2)
        self.layer4 = self._make_layer(512, 3, kernel_size=7, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * OfficialECABottleneck.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = (
                    module.kernel_size[0]
                    * module.kernel_size[1]
                    * module.out_channels
                )
                module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        kernel_size: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        out_channels = planes * OfficialECABottleneck.expansion
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers = [
            OfficialECABottleneck(
                self.inplanes,
                planes,
                stride,
                downsample,
                kernel_size,
            )
        ]
        self.inplanes = out_channels
        for _ in range(1, blocks):
            layers.append(
                OfficialECABottleneck(
                    self.inplanes,
                    planes,
                    kernel_size=kernel_size,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.maxpool(self.relu(self.bn1(self.conv1(inputs))))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        outputs = torch.flatten(outputs, 1)
        return self.fc(outputs)


def load_official_eca_imagenet_weights(
    model: OfficialECAResNet50,
    checkpoint_path: str | Path,
) -> None:
    """Load the official k3557 ImageNet checkpoint, excluding its 1000-way FC."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Official ECA ImageNet checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Unsupported official ECA checkpoint format: {checkpoint_path}"
        )

    normalized_state_dict = {}
    for name, value in state_dict.items():
        if name.startswith("module."):
            name = name[len("module.") :]
        if name in {"fc.weight", "fc.bias"}:
            continue
        normalized_state_dict[name] = value

    incompatible = model.load_state_dict(normalized_state_dict, strict=False)
    expected_missing = {"fc.weight", "fc.bias"}
    actual_missing = set(incompatible.missing_keys)
    if actual_missing != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Official ECA checkpoint is incompatible. "
            f"Missing={sorted(actual_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )


class CBAMChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.activation = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = self.shared_mlp(torch.mean(inputs, dim=(2, 3), keepdim=True))
        maximum = self.shared_mlp(torch.amax(inputs, dim=(2, 3), keepdim=True))
        return inputs * self.activation(average + maximum)


class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.activation = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = torch.mean(inputs, dim=1, keepdim=True)
        maximum = torch.amax(inputs, dim=1, keepdim=True)
        weights = self.activation(self.conv(torch.cat((average, maximum), dim=1)))
        return inputs * weights


class CBAMAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = CBAMChannelAttention(channels, reduction)
        self.spatial = CBAMSpatialAttention(kernel_size=7)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(inputs))


def _build_attention(name: str, channels: int) -> nn.Module:
    if name == "eca":
        return ECAAttention(channels)
    if name == "cbam":
        return CBAMAttention(channels)
    raise ValueError(f"Unsupported attention type: {name!r}")


def model_architecture_version(attention: str) -> str:
    versions = {
        "none": "torchvision_resnet50_v1",
        "eca": "stage_eca_resnet50_v1",
        "eca_mma": "stage_eca_layer4_mma_resnet50_v1",
        "eca_official": "official_eca_resnet50_bottleneck_3557_imagenet_v1",
        "cbam": "stage_cbam_resnet50_v1",
    }
    try:
        return versions[attention.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported attention type: {attention!r}") from error


def build_model(
    num_classes: int,
    pretrained: bool,
    attention: str = "none",
    dropout: float = 0.3,
    official_eca_weights: str | Path | None = None,
) -> nn.Module:
    attention = attention.lower()
    if attention not in {"none", "eca", "eca_mma", "eca_official", "cbam"}:
        raise ValueError(
            f"attention must be one of none, eca, eca_mma, eca_official, cbam; "
            f"got {attention!r}"
        )
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

    if attention == "eca_official":
        model = OfficialECAResNet50(num_classes=1000)
        if pretrained:
            if official_eca_weights is None:
                raise ValueError(
                    "--official-eca-weights is required for pretrained "
                    "eca_official training."
                )
            load_official_eca_imagenet_weights(
                model,
                official_eca_weights,
            )
    else:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)

    if attention in {"eca", "eca_mma", "cbam"}:
        # Apply one lightweight attention module after each complete ResNet stage.
        # The torchvision stage is constructed and pretrained first, so all
        # backbone weights remain directly reusable.
        stage_attention = "eca" if attention == "eca_mma" else attention
        stage_channels = {
            "layer1": 256,
            "layer2": 512,
            "layer3": 1024,
            "layer4": 2048,
        }
        for stage_name, channels in stage_channels.items():
            stage = getattr(model, stage_name)
            modules = [stage, _build_attention(stage_attention, channels)]
            if attention == "eca_mma" and stage_name == "layer4":
                modules.append(MedicalModalityAttention(channels))
            setattr(
                model,
                stage_name,
                nn.Sequential(*modules),
            )
    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


def split_backbone_and_task_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return pretrained backbone parameters and newly initialized task parameters."""
    task_parameter_ids = {id(parameter) for parameter in model.fc.parameters()}
    for module in model.modules():
        if isinstance(
            module,
            (ECAAttention, MedicalModalityAttention, CBAMAttention),
        ):
            task_parameter_ids.update(id(parameter) for parameter in module.parameters())

    backbone_parameters: list[nn.Parameter] = []
    task_parameters: list[nn.Parameter] = []
    for parameter in model.parameters():
        if id(parameter) in task_parameter_ids:
            task_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    return backbone_parameters, task_parameters


def set_parameters_trainable(
    parameters: Iterable[nn.Parameter],
    trainable: bool,
) -> None:
    for parameter in parameters:
        parameter.requires_grad = trainable


def confusion_matrix(
    targets: torch.Tensor, predictions: torch.Tensor, num_classes: int
) -> torch.Tensor:
    indices = targets.to(torch.int64) * num_classes + predictions.to(torch.int64)
    return torch.bincount(indices, minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def metrics_from_confusion_matrix(
    matrix: torch.Tensor, class_names: list[str]
) -> dict[str, Any]:
    matrix = matrix.to(torch.float64)
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    total = matrix.sum().clamp_min(1)

    precision = true_positive / predicted.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    class_weights = support / total

    per_class = {}
    for index, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    return {
        "accuracy": float(true_positive.sum() / total),
        "balanced_accuracy": float(recall.mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * class_weights).sum()),
        "per_class": per_class,
        "confusion_matrix": matrix.to(torch.int64).tolist(),
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    amp_enabled: bool,
    return_predictions: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    matrix = torch.zeros(
        (len(class_names), len(class_names)), dtype=torch.int64, device=device
    )
    prediction_rows: list[dict[str, Any]] = []

    for images, targets, paths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)
        batch_size = targets.size(0)
        total_loss += float(loss) * batch_size
        total_samples += batch_size
        matrix += confusion_matrix(targets, predictions, len(class_names))

        if return_predictions:
            targets_cpu = targets.cpu().tolist()
            predictions_cpu = predictions.cpu().tolist()
            confidence_cpu = confidence.cpu().tolist()
            for path, target, prediction, score in zip(
                paths, targets_cpu, predictions_cpu, confidence_cpu
            ):
                prediction_rows.append(
                    {
                        "image_path": path,
                        "target_id": target,
                        "target_label": class_names[target],
                        "prediction_id": prediction,
                        "prediction_label": class_names[prediction],
                        "confidence": score,
                        "correct": int(target == prediction),
                    }
                )

    metrics = metrics_from_confusion_matrix(matrix, class_names)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["num_samples"] = total_samples
    return metrics, prediction_rows


def balanced_class_weights(targets: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Training split contains no samples for class IDs: {missing}")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def atomic_torch_save(payload: dict[str, Any], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def save_json(payload: Any, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
