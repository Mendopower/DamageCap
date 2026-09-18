#!/usr/bin/env python3
"""ResNet category/GradCAM-guided caption inference with fine-tuned BLIP-3."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from functools import partial
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from peft import PeftModel
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
from transformers.modeling_utils import load_sharded_checkpoint

from open_flamingo import create_model_and_transforms
from open_flamingo.train.any_res_data_utils import process_images


BASE_MODEL_ID = "Salesforce/xgen-mm-phi3-mini-base-r-v1.5"
SYSTEM_MESSAGE = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

CLASS_MAP = {
    "void": "voids",
    "concrete_crack": "cracks",
    "steel_crack": "fatigue cracks",
}


def extract_first_assistant_answer(dual_prompt: str) -> str:
    """Extract the first non-empty assistant response from the dual prompt."""
    if not dual_prompt:
        return ""

    matches = re.findall(
        r"<\|assistant\|>\s*(.*?)(?=<\|end\|>|<\|user\|>|<\|assistant\|>|$)",
        dual_prompt,
        flags=re.DOTALL,
    )
    for text in matches:
        text = text.strip()
        if text:
            return text
    return ""


def add_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def simplify_prediction_record(record: dict) -> dict:
    """Convert one internal prediction record to the requested compact JSON format."""
    image = str(record.get("image", ""))
    image_id = image[len("image/"):] if image.startswith("image/") else image

    predicted_class = str(record.get("predicted_class", "")).strip()
    base_category = CLASS_MAP.get(predicted_class, predicted_class)
    damage_categories = [base_category] if base_category else []

    if predicted_class == "non_damage":
        value = record.get("category_aware_answer", "")
        a1 = "" if value is None else str(value).strip()
    else:
        a1 = extract_first_assistant_answer(str(record.get("dual_prompt", "")))

    description = record.get("model_caption", "")
    if description is None:
        description = ""
    description = str(description)

    evidence = f"{a1} {description}".lower()

    if predicted_class == "spalling":
        if "exposing" in evidence:
            add_unique(damage_categories, "exposing")
        if "rusting" in evidence:
            add_unique(damage_categories, "corrosion")
    elif predicted_class == "pothole":
        if "crack" in evidence:
            add_unique(damage_categories, "cracks")
    elif predicted_class == "corrosion":
        if "peeled" in evidence or "spalled" in evidence:
            add_unique(damage_categories, "peeling")

    return {
        "image_id": image_id,
        "damage_categories": damage_categories,
        "a1": a1,
        "description": description,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--blip-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--resnet-project", type=Path, required=True)
    parser.add_argument("--resnet-checkpoint", type=Path, required=True)
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cam-threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "float32"), default="auto")
    return parser.parse_args()


def choose_device_and_dtype(device_arg: str, dtype_arg: str):
    device = torch.device(
        "cuda" if device_arg == "auto" and torch.cuda.is_available() else
        "cpu" if device_arg == "auto" else device_arg
    )
    dtype = (
        torch.float16 if dtype_arg == "auto" and device.type == "cuda" else
        torch.float32 if dtype_arg == "auto" else
        {"float16": torch.float16, "float32": torch.float32}[dtype_arg]
    )
    return device, dtype


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_annotations(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def unpack_conversation(sample: dict) -> tuple[str, str, str | None, str | None]:
    turns = sample.get("conversations", [])
    roles = [turn.get("from") for turn in turns]
    if len(turns) == 2 and roles == ["human", "gpt"]:
        return str(turns[0]["value"]), str(turns[1]["value"]), None, None
    if len(turns) == 4 and roles == ["human", "gpt", "human", "gpt"]:
        return tuple(str(turn["value"]) for turn in turns)
    raise ValueError(
        f"{sample.get('id')} must contain either one or two human-GPT QA pairs."
    )


def strip_image_token(text: str) -> str:
    return text.replace("<image>", "", 1).strip()


def natural_category(class_name: str) -> str:
    return class_name.replace("_", " ")


def is_non_damage_class(class_name: str | None) -> bool:
    if class_name is None:
        return False
    normalized = class_name.casefold().replace("-", " ").replace("_", " ").strip()
    return normalized in {"non damage", "no damage", "undamaged", "normal"}


def canonical_class_name(class_name: str) -> str:
    return "non_damage" if is_non_damage_class(class_name) else class_name


def build_category_answer(
    class_name: str,
    confidence: float,
    confidence_threshold: float,
) -> str:
    if is_non_damage_class(class_name):
        return "No, there is no damage."
    # Keep A1 close to the annotation style seen during fine-tuning. Confidence is
    # retained in the signature for compatibility/logging but does not alter A1.
    annotation_style = {
        "concrete_crack": "Yes, there is a crack on the concrete surface.",
        "steel_crack": "Yes, a fine crack is visible on the steel component near the weld.",
        "void": "Yes, there are voids or holes on the concrete surface.",
        "spalling": "Yes, spalling and exposed rebar exist on the structural surface.",
        "corrosion": "Yes, there is corrosion on the exposed component.",
        "efflorescence": "Yes, there is efflorescence on the surface.",
        "honeycomb": "Yes, there is a honeycomb defect on the concrete surface.",
        "looseness": "Yes, there is a looseness defect on the concrete surface.",
        "pothole": "Yes, there is a pothole on the road surface.",
    }
    return annotation_style.get(
        class_name,
        f"Yes, there is {natural_category(class_name)} damage.",
    )


def build_location_guidance(location: str) -> str:
    return f"Focus on the defect located in the {location} region of the image."


def build_dual_prompt(
    first_question: str,
    category_answer: str,
    second_question: str,
    location_guidance: str,
) -> tuple[str, str]:
    enhanced_second_question = (
        f"{strip_image_token(second_question)} {location_guidance}"
    )
    prompt = (
        f"<|system|>\n{SYSTEM_MESSAGE}<|end|>\n"
        f"<|user|>\n<image>\n{strip_image_token(first_question)}<|end|>\n"
        f"<|assistant|>\n{category_answer}<|end|>\n"
        f"<|user|>\n{enhanced_second_question}<|end|>\n"
        "<|assistant|>\n"
    )
    return prompt, enhanced_second_question


def clean_answer(text: str) -> str:
    if "<|assistant|>" in text:
        text = text.rsplit("<|assistant|>", 1)[-1]
    text = text.split("<|end|>", 1)[0]
    return text.replace("<s>", "").replace("</s>", "").strip()


def parse_damage_class(answer: str) -> str | None:
    text = answer.casefold()
    if any(
        phrase in text
        for phrase in ("no damage", "not damaged", "non-damaged", "non damaged", "undamaged")
    ):
        return "non_damage"
    patterns = [
        ("efflorescence", ("efflorescen", "salt deposit", "white deposit")),
        ("honeycomb", ("honeycomb",)),
        ("pothole", ("pothole", "pot hole")),
        ("looseness", ("looseness", "loose concrete", "surface is loose", "detached concrete")),
        ("corrosion", ("corrosion", "corroded", "rusting", "rusted", "rust ")),
        ("spalling", ("spalling", "spalled", "exposed rebar", "exposed reinforcement")),
        ("void", ("concrete void", "void in", "voids", "cavity", "air pocket")),
    ]
    for class_name, aliases in patterns:
        if any(alias in text for alias in aliases):
            return class_name
    if "crack" in text or "fracture" in text:
        if any(term in text for term in ("steel", "metal", "weld")):
            return "steel_crack"
        if any(term in text for term in ("concrete", "cement", "masonry")):
            return "concrete_crack"
    return None


def load_classification_utils(project_dir: Path):
    module_path = project_dir / "classification_utils.py"
    spec = importlib.util.spec_from_file_location("classification_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_resnet(
    project_dir: Path,
    checkpoint_path: Path,
    classes_path: Path,
    device: torch.device,
):
    utils = load_classification_utils(project_dir)
    class_names = utils.load_classes(classes_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    attention = checkpoint.get(
        "attention",
        checkpoint.get("args", {}).get("attention", "none"),
    )
    dropout = float(checkpoint.get("args", {}).get("dropout", 0.0))
    model = utils.build_model(
        num_classes=len(class_names),
        pretrained=False,
        attention=attention,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    target_layer = (
        model.layer4[0][-1]
        if attention in {"eca", "eca_mma", "cbam"}
        else model.layer4[-1]
    )
    return model, class_names, target_layer


def prepare_resnet_image(
    image: Image.Image,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
    """Pad the complete image to a square, then resize for ResNet inference."""
    weights = ResNet50_Weights.DEFAULT
    mean = weights.transforms().mean
    std = weights.transforms().std
    fill = tuple(int(round(value * 255)) for value in mean)

    width, height = image.size
    side = max(width, height)
    pad_left = (side - width) // 2
    pad_top = (side - height) // 2

    padded_image = Image.new("RGB", (side, side), fill)
    padded_image.paste(image, (pad_left, pad_top))
    display_image = transforms.Resize((image_size, image_size))(padded_image)

    input_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )(display_image)

    transform_geometry = (
        side,
        pad_left,
        pad_top,
        width,
        height,
    )
    return input_tensor.unsqueeze(0).to(device), transform_geometry


def restore_gradcam_to_original(
    grayscale_cam: np.ndarray,
    original_size: tuple[int, int],
    transform_geometry: tuple[int, int, int, int, int],
) -> np.ndarray:
    """Map a CAM from the padded ResNet input back to original-image coordinates."""
    side, pad_left, pad_top, width, height = transform_geometry
    resampling = getattr(Image, "Resampling", Image).BILINEAR

    square_cam = np.asarray(
        Image.fromarray(grayscale_cam.astype(np.float32), mode="F").resize(
            (side, side),
            resample=resampling,
        ),
        dtype=np.float32,
    )

    original_region = square_cam[
        pad_top:pad_top + height,
        pad_left:pad_left + width,
    ]

    if original_region.shape[::-1] != original_size:
        original_region = np.asarray(
            Image.fromarray(original_region.astype(np.float32), mode="F").resize(
                original_size,
                resample=resampling,
            ),
            dtype=np.float32,
        )

    return np.clip(original_region, 0.0, 1.0)

def locate_gradcam(
    grayscale_cam: np.ndarray,
    activation_threshold: float,
) -> tuple[str, float, float]:
    cutoff = float(grayscale_cam.max()) * activation_threshold
    weights = np.where(grayscale_cam >= cutoff, grayscale_cam, 0.0)
    rows, columns = np.indices(weights.shape)
    total = float(weights.sum())
    center_x = float((weights * columns).sum() / total / (weights.shape[1] - 1))
    center_y = float((weights * rows).sum() / total / (weights.shape[0] - 1))

    if 1 / 3 <= center_x <= 2 / 3 and 1 / 3 <= center_y <= 2 / 3:
        location = "central"
    elif abs(center_x - 0.5) >= abs(center_y - 0.5):
        location = "left" if center_x < 0.5 else "right"
    else:
        location = "upper" if center_y < 0.5 else "lower"
    return location, center_x, center_y


def infer_category_and_gradcam(
    model,
    gradcam,
    class_names: list[str],
    input_tensor: torch.Tensor,
):
    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor)[0], dim=0)
    class_id = int(probabilities.argmax())
    confidence = float(probabilities[class_id])
    grayscale_cam = gradcam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(class_id)],
        aug_smooth=False,
        eigen_smooth=False,
    )[0]
    return (
        class_id,
        class_names[class_id],
        confidence,
        grayscale_cam,
    )


def load_base_checkpoint(model) -> None:
    checkpoint_dir = snapshot_download(
        repo_id=BASE_MODEL_ID,
        allow_patterns=["model*.safetensors", "model.safetensors.index.json"],
    )
    wrapper = torch.nn.Module()
    wrapper.add_module("vlm", model)
    load_sharded_checkpoint(wrapper, checkpoint_dir, strict=True, prefer_safe=True)


def load_blip(checkpoint_dir: Path, device: torch.device, dtype: torch.dtype):
    cfg = OmegaConf.create(
        {
            "image_aspect_ratio": "anyres",
            "anyres_patch_sampling": True,
            "anyres_grids": [(1, 2), (2, 1), (2, 2), (3, 1), (1, 3)],
        }
    )
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="google/siglip-so400m-patch14-384",
        clip_vision_encoder_pretrained="google",
        lang_model_path="microsoft/Phi-3-mini-4k-instruct",
        tokenizer_path=str(checkpoint_dir / "tokenizer"),
        model_family="xgenmm_v1",
        num_vision_tokens=128,
        image_aspect_ratio=cfg.image_aspect_ratio,
        anyres_patch_sampling=cfg.anyres_patch_sampling,
    )
    load_base_checkpoint(model)
    model.lang_model = PeftModel.from_pretrained(
        model.lang_model,
        checkpoint_dir / "phi3_lora_adapter",
    ).merge_and_unload()
    model.vision_tokenizer.load_state_dict(
        torch.load(checkpoint_dir / "vision_token_sampler.pt", map_location="cpu")
    )
    model.anyres_grids = [
        [model.base_img_size * rows, model.base_img_size * columns]
        for rows, columns in cfg.anyres_grids
    ]
    return model.to(device=device, dtype=dtype).eval(), tokenizer, image_processor, cfg


@torch.inference_mode()
def generate_caption(
    model,
    tokenizer,
    image_processor,
    cfg,
    image: Image.Image,
    prompt: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> str:
    image_proc = partial(process_images, image_processor=image_processor, model_cfg=cfg)
    image_tensor = image_proc([image]).to(device=device, dtype=dtype)
    language = tokenizer([prompt], return_tensors="pt")
    generated = model.generate(
        vision_x=[[image_tensor]],
        lang_x=language["input_ids"].to(device),
        attention_mask=language["attention_mask"].to(device),
        image_size=[[image.size]],
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
    )
    return clean_answer(tokenizer.decode(generated[0], skip_special_tokens=False))


def main() -> None:
    args = parse_args()
    set_seed()
    device, dtype = choose_device_and_dtype(args.device, args.dtype)
    samples = load_annotations(args.annotations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cam_dir = args.output_dir / "gradcam"
    cam_dir.mkdir(exist_ok=True)

    resnet, class_names, target_layer = load_resnet(
        args.resnet_project,
        args.resnet_checkpoint,
        args.classes,
        device,
    )
    gradcam = GradCAM(model=resnet, target_layers=[target_layer])
    blip, tokenizer, image_processor, cfg = load_blip(
        args.blip_checkpoint_dir,
        device,
        dtype,
    )
    results = []
    for sample in tqdm(samples, desc="Dual-prompt inference", unit="image"):
        first_question, _, second_question, _ = unpack_conversation(sample)
        image_path = Path(sample["image"])
        if image_path.parts[0] == "defect_dataset":
            image_path = Path(*image_path.parts[1:])
        image = Image.open(args.data_root / image_path).convert("RGB")
        input_tensor, transform_geometry = prepare_resnet_image(
            image,
            args.image_size,
            device,
        )
        (
            _,
            predicted_class,
            confidence,
            grayscale_cam,
        ) = infer_category_and_gradcam(
            resnet,
            gradcam,
            class_names,
            input_tensor,
        )
        grayscale_cam = restore_gradcam_to_original(
            grayscale_cam,
            image.size,
            transform_geometry,
        )
        location, _, _ = locate_gradcam(
            grayscale_cam,
            args.cam_threshold,
        )
        rgb_image = np.asarray(image, dtype=np.float32) / 255.0

        category_answer = build_category_answer(
            predicted_class,
            confidence,
            args.confidence_threshold,
        )
        predicted_class = canonical_class_name(predicted_class)
        predicted_non_damage = is_non_damage_class(predicted_class)

        # Decide whether Q2 exists solely from the classifier prediction.
        # If the classifier predicts a damage class, always run the description stage.
        # For annotations without an original Q2 (typically GT non-damage samples),
        # fall back to a fixed generic description question.
        if second_question is None and not predicted_non_damage:
            second_question = "Please describe the visible damage characteristics in detail."

        skip_q2 = predicted_non_damage
        if skip_q2:
            location_guidance = None
            prompt = None
            caption = None
        else:
            location_guidance = build_location_guidance(location)
            prompt, _ = build_dual_prompt(
                first_question,
                category_answer,
                second_question,
                location_guidance,
            )
            caption = generate_caption(
                blip,
                tokenizer,
                image_processor,
                cfg,
                image,
                prompt,
                device,
                dtype,
                args.max_new_tokens,
            )
        source_filename = Path(sample["image"]).name
        cam_path = cam_dir / source_filename
        Image.fromarray(
            show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
        ).save(cam_path)
        results.append(
            {
                "image": sample["image"],
                "predicted_class": predicted_class,
                "category_aware_answer": category_answer,
                "dual_prompt": prompt,
                "model_caption": caption,
            }
        )

    gradcam.activations_and_grads.release()

    simplified_results = [simplify_prediction_record(result) for result in results]
    (args.output_dir / "predictions.json").write_text(
        json.dumps(simplified_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {len(simplified_results)} predictions to {args.output_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
