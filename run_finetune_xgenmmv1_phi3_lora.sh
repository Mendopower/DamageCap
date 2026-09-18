#!/bin/bash
export PYTHONPATH="/your/path/DamageCap:/your/path/DamageCap/open_flamingo/train:${PYTHONPATH:-}"
set -euo pipefail

project_root="/your/path/DamageCap"
finetune_script="instruction_finetune_lora.py"
dataset_root="dataset"
data_path="annotations/train.json"
pretrained_repo="Salesforce/xgen-mm-phi3-mini-base-r-v1.5"
run_timestamp="$(date +%Y%m%d_%H%M%S)"
exp_name="finetune-xgenmmv1-phi3-lora-defect-${run_timestamp}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE=disabled

cd "$project_root"
mkdir -p "$exp_name"

python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --master_port=9650 \
    "$finetune_script" \
    --lm_path microsoft/Phi-3-mini-4k-instruct \
    --tokenizer_path microsoft/Phi-3-mini-4k-instruct \
    --conv_template_name phi_3 \
    --vision_encoder_path google/siglip-so400m-patch14-384 \
    --vision_encoder_pretrained google \
    --model_family xgenmm_v1 \
    --num_vision_tokens 128 \
    --pretrained_hf "$pretrained_repo" \
    --data_path "$data_path" \
    --image_root "$dataset_root" \
    --image_prefix defect_dataset/ \
    --data_sampler_group_by_length \
    --image_aspect_ratio anyres \
    --anyres_patch_sampling \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --workers 4 \
    --num_epochs 3 \
    --warmup_steps 40 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --lr_scheduler linear \
    --precision amp_bf16 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --run_name "$exp_name" 2>&1 | tee "$exp_name/terminal_output.log"
