#!/bin/bash
set -euo pipefail

run_timestamp="$(date +%Y%m%d_%H%M%S)"

export PYTHONPATH="/your/path/DamageCap:/your/path/DamageCap/open_flamingo/train:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python infer.py \
    --annotations annotations/test.json \
    --data-root dataset/icshm_test \
    --blip-checkpoint-dir checkpoints/finetune-xgenmmv1-phi3-lora-defect/compact_checkpoints/epoch_3 \
    --resnet-project resnet50_classification \
    --resnet-checkpoint resnet50_classification/runs/resnet50_eca_official/best_metric.pt \
    --classes resnet50_classification/annotations/classes.txt \
    --output-dir "/your/path/DamageCap/results_${run_timestamp}" \
    --device cuda \
    --dtype float16
