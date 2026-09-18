#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ATTENTION="${1:-none}"
ANNOTATION_DIR="$SCRIPT_DIR/annotations"
OUTPUT_DIR="runs/resnet50_${ATTENTION}"

case "$ATTENTION" in
  none|eca|eca_mma|eca_official|cbam) ;;
  *)
    echo "Unsupported attention type: $ATTENTION" >&2
    exit 2
    ;;
esac

if [[ "$ATTENTION" == "eca_mma" ]]; then
  BATCH_SIZE=8
else
  BATCH_SIZE=32
fi

EXTRA_ARGS=()
if [[ "$ATTENTION" == "eca_official" ]]; then
  EXTRA_ARGS+=(
    --official-eca-weights "$SCRIPT_DIR/pretrained/eca_resnet50_k3557.pth.tar"
  )
fi

python "$SCRIPT_DIR/train.py" \
  --train-data-root "/your/path/DamageCap/dataset" \
  --train-csv "$ANNOTATION_DIR/train.csv" \
  --eval-csv "$ANNOTATION_DIR/test.csv" \
  --classes "$ANNOTATION_DIR/classes.txt" \
  --output-dir "$OUTPUT_DIR" \
  --attention "$ATTENTION" \
  --epochs 100 \
  --batch-size "$BATCH_SIZE" \
  --num-workers 4 \
  --freeze-backbone-epochs 0 \
  --backbone-learning-rate 1e-4 \
  --learning-rate 1e-4 \
  --dropout 0.3 \
  --weight-decay 5e-4 \
  --eval-every 2 \
  --early-stopping-patience 10 \
  --min-epochs 50 \
  --class-weights balanced \
  --metric-for-best macro_f1 \
  "${EXTRA_ARGS[@]}"
