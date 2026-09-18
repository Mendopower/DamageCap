# ResNet-50 defect classification

This package trains a 10-class ResNet-50 model with optional lightweight
stage-level ECA, stage-level CBAM, or stage-level ECA plus one layer4 Medical
Modality Attention (MMA) module. It also provides the official 16-Bottleneck
ECA-ResNet50 k3557 architecture with its ImageNet checkpoint. It uses
console output and local CSV/text logs; TensorBoard is not required.

The default fine-tuning schedule is:

- epochs 1-5: freeze the pretrained backbone and its BatchNorm statistics;
- epoch 6 onward: train the backbone at `1e-5` and the classification
  head/optional attention modules at `1e-4`;
- place `Dropout(0.3)` before the final linear layer;
- use AdamW weight decay `5e-4`;
- allow at most 100 epochs, but stop after 10 consecutive evaluations without
  a new best Macro-F1; early stopping cannot occur before epoch 50.

## Expected layout

```text
/root/autodl-tmp/
├── dataset/
│   ├── image/
│   └── description.json
└── resnet50_classification/
    ├── annotations/
    │   ├── train.csv
    │   ├── test.csv
    │   └── classes.txt
    ├── classification_utils.py
    ├── train.py
    ├── val.py
    └── run_train.sh
```

Annotation paths are relative to the directory containing `image/`. Training
and evaluation images may use different roots: pass the training root through
`--data-root` and the independent test root through `--eval-data-root`.

`run_train.sh` accepts the independent test root through `EVAL_DATA_ROOT`:

```bash
EVAL_DATA_ROOT="/path/to/Dataset-Project 3-Test/dataset" \
  bash run_train.sh "/path/to/Dataset-Project 3-Updated/dataset" eca_official
```

The current `test.csv` has blank `label` and `class_id` fields. Fill both
fields before using it as `--eval-csv` during training.

## Select the architecture

The second argument to `run_train.sh` selects the model:

```bash
cd /root/autodl-tmp/resnet50_classification

# Baseline ResNet-50
bash run_train.sh "/root/autodl-tmp/dataset" none

# ResNet-50 + ECA
bash run_train.sh "/root/autodl-tmp/dataset" eca

# ResNet-50 + stage-level ECA + one MMA after layer4
bash run_train.sh "/root/autodl-tmp/dataset" eca_mma

# Official Bottleneck-level ECA-ResNet50 + ImageNet k3557 weights
bash run_train.sh "/root/autodl-tmp/dataset" eca_official

# ResNet-50 + CBAM
bash run_train.sh "/root/autodl-tmp/dataset" cbam
```

The default output directories are separated automatically:

```text
/root/autodl-tmp/runs/resnet50_none_defects
/root/autodl-tmp/runs/resnet50_eca_defects
/root/autodl-tmp/runs/resnet50_eca_mma_defects
/root/autodl-tmp/runs/resnet50_eca_official_defects
/root/autodl-tmp/runs/resnet50_cbam_defects
```

Fresh runs never append to an existing run directory. If the requested output
directory already contains files, training automatically creates a new sibling
directory with a timestamp, for example:

```text
/root/autodl-tmp/runs/resnet50_eca_defects_20260731_163000
```

Only `--resume` continues writing to the specified existing directory. The
resolved and originally requested paths are both recorded in `config.json`.
Use a fresh output directory when intentionally branching from an older
checkpoint.

ECA and CBAM use the lightweight stage-level design: one attention module is
placed after each complete ResNet stage (`layer1` through `layer4`), for four
attention modules in total. ECA selects its 1D kernel size adaptively from each
stage's channel count. The torchvision ImageNet backbone is loaded before
attention is attached, so its pretrained weights remain directly reusable.

With `eca_mma`, ECA remains after all four stages and exactly one MMA module is
appended after the layer4 ECA:

```text
layer4 -> ECA(2048 channels) -> MMA(2048 channels) -> avgpool -> classifier
```

MMA follows Fig. 2 and the Section 2.1 equations. Its anatomy branch uses a
7x7 convolution, BatchNorm, ReLU, a 1x1 projection and sigmoid spatial gate.
Its texture branch uses global average pooling and two 1x1 convolutions with a
1/16 channel reduction and sigmoid channel gate. Their product forms intrinsic
modality attention. The multi-scale branch uses three 3x3 dilated convolutions
with dilation rates 1, 2 and 4; their channel-wise concatenation is projected
back to 2048 channels by a 1x1 convolution. The final result is:

```text
Fout = x * Aintr(x) + Mscale(x)
```

The layer4 MMA adds about 21.2 million trainable parameters. Therefore
`run_train.sh` defaults to batch size 8 for `eca_mma`, while the other models
retain batch size 32. A fifth argument overrides the batch size.

## Official ECA-ResNet50 ImageNet weights

Download `eca_resnet50_k3557` from the official ECANet repository:

```text
https://github.com/BangguWu/ECANet
```

Place the downloaded checkpoint at:

```text
/root/autodl-tmp/pretrained/eca_resnet50_k3557.pth.tar
```

The package includes a downloader for the official Google Drive file:

```bash
pip install -r requirements.txt
python download_official_eca_weights.py
```

`eca_official` constructs the official ResNet50 layout with 16 ECA modules,
using stage kernel sizes 3/5/5/7. It loads every ImageNet backbone and ECA
parameter from the checkpoint, discards the original 1000-class `fc` weights,
and randomly initializes the new 10-class classifier. Official ECA parameters
are treated as pretrained backbone parameters: they are frozen with the
backbone for the first five epochs and use the backbone learning rate after
unfreezing.

To use a different checkpoint location, pass it as the sixth argument:

```bash
bash run_train.sh \
  "/root/autodl-tmp/dataset" \
  eca_official \
  "/root/autodl-tmp/resnet50_classification/annotations" \
  "/root/autodl-tmp/runs/resnet50_eca_official_defects" \
  32 \
  "/path/to/eca_resnet50_k3557.pth.tar"
```

## Direct Python command

```bash
python train.py \
  --data-root "/root/autodl-tmp/dataset" \
  --eval-data-root "/root/autodl-tmp/test-dataset" \
  --train-csv "/root/autodl-tmp/resnet50_classification/annotations/train.csv" \
  --eval-csv "/root/autodl-tmp/resnet50_classification/annotations/test.csv" \
  --classes "/root/autodl-tmp/resnet50_classification/annotations/classes.txt" \
  --output-dir "/root/autodl-tmp/runs/resnet50_eca_mma_defects" \
  --attention eca_mma \
  --epochs 100 \
  --batch-size 8 \
  --freeze-backbone-epochs 5 \
  --backbone-learning-rate 1e-5 \
  --learning-rate 1e-4 \
  --dropout 0.3 \
  --weight-decay 5e-4 \
  --eval-every 2 \
  --early-stopping-patience 10 \
  --min-epochs 50
```

Valid values are:

```text
--attention none
--attention eca
--attention eca_mma
--attention eca_official
--attention cbam
```

## Outputs

```text
run_directory/
├── checkpoints/
│   ├── best_metric.pt
│   ├── best_loss.pt
│   └── final.pt
├── eval_metrics/
│   └── epoch_XXXX.json
├── class_mapping.json
├── config.json
├── training.log
└── training_metrics.csv
```

Each checkpoint records its attention type. Do not resume one architecture
from a checkpoint created by another architecture. The dropout setting is also
stored and is restored automatically during evaluation. Earlier stage-level
ECA checkpoints without an `architecture_version` field remain loadable.
Official Bottleneck-level checkpoints created by this package are supported
through the `eca_official` architecture identifier.

Pure stage-level ECA and `eca_mma` checkpoints are also intentionally
incompatible with each other.

## Validate a checkpoint

`val.py` reads the architecture from the checkpoint by default:

```bash
python val.py \
  --data-root "/root/autodl-tmp/dataset" \
  --csv "/root/autodl-tmp/resnet50_classification/annotations/test.csv" \
  --classes "/root/autodl-tmp/resnet50_classification/annotations/classes.txt" \
  --checkpoint "/root/autodl-tmp/runs/resnet50_eca_mma_defects/checkpoints/best_metric.pt" \
  --output-dir "/root/autodl-tmp/runs/resnet50_eca_mma_defects/final_evaluation"
```

It writes `metrics.json`, `predictions.csv`, and `confusion_matrix.csv`.

## GradCAM and FinerCAM inference

Install the CAM dependency with the other requirements:

```bash
pip install -r requirements.txt
```

Generate both visualizations for one image:

```bash
python infer_cam.py \
  --input "/root/autodl-tmp/Dataset-Project 3-Updated/dataset/image/1.jpg" \
  --checkpoint "/root/autodl-tmp/runs/resnet50_eca_official_defects/checkpoints/best_metric.pt" \
  --classes "/root/autodl-tmp/resnet50_classification/annotations/classes.txt" \
  --output-dir "/root/autodl-tmp/cam_results" \
  --method both
```

`--input` can also be a directory. Images are discovered recursively and each
image gets its own output directory containing:

```text
input.png
gradcam.png
finercam.png
metadata.json
```

The root output directory also contains `summary.json`. By default, CAMs
explain the model's predicted class. A different class can be selected by ID or
exact name:

```bash
--target-class corrosion
```

FinerCAM suppresses features shared with competing classes. By default, it
compares the target against the three highest-probability alternative classes.
The comparison can be controlled explicitly:

```bash
--comparison-classes "steel_crack,concrete_crack,corrosion" \
--finer-alpha 1.0
```

Use `--method gradcam` or `--method finercam` to generate only one method.
Optional `--aug-smooth` and `--eigen-smooth` follow the official
`pytorch-grad-cam` API. The target layer is selected automatically:

- baseline and official Bottleneck ECA: final Bottleneck in `layer4`;
- stage-level ECA, ECA+MMA and CBAM: final Bottleneck inside the wrapped
  original `layer4`.

## Methodology note

Using `test.csv` to select the best checkpoint makes that split function as a
validation set. For an unbiased final result, create a separate validation
split and evaluate the true test set only after model selection.
