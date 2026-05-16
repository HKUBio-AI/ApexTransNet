# ApexTransNet

ApexTransNet is an anatomy-aware deep learning framework for apical periodontitis (AP) analysis on panoramic dental radiographs. The model performs pixel-level AP lesion segmentation and case-level AP prediction using a shared encoder-decoder backbone with a Transformer bottleneck, ASPP context aggregation, attention-gated decoding, and auxiliary localization, boundary, and classification heads.

This repository is packaged for reuse: it includes the final trained model checkpoint, core model code, an inference script, environment files, and training/evaluation scripts used in the project.

## Repository structure

```text
ApexTransNet-github/
├── predict.py                         # Standalone inference script for images or folders
├── train.py                           # Training script for Stage 1 / Stage 2 experiments
├── evaluate_casewise_mean.py          # Case-wise mean evaluation script
├── evaluate.py                        # Legacy evaluation script
├── search_postprocess.py              # Threshold/post-processing search utility
├── models/
│   ├── model.py                       # ApexTransNet architecture
│   └── dataset.py                     # Dataset and preprocessing utilities
├── config/
│   ├── train_stage1_server.json       # Stage 1 training configuration
│   └── train_stage2_server.json       # Stage 2 training/inference configuration
├── checkpoints/
│   ├── best_apex_trans_stage2_512.pt  # Final ApexTransNet checkpoint for inference
│   └── pretrained/                    # Optional encoder initialization for retraining, if available
├── scripts/                           # Optional dataset-preparation utilities
├── requirements.txt
├── environment.yml
├── .gitattributes                     # Git LFS rules for model weights
└── .gitignore
```

## Installation

Create a clean Python environment:

```bash
conda env create -f environment.yml
conda activate apextransnet
```

Alternatively, install with pip:

```bash
conda create -n apextransnet python=3.10 -y
conda activate apextransnet
pip install -r requirements.txt
```

For GPU inference or training, install the PyTorch build matching your CUDA version from the official PyTorch installation page. The code also runs on CPU, but inference and training will be slower.

## Quick prediction

Run inference on a single panoramic radiograph:

```bash
python predict.py   --input /path/to/radiograph.png   --checkpoint checkpoints/best_apex_trans_stage2_512.pt   --output outputs/predictions   --threshold 0.55   --cls-threshold 0.5   --save-prob   --save-overlay
```

Run inference on a folder of images:

```bash
python predict.py   --input /path/to/image_folder   --checkpoint checkpoints/best_apex_trans_stage2_512.pt   --output outputs/predictions   --threshold 0.55   --save-prob   --save-overlay
```

The script accepts `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff` images. Each image is converted to grayscale, resized to `512 x 512`, normalized by its own mean and standard deviation, and then passed through the model.

## Prediction outputs

For each input image, `predict.py` saves:

- `<case_id>_mask.png`: binary AP lesion segmentation mask resized back to the original image size.
- `<case_id>_prob.png`: lesion probability heatmap, saved when `--save-prob` is used.
- `<case_id>_overlay.png`: predicted mask overlay on the original grayscale image, saved when `--save-overlay` is used.
- `predictions.csv`: case-level AP probability, classification result, mask area, thresholds, and output file paths.

Important default thresholds:

- Segmentation threshold: `0.55`
- Case-level classification threshold: `0.5`

These are the defaults used for the released Stage 2 checkpoint. If you change thresholds, report them clearly in experiments.

## Model checkpoint and GitHub note

The checkpoint files are large. If this folder is pushed to GitHub, use Git LFS:

```bash
git lfs install
git lfs track "*.pt" "*.pth"
git add .gitattributes checkpoints/
```

Without Git LFS, GitHub may reject model files larger than 100 MB.

## Training

Stage 1 trains lesion segmentation using the hard consensus target:

```bash
python train.py --config config/train_stage1_server.json
```

Stage 2 normally initializes from the best Stage 1 checkpoint during full retraining and performs multi-task refinement with soft-vote lesion supervision, uncertainty weighting, localization supervision, boundary supervision, case-level diagnosis, and anatomy-guided regularization. The released repository keeps the final Stage 2 checkpoint for inference; if reproducing training from scratch, first run Stage 1 and then run Stage 2:

```bash
python train.py --config config/train_stage2_server.json
```

Before training on a new machine, update the dataset paths in the JSON configuration files, especially `metadata_csv`, and ensure that image and mask paths referenced by the metadata file are accessible.

## Evaluation

The current evaluation script reports per-case mean metrics:

```bash
python evaluate_casewise_mean.py   --config config/train_stage2_server.json   --checkpoint checkpoints/best_apex_trans_stage2_512.pt   --split test   --metrics-name test_metrics_apex_trans_stage2_case_mean.csv
```

Pixel-level metrics include Dice, IoU, precision, and recall. Lesion-level metrics are computed by connected components with an IoU matching threshold, controlled by `--iou-thr`.

## Method summary

ApexTransNet uses a ResNet-style encoder, Transformer bottleneck, ASPP context module, apex/root-context fusion, attention-gated decoder, and four prediction heads:

- AP lesion segmentation head.
- Tooth/root localization head.
- Case-level AP classification head.
- Auxiliary lesion boundary head.

The final Stage 2 model was trained with multi-task losses including segmentation, localization, classification, boundary, and anatomy-guided regularization terms. The released checkpoint is intended for research use and should not be used as a standalone clinical diagnostic device.

## Data and privacy

No patient images are included in this packaged project. To reproduce training or evaluation, prepare a metadata CSV and the corresponding image/mask files according to the paths expected by `models/dataset.py` and the configuration files.

## Citation

If you use ApexTransNet, please cite the associated manuscript once available.
