# ApexTransNet

ApexTransNet is a research framework for pixel-level apical periodontitis (AP) lesion segmentation and case-level radiographic AP assessment on periapical dental radiographs.

The repository contains the model implementation and supporting scripts for inference, training, and evaluation. It is provided for research and reproducibility purposes and is not intended for standalone clinical diagnosis or treatment decisions.

## Repository overview

```text
.
├── predict.py                  # Inference entry point
├── train.py                    # Training entry point
├── evaluate_casewise_mean.py   # Case-wise evaluation
├── models/                     # Model and dataset code
├── config/                     # Example configuration files
├── checkpoints/                # Local checkpoint directory
├── scripts/                    # Data-preparation utilities
├── requirements.txt
└── environment.yml
```

## Installation

Using Conda:

```bash
conda env create -f environment.yml
conda activate apextransnet
```

Alternatively:

```bash
conda create -n apextransnet python=3.10 -y
conda activate apextransnet
pip install -r requirements.txt
```

For GPU use, install a PyTorch build compatible with the local CUDA environment.

## Inference

Place a compatible checkpoint in the `checkpoints/` directory, then run:

```bash
python predict.py \
  --input /path/to/image_or_folder \
  --checkpoint /path/to/checkpoint.pt \
  --output outputs/predictions
```

Available options may be inspected with:

```bash
python predict.py --help
```

Depending on the selected options, inference outputs may include segmentation masks, probability maps, overlays, and a CSV summary. Thresholds and preprocessing settings should be reported whenever results are used in an experiment.

## Checkpoint

The released checkpoint is distributed separately from the source repository:

[Download the ApexTransNet checkpoint](https://drive.google.com/drive/folders/155pcqd_6mB1M2NAolCk7i44bjFAY8DYA)

Checkpoint compatibility depends on the model and preprocessing configuration used by the corresponding code version.

## Training

Training is configuration-driven. Review and update the dataset paths and output locations before running:

```bash
python train.py --config config/train_stage1_server.json
```

For the refinement stage, initialize from the selected Stage 1 checkpoint:

```bash
python train.py \
  --config config/train_stage2_server.json \
  --resume /path/to/stage1_checkpoint.pt
```

The supplied configuration files are examples and should be checked against the intended dataset, preprocessing pipeline, and computing environment. Test data should not be used for checkpoint or threshold selection.

## Evaluation

Case-wise evaluation can be run with:

```bash
python evaluate_casewise_mean.py \
  --config config/train_stage2_server.json \
  --checkpoint /path/to/checkpoint.pt \
  --split test
```

Use `--help` to review the options supported by the current script. Report the evaluated checkpoint, dataset split, threshold-selection procedure, and metric definitions with any published results.

## Data

The study's training and evaluation datasets are not distributed through this repository. Users must prepare their own authorized data and follow the metadata and path conventions expected by the dataset code.

Do not commit identifiable patient data, local filesystem paths, credentials, access tokens, or private checkpoint links to the repository. Local data and generated outputs should remain excluded through `.gitignore` or equivalent controls.

## Limitations

The released model represents research-stage technical validation. Performance may vary across acquisition systems, image-quality conditions, patient populations, preprocessing choices, and operating thresholds. Independent external validation and, where appropriate, recalibration are required before clinical use.

## Citation

If you use this code, please cite the associated manuscript once its bibliographic information is available.
