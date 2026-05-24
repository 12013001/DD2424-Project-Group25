# Transfer Learning on the Oxford-IIIT Pet Dataset

This repository contains the code for a deep learning project on transfer learning with the Oxford-IIIT Pet Dataset. The project includes an E-level basic transfer learning pipeline and a B/A-level extension using semi-supervised pseudo-labeling.

## Project Overview

The main task is pet image classification using ImageNet-pretrained ResNet models. The project evaluates how well pretrained visual representations transfer to the Oxford-IIIT Pet Dataset under different training conditions.

The code is organized into two main scripts:

- `project_e.py`: basic transfer learning experiments.
- `project_a.py`: semi-supervised pseudo-labeling extension.

## Dataset

The project uses the Oxford-IIIT Pet Dataset, which contains 37 pet breed classes, including cat and dog breeds. The code downloads the dataset automatically through `torchvision.datasets.OxfordIIITPet`.

The training/validation split is created from the original `trainval` split using stratified sampling. The official test split is used only for final evaluation.

## E-Level Experiments

The E-level script `project_e.py` includes:

1. Binary cat-versus-dog classification.
2. 37-class linear probing with a frozen pretrained ResNet.
3. Fine-tuning Strategy 1: simultaneously unfreezing the last `l` ResNet layer groups.
4. Fine-tuning Strategy 2: gradual unfreezing.
5. Limited labeled-data experiments using 100%, 10%, and 1% of the training data.
6. Class imbalance experiments, where cat-breed samples are reduced and imbalance-handling methods are compared.

The main outputs are saved in the `results/` directory, including CSV summaries and training plots.

## B/A-Level Extension

The B/A-level script `project_a.py` investigates semi-supervised pseudo-labeling. The pipeline is:

1. Train a teacher model using only the labeled subset.
2. Use the teacher to predict labels for the remaining unlabeled training images.
3. Keep only predictions whose maximum softmax confidence is above a threshold.
4. Train a student model using the union of labeled data and accepted pseudo-labeled samples.

The extension evaluates labeled-data fractions of 100%, 50%, 10%, and 1%, and compares pseudo-labeling with supervised baselines under the same labeled-data fraction. It also includes confidence-threshold ablation and failure analysis.

The main outputs are saved in the `results_a/` directory.

## Requirements

The code requires Python 3 and the following main packages:

```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

A CUDA-enabled GPU is recommended. The reported training times are hardware-dependent and should be interpreted only as approximate references.

## How to Run

Run the E-level basic experiments:

```bash
python project_e.py
```

Run the B/A-level pseudo-labeling extension:

```bash
python project_a.py
```

`project_a.py` imports utility functions from `project_e.py`, so both files should be kept in the same directory.

## Output Files

After running the scripts, the following folders are generated:

```text
results/
results_a/
```

These folders contain:

- CSV files with experiment summaries.
- Training and validation curves.
- Comparison plots.
- Per-class F1 plots.
- Confusion-matrix-related analysis.

## Notes

- The random seed is fixed to improve reproducibility.
- The test set is only used for final evaluation.
- The pseudo-labeling experiments use supervised baselines trained with the same labeled-data fraction for fair comparison.
- Some results may vary slightly across hardware and library versions.
- A CUDA-enabled GPU is recommended. The reported training times are hardware-dependent and should be interpreted only as approximate references.

## Repository Structure

```text
.
├── project_e.py          # E-level transfer learning experiments
├── project_a.py          # B/A-level pseudo-labeling extension
├── README.md             # Project description and usage instructions
├── results/              # Generated E-level outputs
└── results_a/            # Generated B/A-level outputs
```

## Authors

This project was developed for a deep learning course project on transfer learning and semi-supervised learning.
