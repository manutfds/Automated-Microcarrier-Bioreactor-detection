# ATENEA: Automated Microcarrier Colonization Detection

[![Paper](https://img.shields.io/badge/Paper-Frontiers-blue)](https://doi.org/XXXX)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)

> **Comparative evaluation of YOLOv8, YOLOv11, and RT-DETR for automated microcarrier colonization assessment in bioreactor cell cultures**

This repository provides comprehensive methodological documentation, configuration files, and template code to support reproducibility of the ATENEA project. Due to institutional intellectual property restrictions, the original training data and model weights cannot be released. However, this repository enables researchers to replicate our methodology on their own microcarrier datasets.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Getting Started](#getting-started)
- [Configuration Files](#configuration-files)
- [Citation](#citation)
- [Contact](#contact)

---

## 🔬 Overview

ATENEA is a deep learning pipeline for automated quantification of mesenchymal stromal cell (MSC) colonization on microcarriers in bioreactor cultures. The system:

- **Detects** individual microcarriers in fluorescence microscopy images
- **Classifies** each microcarrier as FULL (colonized) or EMPTY (non-colonized)
- **Quantifies** colonization percentage per image

### Why This Matters

| Manual Method | ATENEA Pipeline |
|---------------|-----------------|
| ~10 min/image | ~0.14 sec/image |
| >15% inter-operator variability | Consistent, reproducible |
| Subjective assessment | Objective quantification |
| Operator fatigue effects | No degradation over time |

---

## 🎯 Key Findings

Our systematic benchmark of 12 model configurations revealed:

1. **YOLOv8-l outperformed YOLOv11-l** despite YOLOv11 being the newer architecture (mAP₅₀: 0.866 vs 0.797)

2. **Transformer-based RT-DETR failed** on this small-dataset regime (mAP₅₀ < 0.25)

3. **Application-specific threshold calibration is essential** — optimal operating point (conf=0.55, IoU=0.4) differed substantially from defaults

4. **Achieved MAE of 11.75%** in colonization estimation, suitable for routine laboratory monitoring

---

## 📁 Repository Structure

```
ATENEA-methodology/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
│
├── docs/
│   ├── METHODOLOGY.md                 # Complete methodology walkthrough
│   ├── DATASET_PREPARATION.md         # How to prepare your dataset
│   ├── ANNOTATION_GUIDELINES.md       # Annotation protocol for microcarriers
│   ├── MODEL_SELECTION_SHA.md         # SHA algorithm explanation
│   ├── THRESHOLD_CALIBRATION.md       # Post-training calibration procedure
│   └── EVALUATION_METRICS.md          # Metrics definitions and interpretation
│
├── configs/
│   ├── training_config.yaml           # Complete training hyperparameters
│   ├── augmentation_config.yaml       # Data augmentation settings
│   ├── sha_config.yaml                # SHA model selection parameters
│   ├── inference_config.yaml          # Inference settings (calibrated thresholds)
│   └── hardware_config.yaml           # Hardware specifications used
│
├── src/
│   ├── train.py                       # Training script template
│   ├── evaluate.py                    # Evaluation script template
│   ├── inference.py                   # Inference script template
│   ├── sha_model_selection.py         # SHA implementation
│   ├── threshold_calibration.py       # Threshold optimization
│   ├── colonization_metrics.py        # Colonization calculation
│   └── utils/
│       ├── data_utils.py              # Data loading utilities
│       └── visualization.py           # Visualization functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA workflow
│   ├── 02_training_workflow.ipynb     # Training demonstration
│   └── 03_evaluation_analysis.ipynb   # Results analysis
│
└── examples/
    ├── sample_annotation_format.txt   # Example YOLO annotation format
    ├── data_yaml_template.yaml        # Dataset configuration template
    └── expected_outputs/              # Example output formats
```

---

## 🔧 Methodology

### Dataset Characteristics

| Attribute | Value |
|-----------|-------|
| Total images | 699 |
| Total annotations | 46,982 microcarriers |
| Class distribution | 50.18% EMPTY, 49.82% FULL |
| Train/Val/Test split | 70% / 15% / 15% |
| Image resolution | Variable (rescaled to 640×640) |

### Model Selection via SHA

We employed a **Successive Halving Algorithm (SHA)** to efficiently identify the best architecture:

```
Stage 1 (20 epochs):  12 models → Top 4 promoted
Stage 2 (50 epochs):   4 models → Top 2 promoted  
Stage 3 (150 epochs):  2 models → Winner: YOLOv8-l
```

This reduced computational cost by ~60% compared to exhaustive training.

### Best Model Performance

| Metric | Validation | Test |
|--------|------------|------|
| mAP₅₀ | 0.866 | 0.796 |
| mAP₅₀₋₉₅ | 0.856 | 0.787 |
| Precision | 0.762 | 0.823 |
| Recall | 0.855 | 0.761 |
| F1-Score | — | 0.791 |
| **Colonization MAE** | 10.71% | **11.75%** |

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.10+
pip install ultralytics==8.3.225
pip install torch==2.6.0
pip install opencv-python numpy pandas matplotlib seaborn
```

### Hardware Requirements

Our experiments were conducted on modest hardware:

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA T400 (4 GB VRAM) |
| CUDA | 12.4 |
| Precision | FP16 (mixed precision) |

> **Note:** All models required <0.5 GB GPU memory, enabling deployment on standard laboratory computers.

### Quick Start

1. **Prepare your dataset** following [DATASET_PREPARATION.md](docs/DATASET_PREPARATION.md)

2. **Annotate images** using [ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md)

3. **Configure training** by editing `configs/training_config.yaml`

4. **Run SHA model selection:**
```bash
python src/sha_model_selection.py --config configs/sha_config.yaml
```

5. **Calibrate thresholds:**
```bash
python src/threshold_calibration.py --model best.pt --data val_images/
```

6. **Run inference:**
```bash
python src/inference.py --model best.pt --source your_images/ --conf 0.55 --iou 0.4
```

---

## ⚙️ Configuration Files

All hyperparameters are documented in YAML files for transparency and reproducibility:

### Training Configuration (`configs/training_config.yaml`)

```yaml
# Core parameters
image_size: 640
batch_size: 2
epochs: 150  # Final stage

# Optimizer
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0005
lr_final: 0.00001  # 1% of initial

# Schedule
warmup_epochs: 3
lr_scheduler: cosine

# Precision
mixed_precision: true  # FP16
```

### Calibrated Inference (`configs/inference_config.yaml`)

```yaml
# Optimized via grid search on validation set
confidence_threshold: 0.55
iou_threshold: 0.4
image_size: 640
```

See [configs/](configs/) for complete configuration files.

---

## 📊 Reproducing Results

While the original dataset cannot be shared, researchers can replicate our methodology:

1. **Collect fluorescence microscopy images** of microcarrier cultures
2. **Annotate using our guidelines** (see [ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md))
3. **Apply the SHA model selection** to identify optimal architecture for your data
4. **Calibrate inference thresholds** for your specific application

The methodology is architecture-agnostic and can be applied to similar biomedical detection tasks.

---

## 📖 Citation

If you use this methodology in your research, please cite:

```bibtex
@article{fernandez2025atenea,
  title={Comparative evaluation of YOLOv8, YOLOv11, and RT-DETR for automated microcarrier colonization assessment in bioreactor cell cultures},
  author={Fernández de Sevilla, Manuel T. and Valero, Raúl and Menéndez, Beatriz and García, Gonzalo and Blanco, Cristina and Canales, Natalia and Menta, Ramón and Costa, Marta H. G. and Serra, Margarida and Lombardo, Eleuterio and Avivar, Alvaro and Mancheño, Pablo},
  journal={Frontiers in Bioengineering and Biotechnology},
  year={2025},
  doi={10.3389/fbioe.2025.XXXXXX}
}
```

---

## 📧 Contact

For questions about the methodology or potential academic collaborations:

- **Manuel T. Fernández de Sevilla** — manu1torres@gmail.com
- **Pablo Mancheño** — pablomancheno@hotmail.com

**Institution:** Takeda Cell Therapy Technology Center, Tres Cantos, Madrid, Spain

---

## 📄 License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Takeda Cell Therapy Technology Center for funding and computational resources
- iBET (Instituto de Biologia Experimental e Tecnológica) for collaborative research support
- DD&T department at Takeda for valuable contributions

---

<p align="center">
  <i>Developed at Takeda CTTC · Tres Cantos, Madrid · 2025</i>
</p>
