# ATENEA Methodology: Complete Technical Documentation

This document provides a comprehensive, step-by-step description of the methodology used in the ATENEA project for automated microcarrier colonization detection.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Annotation Protocol](#3-annotation-protocol)
4. [Model Architectures](#4-model-architectures)
5. [Training Protocol](#5-training-protocol)
6. [Model Selection via SHA](#6-model-selection-via-sha)
7. [Threshold Calibration](#7-threshold-calibration)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [Colonization Calculation](#9-colonization-calculation)
10. [Error Analysis](#10-error-analysis)

---

## 1. Problem Statement

### Background

Mesenchymal stromal cell (MSC) expansion in bioreactors using microcarriers is a critical step in cell therapy manufacturing. Monitoring colonization—the percentage of microcarriers with at least one cell attached—is essential for process control.

### Current Limitations of Manual Quantification

| Challenge | Impact |
|-----------|--------|
| Time-consuming | ~10 minutes per image |
| Subjective | Dependent on operator interpretation |
| Variable | >15% inter-operator variability |
| Fatiguing | Quality degrades over work sessions |

### Our Solution

An automated deep learning pipeline that:
- Detects individual microcarriers in fluorescence microscopy images
- Classifies each as FULL (colonized) or EMPTY (non-colonized)
- Calculates colonization percentage in seconds

---

## 2. Dataset Preparation

### 2.1 Image Acquisition

**Microscopy Setup:**
- Fluorescence-inverted microscope (e.g., Leica DMI6000)
- Green channel for cell visualization
- Fluorescein diacetate staining (20 µg/mL)

**Image Characteristics:**
- Format: JPEG, PNG, or TIFF
- Variable native resolution (standardized to 640×640 for training)
- Single focal plane recommended

### 2.2 Dataset Composition

Our dataset included:
- **699 total images**
- Multiple bioreactor runs
- Different cell donors
- Various culture days
- Multiple fields of view per sampling point

### 2.3 Data Splitting

```
Total: 699 images
├── Training:   489 images (70%)
├── Validation: 105 images (15%)
└── Test:       105 images (15%)
```

**Split Strategy:**
- Random assignment at image level
- Verified similar colonization distributions across splits
- No data leakage due to experimental diversity

### 2.4 YOLO Dataset Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   ├── img002.txt
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── data.yaml
```

### 2.5 data.yaml Configuration

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 2  # Number of classes
names:
  0: EMPTY
  1: FULL
```

---

## 3. Annotation Protocol

### 3.1 Class Definitions

| Class | ID | Description | Visual Criteria |
|-------|-----|-------------|-----------------|
| **EMPTY** | 0 | Non-colonized microcarrier | No internal green fluorescent signal |
| **FULL** | 1 | Colonized microcarrier | Visible green fluorescent signal inside |

### 3.2 Annotation Guidelines

**Include:**
- All fully visible microcarriers
- Microcarriers with any degree of cell coverage (even partial)

**Exclude:**
- Microcarriers cropped at image borders
- Debris or artifacts
- Out-of-focus objects that cannot be classified

### 3.3 YOLO Annotation Format

Each annotation file (`.txt`) contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

**All coordinates are normalized (0-1):**
- `x_center`: Center X / Image width
- `y_center`: Center Y / Image height
- `width`: Box width / Image width
- `height`: Box height / Image height

**Example:**
```
0 0.4521 0.3125 0.0856 0.0923
1 0.7234 0.5678 0.0912 0.0887
0 0.1567 0.8234 0.0834 0.0901
```

### 3.4 Annotation Tool

We used **LabelImg** (version 1.8.6) with YOLO format export.

**Alternative tools:**
- CVAT
- Roboflow
- Label Studio

### 3.5 Quality Control

- **Multiple annotators** (5 laboratory experts)
- **Consensus-based review** for ambiguous cases
- **EDA verification** to detect annotation inconsistencies

---

## 4. Model Architectures

### 4.1 Models Evaluated

| Family | Variants | Type | Key Features |
|--------|----------|------|--------------|
| **YOLOv8** | n, s, m, l, x | CNN | Anchor-free, decoupled head, C2f blocks |
| **YOLOv11** | n, s, m, l, x | CNN | C3k2 blocks, C2PSA attention modules |
| **RT-DETR** | l, x | Transformer | End-to-end, no NMS, hybrid encoder |

### 4.2 Model Complexity

| Model | Parameters (M) | GFLOPs | GPU Memory (GB) |
|-------|---------------|--------|-----------------|
| YOLOv8-n | 3.16 | 8.9 | 0.07 |
| YOLOv8-s | 11.17 | 28.8 | 0.11 |
| YOLOv8-m | 25.90 | 79.3 | 0.19 |
| **YOLOv8-l** | **43.69** | **165.7** | **0.28** |
| YOLOv8-x | 68.23 | 258.5 | 0.40 |
| YOLOv11-n | 2.62 | 6.6 | 0.10 |
| YOLOv11-s | 9.46 | 21.7 | 0.14 |
| YOLOv11-m | 20.11 | 68.5 | 0.21 |
| YOLOv11-l | 25.37 | 87.6 | 0.24 |
| YOLOv11-x | 56.97 | 196.0 | 0.41 |
| RT-DETR-l | 32.97 | 108.3 | 0.28 |
| RT-DETR-x | 67.47 | 232.7 | 0.44 |

### 4.3 Pretrained Initialization

All models initialized with **COCO-pretrained weights** from Ultralytics:
- Accelerates convergence
- Provides robust low-level features
- Reduces overfitting risk on small datasets

---

## 5. Training Protocol

### 5.1 Hardware Configuration

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA T400 (4 GB VRAM) |
| Framework | PyTorch 2.6.0 |
| CUDA | 12.4 |
| Training Framework | Ultralytics 8.3.225 |
| Python | 3.10.19 |

### 5.2 Training Hyperparameters

```yaml
# Image
image_size: 640

# Batch
batch_size: 2  # Limited by GPU memory

# Optimizer
optimizer: AdamW
learning_rate: 0.001      # Initial
weight_decay: 0.0005
final_lr_ratio: 0.01      # Final LR = 0.00001

# Schedule
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.1
lr_scheduler: cosine

# Precision
mixed_precision: true     # FP16 via AMP

# Reproducibility
seed: 0
deterministic: true
```

### 5.3 Data Augmentation

```yaml
# Geometric
horizontal_flip: 0.5      # Probability
rotation: 15              # Degrees (±)
scale: 0.8                # Factor

# Color
hsv_h: 0.015              # Hue adjustment (±1.5%)
hsv_s: 0.7                # Saturation adjustment (±70%)
hsv_v: 0.4                # Value adjustment (±40%)

# Advanced
mosaic: 1.0               # Enabled
close_mosaic: 10          # Disable last 10 epochs
erasing: 0.4              # Random erasing probability

# Auto augmentation
auto_augment: randaugment
```

### 5.4 Loss Functions

YOLOv8/v11 uses a combination of:
- **Box Loss:** CIoU (Complete IoU)
- **Classification Loss:** Binary Cross-Entropy
- **Distribution Focal Loss (DFL):** For precise bounding box regression

---

## 6. Model Selection via SHA

### 6.1 Successive Halving Algorithm (SHA)

SHA is a principled early-stopping method that:
1. Trains all candidates for a small budget
2. Evaluates on validation metric
3. Promotes top performers to next stage with larger budget
4. Repeats until final selection

**Key advantage:** ~60% computational savings vs. exhaustive training

### 6.2 SHA Configuration

```yaml
# Reduction factor
eta: 3  # Promote top 1/3 at each stage

# Stages
stages:
  - name: stage_1
    epochs: 20
    models: 12
    promote: 4
    
  - name: stage_2
    epochs: 50
    models: 4
    promote: 2
    
  - name: stage_3
    epochs: 150
    models: 2
    promote: 1  # Winner

# Selection metric
metric: mAP50-95
```

### 6.3 SHA Results

**Stage 1 (20 epochs) — 12 models:**
| Model | mAP₅₀₋₉₅ | Time (h) | Status |
|-------|----------|----------|--------|
| YOLOv8-l | 0.577 | 3.22 | ✓ Promoted |
| YOLOv8-x | 0.568 | 10.13 | ✓ Promoted |
| YOLOv11-l | 0.566 | 2.71 | ✓ Promoted |
| YOLOv11-x | 0.563 | 10.30 | ✓ Promoted |
| YOLOv11-m | 0.562 | 2.25 | ✗ |
| YOLOv8-m | 0.548 | 2.18 | ✗ |
| YOLOv8-s | 0.545 | 1.02 | ✗ |
| YOLOv11-s | 0.531 | 1.07 | ✗ |
| YOLOv8-n | 0.489 | 0.46 | ✗ |
| YOLOv11-n | 0.488 | 0.51 | ✗ |
| RT-DETR-l | 0.241 | 4.42 | ✗ |
| RT-DETR-x | 0.219 | 12.48 | ✗ |

**Stage 2 (50 epochs) — 4 models:**
| Model | mAP₅₀₋₉₅ | Δ from Stage 1 | Status |
|-------|----------|----------------|--------|
| YOLOv8-l | 0.655 | +0.078 | ✓ Promoted |
| YOLOv11-l | 0.641 | +0.075 | ✓ Promoted |
| YOLOv8-x | 0.637 | +0.069 | ✗ |
| YOLOv11-x | 0.631 | +0.068 | ✗ |

**Stage 3 (150 epochs) — 2 models:**
| Model | mAP₅₀₋₉₅ | Δ from Stage 2 | Status |
|-------|----------|----------------|--------|
| **YOLOv8-l** | **0.856** | +0.201 | 🏆 Winner |
| YOLOv11-l | 0.790 | +0.149 | — |

### 6.4 Implementation Note

Each stage **restarts training from COCO pretrained weights**, not from previous checkpoints. This ensures:
- Full benefit of learning rate warmup at each stage
- Complete cosine annealing schedule
- Fair comparison across stages

---

## 7. Threshold Calibration

### 7.1 Why Calibration?

Default detection thresholds are optimized for general benchmarks (COCO), not domain-specific applications. For colonization estimation, we need to minimize **colonization error**, not maximize detection metrics.

### 7.2 Calibration Procedure

```python
# Grid search over threshold combinations
confidence_values = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
iou_values = [0.3, 0.4, 0.5, 0.6, 0.7]

# Total combinations evaluated: 44

# Selection criterion
optimal = argmin(MAE_colonization)
```

### 7.3 Calibration Results

| Threshold Combo | Validation MAE | Notes |
|-----------------|---------------|-------|
| conf=0.25, iou=0.5 | 14.23% | Too many FPs |
| conf=0.55, iou=0.4 | **10.71%** | **Optimal** |
| conf=0.70, iou=0.5 | 12.89% | Too many FNs |

**Selected Operating Point:**
- Confidence threshold: **0.55**
- IoU threshold: **0.4**

### 7.4 Important Notes

- Calibration is performed **only on validation set**
- Model weights are **not modified**
- Calibrated thresholds are then applied to test set

---

## 8. Evaluation Protocol

### 8.1 Detection Metrics

**Mean Average Precision (mAP):**
- **mAP₅₀:** AP at IoU threshold 0.5
- **mAP₅₀₋₉₅:** Average AP across IoU 0.5 to 0.95 (step 0.05)

**Per-class metrics:**
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** 2 × (P × R) / (P + R)

### 8.2 Colonization Metrics

**Mean Absolute Error (MAE):**
```
MAE = (1/N) × Σ|predicted_col - gt_col|
```

**Mean Absolute Percentage Error (MAPE):**
```
MAPE = (100/N) × Σ|predicted_col - gt_col| / gt_col
```

**Bias (Systematic Error):**
```
Bias = (1/N) × Σ(predicted_col - gt_col)
```

**Coefficient of Determination (R²):**
```
R² = 1 - (SS_res / SS_tot)
```

### 8.3 Final Test Results

| Metric | Value |
|--------|-------|
| mAP₅₀ | 0.796 |
| mAP₅₀₋₉₅ | 0.787 |
| Precision | 0.823 |
| Recall | 0.761 |
| F1-Score | 0.791 |
| **Colonization MAE** | **11.75%** |
| Colonization Bias | -5.28% |
| R² | 0.316 |

---

## 9. Colonization Calculation

### 9.1 Formula

```
Colonization (%) = (N_FULL / (N_FULL + N_EMPTY)) × 100
```

Where:
- `N_FULL`: Number of detected colonized microcarriers
- `N_EMPTY`: Number of detected non-colonized microcarriers

### 9.2 Implementation

```python
def calculate_colonization(detections):
    """
    Calculate colonization percentage from detections.
    
    Args:
        detections: List of detections with class_id
                   (0 = EMPTY, 1 = FULL)
    
    Returns:
        float: Colonization percentage (0-100)
    """
    n_full = sum(1 for d in detections if d['class_id'] == 1)
    n_empty = sum(1 for d in detections if d['class_id'] == 0)
    
    total = n_full + n_empty
    
    if total == 0:
        return 0.0
    
    return (n_full / total) * 100
```

---

## 10. Error Analysis

### 10.1 Error Categories

| Category | Definition |
|----------|------------|
| **True Positive (TP)** | Correct detection with correct class |
| **False Positive (FP)** | Detection without corresponding GT |
| **False Negative (FN)** | GT without corresponding detection |
| **Misclassification** | Correct detection, wrong class |

### 10.2 Error Distribution (Test Set)

| Category | Count | Percentage |
|----------|-------|------------|
| True Positives | 5,497 | 67.7% |
| False Positives | 878 | 10.8% |
| False Negatives | 814 | 10.0% |
| Misclassifications | 930 | 11.5% |

### 10.3 Misclassification Direction

| Direction | Count | Implication |
|-----------|-------|-------------|
| FULL → EMPTY | 582 | Underestimation of colonization |
| EMPTY → FULL | 348 | Overestimation of colonization |

**Net effect:** Systematic underestimation bias (-5.28%)

### 10.4 Error Sources

**Common causes of errors:**
1. **Weak fluorescence:** Low-intensity FULL microcarriers misclassified as EMPTY
2. **Partial colonization:** Ambiguous cases at classification boundary
3. **Overlapping objects:** Adjacent microcarriers with merged signals
4. **Edge effects:** Partially visible microcarriers at image borders
5. **Out-of-focus regions:** Reduced signal clarity in certain focal planes

### 10.5 Practical Implications

- Model predictions should be interpreted as **conservative lower bounds**
- Additional verification recommended for high-colonization samples (>80%)
- Consistent bias allows for potential post-hoc correction if needed

---

## Summary

This methodology enables automated, reproducible quantification of microcarrier colonization with:

- **~100× speedup** over manual analysis
- **11.75% MAE** in colonization estimation
- **Elimination** of operator variability
- **Deployment** on standard laboratory hardware

The approach is generalizable to other biomedical detection tasks with appropriate dataset preparation and threshold calibration.

---

*For questions, contact: manu1torres@gmail.com or pablomancheno@hotmail.com*
