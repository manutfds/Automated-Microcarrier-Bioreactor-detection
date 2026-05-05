# ATENEA: Automated Microcarrier Colonization Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3.225-purple.svg)](https://ultralytics.com/)

**Methodology repository** for the paper:

> *"Comparative evaluation of YOLOv8, YOLOv11, and RT-DETR for automated microcarrier colonization assessment in bioreactor cell cultures"*
>
> Fernández de Sevilla, M.T., Valero, R., et al. — Frontiers in Bioengineering and Biotechnology (2025)

---

## 🎯 Purpose

This repository provides **complete methodological documentation** to enable reproducibility of our approach. Due to institutional IP restrictions, the original dataset and trained weights cannot be released. However, researchers can replicate our methodology on their own microcarrier datasets using the materials provided here.

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Best Model | YOLOv8-l |
| Test mAP₅₀ | 0.796 |
| Test Precision | 0.823 |
| Test Recall | 0.761 |
| **Colonization MAE** | **11.75%** |
| Processing Time | ~137 ms/image |

**Key finding:** YOLOv8-l outperformed YOLOv11-l despite newer architecture. RT-DETR failed on this small-dataset regime.

---

## 📁 Repository Structure

```
├── docs/
│   ├── METHODOLOGY.md           # Complete methodology (10 sections)
│   └── ANNOTATION_GUIDELINES.md # How to annotate microcarriers
│
├── configs/
│   ├── training_config.yaml     # All training hyperparameters
│   ├── sha_config.yaml          # SHA model selection configuration
│   └── inference_config.yaml    # Calibrated thresholds (conf=0.55, iou=0.4)
│
├── src/
│   ├── sha_model_selection.py   # Successive Halving implementation
│   └── threshold_calibration.py # Threshold optimization script
│
├── examples/
│   ├── data.yaml                # Dataset configuration template
│   └── sample_annotations.txt   # Example YOLO format annotations
│
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset

Follow [ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md) to annotate your microcarrier images, then organize as:

```
your_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

See [examples/data.yaml](examples/data.yaml) for the configuration template.

### 3. Run SHA model selection

```bash
python src/sha_model_selection.py --data your_dataset/data.yaml --output results/
```

### 4. Calibrate thresholds

```bash
python src/threshold_calibration.py \
    --model results/best.pt \
    --val_images your_dataset/images/val \
    --val_labels your_dataset/labels/val
```

### 5. Run inference with Ultralytics

```bash
yolo detect predict model=best.pt source=images/ conf=0.55 iou=0.4
```

---

## ⚙️ Key Parameters

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Image size | 640×640 |
| Batch size | 2 |
| Optimizer | AdamW |
| Learning rate | 0.001 → 0.00001 |
| Epochs (final) | 150 |
| Precision | FP16 |

### Calibrated Inference Thresholds

| Parameter | Value | Note |
|-----------|-------|------|
| Confidence | **0.55** | Higher than default (0.25) |
| IoU (NMS) | **0.4** | Lower than default (0.5) |

These thresholds were optimized via grid search to minimize colonization estimation error.

---

## 📈 SHA Model Selection

We used **Successive Halving Algorithm (SHA)** to efficiently identify the best architecture:

```
Stage 1 (20 ep):  12 models → 4 promoted
Stage 2 (50 ep):   4 models → 2 promoted
Stage 3 (150 ep):  2 models → Winner: YOLOv8-l
```

This reduced computational cost by ~60% vs. training all models for 150 epochs.

See [configs/sha_config.yaml](configs/sha_config.yaml) for complete configuration.

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [METHODOLOGY.md](docs/METHODOLOGY.md) | Complete technical walkthrough (dataset, training, evaluation, error analysis) |
| [ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md) | How to annotate FULL vs EMPTY microcarriers |

---

## ⚠️ Limitations

1. **No public dataset** — Original images are proprietary; methodology must be applied to your own data
2. **Single laboratory** — Results validated on images from one microscopy setup
3. **Systematic bias** — Model tends to slightly underestimate colonization (-5.28% bias)

---

## 🔧 Hardware Used

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA T400 (4 GB) |
| Framework | PyTorch 2.6.0, Ultralytics 8.3.225 |
| CUDA | 12.4 |

All models required <0.5 GB GPU memory, enabling deployment on standard laboratory computers.

---

## 📝 Citation

```bibtex
@article{fernandez2025microcarrier,
  title={Comparative evaluation of YOLOv8, YOLOv11, and RT-DETR for 
         automated microcarrier colonization assessment in bioreactor 
         cell cultures},
  author={Fernández de Sevilla, Manuel T. and Valero, Raúl and 
          Menéndez, Beatriz and García, Gonzalo and others},
  journal={Frontiers in Bioengineering and Biotechnology},
  year={2025}
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

[MIT License](LICENSE) — Free to use with attribution.

---

## 🙏 Acknowledgments

- Takeda Cell Therapy Technology Center for funding and computational resources
- iBET (Instituto de Biologia Experimental e Tecnológica) for collaborative research support
