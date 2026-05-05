# =============================================================================
# ATENEA Project - Training Configuration
# =============================================================================
# Complete hyperparameter configuration for YOLO model training
# as used in the microcarrier colonization detection study.
#
# Reference: Fernández de Sevilla et al. (2025)
# =============================================================================

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
model:
  # Supported: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  #            yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
  #            rtdetr-l, rtdetr-x
  architecture: yolov8l
  
  # Pretrained weights (recommended for transfer learning)
  pretrained: true
  weights: yolov8l.pt  # COCO pretrained
  
  # Number of classes
  nc: 2
  
  # Class names
  names:
    0: EMPTY
    1: FULL

# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------
dataset:
  # Path to data.yaml
  data_yaml: /path/to/data.yaml
  
  # Image size (square, will be resized)
  image_size: 640
  
  # Data splits (percentages)
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
training:
  # Epochs
  epochs: 150           # Final training (Stage 3 of SHA)
  
  # Batch size (limited by GPU memory)
  batch_size: 2
  
  # Optimizer
  optimizer: AdamW
  
  # Learning rate
  lr0: 0.001            # Initial learning rate
  lrf: 0.01             # Final LR = lr0 * lrf = 0.00001
  
  # Momentum (for SGD, not used with AdamW)
  momentum: 0.937
  
  # Weight decay
  weight_decay: 0.0005
  
  # Warmup
  warmup_epochs: 3
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1
  
  # Precision
  amp: true             # Automatic Mixed Precision (FP16)
  
  # Reproducibility
  seed: 0
  deterministic: true
  
  # Workers for data loading
  workers: 8

# -----------------------------------------------------------------------------
# Data Augmentation
# -----------------------------------------------------------------------------
augmentation:
  # Geometric transformations
  hsv_h: 0.015          # Hue augmentation (fraction)
  hsv_s: 0.7            # Saturation augmentation (fraction)
  hsv_v: 0.4            # Value augmentation (fraction)
  
  degrees: 15.0         # Rotation (±degrees)
  translate: 0.1        # Translation (fraction)
  scale: 0.8            # Scale (gain)
  shear: 0.0            # Shear (degrees)
  perspective: 0.0      # Perspective (fraction)
  
  # Flip
  flipud: 0.0           # Vertical flip probability
  fliplr: 0.5           # Horizontal flip probability
  
  # Mosaic & mixup
  mosaic: 1.0           # Mosaic augmentation probability
  mixup: 0.0            # Mixup augmentation probability
  close_mosaic: 10      # Disable mosaic for final N epochs
  
  # Copy-paste
  copy_paste: 0.0       # Copy-paste augmentation probability
  
  # Random erasing
  erasing: 0.4          # Random erasing probability
  
  # Auto augmentation
  auto_augment: randaugment  # Options: randaugment, autoaugment, augmix

# -----------------------------------------------------------------------------
# Loss Configuration
# -----------------------------------------------------------------------------
loss:
  # Box loss weight
  box: 7.5
  
  # Classification loss weight
  cls: 0.5
  
  # DFL loss weight
  dfl: 1.5

# -----------------------------------------------------------------------------
# Validation Configuration
# -----------------------------------------------------------------------------
validation:
  # Validation frequency
  val_period: 1         # Validate every N epochs
  
  # Save best model based on
  save_best_metric: mAP50-95
  
  # Patience for early stopping (0 = disabled)
  patience: 0           # We use SHA instead

# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
checkpoint:
  # Save checkpoints
  save: true
  
  # Save period (0 = only best and last)
  save_period: 0
  
  # Keep only best checkpoint
  exist_ok: true

# -----------------------------------------------------------------------------
# Hardware Configuration
# -----------------------------------------------------------------------------
hardware:
  # Device
  device: 0             # GPU index or 'cpu'
  
  # Expected GPU memory usage
  expected_gpu_memory_gb: 0.28  # For YOLOv8-l

# -----------------------------------------------------------------------------
# Output Configuration
# -----------------------------------------------------------------------------
output:
  # Project name
  project: runs/detect
  
  # Experiment name
  name: microcarrier_detection
  
  # Verbose output
  verbose: true
  
  # Plot training curves
  plots: true
