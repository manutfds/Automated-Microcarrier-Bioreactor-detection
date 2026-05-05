# =============================================================================
# ATENEA Project - SHA Model Selection Configuration
# =============================================================================
# Configuration for Successive Halving Algorithm (SHA) used to efficiently
# identify the best-performing model architecture.
#
# SHA is a principled early-stopping method that progressively allocates
# computational resources to the most promising candidates.
#
# Reference: Zhang & Duh (2024). Best Practices of Successive Halving
# =============================================================================

# -----------------------------------------------------------------------------
# SHA Parameters
# -----------------------------------------------------------------------------
sha:
  # Reduction factor (η)
  # Top 1/η models are promoted at each stage
  eta: 3
  
  # Selection metric for ranking models
  # Options: mAP50, mAP50-95, precision, recall, f1
  ranking_metric: mAP50-95
  
  # Higher is better for the ranking metric
  metric_mode: max

# -----------------------------------------------------------------------------
# Candidate Models
# -----------------------------------------------------------------------------
candidates:
  # YOLOv8 family
  yolov8:
    - yolov8n
    - yolov8s
    - yolov8m
    - yolov8l
    - yolov8x
  
  # YOLOv11 family
  yolo11:
    - yolo11n
    - yolo11s
    - yolo11m
    - yolo11l
    - yolo11x
  
  # RT-DETR family
  rtdetr:
    - rtdetr-l
    - rtdetr-x

# Total: 12 candidate models

# -----------------------------------------------------------------------------
# Stage Configuration
# -----------------------------------------------------------------------------
stages:
  # Stage 1: Initial screening
  stage_1:
    epochs: 20
    num_models_in: 12    # All candidates
    num_models_out: 4    # Top 4 promoted (12 / η ≈ 4)
    description: "Initial screening - quick elimination of weak performers"
  
  # Stage 2: Intermediate evaluation
  stage_2:
    epochs: 50
    num_models_in: 4
    num_models_out: 2    # Top 2 promoted (4 / η ≈ 2)
    description: "Extended training to refine rankings"
  
  # Stage 3: Final evaluation
  stage_3:
    epochs: 150
    num_models_in: 2
    num_models_out: 1    # Winner
    description: "Full training budget for final selection"

# -----------------------------------------------------------------------------
# Training Protocol for Each Stage
# -----------------------------------------------------------------------------
stage_training:
  # IMPORTANT: Each stage restarts from pretrained weights
  # We do NOT continue from previous stage checkpoints
  restart_from_pretrained: true
  
  # Common hyperparameters across stages
  batch_size: 2
  image_size: 640
  optimizer: AdamW
  lr0: 0.001
  weight_decay: 0.0005
  amp: true
  seed: 0
  deterministic: true

# -----------------------------------------------------------------------------
# Computational Budget
# -----------------------------------------------------------------------------
budget:
  # Estimated training times per model (on NVIDIA T400)
  # These are rough estimates - actual times vary by model size
  
  stage_1_hours_per_model:
    yolo_nano: 0.5
    yolo_small: 1.0
    yolo_medium: 2.2
    yolo_large: 3.0
    yolo_xlarge: 10.0
    rtdetr_large: 4.5
    rtdetr_xlarge: 12.5
  
  # Total estimated budget
  stage_1_total_hours: 50    # 12 models × variable times
  stage_2_total_hours: 65    # 4 models × ~16h average
  stage_3_total_hours: 45    # 2 models × ~22h average
  
  # Total project hours
  total_hours: 160
  
  # Savings vs exhaustive training (all 12 models for 150 epochs)
  savings_percent: 60

# -----------------------------------------------------------------------------
# Expected Results (from our experiments)
# -----------------------------------------------------------------------------
expected_results:
  stage_1:
    # Models promoted to Stage 2
    promoted:
      - model: yolov8l
        mAP50-95: 0.577
      - model: yolov8x
        mAP50-95: 0.568
      - model: yolo11l
        mAP50-95: 0.566
      - model: yolo11x
        mAP50-95: 0.563
    
    # Models eliminated
    eliminated:
      - model: yolo11m
        mAP50-95: 0.562
      - model: yolov8m
        mAP50-95: 0.548
      - model: yolov8s
        mAP50-95: 0.545
      - model: yolo11s
        mAP50-95: 0.531
      - model: yolov8n
        mAP50-95: 0.489
      - model: yolo11n
        mAP50-95: 0.488
      - model: rtdetr-l
        mAP50-95: 0.241
      - model: rtdetr-x
        mAP50-95: 0.219
  
  stage_2:
    promoted:
      - model: yolov8l
        mAP50-95: 0.655
      - model: yolo11l
        mAP50-95: 0.641
    eliminated:
      - model: yolov8x
        mAP50-95: 0.637
      - model: yolo11x
        mAP50-95: 0.631
  
  stage_3:
    winner:
      model: yolov8l
      mAP50-95: 0.856
      mAP50: 0.866
    runner_up:
      model: yolo11l
      mAP50-95: 0.790
      mAP50: 0.797

# -----------------------------------------------------------------------------
# Key Observations
# -----------------------------------------------------------------------------
observations:
  - "RT-DETR failed completely (mAP < 0.25) due to data-hungry nature"
  - "YOLOv8-l outperformed YOLOv11-l despite newer architecture"
  - "Performance gap widened with extended training (0.014 → 0.066)"
  - "Larger models (x variants) did not justify computational cost"
  - "SHA reduced total compute by ~60% vs exhaustive search"
