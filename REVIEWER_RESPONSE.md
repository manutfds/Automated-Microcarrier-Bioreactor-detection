# =============================================================================
# ATENEA Project - Inference Configuration
# =============================================================================
# Calibrated inference parameters optimized for colonization estimation.
#
# IMPORTANT: These thresholds were calibrated specifically for the
# microcarrier colonization task using grid search on the validation set.
# Default YOLO thresholds are NOT optimal for this application.
# =============================================================================

# -----------------------------------------------------------------------------
# Calibrated Thresholds
# -----------------------------------------------------------------------------
thresholds:
  # Confidence threshold
  # Detections below this confidence are discarded
  # Calibrated value: 0.55 (higher than default 0.25)
  confidence: 0.55
  
  # IoU threshold for NMS
  # Overlapping detections above this IoU are suppressed
  # Calibrated value: 0.4 (lower than default 0.5)
  iou: 0.4

# -----------------------------------------------------------------------------
# Calibration Details
# -----------------------------------------------------------------------------
calibration:
  # Method used
  method: grid_search
  
  # Search space
  confidence_values: [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
  iou_values: [0.3, 0.4, 0.5, 0.6, 0.7]
  
  # Total combinations evaluated
  total_combinations: 44
  
  # Optimization criterion
  criterion: minimize_colonization_MAE
  
  # Validation set results at optimal thresholds
  validation_results:
    MAE: 10.71
    R2: 0.37
    bias: -4.24

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
model:
  # Best performing model
  architecture: yolov8l
  
  # Weights file
  weights: best.pt
  
  # Input size
  image_size: 640
  
  # Precision
  half: true  # FP16 inference

# -----------------------------------------------------------------------------
# Inference Parameters
# -----------------------------------------------------------------------------
inference:
  # Batch size for inference
  batch_size: 1
  
  # Device
  device: 0  # GPU index or 'cpu'
  
  # Verbose output
  verbose: false
  
  # Visualization
  save_visualizations: true
  line_width: 2
  
  # Class colors (BGR for OpenCV)
  colors:
    EMPTY: [255, 150, 50]   # Blue
    FULL: [50, 205, 50]     # Green

# -----------------------------------------------------------------------------
# Output Configuration
# -----------------------------------------------------------------------------
output:
  # Save format
  save_txt: true           # Save predictions as .txt
  save_conf: true          # Include confidence in .txt
  save_crop: false         # Save cropped detections
  
  # Colonization output
  save_colonization_csv: true
  
  # Columns in output CSV
  csv_columns:
    - image_name
    - n_empty
    - n_full
    - total_detections
    - colonization_percent
    - inference_time_ms

# -----------------------------------------------------------------------------
# Expected Performance
# -----------------------------------------------------------------------------
expected_performance:
  # Detection metrics (test set)
  detection:
    mAP50: 0.796
    mAP50-95: 0.787
    precision: 0.823
    recall: 0.761
    f1_score: 0.791
  
  # Colonization metrics (test set)
  colonization:
    MAE: 11.75
    MAPE: 27.92
    bias: -5.28
    R2: 0.316
    RMSE: 20.748
    pearson_r: 0.6548
  
  # Computational performance (NVIDIA T400)
  latency:
    mean_ms: 137.1
    std_ms: 0.9
    fps: 7.3
    gpu_memory_gb: 0.28

# -----------------------------------------------------------------------------
# Threshold Sensitivity Analysis
# -----------------------------------------------------------------------------
threshold_sensitivity:
  # Effect of confidence threshold (at IoU=0.4)
  confidence_sweep:
    - conf: 0.25
      precision: 0.65
      recall: 0.91
      MAE: 14.2
    - conf: 0.40
      precision: 0.75
      recall: 0.84
      MAE: 12.1
    - conf: 0.55
      precision: 0.82
      recall: 0.76
      MAE: 11.75  # Optimal
    - conf: 0.70
      precision: 0.88
      recall: 0.63
      MAE: 14.8
  
  # Key insight
  notes: >
    Lower confidence increases recall but also FPs.
    Higher confidence increases precision but misses detections.
    Optimal point (0.55) minimizes colonization estimation error.

# -----------------------------------------------------------------------------
# Usage Example
# -----------------------------------------------------------------------------
usage:
  python: |
    from ultralytics import YOLO
    
    # Load model
    model = YOLO('best.pt')
    
    # Run inference with calibrated thresholds
    results = model.predict(
        source='images/',
        conf=0.55,      # Calibrated confidence
        iou=0.4,        # Calibrated IoU
        imgsz=640,
        half=True,      # FP16 inference
        device=0
    )
    
    # Calculate colonization
    for r in results:
        classes = r.boxes.cls.cpu().numpy()
        n_full = (classes == 1).sum()
        n_empty = (classes == 0).sum()
        colonization = n_full / (n_full + n_empty) * 100
        print(f"Colonization: {colonization:.1f}%")
  
  cli: |
    yolo detect predict \
      model=best.pt \
      source=images/ \
      conf=0.55 \
      iou=0.4 \
      imgsz=640 \
      half=True
