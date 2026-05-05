"""
================================================================================
ATENEA Project - Threshold Calibration for Colonization Estimation
================================================================================

This script performs post-training threshold calibration to optimize
confidence and IoU thresholds for colonization estimation accuracy.

Key insight: Default YOLO thresholds are optimized for general detection
benchmarks (COCO), not for domain-specific applications. Calibration on
validation set can significantly improve downstream task performance.

Method:
    Grid search over confidence and IoU threshold combinations,
    selecting the pair that minimizes Mean Absolute Error (MAE)
    of colonization percentage estimation.

Usage:
    python threshold_calibration.py --model best.pt --val_images val/ --val_labels val_labels/

Author: ATENEA Project Team
Institution: Takeda CTTC, Tres Cantos, Madrid
================================================================================
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from itertools import product

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def load_yolo_labels(label_path: Path, img_width: int, img_height: int) -> List[Dict]:
    """
    Load YOLO format labels and convert to absolute coordinates.
    
    Args:
        label_path: Path to label .txt file
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        List of dictionaries with box coordinates and class
    """
    boxes = []
    
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx = float(parts[1]) * img_width
                cy = float(parts[2]) * img_height
                w = float(parts[3]) * img_width
                h = float(parts[4]) * img_height
                
                boxes.append({
                    'class_id': class_id,
                    'x_center': cx,
                    'y_center': cy,
                    'width': w,
                    'height': h
                })
    
    return boxes


def calculate_colonization(detections: List[Dict]) -> float:
    """
    Calculate colonization percentage from detections.
    
    Args:
        detections: List of detection dictionaries with class_id
                   (0 = EMPTY, 1 = FULL)
                   
    Returns:
        Colonization percentage (0-100)
    """
    if not detections:
        return 0.0
    
    n_full = sum(1 for d in detections if d['class_id'] == 1)
    n_empty = sum(1 for d in detections if d['class_id'] == 0)
    total = n_full + n_empty
    
    if total == 0:
        return 0.0
    
    return (n_full / total) * 100


def calculate_gt_colonization(labels: List[Dict]) -> float:
    """Calculate ground truth colonization from labels."""
    return calculate_colonization(labels)


# =============================================================================
# Threshold Calibration
# =============================================================================

class ThresholdCalibrator:
    """
    Calibrate confidence and IoU thresholds for optimal colonization estimation.
    
    Attributes:
        model: YOLO model
        val_images_dir: Directory with validation images
        val_labels_dir: Directory with validation labels
        image_size: Input image size for inference
    """
    
    def __init__(
        self,
        model_path: str,
        val_images_dir: str,
        val_labels_dir: str,
        image_size: int = 640
    ):
        """
        Initialize threshold calibrator.
        
        Args:
            model_path: Path to trained model weights
            val_images_dir: Directory containing validation images
            val_labels_dir: Directory containing validation labels
            image_size: Input size for inference
        """
        logger.info(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        
        self.val_images_dir = Path(val_images_dir)
        self.val_labels_dir = Path(val_labels_dir)
        self.image_size = image_size
        
        # Get image files
        self.image_files = list(self.val_images_dir.glob('*.jpg'))
        self.image_files += list(self.val_images_dir.glob('*.png'))
        
        logger.info(f"Found {len(self.image_files)} validation images")
        
        # Results storage
        self.calibration_results = []
    
    def evaluate_thresholds(
        self,
        conf_threshold: float,
        iou_threshold: float
    ) -> Dict:
        """
        Evaluate a specific threshold combination.
        
        Args:
            conf_threshold: Confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
            
        Returns:
            Dictionary with evaluation metrics
        """
        errors = []
        gt_colonizations = []
        pred_colonizations = []
        
        for img_path in self.image_files:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            
            # Load ground truth
            label_path = self.val_labels_dir / f"{img_path.stem}.txt"
            gt_labels = load_yolo_labels(label_path, img_w, img_h)
            gt_col = calculate_gt_colonization(gt_labels)
            
            # Run inference
            results = self.model.predict(
                img,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=self.image_size,
                verbose=False
            )
            
            # Extract detections
            detections = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    detections.append({
                        'class_id': int(box.cls.item())
                    })
            
            # Calculate predicted colonization
            pred_col = calculate_colonization(detections)
            
            # Store
            gt_colonizations.append(gt_col)
            pred_colonizations.append(pred_col)
            errors.append(pred_col - gt_col)
        
        # Calculate metrics
        errors = np.array(errors)
        abs_errors = np.abs(errors)
        
        metrics = {
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'MAE': np.mean(abs_errors),
            'RMSE': np.sqrt(np.mean(errors**2)),
            'bias': np.mean(errors),
            'std': np.std(errors),
            'max_error': np.max(abs_errors),
            'n_images': len(errors)
        }
        
        # Calculate R²
        if len(gt_colonizations) > 1:
            ss_res = np.sum((np.array(pred_colonizations) - np.array(gt_colonizations))**2)
            ss_tot = np.sum((np.array(gt_colonizations) - np.mean(gt_colonizations))**2)
            metrics['R2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            metrics['R2'] = 0
        
        return metrics
    
    def run_grid_search(
        self,
        conf_values: List[float] = None,
        iou_values: List[float] = None
    ) -> pd.DataFrame:
        """
        Run grid search over threshold combinations.
        
        Args:
            conf_values: List of confidence thresholds to try
            iou_values: List of IoU thresholds to try
            
        Returns:
            DataFrame with results for all combinations
        """
        # Default search space
        if conf_values is None:
            conf_values = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
        
        if iou_values is None:
            iou_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        total_combinations = len(conf_values) * len(iou_values)
        logger.info(f"Running grid search: {total_combinations} combinations")
        
        results = []
        
        for conf, iou in tqdm(
            product(conf_values, iou_values),
            total=total_combinations,
            desc="Calibrating"
        ):
            metrics = self.evaluate_thresholds(conf, iou)
            results.append(metrics)
        
        self.calibration_results = pd.DataFrame(results)
        
        return self.calibration_results
    
    def get_optimal_thresholds(self) -> Tuple[float, float, float]:
        """
        Get optimal thresholds based on minimum MAE.
        
        Returns:
            Tuple of (optimal_conf, optimal_iou, min_MAE)
        """
        if len(self.calibration_results) == 0:
            raise ValueError("Run grid search first")
        
        best_idx = self.calibration_results['MAE'].idxmin()
        best_row = self.calibration_results.loc[best_idx]
        
        return (
            best_row['conf_threshold'],
            best_row['iou_threshold'],
            best_row['MAE']
        )
    
    def plot_results(self, output_dir: str = '.'):
        """
        Generate visualization of calibration results.
        
        Args:
            output_dir: Directory to save plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = self.calibration_results
        
        # Create heatmap
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # MAE heatmap
        pivot_mae = df.pivot(
            index='conf_threshold',
            columns='iou_threshold',
            values='MAE'
        )
        
        ax = axes[0]
        sns.heatmap(
            pivot_mae,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',
            ax=ax
        )
        ax.set_title('MAE (%) by Threshold Combination')
        ax.set_xlabel('IoU Threshold')
        ax.set_ylabel('Confidence Threshold')
        
        # Find and mark optimal
        opt_conf, opt_iou, opt_mae = self.get_optimal_thresholds()
        
        # R² heatmap
        pivot_r2 = df.pivot(
            index='conf_threshold',
            columns='iou_threshold',
            values='R2'
        )
        
        ax = axes[1]
        sns.heatmap(
            pivot_r2,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            ax=ax
        )
        ax.set_title('R² by Threshold Combination')
        ax.set_xlabel('IoU Threshold')
        ax.set_ylabel('Confidence Threshold')
        
        plt.suptitle(
            f'Threshold Calibration Results\n'
            f'Optimal: conf={opt_conf}, iou={opt_iou}, MAE={opt_mae:.2f}%',
            fontsize=12
        )
        plt.tight_layout()
        
        # Save
        plt.savefig(output_dir / 'calibration_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to {output_dir / 'calibration_heatmap.png'}")
    
    def save_results(self, output_dir: str = '.'):
        """
        Save calibration results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        self.calibration_results.to_csv(
            output_dir / 'calibration_grid_search.csv',
            index=False
        )
        
        # Save optimal configuration
        opt_conf, opt_iou, opt_mae = self.get_optimal_thresholds()
        
        optimal_config = {
            'confidence_threshold': opt_conf,
            'iou_threshold': opt_iou,
            'validation_MAE': opt_mae,
            'total_combinations_tested': len(self.calibration_results)
        }
        
        import yaml
        with open(output_dir / 'optimal_thresholds.yaml', 'w') as f:
            yaml.dump(optimal_config, f, default_flow_style=False)
        
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Optimal: conf={opt_conf}, iou={opt_iou}, MAE={opt_mae:.2f}%")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for threshold calibration."""
    
    parser = argparse.ArgumentParser(
        description='Calibrate detection thresholds for colonization estimation'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model weights (best.pt)'
    )
    parser.add_argument(
        '--val_images',
        type=str,
        required=True,
        help='Directory containing validation images'
    )
    parser.add_argument(
        '--val_labels',
        type=str,
        required=True,
        help='Directory containing validation labels'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='calibration_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size for inference'
    )
    
    args = parser.parse_args()
    
    # Initialize calibrator
    calibrator = ThresholdCalibrator(
        model_path=args.model,
        val_images_dir=args.val_images,
        val_labels_dir=args.val_labels,
        image_size=args.imgsz
    )
    
    # Run grid search
    results = calibrator.run_grid_search()
    
    # Get optimal
    opt_conf, opt_iou, opt_mae = calibrator.get_optimal_thresholds()
    
    print("\n" + "=" * 50)
    print("CALIBRATION COMPLETE")
    print("=" * 50)
    print(f"Optimal confidence threshold: {opt_conf}")
    print(f"Optimal IoU threshold: {opt_iou}")
    print(f"Validation MAE: {opt_mae:.2f}%")
    print("=" * 50)
    
    # Save results
    calibrator.save_results(args.output)
    calibrator.plot_results(args.output)


if __name__ == '__main__':
    main()
