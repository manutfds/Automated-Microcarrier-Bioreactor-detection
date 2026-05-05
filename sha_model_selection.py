"""
================================================================================
ATENEA Project - Successive Halving Algorithm (SHA) for Model Selection
================================================================================

This script implements the SHA model selection strategy used in the ATENEA
project for efficiently identifying the best-performing YOLO architecture.

SHA is a principled early-stopping method that:
1. Trains all candidates for a small budget
2. Evaluates on validation metric
3. Promotes top performers to next stage with larger budget
4. Repeats until final selection

Reference:
    Zhang & Duh (2024). Best Practices of Successive Halving on Neural Machine
    Translation and Large Language Models.

Usage:
    python sha_model_selection.py --config configs/sha_config.yaml

Author: ATENEA Project Team
Institution: Takeda CTTC, Tres Cantos, Madrid
================================================================================
"""

import os
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SHA Configuration
# =============================================================================

class SHAConfig:
    """Configuration for SHA model selection."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize SHA configuration."""
        
        # Default configuration
        self.candidates = [
            'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
            'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x',
            'rtdetr-l', 'rtdetr-x'
        ]
        
        self.stages = [
            {'name': 'stage_1', 'epochs': 20, 'keep_top': 4},
            {'name': 'stage_2', 'epochs': 50, 'keep_top': 2},
            {'name': 'stage_3', 'epochs': 150, 'keep_top': 1},
        ]
        
        self.eta = 3  # Reduction factor
        self.ranking_metric = 'metrics/mAP50-95(B)'
        self.data_yaml = 'data.yaml'
        self.image_size = 640
        self.batch_size = 2
        self.device = 0
        self.seed = 0
        
        # Load from config file if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update attributes from config
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)


# =============================================================================
# SHA Model Selector
# =============================================================================

class SHAModelSelector:
    """
    Successive Halving Algorithm for YOLO model selection.
    
    Attributes:
        config: SHA configuration
        results: Dictionary storing results from all stages
        output_dir: Directory for saving results
    """
    
    def __init__(self, config: SHAConfig, output_dir: str = 'sha_results'):
        """
        Initialize SHA model selector.
        
        Args:
            config: SHAConfig object with selection parameters
            output_dir: Directory for saving results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {
            'all_runs': [],
            'stage_summaries': [],
            'winner': None
        }
        
        logger.info(f"SHA Model Selector initialized")
        logger.info(f"  Candidates: {len(config.candidates)} models")
        logger.info(f"  Stages: {len(config.stages)}")
        logger.info(f"  Output: {self.output_dir}")
    
    def train_model(
        self,
        model_name: str,
        epochs: int,
        stage_name: str
    ) -> Dict:
        """
        Train a single model and return results.
        
        Args:
            model_name: Name of model to train (e.g., 'yolov8l')
            epochs: Number of training epochs
            stage_name: Name of current SHA stage
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {model_name} for {epochs} epochs ({stage_name})")
        
        # Initialize model with pretrained weights
        model = YOLO(f'{model_name}.pt')
        
        # Create experiment name
        exp_name = f"{stage_name}/{model_name}_{epochs}ep"
        
        # Train
        start_time = datetime.now()
        
        try:
            results = model.train(
                data=self.config.data_yaml,
                epochs=epochs,
                imgsz=self.config.image_size,
                batch=self.config.batch_size,
                device=self.config.device,
                seed=self.config.seed,
                deterministic=True,
                amp=True,  # Mixed precision
                project=str(self.output_dir / 'runs'),
                name=exp_name,
                exist_ok=True,
                verbose=False
            )
            
            # Extract metrics
            metrics = {
                'model': model_name,
                'stage': stage_name,
                'epochs': epochs,
                'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
                'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'precision': results.results_dict.get('metrics/precision(B)', 0),
                'recall': results.results_dict.get('metrics/recall(B)', 0),
                'training_time_hours': (datetime.now() - start_time).total_seconds() / 3600,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            metrics = {
                'model': model_name,
                'stage': stage_name,
                'epochs': epochs,
                'mAP50': 0,
                'mAP50-95': 0,
                'precision': 0,
                'recall': 0,
                'training_time_hours': (datetime.now() - start_time).total_seconds() / 3600,
                'status': 'failed'
            }
        
        # Store result
        self.results['all_runs'].append(metrics)
        
        return metrics
    
    def run_stage(
        self,
        models: List[str],
        stage_config: Dict
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Run a single SHA stage.
        
        Args:
            models: List of model names to evaluate
            stage_config: Stage configuration dict
            
        Returns:
            Tuple of (promoted model names, stage results dataframe)
        """
        stage_name = stage_config['name']
        epochs = stage_config['epochs']
        keep_top = stage_config['keep_top']
        
        logger.info("=" * 60)
        logger.info(f"SHA {stage_name.upper()}")
        logger.info(f"  Models: {len(models)}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Promote top: {keep_top}")
        logger.info("=" * 60)
        
        # Train all models in this stage
        stage_results = []
        for model_name in models:
            result = self.train_model(model_name, epochs, stage_name)
            stage_results.append(result)
        
        # Create dataframe and rank
        df = pd.DataFrame(stage_results)
        df = df.sort_values('mAP50-95', ascending=False)
        
        # Select top performers
        promoted_df = df.head(keep_top)
        promoted_models = promoted_df['model'].tolist()
        eliminated_models = df.tail(len(models) - keep_top)['model'].tolist()
        
        # Log results
        logger.info(f"\n{stage_name} Results:")
        logger.info(df[['model', 'mAP50-95', 'mAP50', 'training_time_hours']].to_string())
        logger.info(f"\nPromoted: {promoted_models}")
        logger.info(f"Eliminated: {eliminated_models}")
        
        # Store stage summary
        self.results['stage_summaries'].append({
            'stage': stage_name,
            'epochs': epochs,
            'models_in': len(models),
            'models_out': keep_top,
            'promoted': promoted_models,
            'eliminated': eliminated_models,
            'best_mAP': df['mAP50-95'].max(),
            'worst_mAP': df['mAP50-95'].min()
        })
        
        # Save stage results
        df.to_csv(self.output_dir / f'{stage_name}_results.csv', index=False)
        
        return promoted_models, df
    
    def run_selection(self) -> Dict:
        """
        Run complete SHA model selection.
        
        Returns:
            Dictionary with final results including winner
        """
        logger.info("=" * 60)
        logger.info("STARTING SHA MODEL SELECTION")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Start with all candidates
        current_models = self.config.candidates.copy()
        
        # Run each stage
        for stage_config in self.config.stages:
            promoted_models, _ = self.run_stage(current_models, stage_config)
            current_models = promoted_models
        
        # Final winner
        winner = current_models[0]
        self.results['winner'] = winner
        
        total_time = (datetime.now() - start_time).total_seconds() / 3600
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SHA SELECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Winner: {winner}")
        logger.info(f"Total time: {total_time:.2f} hours")
        
        # Save final results
        self._save_final_results()
        
        return self.results
    
    def _save_final_results(self):
        """Save all results to files."""
        
        # All runs
        df_all = pd.DataFrame(self.results['all_runs'])
        df_all.to_csv(self.output_dir / 'all_training_runs.csv', index=False)
        
        # Stage summaries
        df_stages = pd.DataFrame(self.results['stage_summaries'])
        df_stages.to_csv(self.output_dir / 'stage_summaries.csv', index=False)
        
        # Winner summary
        with open(self.output_dir / 'winner.txt', 'w') as f:
            f.write(f"SHA Model Selection Winner\n")
            f.write(f"=" * 40 + "\n\n")
            f.write(f"Winner: {self.results['winner']}\n\n")
            f.write(f"Stage progression:\n")
            for stage in self.results['stage_summaries']:
                f.write(f"  {stage['stage']}: {stage['promoted']}\n")
        
        logger.info(f"Results saved to {self.output_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for SHA model selection."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run SHA model selection for YOLO architectures'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to SHA configuration YAML file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='sha_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data.yaml',
        help='Path to dataset YAML file'
    )
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = SHAConfig(args.config)
    config.data_yaml = args.data
    
    # Initialize selector
    selector = SHAModelSelector(config, args.output)
    
    # Run selection
    results = selector.run_selection()
    
    print(f"\n🏆 Winner: {results['winner']}")
    print(f"📁 Results saved to: {args.output}")


if __name__ == '__main__':
    main()
