"""
Phase 4: Train fault detection ML model
Usage: python scripts/train_fault_detector.py --data data/dataset_v1 --data-source dataset_v1 --model lstm
"""
import argparse
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.phase4_config import Phase4Config
from src.ml.trainer import Trainer
from src.ml.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 4: Fault Detection ML Pipeline"
    )
    parser.add_argument(
        "--data", type=str, default="data/dataset_v1",
        help="Path to data (directory for dataset_v1 or CSV file for single_csv)"
    )
    parser.add_argument(
        "--data-source", type=str, choices=["dataset_v1", "single_csv"],
        default="dataset_v1",
        help="Data source type: dataset_v1 (telemetry parquet runs) or single_csv"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/phase4",
        help="Output directory for artifacts"
    )
    parser.add_argument(
        "--model", type=str, choices=["lstm", "cnn1d"], 
        default="lstm", help="Model architecture"
    )
    parser.add_argument(
        "--window-seconds", type=float, default=5.0,
        help="Window duration in seconds"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Run evaluation only (skip training)"
    )
    
    args = parser.parse_args()
    
    config = Phase4Config(
        data_path=Path(args.data),
        data_source=args.data_source,
        output_dir=Path(args.output_dir),
        model_type=args.model,
        window_seconds=args.window_seconds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    if args.eval_only:
        logger.info("Running evaluation only...")
        evaluator = Evaluator(config)
        evaluator.evaluate()
        evaluator.plot_training_history()
    else:
        logger.info("Running training...")
        trainer = Trainer(config)
        trainer.train()
        
        logger.info("Running evaluation...")
        evaluator = Evaluator(config)
        evaluator.evaluate()
        evaluator.plot_training_history()

if __name__ == "__main__":
    main()
