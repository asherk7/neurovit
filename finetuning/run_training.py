#!/usr/bin/env python3
"""
Production-ready training script for fine-tuning Gemma 2B on MedQA dataset.
This script includes comprehensive logging, error handling, and monitoring.
"""

import os
import sys
import json
import logging
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import psutil
import signal
from transformers import set_seed

# Add finetuning directory to path
sys.path.append(str(Path(__file__).parent))

from config import (
    FineTuningConfig,
    get_quick_test_config,
    get_production_config,
    get_memory_efficient_config,
    get_config_from_env
)
from train import GemmaMedQATrainer
from evaluation import MedQAEvaluator
from datasets import load_dataset

class TrainingMonitor:
    """Monitor training progress and system resources"""

    def __init__(self, log_file: str):
        self.log_file = log_file
        self.start_time = None
        self.training_interrupted = False

    def start_monitoring(self):
        """Start monitoring training"""
        self.start_time = datetime.now()
        self.log_system_info()

    def log_system_info(self):
        """Log system information"""
        logging.info("="*60)
        logging.info("SYSTEM INFORMATION")
        logging.info("="*60)

        # CPU Info
        logging.info(f"CPU Count: {psutil.cpu_count()}")
        logging.info(f"CPU Usage: {psutil.cpu_percent()}%")

        # Memory Info
        memory = psutil.virtual_memory()
        logging.info(f"Total Memory: {memory.total / (1024**3):.2f} GB")
        logging.info(f"Available Memory: {memory.available / (1024**3):.2f} GB")
        logging.info(f"Memory Usage: {memory.percent}%")

        # GPU Info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logging.info(f"GPU {i}: {props.name}")
                logging.info(f"GPU {i} Memory: {props.total_memory / (1024**3):.2f} GB")

                # Current GPU memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    logging.info(f"GPU {i} Allocated: {allocated:.2f} GB")
                    logging.info(f"GPU {i} Cached: {cached:.2f} GB")
        else:
            logging.warning("No CUDA devices available")

        # Disk Info
        disk = psutil.disk_usage('/')
        logging.info(f"Disk Total: {disk.total / (1024**3):.2f} GB")
        logging.info(f"Disk Free: {disk.free / (1024**3):.2f} GB")
        logging.info(f"Disk Usage: {disk.percent}%")

        logging.info("="*60)

    def log_training_progress(self, epoch: int, step: int, loss: float, lr: float):
        """Log training progress"""
        elapsed = datetime.now() - self.start_time

        logging.info(f"Epoch {epoch}, Step {step}: Loss={loss:.4f}, LR={lr:.2e}, "
                    f"Elapsed={elapsed}")

        # Log GPU memory usage if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                logging.info(f"GPU {i} Memory: {allocated:.2f}GB allocated, "
                           f"{cached:.2f}GB cached")

    def handle_interruption(self):
        """Handle training interruption"""
        self.training_interrupted = True
        logging.warning("Training interrupted by user")

    def log_completion(self, success: bool, error_msg: Optional[str] = None):
        """Log training completion"""
        elapsed = datetime.now() - self.start_time

        if success:
            logging.info(f"Training completed successfully in {elapsed}")
        else:
            logging.error(f"Training failed after {elapsed}")
            if error_msg:
                logging.error(f"Error: {error_msg}")

def setup_logging(output_dir: str, log_level: str = "INFO") -> str:
    """Setup logging configuration"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file

def validate_environment():
    """Validate training environment"""

    logging.info("Validating training environment...")

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        raise RuntimeError(f"Python 3.8+ required, got {python_version}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, training will be very slow")

    # Check available memory
    memory = psutil.virtual_memory()
    if memory.available < 8 * (1024**3):  # 8GB
        logging.warning(f"Low system memory: {memory.available / (1024**3):.2f} GB available")

    # Check disk space
    disk = psutil.disk_usage('/')
    if disk.free < 20 * (1024**3):  # 20GB
        raise RuntimeError(f"Insufficient disk space: {disk.free / (1024**3):.2f} GB available")

    logging.info("Environment validation completed")

def validate_configuration(config: FineTuningConfig) -> bool:
    """Validate training configuration"""

    logging.info("Validating configuration...")

    # Check model availability
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
        logging.info(f"Model {config.model.model_name} is accessible")
    except Exception as e:
        logging.error(f"Cannot access model {config.model.model_name}: {e}")
        return False

    # Check dataset availability
    try:
        dataset = load_dataset(config.dataset.dataset_name)
        logging.info(f"Dataset {config.dataset.dataset_name} loaded successfully")
        logging.info(f"Dataset splits: {list(dataset.keys())}")
    except Exception as e:
        logging.error(f"Cannot load dataset {config.dataset.dataset_name}: {e}")
        return False

    # Validate batch size and memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        total_batch_size = (config.training.per_device_train_batch_size *
                           config.training.gradient_accumulation_steps)

        if gpu_memory < 8 and total_batch_size > 8:
            logging.warning(f"Large batch size ({total_batch_size}) for GPU memory ({gpu_memory:.1f}GB)")

    # Validate paths
    if not os.path.exists(os.path.dirname(config.training.output_dir)):
        try:
            os.makedirs(os.path.dirname(config.training.output_dir), exist_ok=True)
        except Exception as e:
            logging.error(f"Cannot create output directory: {e}")
            return False

    logging.info("Configuration validation completed")
    return True

def save_training_metadata(config: FineTuningConfig, output_dir: str):
    """Save training metadata"""

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config": config.to_dict(),
        "system_info": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": psutil.cpu_count(),
            "total_memory": psutil.virtual_memory().total / (1024**3)
        },
        "git_info": get_git_info()
    }

    # Save metadata
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    logging.info(f"Training metadata saved to {metadata_path}")

def get_git_info() -> Dict[str, str]:
    """Get git information if available"""
    try:
        import subprocess

        git_info = {}

        # Get commit hash
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                               capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            git_info['commit_hash'] = result.stdout.strip()

        # Get branch name
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                               capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()

        # Check for uncommitted changes
        result = subprocess.run(['git', 'diff', '--quiet'],
                               capture_output=True, cwd=Path(__file__).parent)
        git_info['has_uncommitted_changes'] = result.returncode != 0

        return git_info
    except Exception:
        return {"error": "Git information not available"}

def run_post_training_evaluation(model_path: str, config: FineTuningConfig) -> Dict[str, Any]:
    """Run evaluation after training"""

    logging.info("Running post-training evaluation...")

    try:
        # Load test dataset
        dataset = load_dataset(config.dataset.dataset_name)
        test_dataset = dataset['test']

        # Initialize evaluator
        evaluator = MedQAEvaluator(model_path)

        # Run evaluation
        eval_size = min(1000, len(test_dataset))
        results = evaluator.evaluate_dataset(test_dataset, max_examples=eval_size)

        # Log results
        metrics = results['metrics']
        logging.info(f"Evaluation completed on {eval_size} examples")
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"Recall: {metrics['recall']:.4f}")
        logging.info(f"F1 Score: {metrics['f1']:.4f}")

        # Save evaluation results
        eval_path = os.path.join(model_path, "evaluation_results.json")
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return results

    except Exception as e:
        logging.error(f"Post-training evaluation failed: {e}")
        return {"error": str(e)}

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    logging.warning(f"Received signal {signum}, initiating graceful shutdown...")
    sys.exit(1)

def main():
    """Main training function"""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Production training for Gemma 2B MedQA fine-tuning")
    parser.add_argument("--config", choices=["quick", "production", "memory_efficient", "env"],
                       default="quick", help="Configuration preset")
    parser.add_argument("--config_file", help="Path to custom configuration file")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--model_name", help="Override model name")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases")
    parser.add_argument("--wandb_project", help="Weights & Biases project name")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip post-training evaluation")
    parser.add_argument("--resume_from", help="Resume training from checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="Validate configuration without training")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Load configuration
    try:
        if args.config_file:
            with open(args.config_file, 'r') as f:
                config_dict = json.load(f)
            config = FineTuningConfig.from_dict(config_dict)
        elif args.config == "quick":
            config = get_quick_test_config()
        elif args.config == "production":
            config = get_production_config()
        elif args.config == "memory_efficient":
            config = get_memory_efficient_config()
        elif args.config == "env":
            config = get_config_from_env()
        else:
            config = FineTuningConfig()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Apply command line overrides
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.model_name:
        config.model.model_name = args.model_name
    if args.epochs:
        config.training.num_train_epochs = args.epochs
    if args.batch_size:
        config.training.per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.use_wandb:
        config.wandb.use_wandb = True
        config.training.report_to = "wandb"
    if args.wandb_project:
        config.wandb.project_name = args.wandb_project

    # Setup logging
    log_file = setup_logging(config.training.output_dir, args.log_level)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize monitor
    monitor = TrainingMonitor(log_file)
    monitor.start_monitoring()

    # Log configuration
    logging.info("="*80)
    logging.info("STARTING GEMMA 2B MEDQA FINE-TUNING")
    logging.info("="*80)
    logging.info(f"Configuration: {args.config}")
    logging.info(f"Output directory: {config.training.output_dir}")
    logging.info(f"Model: {config.model.model_name}")
    logging.info(f"Epochs: {config.training.num_train_epochs}")
    logging.info(f"Batch size: {config.training.per_device_train_batch_size}")
    logging.info(f"Learning rate: {config.training.learning_rate}")
    logging.info(f"Random seed: {args.seed}")

    success = False
    error_msg = None

    try:
        # Validate environment
        validate_environment()

        # Validate configuration
        if not validate_configuration(config):
            raise RuntimeError("Configuration validation failed")

        # Save training metadata
        save_training_metadata(config, config.training.output_dir)

        if args.dry_run:
            logging.info("Dry run completed successfully")
            return

        # Initialize trainer
        logging.info("Initializing trainer...")
        trainer = GemmaMedQATrainer(config)

        # Start training
        logging.info("Starting training...")
        trainer.train()

        success = True
        logging.info("Training completed successfully!")

        # Run post-training evaluation
        if not args.skip_evaluation:
            evaluation_results = run_post_training_evaluation(
                config.training.output_dir, config
            )

            if "error" not in evaluation_results:
                logging.info("Post-training evaluation completed successfully")
            else:
                logging.warning("Post-training evaluation failed")

    except KeyboardInterrupt:
        error_msg = "Training interrupted by user"
        logging.warning(error_msg)
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Training failed: {error_msg}")
        logging.error(f"Traceback: {traceback.format_exc()}")

    finally:
        # Log completion
        monitor.log_completion(success, error_msg)

        # Final system info
        monitor.log_system_info()

        if success:
            logging.info(f"Model saved to: {config.training.output_dir}")
            logging.info(f"Logs saved to: {log_file}")
            logging.info("Training completed successfully!")
        else:
            logging.error("Training failed. Check logs for details.")
            sys.exit(1)

if __name__ == "__main__":
    main()
