#!/usr/bin/env python3
"""
Quick start example for fine-tuning Gemma 2B on MedQA dataset.
This script demonstrates the basic usage of the fine-tuning pipeline.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from train import GemmaMedQATrainer
from config import get_quick_test_config, get_memory_efficient_config
from evaluation import MedQAEvaluator, evaluate_sample_questions
from data_utils import MedQADataLoader
from datasets import load_dataset
import torch

def main():
    """Quick start example"""

    print("="*80)
    print("GEMMA 2B MEDQA FINE-TUNING - QUICK START")
    print("="*80)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ Warning: No GPU detected. Training will be very slow.")

    # Step 1: Load and examine dataset
    print("\n" + "="*60)
    print("STEP 1: LOADING DATASET")
    print("="*60)

    data_loader = MedQADataLoader()
    dataset = data_loader.load_dataset()

    # Show sample examples
    print("\nSample examples:")
    samples = data_loader.get_sample_examples(num_samples=2)
    for i, sample in enumerate(samples, 1):
        print(f"\nExample {i}:")
        print(f"Q: {sample['question'][:100]}...")
        print(f"Options: {list(sample['options'].keys())}")
        print(f"Answer: {sample['answer']}")

    # Step 2: Configure training
    print("\n" + "="*60)
    print("STEP 2: CONFIGURATION")
    print("="*60)

    # Choose config based on available GPU memory
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

    if gpu_memory_gb >= 12:
        config = get_quick_test_config()
        config_name = "quick_test"
        print(f"✓ Using quick test config (GPU memory: {gpu_memory_gb:.1f} GB)")
    else:
        config = get_memory_efficient_config()
        config_name = "memory_efficient"
        print(f"✓ Using memory efficient config (GPU memory: {gpu_memory_gb:.1f} GB)")

    # Customize output directory
    config.training.output_dir = f"./quick_start_model_{config_name}"
    config.training.num_train_epochs = 1  # Very quick for demo
    config.dataset.train_size_limit = 50  # Small dataset for quick demo
    config.dataset.eval_size_limit = 20

    print(f"✓ Output directory: {config.training.output_dir}")
    print(f"✓ Training epochs: {config.training.num_train_epochs}")
    print(f"✓ Training examples: {config.dataset.train_size_limit}")
    print(f"✓ Evaluation examples: {config.dataset.eval_size_limit}")

    # Step 3: Initialize trainer
    print("\n" + "="*60)
    print("STEP 3: INITIALIZING TRAINER")
    print("="*60)

    trainer = GemmaMedQATrainer(config)

    # Step 4: Training
    print("\n" + "="*60)
    print("STEP 4: TRAINING")
    print("="*60)

    print("Starting training... This may take a few minutes.")
    print("Note: This is a minimal demo. For production use, increase epochs and dataset size.")

    try:
        trainer.train()
        print("✓ Training completed successfully!")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        print("Try reducing batch size or using memory_efficient config")
        return

    # Step 5: Evaluation
    print("\n" + "="*60)
    print("STEP 5: EVALUATION")
    print("="*60)

    # Sample questions for evaluation
    sample_questions = [
        {
            "question": "A 65-year-old man presents with chest pain and shortness of breath. ECG shows ST-elevation in leads II, III, and aVF. What is the most likely diagnosis?",
            "options": {
                "A": "Anterior wall myocardial infarction",
                "B": "Inferior wall myocardial infarction",
                "C": "Lateral wall myocardial infarction",
                "D": "Posterior wall myocardial infarction"
            },
            "answer": "B"
        },
        {
            "question": "A 30-year-old woman presents with sudden severe headache described as 'worst headache of her life'. What is the most appropriate immediate diagnostic test?",
            "options": {
                "A": "MRI of the brain",
                "B": "CT scan of the head",
                "C": "Lumbar puncture",
                "D": "Carotid ultrasound"
            },
            "answer": "B"
        }
    ]

    print("Evaluating model on sample questions...")
    evaluate_sample_questions(config.training.output_dir, sample_questions)

    # Step 6: Quick dataset evaluation
    print("\n" + "="*60)
    print("STEP 6: DATASET EVALUATION")
    print("="*60)

    try:
        evaluator = MedQAEvaluator(config.training.output_dir)
        test_dataset = dataset['test']

        print("Running evaluation on test dataset (limited to 20 examples)...")
        results = evaluator.evaluate_dataset(test_dataset, max_examples=20)

        print(f"✓ Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"✓ Precision: {results['metrics']['precision']:.4f}")
        print(f"✓ Recall: {results['metrics']['recall']:.4f}")
        print(f"✓ F1 Score: {results['metrics']['f1']:.4f}")

    except Exception as e:
        print(f"⚠ Evaluation failed: {e}")
        print("This is normal for very short training runs")

    # Step 7: Next steps
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)

    print("🎉 Quick start completed successfully!")
    print("\nFor production training, consider:")
    print("1. Use production config: python train.py --config production")
    print("2. Enable wandb tracking: python train.py --use_wandb")
    print("3. Increase training epochs: python train.py --epochs 5")
    print("4. Use full dataset: remove train_size_limit in config")
    print("\nFor evaluation:")
    print("1. Run comprehensive evaluation: python evaluation.py")
    print("2. Compare with baseline models")
    print("3. Analyze error patterns")

    print(f"\nModel saved to: {config.training.output_dir}")
    print("You can now use this model for inference!")

if __name__ == "__main__":
    main()
