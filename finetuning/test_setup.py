#!/usr/bin/env python3
"""
Test script to verify fine-tuning setup is working correctly.
This script performs basic checks on dependencies, model access, and dataset loading.
"""

import sys
import os
import warnings
from pathlib import Path
import importlib.util

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")

    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required, got {sys.version_info}")
        return False

    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n📦 Checking dependencies...")

    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "bitsandbytes",
        "accelerate",
        "scikit-learn",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "psutil"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("✅ All dependencies installed")
    return True

def check_cuda():
    """Check CUDA availability"""
    print("\n🔥 Checking CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✅ CUDA available")
            print(f"   Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"   GPU {i}: {props.name} ({memory_gb:.1f}GB)")
            return True
        else:
            print("⚠️  CUDA not available - training will be very slow")
            return False

    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def check_model_access():
    """Check if we can access the Gemma model"""
    print("\n🤖 Checking model access...")

    try:
        from transformers import AutoTokenizer

        model_name = "google/gemma-2b-it"
        print(f"   Attempting to load: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Model accessible")
        print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
        return True

    except Exception as e:
        print(f"❌ Cannot access model: {e}")
        print("   Make sure you have access to Gemma model")
        print("   Run: huggingface-cli login")
        return False

def check_dataset_access():
    """Check if we can access the MedQA dataset"""
    print("\n📊 Checking dataset access...")

    try:
        from datasets import load_dataset

        dataset_name = "GBaker/MedQA-USMLE-4-options"
        print(f"   Attempting to load: {dataset_name}")

        dataset = load_dataset(dataset_name)
        print("✅ Dataset accessible")

        for split in dataset.keys():
            print(f"   {split}: {len(dataset[split])} examples")

        # Check sample
        sample = dataset['train'][0]
        print(f"   Sample keys: {list(sample.keys())}")

        return True

    except Exception as e:
        print(f"❌ Cannot access dataset: {e}")
        return False

def check_configuration():
    """Check if configuration system works"""
    print("\n⚙️  Checking configuration...")

    try:
        from config import FineTuningConfig, get_quick_test_config

        # Test basic config
        config = FineTuningConfig()
        print("✅ Basic configuration created")

        # Test preset config
        quick_config = get_quick_test_config()
        print("✅ Quick test configuration created")

        # Test serialization
        config_dict = config.to_dict()
        reconstructed = FineTuningConfig.from_dict(config_dict)
        print("✅ Configuration serialization works")

        return True

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_data_processing():
    """Check if data processing works"""
    print("\n🔧 Checking data processing...")

    try:
        from data_utils import MedQADataProcessor, MedQADataLoader
        from transformers import AutoTokenizer

        # Test data loader
        loader = MedQADataLoader()
        print("✅ Data loader created")

        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
        processor = MedQADataProcessor(tokenizer)
        print("✅ Data processor created")

        # Test sample processing
        sample_example = {
            "question": "What is the most common cause of chest pain?",
            "options": {
                "A": "Myocardial infarction",
                "B": "Gastroesophageal reflux",
                "C": "Anxiety",
                "D": "Musculoskeletal"
            },
            "answer": "B"
        }

        formatted = processor.format_conversation(
            sample_example['question'],
            sample_example['options'],
            sample_example['answer']
        )
        print("✅ Sample formatting works")

        return True

    except Exception as e:
        print(f"❌ Data processing error: {e}")
        return False

def check_memory_requirements():
    """Check memory requirements"""
    print("\n💾 Checking memory requirements...")

    try:
        import psutil

        # System memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"   System memory: {memory_gb:.1f}GB")

        if memory_gb < 8:
            print("⚠️  Low system memory - consider memory_efficient config")
        else:
            print("✅ Sufficient system memory")

        # GPU memory
        if torch.cuda.is_available():
            import torch
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   GPU memory: {gpu_memory:.1f}GB")

            if gpu_memory < 8:
                print("⚠️  Low GPU memory - use memory_efficient config")
            elif gpu_memory < 16:
                print("⚠️  Moderate GPU memory - consider quick config")
            else:
                print("✅ Sufficient GPU memory for production config")

        # Disk space
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        print(f"   Free disk space: {disk_free_gb:.1f}GB")

        if disk_free_gb < 10:
            print("⚠️  Low disk space - need at least 10GB")
        else:
            print("✅ Sufficient disk space")

        return True

    except Exception as e:
        print(f"❌ Memory check error: {e}")
        return False

def run_minimal_test():
    """Run a minimal training test"""
    print("\n🧪 Running minimal test...")

    try:
        from config import get_quick_test_config
        from train import GemmaMedQATrainer

        # Create minimal config
        config = get_quick_test_config()
        config.training.output_dir = "./test_output"
        config.training.num_train_epochs = 1
        config.dataset.train_size_limit = 2
        config.dataset.eval_size_limit = 2
        config.training.per_device_train_batch_size = 1
        config.training.gradient_accumulation_steps = 1
        config.training.save_steps = 1
        config.training.eval_steps = 1
        config.training.logging_steps = 1

        print("✅ Minimal config created")

        # Note: We don't actually run training in the test
        # as it would require significant resources
        print("✅ Test configuration validated")
        print("   (Skipping actual training in test)")

        return True

    except Exception as e:
        print(f"❌ Minimal test error: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("GEMMA 2B MEDQA FINE-TUNING SETUP TEST")
    print("="*60)

    tests = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA", check_cuda),
        ("Model Access", check_model_access),
        ("Dataset Access", check_dataset_access),
        ("Configuration", check_configuration),
        ("Data Processing", check_data_processing),
        ("Memory Requirements", check_memory_requirements),
        ("Minimal Test", run_minimal_test)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! Setup is ready for fine-tuning.")
        print("\nNext steps:")
        print("1. Run quick test: python train.py --config quick")
        print("2. Run production training: python train.py --config production")
        print("3. Check examples/ directory for more usage examples")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix issues before proceeding.")
        print("\nCommon fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Login to HuggingFace: huggingface-cli login")
        print("3. Check internet connection for dataset/model access")
        print("4. Ensure sufficient disk space and memory")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
