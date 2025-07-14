# Getting Started with Gemma 2B Fine-tuning on MedQA

This guide will walk you through the process of fine-tuning Google's Gemma 2B IT model on the MedQA dataset using QLoRA (Quantized Low-Rank Adaptation).

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with at least 8GB VRAM (recommended: 16GB+)
- **RAM**: 16GB system RAM (recommended: 32GB+)
- **Storage**: 20GB free space
- **CUDA**: Compatible CUDA installation

### Account Setup
1. **Hugging Face Account**: Create account at [huggingface.co](https://huggingface.co)
2. **Gemma Access**: Request access to Gemma models
3. **Weights & Biases** (optional): For experiment tracking

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
# Navigate to finetuning directory
cd neurovit/finetuning

# Install required packages
pip install -r requirements.txt
```

### Step 2: Login to Hugging Face
```bash
# Login to access Gemma model
huggingface-cli login
```

### Step 3: Test Setup
```bash
# Verify everything is working
python test_setup.py
```

### Step 4: Run Quick Training
```bash
# Start with a quick test (30 minutes)
python train.py --config quick
```

### Step 5: Evaluate Results
```bash
# Test the fine-tuned model
python train.py --evaluate_only ./quick_test_model_quick_test
```

## 📖 Detailed Setup

### 1. Environment Setup

Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Authentication

Login to Hugging Face:
```bash
huggingface-cli login
```

When prompted, paste your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 3. Verify Access

Test model access:
```bash
python -c "from transformers import AutoTokenizer; print('✅ Model accessible')"
```

Test dataset access:
```bash
python -c "from datasets import load_dataset; dataset = load_dataset('GBaker/MedQA-USMLE-4-options'); print('✅ Dataset accessible')"
```

## 🏃‍♂️ Training Options

### Option 1: Quick Test (Recommended for first time)
```bash
python train.py --config quick
```
- **Time**: ~30 minutes
- **GPU Memory**: ~6GB
- **Purpose**: Verify setup works

### Option 2: Memory Efficient
```bash
python train.py --config memory_efficient
```
- **Time**: 4-6 hours
- **GPU Memory**: ~8GB
- **Purpose**: Full training on limited hardware

### Option 3: Production Training
```bash
python train.py --config production --use_wandb
```
- **Time**: 2-3 hours
- **GPU Memory**: ~16GB
- **Purpose**: Best performance

### Option 4: Custom Configuration
```bash
python train.py \
  --config production \
  --epochs 5 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --output_dir ./my_model
```

## 📊 Monitoring Training

### Using Weights & Biases
```bash
# Login to wandb
wandb login

# Run with tracking
python train.py --config production --use_wandb --wandb_project my-medqa-experiment
```

### Using Logs
```bash
# Monitor training logs
tail -f ./finetuned_gemma_medqa/training_*.log
```

### Using GPU Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi
```

## 🔍 Evaluation

### Basic Evaluation
```bash
python train.py --evaluate_only ./finetuned_gemma_medqa
```

### Comprehensive Evaluation
```bash
python examples/custom_evaluation.py
```

### Sample Questions
```bash
python -c "
from evaluation import evaluate_sample_questions

questions = [
    {
        'question': 'A 65-year-old man presents with chest pain. What is the most likely cause?',
        'options': {'A': 'MI', 'B': 'GERD', 'C': 'Anxiety', 'D': 'MSK'},
        'answer': 'A'
    }
]

evaluate_sample_questions('./finetuned_gemma_medqa', questions)
"
```

## 🛠 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Solution: Use memory efficient config
python train.py --config memory_efficient

# Or reduce batch size
python train.py --batch_size 1
```

**2. Model Access Denied**
```bash
# Solution: Login and check access
huggingface-cli login
# Request access to Gemma at https://huggingface.co/google/gemma-2b-it
```

**3. Dataset Loading Issues**
```bash
# Solution: Clear cache and retry
python -c "from datasets import load_dataset; load_dataset('GBaker/MedQA-USMLE-4-options', download_mode='force_redownload')"
```

**4. Import Errors**
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Performance Tips

**1. Optimize Batch Size**
```bash
# Find largest batch size that fits
python train.py --batch_size 8  # Try 8, 4, 2, 1
```

**2. Use Mixed Precision**
```bash
# Already enabled by default with bf16=True
```

**3. Monitor Memory Usage**
```bash
# Use memory profiler
pip install memory-profiler
python -m memory_profiler train.py --config quick
```

## 📈 Advanced Usage

### Custom Configuration
```python
from config import FineTuningConfig

config = FineTuningConfig()
config.training.num_train_epochs = 10
config.training.learning_rate = 5e-5
config.lora.r = 32
config.lora.lora_alpha = 64

# Save and use
import json
with open('custom_config.json', 'w') as f:
    json.dump(config.to_dict(), f, indent=2, default=str)
```

### Environment Variables
```bash
export MODEL_NAME="google/gemma-2b-it"
export NUM_EPOCHS=5
export LEARNING_RATE=1e-4
export USE_WANDB=true
export WANDB_PROJECT=my-experiment

python train.py --config env
```

### Resuming Training
```bash
# Resume from checkpoint
python train.py --resume_from ./finetuned_gemma_medqa/checkpoint-500
```

## 🎯 Expected Results

### Performance Benchmarks
- **Baseline (no fine-tuning)**: ~40-50% accuracy
- **After fine-tuning**: ~70-80% accuracy
- **Training time**: 2-6 hours depending on configuration

### Sample Output
```
==============================================================
EVALUATION SUMMARY
==============================================================
Model: ./finetuned_gemma_medqa
Total Examples: 1000
Accuracy: 0.7650
Precision: 0.7623
Recall: 0.7650
F1 Score: 0.7635
Extraction Success Rate: 0.9800
==============================================================
```

## 🔄 Integration with Main Project

### Update LLM Configuration
```python
# In llm/llm_wrapper.py
model_path = "./finetuning/finetuned_gemma_medqa"
```

### Use in API
```python
from finetuning.evaluation import MedQAEvaluator

evaluator = MedQAEvaluator("./finetuning/finetuned_gemma_medqa")
answer, response = evaluator.generate_answer(question, options)
```

## 📚 Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes
2. **Add more data**: Include additional medical datasets
3. **Evaluate on specific domains**: Test performance on different medical specialties
4. **Deploy the model**: Integrate with the main NeuroViT application
5. **Monitor in production**: Set up monitoring for deployed model

## 🆘 Getting Help

- **Check logs**: Always check training logs for detailed error messages
- **Run test_setup.py**: Verify your environment is correctly configured
- **Review examples/**: Look at example scripts for common use cases
- **Check documentation**: Read the full README.md for comprehensive information

## 📝 Example Commands Summary

```bash
# Quick start
python test_setup.py                    # Test setup
python train.py --config quick          # Quick training
python train.py --evaluate_only ./model # Evaluate

# Production training
python train.py --config production --use_wandb

# Custom training
python train.py --epochs 5 --batch_size 4 --learning_rate 2e-4

# Advanced evaluation
python examples/custom_evaluation.py
```

Happy fine-tuning! 🚀