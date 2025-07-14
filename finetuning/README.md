# Fine-tuning Gemma 2B on MedQA Dataset using QLoRA

This directory contains a comprehensive fine-tuning pipeline for adapting Google's Gemma 2B IT model to medical question answering using the MedQA dataset and QLoRA (Quantized Low-Rank Adaptation).

## Overview

The fine-tuning pipeline is designed to:
- Fine-tune Gemma 2B IT on medical Q&A data using memory-efficient QLoRA
- Provide configurable training parameters for different hardware setups
- Include comprehensive evaluation metrics and utilities
- Support experiment tracking with Weights & Biases
- Ensure reproducible results across different environments

## Features

- **QLoRA Implementation**: Memory-efficient 4-bit quantization with LoRA adapters
- **Configurable Training**: Multiple preset configurations for different use cases
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, and error analysis
- **Modular Design**: Separate modules for configuration, data processing, and evaluation
- **Experiment Tracking**: Optional Weights & Biases integration
- **Medical Focus**: Optimized for medical question answering tasks

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or if you're using the main project requirements
pip install -r ../requirements.txt
```

### 2. Basic Training

```bash
# Quick test run (minimal resources)
python train.py --config quick

# Production training (full dataset)
python train.py --config production --use_wandb

# Memory-efficient training (low GPU memory)
python train.py --config memory_efficient
```

### 3. Evaluation

```bash
# Evaluate a trained model
python train.py --evaluate_only ./finetuned_gemma_medqa

# Or use the evaluation script directly
python -c "
from evaluation import MedQAEvaluator
from datasets import load_dataset

evaluator = MedQAEvaluator('./finetuned_gemma_medqa')
dataset = load_dataset('GBaker/MedQA-USMLE-4-options')
results = evaluator.evaluate_dataset(dataset['test'], max_examples=100)
print(f'Accuracy: {results["metrics"]["accuracy"]:.4f}')
"
```

## Configuration

The training pipeline supports multiple configuration presets:

### Quick Test Configuration
- **Purpose**: Fast testing and development
- **Resources**: Minimal GPU memory required
- **Dataset**: Limited to 100 training examples
- **Time**: ~30 minutes on single GPU

```bash
python train.py --config quick
```

### Production Configuration
- **Purpose**: Full-scale training for best performance
- **Resources**: High GPU memory recommended
- **Dataset**: Full MedQA dataset
- **Time**: Several hours on high-end GPU

```bash
python train.py --config production --use_wandb
```

### Memory-Efficient Configuration
- **Purpose**: Training on limited GPU memory
- **Resources**: Works with 8GB GPU memory
- **Dataset**: Full dataset with optimized settings
- **Time**: Extended training time due to smaller batches

```bash
python train.py --config memory_efficient
```

## Advanced Usage

### Custom Configuration

Create a custom configuration file:

```python
from config import FineTuningConfig

# Create custom config
config = FineTuningConfig()
config.training.num_train_epochs = 5
config.training.learning_rate = 1e-4
config.lora.r = 32
config.lora.lora_alpha = 64

# Save config
import json
with open('my_config.json', 'w') as f:
    json.dump(config.to_dict(), f, indent=2, default=str)
```

Use custom configuration:

```bash
python train.py --config_file my_config.json
```

### Environment Variables

Set training parameters via environment variables:

```bash
export MODEL_NAME="google/gemma-2b-it"
export OUTPUT_DIR="./my_finetuned_model"
export NUM_EPOCHS=3
export BATCH_SIZE=4
export LEARNING_RATE=2e-4
export USE_WANDB=true
export WANDB_PROJECT="my-medqa-experiment"

python train.py --config env
```

### Command Line Overrides

Override specific parameters:

```bash
python train.py \
  --config production \
  --epochs 5 \
  --batch_size 2 \
  --learning_rate 1e-4 \
  --output_dir ./my_model \
  --use_wandb \
  --wandb_project my-experiment
```

## File Structure

```
finetuning/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration classes and presets
├── train.py                  # Main training script
├── data_utils.py            # Data processing utilities
├── evaluation.py            # Evaluation utilities
├── finetune_gemma_medqa.py  # Legacy standalone script
└── examples/                # Example usage scripts
    ├── quick_start.py
    ├── custom_evaluation.py
    └── model_comparison.py
```

## Training Parameters

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 2e-4 | Learning rate for AdamW optimizer |
| `num_train_epochs` | 3 | Number of training epochs |
| `per_device_train_batch_size` | 4 | Batch size per device |
| `gradient_accumulation_steps` | 4 | Steps to accumulate gradients |
| `max_length` | 1024 | Maximum sequence length |
| `warmup_steps` | 100 | Number of warmup steps |

### LoRA Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | 16 | LoRA rank (lower = more efficient) |
| `lora_alpha` | 32 | LoRA alpha parameter |
| `lora_dropout` | 0.1 | Dropout rate for LoRA layers |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Target modules for LoRA |

### Quantization Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `load_in_4bit` | True | Enable 4-bit quantization |
| `bnb_4bit_quant_type` | "nf4" | Quantization type |
| `bnb_4bit_compute_dtype` | bfloat16 | Compute dtype for quantized model |

## Evaluation Metrics

The evaluation system provides comprehensive metrics:

### Core Metrics
- **Accuracy**: Overall accuracy on multiple choice questions
- **Precision**: Per-class and weighted precision
- **Recall**: Per-class and weighted recall
- **F1 Score**: Harmonic mean of precision and recall

### Additional Metrics
- **Extraction Success Rate**: Percentage of responses where answer was successfully extracted
- **Per-class Performance**: Individual performance for options A, B, C, D
- **Confusion Matrix**: Visual representation of prediction patterns
- **Error Analysis**: Detailed analysis of failure cases

### Sample Evaluation Output

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

## Hardware Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (RTX 3070/4060 Ti or better)
- **RAM**: 16GB system RAM
- **Storage**: 10GB free space

### Recommended Requirements
- **GPU**: 16GB+ VRAM (RTX 4080/4090, A100, etc.)
- **RAM**: 32GB+ system RAM
- **Storage**: 50GB+ free space (for datasets and checkpoints)

### Memory Usage by Configuration

| Configuration | GPU Memory | Training Time | Dataset Size |
|---------------|------------|---------------|--------------|
| Quick | ~6GB | 30 min | 100 examples |
| Memory Efficient | ~8GB | 4-6 hours | Full dataset |
| Production | ~16GB | 2-3 hours | Full dataset |

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use memory-efficient config
   python train.py --config memory_efficient
   
   # Or reduce batch size
   python train.py --batch_size 1
   ```

2. **Model Loading Issues**
   ```bash
   # Ensure you have access to Gemma model
   huggingface-cli login
   
   # Check model access
   python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/gemma-2b-it')"
   ```

3. **Dataset Loading Problems**
   ```bash
   # Clear cache and reload
   python -c "from datasets import load_dataset; load_dataset('GBaker/MedQA-USMLE-4-options', download_mode='force_redownload')"
   ```

### Performance Tips

1. **Enable Mixed Precision**: Set `bf16=True` in training config
2. **Use Gradient Checkpointing**: Reduces memory usage
3. **Optimize Batch Size**: Find the largest batch size that fits in memory
4. **Monitor GPU Usage**: Use `nvidia-smi` or `gpustat` to monitor utilization

## Integration with Main Project

To integrate the fine-tuned model with the main NeuroViT project:

1. **Update LLM Configuration**:
   ```python
   # In llm/llm_wrapper.py
   model_path = "./finetuning/finetuned_gemma_medqa"
   ```

2. **Modify API Endpoints**:
   ```python
   # In api/routes/chat.py
   from ..finetuning.evaluation import MedQAEvaluator
   
   evaluator = MedQAEvaluator(model_path)
   ```

3. **Update Docker Configuration**:
   ```dockerfile
   # Copy fine-tuned model
   COPY finetuning/finetuned_gemma_medqa /app/models/
   ```

## Experiment Tracking

### Weights & Biases Integration

```bash
# Login to wandb
wandb login

# Run with tracking
python train.py --config production --use_wandb --wandb_project my-medqa-experiment
```

### Metrics Logged
- Training loss
- Validation loss
- Learning rate
- GPU memory usage
- Training speed (samples/sec)
- Evaluation metrics

## Contributing

1. **Code Style**: Follow PEP 8 guidelines
2. **Testing**: Add tests for new functionality
3. **Documentation**: Update README for new features
4. **Configuration**: Add new presets to `config.py`

## License

This fine-tuning pipeline is part of the NeuroViT project and follows the same license terms.

## Citation

If you use this fine-tuning pipeline in your research, please cite:

```bibtex
@software{neurovit_finetuning,
  title={NeuroViT Fine-tuning Pipeline for Medical Question Answering},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/neurovit}
}
```

## References

- [Gemma Model](https://huggingface.co/google/gemma-2b-it)
- [MedQA Dataset](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)