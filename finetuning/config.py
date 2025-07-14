import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from transformers import TrainingArguments
from peft import LoraConfig
from transformers import BitsAndBytesConfig
import torch

@dataclass
class ModelConfig:
    """Configuration for the base model"""
    model_name: str = "google/gemma-2b-it"
    trust_remote_code: bool = True
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"
    use_cache: bool = False  # Required for gradient checkpointing

@dataclass
class QuantizationConfig:
    """Configuration for 4-bit quantization"""
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16

    def to_bnb_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype
        )

@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation"""
    r: int = 16  # Rank
    lora_alpha: int = 32  # Alpha parameter for LoRA scaling
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            # Default target modules for Gemma
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

    def to_peft_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
        )

@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    dataset_name: str = "GBaker/MedQA-USMLE-4-options"
    max_length: int = 1024
    padding: str = "max_length"
    truncation: bool = True
    train_size_limit: Optional[int] = 10000  # None for no limit
    eval_size_limit: Optional[int] = 1000   # None for no limit

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    output_dir: str = "./finetuned_gemma_medqa"
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    lr_scheduler_type: str = "linear"
    optim: str = "paged_adamw_32bit"

    # Mixed precision
    fp16: bool = False
    bf16: bool = True

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    evaluation_strategy: str = "steps"
    save_total_limit: int = 2

    # Best model selection
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Performance optimizations
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    gradient_checkpointing: bool = True

    # Experiment tracking
    report_to: Optional[str] = None  # Set to "wandb" if using wandb
    push_to_hub: bool = False

    def to_training_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            num_train_epochs=self.num_train_epochs,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            optim=self.optim,
            fp16=self.fp16,
            bf16=self.bf16,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            evaluation_strategy=self.evaluation_strategy,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            dataloader_num_workers=self.dataloader_num_workers,
            remove_unused_columns=self.remove_unused_columns,
            gradient_checkpointing=self.gradient_checkpointing,
            report_to=self.report_to,
            push_to_hub=self.push_to_hub,
        )

@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging"""
    use_wandb: bool = False
    project_name: str = "gemma-medqa-finetune"
    run_name: Optional[str] = None
    entity: Optional[str] = None  # Your wandb username or team name
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = ["gemma-2b", "medqa", "qlora", "medical-qa"]

        if self.run_name is None:
            self.run_name = f"gemma-2b-medqa-qlora"

@dataclass
class GenerationConfig:
    """Configuration for text generation during evaluation"""
    max_new_tokens: int = 200
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    pad_token_id: Optional[int] = None  # Will be set to tokenizer.eos_token_id

@dataclass
class FineTuningConfig:
    """Main configuration class that combines all configs"""
    model: ModelConfig = None
    quantization: QuantizationConfig = None
    lora: LoRAConfig = None
    dataset: DatasetConfig = None
    training: TrainingConfig = None
    wandb: WandbConfig = None
    generation: GenerationConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.quantization is None:
            self.quantization = QuantizationConfig()
        if self.lora is None:
            self.lora = LoRAConfig()
        if self.dataset is None:
            self.dataset = DatasetConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.wandb is None:
            self.wandb = WandbConfig()
        if self.generation is None:
            self.generation = GenerationConfig()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FineTuningConfig":
        """Create config from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            quantization=QuantizationConfig(**config_dict.get("quantization", {})),
            lora=LoRAConfig(**config_dict.get("lora", {})),
            dataset=DatasetConfig(**config_dict.get("dataset", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            wandb=WandbConfig(**config_dict.get("wandb", {})),
            generation=GenerationConfig(**config_dict.get("generation", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model": self.model.__dict__,
            "quantization": self.quantization.__dict__,
            "lora": self.lora.__dict__,
            "dataset": self.dataset.__dict__,
            "training": self.training.__dict__,
            "wandb": self.wandb.__dict__,
            "generation": self.generation.__dict__,
        }

# Predefined configurations for different use cases
def get_quick_test_config() -> FineTuningConfig:
    """Configuration for quick testing with minimal resources"""
    config = FineTuningConfig()
    config.training.num_train_epochs = 1
    config.training.per_device_train_batch_size = 2
    config.training.gradient_accumulation_steps = 2
    config.training.eval_steps = 50
    config.training.save_steps = 50
    config.training.logging_steps = 5
    config.dataset.train_size_limit = 100
    config.dataset.eval_size_limit = 50
    config.dataset.max_length = 512
    return config

def get_production_config() -> FineTuningConfig:
    """Configuration for production training with full dataset"""
    config = FineTuningConfig()
    config.training.num_train_epochs = 5
    config.training.per_device_train_batch_size = 8
    config.training.gradient_accumulation_steps = 2
    config.training.eval_steps = 200
    config.training.save_steps = 200
    config.training.learning_rate = 1e-4
    config.dataset.train_size_limit = None  # Use full dataset
    config.dataset.eval_size_limit = None
    config.dataset.max_length = 1024
    config.lora.r = 32
    config.lora.lora_alpha = 64
    return config

def get_memory_efficient_config() -> FineTuningConfig:
    """Configuration optimized for low memory usage"""
    config = FineTuningConfig()
    config.training.per_device_train_batch_size = 1
    config.training.gradient_accumulation_steps = 8
    config.training.gradient_checkpointing = True
    config.dataset.max_length = 512
    config.lora.r = 8
    config.lora.lora_alpha = 16
    config.quantization.bnb_4bit_compute_dtype = torch.float16
    return config

# Environment-based configuration
def get_config_from_env() -> FineTuningConfig:
    """Get configuration from environment variables"""
    config = FineTuningConfig()

    # Model configuration
    config.model.model_name = os.getenv("MODEL_NAME", config.model.model_name)

    # Training configuration
    config.training.output_dir = os.getenv("OUTPUT_DIR", config.training.output_dir)
    config.training.num_train_epochs = int(os.getenv("NUM_EPOCHS", config.training.num_train_epochs))
    config.training.learning_rate = float(os.getenv("LEARNING_RATE", config.training.learning_rate))
    config.training.per_device_train_batch_size = int(os.getenv("BATCH_SIZE", config.training.per_device_train_batch_size))

    # Dataset configuration
    config.dataset.max_length = int(os.getenv("MAX_LENGTH", config.dataset.max_length))

    # LoRA configuration
    config.lora.r = int(os.getenv("LORA_R", config.lora.r))
    config.lora.lora_alpha = int(os.getenv("LORA_ALPHA", config.lora.lora_alpha))

    # Wandb configuration
    config.wandb.use_wandb = os.getenv("USE_WANDB", "false").lower() == "true"
    config.wandb.project_name = os.getenv("WANDB_PROJECT", config.wandb.project_name)
    config.wandb.entity = os.getenv("WANDB_ENTITY", config.wandb.entity)

    if config.wandb.use_wandb:
        config.training.report_to = "wandb"

    return config
