#!/usr/bin/env python3
"""
Modular fine-tuning script for Gemma 2B on MedQA dataset using QLoRA.
This script uses the configuration system for better modularity and flexibility.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json
import os
import sys
from pathlib import Path
import argparse
import logging
from typing import Dict, Any, Optional
import wandb
from tqdm import tqdm

# Add the finetuning directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import (
    FineTuningConfig,
    get_quick_test_config,
    get_production_config,
    get_memory_efficient_config,
    get_config_from_env
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GemmaMedQATrainer:
    """Fine-tuning trainer for Gemma 2B on MedQA dataset"""

    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None

        # Create output directory
        os.makedirs(config.training.output_dir, exist_ok=True)

        # Save config to output directory
        self._save_config()

        # Initialize wandb if requested
        if config.wandb.use_wandb:
            self._init_wandb()

    def _save_config(self):
        """Save configuration to output directory"""
        config_path = os.path.join(self.config.training.output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)
        logger.info(f"Configuration saved to {config_path}")

    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.config.wandb.project_name,
            name=self.config.wandb.run_name,
            entity=self.config.wandb.entity,
            tags=self.config.wandb.tags,
            config=self.config.to_dict()
        )
        logger.info("Weights & Biases initialized")

    def load_model_and_tokenizer(self):
        """Load and prepare the model and tokenizer"""
        logger.info(f"Loading model: {self.config.model.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            quantization_config=self.config.quantization.to_bnb_config(),
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=self.config.model.torch_dtype,
            use_cache=self.config.model.use_cache
        )

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        self.model = get_peft_model(self.model, self.config.lora.to_peft_config())

        # Print trainable parameters
        self.model.print_trainable_parameters()

        logger.info("Model and tokenizer loaded successfully")

    def format_medqa_example(self, example: Dict[str, Any]) -> str:
        """Format a MedQA example into Gemma conversation format"""
        question = example['question']
        options = example['options']
        answer = example['answer']

        # Create options string
        options_str = ""
        for key, value in options.items():
            options_str += f"{key}: {value}\n"

        # Create the conversation format
        prompt = f"""<bos><start_of_turn>user
Medical Question: {question}

Options:
{options_str.strip()}

Please provide the correct answer and explain your reasoning.<end_of_turn>
<start_of_turn>model
The correct answer is {answer}.

Explanation: This question tests medical knowledge. The correct answer is {answer} because it represents the most appropriate clinical approach based on the presented scenario.<end_of_turn><eos>"""

        return prompt

    def load_and_process_dataset(self):
        """Load and process the MedQA dataset"""
        logger.info(f"Loading dataset: {self.config.dataset.dataset_name}")

        # Load dataset
        dataset = load_dataset(self.config.dataset.dataset_name)

        def process_example(example):
            """Process a single example"""
            formatted_text = self.format_medqa_example(example)

            # Tokenize
            tokenized = self.tokenizer(
                formatted_text,
                truncation=self.config.dataset.truncation,
                max_length=self.config.dataset.max_length,
                padding=self.config.dataset.padding,
                return_tensors="pt"
            )

            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        # Process training dataset
        logger.info("Processing training dataset...")
        self.train_dataset = dataset["train"].map(
            process_example,
            batched=False,
            remove_columns=dataset["train"].column_names,
            desc="Processing training examples"
        )

        # Process validation dataset
        logger.info("Processing validation dataset...")
        self.eval_dataset = dataset["validation"].map(
            process_example,
            batched=False,
            remove_columns=dataset["validation"].column_names,
            desc="Processing validation examples"
        )

        # Apply size limits if specified
        if self.config.dataset.train_size_limit:
            original_size = len(self.train_dataset)
            self.train_dataset = self.train_dataset.select(
                range(min(self.config.dataset.train_size_limit, original_size))
            )
            logger.info(f"Limited training dataset from {original_size} to {len(self.train_dataset)}")

        if self.config.dataset.eval_size_limit:
            original_size = len(self.eval_dataset)
            self.eval_dataset = self.eval_dataset.select(
                range(min(self.config.dataset.eval_size_limit, original_size))
            )
            logger.info(f"Limited validation dataset from {original_size} to {len(self.eval_dataset)}")

        logger.info(f"Final dataset sizes - Train: {len(self.train_dataset)}, Validation: {len(self.eval_dataset)}")

    def create_trainer(self):
        """Create the Trainer instance"""
        logger.info("Creating trainer...")

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt",
            pad_to_multiple_of=8
        )

        # Training arguments
        training_args = self.config.training.to_training_args()

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        logger.info("Trainer created successfully")

    def train(self):
        """Execute the training process"""
        logger.info("Starting training process...")

        # Load model and tokenizer
        self.load_model_and_tokenizer()

        # Load and process dataset
        self.load_and_process_dataset()

        # Create trainer
        self.create_trainer()

        # Train the model
        logger.info("Beginning training...")
        self.trainer.train()

        # Save the final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.training.output_dir)

        logger.info(f"Training completed! Model saved to {self.config.training.output_dir}")

        # Finish wandb if used
        if self.config.wandb.use_wandb:
            wandb.finish()

    def evaluate(self, custom_examples: Optional[list] = None):
        """Evaluate the model on sample questions"""
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Please train first or load a trained model.")
            return

        logger.info("Evaluating model...")

        # Sample test questions if not provided
        if custom_examples is None:
            custom_examples = [
                {
                    "question": "A 65-year-old woman presents with sudden onset of severe headache. CT scan shows subarachnoid hemorrhage. What is the most likely cause?",
                    "options": {
                        "A": "Hypertensive crisis",
                        "B": "Ruptured cerebral aneurysm",
                        "C": "Arteriovenous malformation",
                        "D": "Cerebral contusion"
                    },
                    "answer": "B"
                },
                {
                    "question": "A 30-year-old man presents with chest pain and shortness of breath. ECG shows ST-segment elevation in leads V1-V4. What is the most likely diagnosis?",
                    "options": {
                        "A": "Posterior wall MI",
                        "B": "Anterior wall MI",
                        "C": "Lateral wall MI",
                        "D": "Inferior wall MI"
                    },
                    "answer": "B"
                }
            ]

        self.model.eval()

        for i, example in enumerate(custom_examples):
            # Format question for generation
            question_prompt = f"""<bos><start_of_turn>user
Medical Question: {example['question']}

Options:
"""
            for key, value in example['options'].items():
                question_prompt += f"{key}: {value}\n"

            question_prompt += "\nPlease provide the correct answer and explain your reasoning.<end_of_turn>\n<start_of_turn>model\n"

            # Tokenize
            inputs = self.tokenizer(question_prompt, return_tensors="pt").to(self.model.device)

            # Generate response
            generation_config = self.config.generation
            if generation_config.pad_token_id is None:
                generation_config.pad_token_id = self.tokenizer.eos_token_id

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=generation_config.max_new_tokens,
                    temperature=generation_config.temperature,
                    do_sample=generation_config.do_sample,
                    top_p=generation_config.top_p,
                    top_k=generation_config.top_k,
                    repetition_penalty=generation_config.repetition_penalty,
                    pad_token_id=generation_config.pad_token_id
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract model response
            model_response = response.split("<start_of_turn>model\n")[-1]

            print(f"\n{'='*80}")
            print(f"Question {i+1}: {example['question']}")
            print(f"Correct Answer: {example['answer']}")
            print(f"Model Response:\n{model_response}")
            print(f"{'='*80}")

    def load_trained_model(self, model_path: str):
        """Load a previously trained model"""
        logger.info(f"Loading trained model from {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=self.config.quantization.to_bnb_config(),
            device_map=self.config.model.device_map,
            torch_dtype=self.config.model.torch_dtype,
        )

        logger.info("Trained model loaded successfully")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 2B on MedQA using QLoRA")
    parser.add_argument("--config", choices=["quick", "production", "memory_efficient", "env"],
                       default="quick", help="Configuration preset to use")
    parser.add_argument("--output_dir", help="Override output directory")
    parser.add_argument("--model_name", help="Override model name")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", help="Weights & Biases project name")
    parser.add_argument("--evaluate_only", help="Path to trained model for evaluation only")
    parser.add_argument("--config_file", help="Path to JSON configuration file")

    args = parser.parse_args()

    # Load configuration
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

    # Create trainer
    trainer = GemmaMedQATrainer(config)

    if args.evaluate_only:
        # Load trained model and evaluate
        trainer.load_trained_model(args.evaluate_only)
        trainer.evaluate()
    else:
        # Train the model
        trainer.train()

        # Evaluate after training
        trainer.evaluate()


if __name__ == "__main__":
    main()
