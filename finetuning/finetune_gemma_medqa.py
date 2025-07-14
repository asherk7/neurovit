import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json
import os
from tqdm import tqdm
import argparse
import logging
from typing import Dict, Any
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedQAFineTuner:
    def __init__(self,
                 model_name: str = "google/gemma-2b-it",
                 output_dir: str = "./finetuned_gemma_medqa",
                 use_wandb: bool = False,
                 wandb_project: str = "gemma-medqa-finetune"):

        self.model_name = model_name
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(project=self.wandb_project, name=f"gemma-2b-medqa-qlora")

        # QLoRA configuration
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # LoRA configuration
        self.lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Alpha parameter for LoRA scaling
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            warmup_steps=100,
            lr_scheduler_type="linear",
            report_to="wandb" if self.use_wandb else None,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            push_to_hub=False,
            gradient_checkpointing=True,
        )

        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None

    def load_model_and_tokenizer(self):
        """Load and prepare the model and tokenizer"""
        logger.info(f"Loading model and tokenizer: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Load model with QLoRA config
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False  # Required for gradient checkpointing
        )

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        logger.info("Model and tokenizer loaded successfully!")

    def format_medqa_prompt(self, question: str, options: Dict[str, str], answer: str) -> str:
        """Format MedQA data into a conversation format suitable for Gemma"""

        # Create options string
        options_str = ""
        for key, value in options.items():
            options_str += f"{key}: {value}\n"

        # Create the prompt in a medical Q&A format
        prompt = f"""<bos><start_of_turn>user
Medical Question: {question}

Options:
{options_str.strip()}

Please provide the correct answer and explain your reasoning.<end_of_turn>
<start_of_turn>model
The correct answer is {answer}.

Explanation: This is a medical question that requires understanding of clinical concepts. The correct answer is {answer} because """

        return prompt

    def load_and_process_dataset(self):
        """Load and process the MedQA dataset"""
        logger.info("Loading MedQA dataset...")

        # Load MedQA dataset
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

        def process_example(example):
            """Process a single example"""
            question = example['question']
            options = example['options']
            answer = example['answer']

            # Format the prompt
            formatted_text = self.format_medqa_prompt(question, options, answer)

            # Tokenize
            tokenized = self.tokenizer(
                formatted_text,
                truncation=True,
                max_length=1024,
                padding="max_length",
                return_tensors="pt"
            )

            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()

            return tokenized

        # Process the datasets
        logger.info("Processing training dataset...")
        self.train_dataset = dataset["train"].map(
            process_example,
            batched=False,
            remove_columns=dataset["train"].column_names,
            desc="Processing training examples"
        )

        logger.info("Processing validation dataset...")
        self.eval_dataset = dataset["validation"].map(
            process_example,
            batched=False,
            remove_columns=dataset["validation"].column_names,
            desc="Processing validation examples"
        )

        # Limit dataset size for faster training (optional)
        if len(self.train_dataset) > 10000:
            self.train_dataset = self.train_dataset.select(range(10000))
        if len(self.eval_dataset) > 1000:
            self.eval_dataset = self.eval_dataset.select(range(1000))

        logger.info(f"Training dataset size: {len(self.train_dataset)}")
        logger.info(f"Validation dataset size: {len(self.eval_dataset)}")

    def create_trainer(self):
        """Create the trainer"""
        logger.info("Creating trainer...")

        # Data collator
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            return_tensors="pt",
            pad_to_multiple_of=8
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        return trainer

    def train(self):
        """Run the training process"""
        logger.info("Starting training...")

        # Load model and tokenizer
        self.load_model_and_tokenizer()

        # Load and process dataset
        self.load_and_process_dataset()

        # Create trainer
        trainer = self.create_trainer()

        # Train the model
        trainer.train()

        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()

        # Save tokenizer
        self.tokenizer.save_pretrained(self.output_dir)

        logger.info(f"Training completed! Model saved to {self.output_dir}")

        if self.use_wandb:
            wandb.finish()

    def evaluate_model(self, test_questions: list = None):
        """Evaluate the fine-tuned model on sample questions"""
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Please run train() first.")
            return

        logger.info("Evaluating model...")

        # Sample test questions if not provided
        if test_questions is None:
            test_questions = [
                {
                    "question": "A 45-year-old man presents with chest pain. What is the most likely diagnosis?",
                    "options": {
                        "A": "Myocardial infarction",
                        "B": "Gastroesophageal reflux",
                        "C": "Pulmonary embolism",
                        "D": "Anxiety disorder"
                    },
                    "answer": "A"
                }
            ]

        self.model.eval()

        for i, example in enumerate(test_questions):
            # Format the question (without answer for generation)
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
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract just the model's response
            model_response = response.split("<start_of_turn>model\n")[-1]

            print(f"\n{'='*50}")
            print(f"Question {i+1}: {example['question']}")
            print(f"Correct Answer: {example['answer']}")
            print(f"Model Response: {model_response}")
            print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 2B on MedQA dataset using QLoRA")
    parser.add_argument("--model_name", default="google/gemma-2b-it", help="Model name from HuggingFace")
    parser.add_argument("--output_dir", default="./finetuned_gemma_medqa", help="Output directory for the fine-tuned model")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", default="gemma-medqa-finetune", help="Weights & Biases project name")
    parser.add_argument("--evaluate_only", action="store_true", help="Only evaluate the model without training")

    args = parser.parse_args()

    # Create fine-tuner
    fine_tuner = MedQAFineTuner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project
    )

    if args.evaluate_only:
        # Load the fine-tuned model for evaluation
        fine_tuner.model_name = args.output_dir  # Use the fine-tuned model path
        fine_tuner.load_model_and_tokenizer()
        fine_tuner.evaluate_model()
    else:
        # Run training
        fine_tuner.train()

        # Evaluate after training
        fine_tuner.evaluate_model()

if __name__ == "__main__":
    main()
