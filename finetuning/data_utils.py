"""
Data processing utilities for MedQA dataset fine-tuning.
This module provides functions to load, process, and format the MedQA dataset
for training with the Gemma model.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import random
import re

logger = logging.getLogger(__name__)

class MedQADataProcessor:
    """Data processor for MedQA dataset"""

    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_conversation(self, question: str, options: Dict[str, str],
                          answer: str, include_answer: bool = True) -> str:
        """Format a MedQA example into Gemma conversation format"""

        # Create options string
        options_str = ""
        for key, value in options.items():
            options_str += f"{key}: {value}\n"

        # Create the user turn
        user_turn = f"""<bos><start_of_turn>user
Medical Question: {question}

Options:
{options_str.strip()}

Please provide the correct answer and explain your reasoning.<end_of_turn>"""

        if include_answer:
            # Create the assistant turn with answer
            assistant_turn = f"""<start_of_turn>model
The correct answer is {answer}.

Explanation: This question tests medical knowledge and clinical reasoning. The correct answer is {answer} because it represents the most appropriate clinical approach based on the presented scenario and established medical principles.<end_of_turn><eos>"""

            return user_turn + "\n" + assistant_turn
        else:
            # Return only user turn for inference
            return user_turn + "\n<start_of_turn>model\n"

    def format_conversation_detailed(self, question: str, options: Dict[str, str],
                                   answer: str, explanation: str = None) -> str:
        """Format with detailed explanation if available"""

        options_str = ""
        for key, value in options.items():
            options_str += f"{key}: {value}\n"

        user_turn = f"""<bos><start_of_turn>user
Medical Question: {question}

Options:
{options_str.strip()}

Please provide the correct answer and explain your reasoning.<end_of_turn>"""

        if explanation:
            assistant_turn = f"""<start_of_turn>model
The correct answer is {answer}.

Explanation: {explanation}<end_of_turn><eos>"""
        else:
            assistant_turn = f"""<start_of_turn>model
The correct answer is {answer}.

Explanation: This question tests medical knowledge and clinical reasoning. The correct answer is {answer} because it represents the most appropriate clinical approach based on the presented scenario.<end_of_turn><eos>"""

        return user_turn + "\n" + assistant_turn

    def tokenize_example(self, text: str, add_labels: bool = True) -> Dict[str, torch.Tensor]:
        """Tokenize a single example"""

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        if add_labels:
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()

            # Mask padding tokens in labels
            tokenized["labels"][tokenized["input_ids"] == self.tokenizer.pad_token_id] = -100

        return tokenized

    def process_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Process a single MedQA example"""

        # Format the conversation
        formatted_text = self.format_conversation(
            example['question'],
            example['options'],
            example['answer']
        )

        # Tokenize
        return self.tokenize_example(formatted_text)


class MedQADataLoader:
    """Data loader for MedQA dataset"""

    def __init__(self, dataset_name: str = "GBaker/MedQA-USMLE-4-options"):
        self.dataset_name = dataset_name
        self.raw_dataset = None

    def load_dataset(self) -> Dataset:
        """Load the MedQA dataset"""
        logger.info(f"Loading dataset: {self.dataset_name}")

        self.raw_dataset = load_dataset(self.dataset_name)

        logger.info(f"Dataset loaded successfully")
        logger.info(f"Train size: {len(self.raw_dataset['train'])}")
        logger.info(f"Validation size: {len(self.raw_dataset['validation'])}")
        logger.info(f"Test size: {len(self.raw_dataset['test'])}")

        return self.raw_dataset

    def get_sample_examples(self, split: str = "train", num_samples: int = 5) -> List[Dict[str, Any]]:
        """Get sample examples from the dataset"""
        if self.raw_dataset is None:
            self.load_dataset()

        examples = []
        dataset_split = self.raw_dataset[split]

        for i in range(min(num_samples, len(dataset_split))):
            examples.append(dataset_split[i])

        return examples

    def analyze_dataset(self) -> Dict[str, Any]:
        """Analyze the dataset structure and statistics"""
        if self.raw_dataset is None:
            self.load_dataset()

        analysis = {}

        for split in ['train', 'validation', 'test']:
            split_data = self.raw_dataset[split]

            # Basic statistics
            analysis[split] = {
                'size': len(split_data),
                'columns': split_data.column_names,
                'sample_example': split_data[0] if len(split_data) > 0 else None
            }

            # Analyze question lengths
            question_lengths = [len(example['question']) for example in split_data]
            analysis[split]['question_stats'] = {
                'min_length': min(question_lengths),
                'max_length': max(question_lengths),
                'avg_length': sum(question_lengths) / len(question_lengths)
            }

            # Analyze answer distribution
            answers = [example['answer'] for example in split_data]
            answer_counts = {}
            for answer in answers:
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
            analysis[split]['answer_distribution'] = answer_counts

        return analysis

    def create_processed_dataset(self, processor: MedQADataProcessor,
                               split: str = "train",
                               limit: Optional[int] = None) -> Dataset:
        """Create a processed dataset ready for training"""

        if self.raw_dataset is None:
            self.load_dataset()

        dataset_split = self.raw_dataset[split]

        # Limit dataset size if specified
        if limit:
            dataset_split = dataset_split.select(range(min(limit, len(dataset_split))))

        # Process examples
        processed_examples = []

        logger.info(f"Processing {len(dataset_split)} examples from {split} split...")

        for example in dataset_split:
            try:
                processed = processor.process_example(example)
                processed_examples.append(processed)
            except Exception as e:
                logger.warning(f"Failed to process example: {e}")
                continue

        # Convert to dataset
        if processed_examples:
            # Stack tensors
            stacked_data = {}
            for key in processed_examples[0].keys():
                stacked_data[key] = torch.stack([ex[key].squeeze() for ex in processed_examples])

            # Create dataset
            processed_dataset = Dataset.from_dict({
                k: v.tolist() for k, v in stacked_data.items()
            })

            logger.info(f"Successfully processed {len(processed_dataset)} examples")
            return processed_dataset
        else:
            logger.error("No examples were successfully processed")
            return None


def create_medical_qa_prompts(examples: List[Dict[str, Any]],
                            format_type: str = "conversation") -> List[str]:
    """Create formatted prompts from MedQA examples"""

    prompts = []

    for example in examples:
        question = example['question']
        options = example['options']
        answer = example['answer']

        if format_type == "conversation":
            # Gemma conversation format
            options_str = ""
            for key, value in options.items():
                options_str += f"{key}: {value}\n"

            prompt = f"""<bos><start_of_turn>user
Medical Question: {question}

Options:
{options_str.strip()}

Please provide the correct answer and explain your reasoning.<end_of_turn>
<start_of_turn>model
The correct answer is {answer}.

Explanation: This question tests medical knowledge and clinical reasoning. The correct answer is {answer} because it represents the most appropriate clinical approach based on the presented scenario.<end_of_turn><eos>"""

        elif format_type == "instruction":
            # Instruction format
            options_str = ""
            for key, value in options.items():
                options_str += f"{key}: {value}\n"

            prompt = f"""### Instruction:
Answer the following medical question and provide an explanation.

### Question:
{question}

### Options:
{options_str.strip()}

### Response:
The correct answer is {answer}.

Explanation: This question tests medical knowledge and clinical reasoning. The correct answer is {answer} because it represents the most appropriate clinical approach based on the presented scenario."""

        else:
            # Simple format
            options_str = " ".join([f"{k}: {v}" for k, v in options.items()])
            prompt = f"Question: {question} Options: {options_str} Answer: {answer}"

        prompts.append(prompt)

    return prompts


def validate_dataset_format(dataset: Dataset) -> Tuple[bool, List[str]]:
    """Validate that the dataset has the correct format"""

    errors = []

    # Check required columns
    required_columns = ['question', 'options', 'answer']
    for col in required_columns:
        if col not in dataset.column_names:
            errors.append(f"Missing required column: {col}")

    if errors:
        return False, errors

    # Check sample entries
    sample_size = min(10, len(dataset))
    for i in range(sample_size):
        example = dataset[i]

        # Check question
        if not isinstance(example['question'], str) or not example['question'].strip():
            errors.append(f"Example {i}: Invalid question")

        # Check options
        if not isinstance(example['options'], dict):
            errors.append(f"Example {i}: Options must be a dictionary")
        else:
            # Check if options contain A, B, C, D
            expected_keys = set(['A', 'B', 'C', 'D'])
            if set(example['options'].keys()) != expected_keys:
                errors.append(f"Example {i}: Options must contain exactly A, B, C, D")

        # Check answer
        if example['answer'] not in ['A', 'B', 'C', 'D']:
            errors.append(f"Example {i}: Answer must be A, B, C, or D")

    return len(errors) == 0, errors


def save_processed_examples(examples: List[Dict[str, Any]],
                          output_path: str,
                          format_type: str = "json"):
    """Save processed examples to file"""

    if format_type == "json":
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)
    elif format_type == "jsonl":
        with open(output_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

    logger.info(f"Saved {len(examples)} examples to {output_path}")


def load_processed_examples(input_path: str,
                          format_type: str = "json") -> List[Dict[str, Any]]:
    """Load processed examples from file"""

    if format_type == "json":
        with open(input_path, 'r') as f:
            examples = json.load(f)
    elif format_type == "jsonl":
        examples = []
        with open(input_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line.strip()))
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

    logger.info(f"Loaded {len(examples)} examples from {input_path}")
    return examples


def create_train_val_split(dataset: Dataset,
                         val_ratio: float = 0.1,
                         seed: int = 42) -> Tuple[Dataset, Dataset]:
    """Create train/validation split from dataset"""

    random.seed(seed)

    # Create indices
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)

    indices = list(range(total_size))
    random.shuffle(indices)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    # Create splits
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)

    logger.info(f"Created train/val split: {len(train_dataset)}/{len(val_dataset)}")

    return train_dataset, val_dataset


def get_dataset_statistics(dataset: Dataset) -> Dict[str, Any]:
    """Get comprehensive statistics about the dataset"""

    stats = {
        'total_examples': len(dataset),
        'columns': dataset.column_names
    }

    # Text length statistics
    if 'question' in dataset.column_names:
        questions = [ex['question'] for ex in dataset]
        question_lengths = [len(q) for q in questions]

        stats['question_lengths'] = {
            'min': min(question_lengths),
            'max': max(question_lengths),
            'mean': sum(question_lengths) / len(question_lengths),
            'median': sorted(question_lengths)[len(question_lengths) // 2]
        }

    # Answer distribution
    if 'answer' in dataset.column_names:
        answers = [ex['answer'] for ex in dataset]
        answer_counts = {}
        for answer in answers:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1

        stats['answer_distribution'] = answer_counts

    return stats
