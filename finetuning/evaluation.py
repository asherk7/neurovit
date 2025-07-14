"""
Evaluation utilities for fine-tuned Gemma model on MedQA dataset.
This module provides comprehensive evaluation metrics and utilities.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import json
import logging
from pathlib import Path
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)

class MedQAEvaluator:
    """Evaluator for fine-tuned Gemma model on MedQA dataset"""

    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

        logger.info("Model loaded successfully")

    def format_question_for_inference(self, question: str, options: Dict[str, str]) -> str:
        """Format a question for inference"""
        options_str = ""
        for key, value in options.items():
            options_str += f"{key}: {value}\n"

        prompt = f"""<bos><start_of_turn>user
Medical Question: {question}

Options:
{options_str.strip()}

Please provide the correct answer and explain your reasoning.<end_of_turn>
<start_of_turn>model
"""
        return prompt

    def extract_answer_from_response(self, response: str) -> Optional[str]:
        """Extract the answer choice (A, B, C, or D) from model response"""
        # Remove the input prompt part
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1]

        # Look for patterns like "The correct answer is A" or "Answer: A"
        patterns = [
            r"(?:correct answer is|answer is)\s*([ABCD])",
            r"(?:Answer:|answer:)\s*([ABCD])",
            r"^([ABCD])\b",  # Answer at the beginning
            r"\b([ABCD])\b.*(?:correct|right|answer)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # If no clear pattern, look for the first occurrence of A, B, C, or D
        for char in ['A', 'B', 'C', 'D']:
            if char in response.upper():
                return char

        return None

    def generate_answer(self, question: str, options: Dict[str, str],
                       max_new_tokens: int = 200, temperature: float = 0.3) -> Tuple[str, str]:
        """Generate an answer for a given question"""
        if self.model is None:
            self.load_model()

        # Format the prompt
        prompt = self.format_question_for_inference(question, options)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the full response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the model's response
        model_response = full_response.split("<start_of_turn>model")[-1].strip()

        # Extract the answer
        predicted_answer = self.extract_answer_from_response(model_response)

        return predicted_answer, model_response

    def evaluate_dataset(self, dataset: Dataset,
                        max_examples: Optional[int] = None,
                        batch_size: int = 1) -> Dict[str, Any]:
        """Evaluate the model on a dataset"""
        if self.model is None:
            self.load_model()

        # Limit dataset size if specified
        if max_examples and len(dataset) > max_examples:
            dataset = dataset.select(range(max_examples))

        predictions = []
        true_answers = []
        responses = []
        failed_extractions = 0

        logger.info(f"Evaluating on {len(dataset)} examples...")

        for i, example in enumerate(tqdm(dataset, desc="Evaluating")):
            try:
                # Generate answer
                predicted_answer, response = self.generate_answer(
                    example['question'],
                    example['options']
                )

                predictions.append(predicted_answer)
                true_answers.append(example['answer'])
                responses.append(response)

                if predicted_answer is None:
                    failed_extractions += 1
                    logger.warning(f"Failed to extract answer from response {i}")

            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                predictions.append(None)
                true_answers.append(example['answer'])
                responses.append("")
                failed_extractions += 1

        # Calculate metrics
        metrics = self.calculate_metrics(predictions, true_answers)

        # Add additional info
        metrics['total_examples'] = len(dataset)
        metrics['failed_extractions'] = failed_extractions
        metrics['extraction_success_rate'] = 1 - (failed_extractions / len(dataset))

        return {
            'metrics': metrics,
            'predictions': predictions,
            'true_answers': true_answers,
            'responses': responses
        }

    def calculate_metrics(self, predictions: List[str], true_answers: List[str]) -> Dict[str, float]:
        """Calculate evaluation metrics"""

        # Filter out None predictions for metric calculation
        valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_true_answers = [true_answers[i] for i in valid_indices]

        if not valid_predictions:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'valid_predictions': 0
            }

        # Calculate accuracy
        accuracy = accuracy_score(valid_true_answers, valid_predictions)

        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_true_answers, valid_predictions, average='weighted', zero_division=0
        )

        # Calculate per-class metrics
        labels = ['A', 'B', 'C', 'D']
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            valid_true_answers, valid_predictions, labels=labels, zero_division=0
        )

        per_class_metrics = {}
        for i, label in enumerate(labels):
            per_class_metrics[f'precision_{label}'] = per_class_precision[i]
            per_class_metrics[f'recall_{label}'] = per_class_recall[i]
            per_class_metrics[f'f1_{label}'] = per_class_f1[i]
            per_class_metrics[f'support_{label}'] = support[i]

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'valid_predictions': len(valid_predictions),
            **per_class_metrics
        }

    def create_confusion_matrix(self, predictions: List[str], true_answers: List[str],
                               save_path: Optional[str] = None) -> np.ndarray:
        """Create and optionally save confusion matrix"""

        # Filter out None predictions
        valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
        valid_predictions = [predictions[i] for i in valid_indices]
        valid_true_answers = [true_answers[i] for i in valid_indices]

        if not valid_predictions:
            logger.warning("No valid predictions to create confusion matrix")
            return np.array([])

        # Create confusion matrix
        labels = ['A', 'B', 'C', 'D']
        cm = confusion_matrix(valid_true_answers, valid_predictions, labels=labels)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

        return cm

    def analyze_errors(self, predictions: List[str], true_answers: List[str],
                      responses: List[str], dataset: Dataset) -> Dict[str, Any]:
        """Analyze prediction errors"""

        errors = []
        correct_predictions = 0

        for i, (pred, true, response) in enumerate(zip(predictions, true_answers, responses)):
            if pred != true:
                errors.append({
                    'index': i,
                    'question': dataset[i]['question'],
                    'options': dataset[i]['options'],
                    'true_answer': true,
                    'predicted_answer': pred,
                    'response': response
                })
            else:
                correct_predictions += 1

        # Analyze error patterns
        error_patterns = {
            'extraction_failures': len([e for e in errors if e['predicted_answer'] is None]),
            'wrong_predictions': len([e for e in errors if e['predicted_answer'] is not None]),
            'most_confused_pairs': self._find_confusion_pairs(predictions, true_answers)
        }

        return {
            'total_errors': len(errors),
            'correct_predictions': correct_predictions,
            'error_rate': len(errors) / len(predictions) if predictions else 0,
            'error_patterns': error_patterns,
            'error_examples': errors[:10]  # First 10 errors for inspection
        }

    def _find_confusion_pairs(self, predictions: List[str], true_answers: List[str]) -> List[Tuple[str, str, int]]:
        """Find the most common confusion pairs"""
        confusion_counts = {}

        for pred, true in zip(predictions, true_answers):
            if pred != true and pred is not None:
                pair = (true, pred)
                confusion_counts[pair] = confusion_counts.get(pair, 0) + 1

        # Sort by frequency
        sorted_pairs = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)

        return [(pair[0], pair[1], count) for pair, count in sorted_pairs[:5]]

    def benchmark_inference_speed(self, dataset: Dataset, num_samples: int = 100) -> Dict[str, float]:
        """Benchmark inference speed"""
        if self.model is None:
            self.load_model()

        # Select random samples
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        sample_dataset = dataset.select(indices)

        times = []

        logger.info(f"Benchmarking inference speed on {len(sample_dataset)} samples...")

        for example in tqdm(sample_dataset, desc="Benchmarking"):
            start_time = time.time()

            try:
                self.generate_answer(example['question'], example['options'])
                inference_time = time.time() - start_time
                times.append(inference_time)
            except Exception as e:
                logger.error(f"Error during benchmarking: {e}")
                continue

        if times:
            return {
                'mean_inference_time': np.mean(times),
                'median_inference_time': np.median(times),
                'std_inference_time': np.std(times),
                'min_inference_time': np.min(times),
                'max_inference_time': np.max(times),
                'samples_per_second': 1.0 / np.mean(times)
            }
        else:
            return {}

    def generate_evaluation_report(self, evaluation_results: Dict[str, Any],
                                 output_path: str):
        """Generate a comprehensive evaluation report"""

        metrics = evaluation_results['metrics']

        report = {
            'model_path': self.model_path,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_examples': metrics.get('total_examples', 0),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1'],
                'extraction_success_rate': metrics.get('extraction_success_rate', 0),
                'failed_extractions': metrics.get('failed_extractions', 0)
            },
            'per_class_metrics': {
                label: {
                    'precision': metrics.get(f'precision_{label}', 0),
                    'recall': metrics.get(f'recall_{label}', 0),
                    'f1': metrics.get(f'f1_{label}', 0),
                    'support': metrics.get(f'support_{label}', 0)
                }
                for label in ['A', 'B', 'C', 'D']
            },
            'detailed_results': evaluation_results
        }

        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to {output_path}")

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Model: {self.model_path}")
        print(f"Total Examples: {metrics.get('total_examples', 0)}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Extraction Success Rate: {metrics.get('extraction_success_rate', 0):.4f}")
        print("="*60)

        return report

def compare_models(model_paths: List[str], dataset: Dataset,
                  max_examples: Optional[int] = None) -> Dict[str, Any]:
    """Compare multiple models on the same dataset"""

    results = {}

    for model_path in model_paths:
        logger.info(f"Evaluating model: {model_path}")

        evaluator = MedQAEvaluator(model_path)
        evaluation_results = evaluator.evaluate_dataset(dataset, max_examples)

        results[model_path] = evaluation_results['metrics']

    # Create comparison summary
    comparison = {
        'models': list(results.keys()),
        'metrics_comparison': {}
    }

    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']

    for metric in metrics_to_compare:
        comparison['metrics_comparison'][metric] = {
            model: results[model][metric]
            for model in results.keys()
        }

    return comparison

def evaluate_sample_questions(model_path: str, questions: List[Dict[str, Any]]) -> None:
    """Evaluate model on sample questions and print results"""

    evaluator = MedQAEvaluator(model_path)
    evaluator.load_model()

    print("\n" + "="*80)
    print("SAMPLE EVALUATION")
    print("="*80)

    for i, example in enumerate(questions, 1):
        print(f"\nQuestion {i}:")
        print(f"Q: {example['question']}")
        print("\nOptions:")
        for key, value in example['options'].items():
            print(f"  {key}: {value}")

        predicted_answer, response = evaluator.generate_answer(
            example['question'],
            example['options']
        )

        print(f"\nCorrect Answer: {example['answer']}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Correct: {'✓' if predicted_answer == example['answer'] else '✗'}")
        print(f"\nModel Response:")
        print(response)
        print("-" * 80)
