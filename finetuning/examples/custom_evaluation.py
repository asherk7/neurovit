#!/usr/bin/env python3
"""
Custom evaluation example for fine-tuned Gemma 2B on MedQA dataset.
This script demonstrates advanced evaluation techniques and custom metrics.
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation import MedQAEvaluator, compare_models
from data_utils import MedQADataLoader
from datasets import load_dataset
import torch

def evaluate_by_question_type(evaluator: MedQAEvaluator, dataset, output_dir: str):
    """Evaluate model performance by question type/medical domain"""

    print("\n" + "="*60)
    print("EVALUATION BY QUESTION TYPE")
    print("="*60)

    # Medical domain keywords for classification
    domain_keywords = {
        'cardiology': ['heart', 'cardiac', 'coronary', 'myocardial', 'ecg', 'chest pain', 'arrhythmia'],
        'neurology': ['brain', 'neurological', 'seizure', 'stroke', 'headache', 'consciousness'],
        'pulmonology': ['lung', 'respiratory', 'breathing', 'pneumonia', 'asthma', 'copd'],
        'gastroenterology': ['stomach', 'intestinal', 'liver', 'digestive', 'abdominal', 'nausea'],
        'endocrinology': ['diabetes', 'thyroid', 'hormone', 'insulin', 'glucose', 'endocrine'],
        'infectious_disease': ['infection', 'fever', 'antibiotic', 'bacterial', 'viral', 'sepsis'],
        'oncology': ['cancer', 'tumor', 'malignant', 'chemotherapy', 'metastasis', 'biopsy'],
        'psychiatry': ['depression', 'anxiety', 'psychiatric', 'mental', 'mood', 'psychotic'],
        'emergency': ['emergency', 'trauma', 'acute', 'shock', 'collapse', 'unconscious']
    }

    # Classify questions by domain
    domain_results = {}

    for domain, keywords in domain_keywords.items():
        domain_examples = []

        for example in dataset:
            question_lower = example['question'].lower()
            if any(keyword in question_lower for keyword in keywords):
                domain_examples.append(example)

        if len(domain_examples) >= 5:  # Only evaluate domains with sufficient examples
            print(f"\nEvaluating {domain} ({len(domain_examples)} examples)...")

            # Create dataset for this domain
            domain_dataset = dataset.filter(lambda x: any(keyword in x['question'].lower() for keyword in keywords))

            # Evaluate
            results = evaluator.evaluate_dataset(domain_dataset, max_examples=min(50, len(domain_examples)))

            domain_results[domain] = {
                'num_examples': len(domain_examples),
                'accuracy': results['metrics']['accuracy'],
                'precision': results['metrics']['precision'],
                'recall': results['metrics']['recall'],
                'f1': results['metrics']['f1']
            }

            print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
            print(f"  F1 Score: {results['metrics']['f1']:.4f}")

    # Save domain results
    with open(f"{output_dir}/domain_evaluation.json", 'w') as f:
        json.dump(domain_results, f, indent=2)

    # Create visualization
    if domain_results:
        domains = list(domain_results.keys())
        accuracies = [domain_results[d]['accuracy'] for d in domains]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(domains, accuracies, color='skyblue', alpha=0.7)
        plt.title('Model Performance by Medical Domain')
        plt.xlabel('Medical Domain')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/domain_performance.png", dpi=300, bbox_inches='tight')
        plt.show()

    return domain_results

def evaluate_by_difficulty(evaluator: MedQAEvaluator, dataset, output_dir: str):
    """Evaluate model performance by question difficulty"""

    print("\n" + "="*60)
    print("EVALUATION BY QUESTION DIFFICULTY")
    print("="*60)

    # Simple heuristics for difficulty classification
    def classify_difficulty(question: str, options: Dict[str, str]) -> str:
        question_lower = question.lower()

        # Complex indicators
        complex_indicators = [
            'differential diagnosis', 'most likely', 'next step', 'complications',
            'pathophysiology', 'mechanism', 'side effects', 'contraindications'
        ]

        # Simple indicators
        simple_indicators = [
            'definition', 'caused by', 'characteristic', 'typical', 'common'
        ]

        # Length-based difficulty
        if len(question) > 200:
            return 'hard'
        elif len(question) < 100:
            return 'easy'

        # Content-based difficulty
        if any(indicator in question_lower for indicator in complex_indicators):
            return 'hard'
        elif any(indicator in question_lower for indicator in simple_indicators):
            return 'easy'

        return 'medium'

    # Classify questions by difficulty
    difficulty_groups = {'easy': [], 'medium': [], 'hard': []}

    for example in dataset:
        difficulty = classify_difficulty(example['question'], example['options'])
        difficulty_groups[difficulty].append(example)

    difficulty_results = {}

    for difficulty, examples in difficulty_groups.items():
        if len(examples) >= 10:  # Only evaluate groups with sufficient examples
            print(f"\nEvaluating {difficulty} questions ({len(examples)} examples)...")

            # Create dataset for this difficulty
            difficulty_dataset = dataset.filter(
                lambda x: classify_difficulty(x['question'], x['options']) == difficulty
            )

            # Evaluate
            results = evaluator.evaluate_dataset(difficulty_dataset, max_examples=min(100, len(examples)))

            difficulty_results[difficulty] = {
                'num_examples': len(examples),
                'accuracy': results['metrics']['accuracy'],
                'precision': results['metrics']['precision'],
                'recall': results['metrics']['recall'],
                'f1': results['metrics']['f1']
            }

            print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
            print(f"  F1 Score: {results['metrics']['f1']:.4f}")

    # Save difficulty results
    with open(f"{output_dir}/difficulty_evaluation.json", 'w') as f:
        json.dump(difficulty_results, f, indent=2)

    # Create visualization
    if difficulty_results:
        difficulties = ['easy', 'medium', 'hard']
        accuracies = [difficulty_results.get(d, {}).get('accuracy', 0) for d in difficulties]

        plt.figure(figsize=(8, 6))
        colors = ['green', 'orange', 'red']
        bars = plt.bar(difficulties, accuracies, color=colors, alpha=0.7)
        plt.title('Model Performance by Question Difficulty')
        plt.xlabel('Question Difficulty')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/difficulty_performance.png", dpi=300, bbox_inches='tight')
        plt.show()

    return difficulty_results

def analyze_response_patterns(evaluator: MedQAEvaluator, dataset, output_dir: str):
    """Analyze model response patterns and common mistakes"""

    print("\n" + "="*60)
    print("RESPONSE PATTERN ANALYSIS")
    print("="*60)

    # Evaluate a sample of the dataset
    sample_size = min(200, len(dataset))
    sample_dataset = dataset.shuffle(seed=42).select(range(sample_size))

    results = evaluator.evaluate_dataset(sample_dataset)

    predictions = results['predictions']
    true_answers = results['true_answers']
    responses = results['responses']

    # Analyze response characteristics
    response_analysis = {
        'total_responses': len(responses),
        'successful_extractions': sum(1 for p in predictions if p is not None),
        'average_response_length': np.mean([len(r) for r in responses]),
        'confidence_patterns': {},
        'explanation_quality': {}
    }

    # Analyze confidence patterns
    confidence_phrases = {
        'high_confidence': ['correct answer is', 'definitely', 'clearly', 'obviously'],
        'medium_confidence': ['likely', 'probably', 'most appropriate'],
        'low_confidence': ['possibly', 'may be', 'could be', 'uncertain']
    }

    for confidence_level, phrases in confidence_phrases.items():
        count = sum(1 for response in responses
                   if any(phrase in response.lower() for phrase in phrases))
        response_analysis['confidence_patterns'][confidence_level] = {
            'count': count,
            'percentage': count / len(responses) * 100
        }

    # Analyze explanation quality
    explanation_indicators = {
        'medical_reasoning': ['because', 'pathophysiology', 'mechanism', 'clinical'],
        'evidence_based': ['studies', 'research', 'guidelines', 'evidence'],
        'differential': ['differential', 'alternative', 'other options', 'ruled out']
    }

    for indicator_type, indicators in explanation_indicators.items():
        count = sum(1 for response in responses
                   if any(indicator in response.lower() for indicator in indicators))
        response_analysis['explanation_quality'][indicator_type] = {
            'count': count,
            'percentage': count / len(responses) * 100
        }

    # Find common error patterns
    error_patterns = {}
    incorrect_indices = [i for i, (pred, true) in enumerate(zip(predictions, true_answers))
                        if pred != true and pred is not None]

    if incorrect_indices:
        # Analyze incorrect responses
        incorrect_responses = [responses[i] for i in incorrect_indices]

        error_keywords = {
            'overconfidence': ['definitely', 'clearly', 'obviously'],
            'incomplete_reasoning': ['because', 'due to', 'therefore'],
            'factual_errors': ['incorrect', 'wrong', 'mistake']
        }

        for error_type, keywords in error_keywords.items():
            count = sum(1 for response in incorrect_responses
                       if any(keyword in response.lower() for keyword in keywords))
            error_patterns[error_type] = {
                'count': count,
                'percentage': count / len(incorrect_responses) * 100 if incorrect_responses else 0
            }

    response_analysis['error_patterns'] = error_patterns

    # Save analysis
    with open(f"{output_dir}/response_analysis.json", 'w') as f:
        json.dump(response_analysis, f, indent=2)

    # Print summary
    print(f"✓ Analyzed {response_analysis['total_responses']} responses")
    print(f"✓ Successful extractions: {response_analysis['successful_extractions']}")
    print(f"✓ Average response length: {response_analysis['average_response_length']:.1f} characters")

    print("\nConfidence Patterns:")
    for level, data in response_analysis['confidence_patterns'].items():
        print(f"  {level}: {data['count']} responses ({data['percentage']:.1f}%)")

    print("\nExplanation Quality:")
    for quality, data in response_analysis['explanation_quality'].items():
        print(f"  {quality}: {data['count']} responses ({data['percentage']:.1f}%)")

    return response_analysis

def create_custom_evaluation_report(model_path: str, output_dir: str):
    """Create a comprehensive custom evaluation report"""

    print("\n" + "="*80)
    print("CUSTOM EVALUATION REPORT")
    print("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load evaluator and dataset
    evaluator = MedQAEvaluator(model_path)
    data_loader = MedQADataLoader()
    dataset = data_loader.load_dataset()

    # Use test dataset for evaluation
    test_dataset = dataset['test']

    # 1. Basic evaluation
    print("\n1. Running basic evaluation...")
    basic_results = evaluator.evaluate_dataset(test_dataset, max_examples=500)

    # 2. Domain-specific evaluation
    print("\n2. Running domain-specific evaluation...")
    domain_results = evaluate_by_question_type(evaluator, test_dataset, output_dir)

    # 3. Difficulty-based evaluation
    print("\n3. Running difficulty-based evaluation...")
    difficulty_results = evaluate_by_difficulty(evaluator, test_dataset, output_dir)

    # 4. Response pattern analysis
    print("\n4. Running response pattern analysis...")
    response_analysis = analyze_response_patterns(evaluator, test_dataset, output_dir)

    # 5. Create confusion matrix
    print("\n5. Creating confusion matrix...")
    confusion_matrix = evaluator.create_confusion_matrix(
        basic_results['predictions'],
        basic_results['true_answers'],
        save_path=f"{output_dir}/confusion_matrix.png"
    )

    # 6. Error analysis
    print("\n6. Running error analysis...")
    error_analysis = evaluator.analyze_errors(
        basic_results['predictions'],
        basic_results['true_answers'],
        basic_results['responses'],
        test_dataset.select(range(min(500, len(test_dataset))))
    )

    # 7. Speed benchmark
    print("\n7. Running speed benchmark...")
    speed_metrics = evaluator.benchmark_inference_speed(test_dataset, num_samples=50)

    # Compile comprehensive report
    comprehensive_report = {
        'model_path': model_path,
        'evaluation_summary': {
            'basic_metrics': basic_results['metrics'],
            'domain_performance': domain_results,
            'difficulty_performance': difficulty_results,
            'response_patterns': response_analysis,
            'error_analysis': error_analysis,
            'speed_metrics': speed_metrics
        },
        'recommendations': generate_recommendations(basic_results, domain_results, difficulty_results)
    }

    # Save comprehensive report
    with open(f"{output_dir}/comprehensive_evaluation_report.json", 'w') as f:
        json.dump(comprehensive_report, f, indent=2, default=str)

    # Generate final summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    print(f"Overall Accuracy: {basic_results['metrics']['accuracy']:.4f}")
    print(f"Overall F1 Score: {basic_results['metrics']['f1']:.4f}")

    if domain_results:
        best_domain = max(domain_results.keys(), key=lambda x: domain_results[x]['accuracy'])
        worst_domain = min(domain_results.keys(), key=lambda x: domain_results[x]['accuracy'])
        print(f"Best Domain: {best_domain} ({domain_results[best_domain]['accuracy']:.4f})")
        print(f"Worst Domain: {worst_domain} ({domain_results[worst_domain]['accuracy']:.4f})")

    if speed_metrics:
        print(f"Average Inference Time: {speed_metrics['mean_inference_time']:.2f}s")
        print(f"Throughput: {speed_metrics['samples_per_second']:.2f} samples/sec")

    print(f"\nDetailed results saved to: {output_dir}/")
    print("Files generated:")
    print("- comprehensive_evaluation_report.json")
    print("- domain_evaluation.json")
    print("- difficulty_evaluation.json")
    print("- response_analysis.json")
    print("- confusion_matrix.png")
    print("- domain_performance.png")
    print("- difficulty_performance.png")

    return comprehensive_report

def generate_recommendations(basic_results, domain_results, difficulty_results):
    """Generate recommendations based on evaluation results"""

    recommendations = []

    # Overall performance recommendations
    accuracy = basic_results['metrics']['accuracy']
    if accuracy < 0.6:
        recommendations.append("Consider increasing training epochs or improving data quality")
    elif accuracy < 0.8:
        recommendations.append("Good performance, consider fine-tuning hyperparameters")
    else:
        recommendations.append("Excellent performance, model is ready for deployment")

    # Domain-specific recommendations
    if domain_results:
        domain_accuracies = [domain_results[d]['accuracy'] for d in domain_results.keys()]
        if max(domain_accuracies) - min(domain_accuracies) > 0.2:
            recommendations.append("Large performance variance across domains - consider domain-specific training")

    # Difficulty-specific recommendations
    if difficulty_results:
        if 'hard' in difficulty_results and difficulty_results['hard']['accuracy'] < 0.5:
            recommendations.append("Poor performance on difficult questions - consider adding more complex examples")

    return recommendations

def main():
    """Main function for custom evaluation"""

    # Configuration
    MODEL_PATH = "./finetuned_gemma_medqa"  # Update this path
    OUTPUT_DIR = "./custom_evaluation_results"

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train a model first or update the MODEL_PATH variable")
        return

    # Run comprehensive evaluation
    try:
        report = create_custom_evaluation_report(MODEL_PATH, OUTPUT_DIR)
        print("\n✓ Custom evaluation completed successfully!")

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        print("Please check the model path and ensure all dependencies are installed")

if __name__ == "__main__":
    main()
