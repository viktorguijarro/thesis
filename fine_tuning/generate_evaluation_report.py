#!/usr/bin/env python3
"""
Generate comprehensive evaluation report from four-way comparison results.
Updated to use the correct model and configuration names for Qwen2-0.5B.
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

def generate_report(results_file: str, output_file: str = None):
    """Generate comprehensive evaluation report."""

    # Load results
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at '{results_file}'")
        print("Please run the evaluation script first.")
        return

    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_report_{timestamp}.txt"

    # --- DEFINE THE EXACT CONFIGURATION NAMES USED IN THE EVALUATION SCRIPT ---
    # This is the key change to make the script work with the new results file.
    BASE_MODEL_NAME = "Base Model (Qwen2-0.5B)"
    FINETUNED_MODEL_NAME = "Fine-tuned Model (qwen2finetuned)"
    BASE_RAG_NAME = "Base Model + RAG"
    FINETUNED_RAG_NAME = "Fine-tuned Model + RAG"

    # Generate report
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("FOUR-WAY MODEL COMPARISON EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Test Questions: {results['metadata']['test_questions_count']}")
    report_lines.append(f"Knowledge Base Size: {results['metadata']['knowledge_base_size']}")
    report_lines.append(f"Test File: {results['metadata']['test_file_used']}")
    report_lines.append("")

    # Configuration details
    report_lines.append("CONFIGURATIONS TESTED:")
    report_lines.append("-" * 40)
    for i, config in enumerate(results['configurations'], 1):
        report_lines.append(f"{i}. {config['name']}")
        report_lines.append(f"   Model: {config['model']}")
        report_lines.append(f"   RAG: {'Yes' if config['use_rag'] else 'No'}")
    report_lines.append("")

    # Performance summary
    report_lines.append("PERFORMANCE SUMMARY:")
    report_lines.append("-" * 40)

    metrics = ['keyword_relevance', 'domain_specificity', 'contextual_accuracy',
              'response_completeness', 'informativeness', 'linguistic_quality', 'factual_consistency']

    config_scores = {}

    for config in results['configurations']:
        config_name = config['name']
        config_results = results['results'][config_name]

        # Calculate metrics
        metric_scores = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in config_results] # Use .get(metric, 0) for safety
            metric_scores[metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }

        # Overall score
        overall_scores = []
        for result in config_results:
            score = np.mean([result.get(metric, 0) for metric in metrics])
            overall_scores.append(score)

        overall_mean = np.mean(overall_scores)
        overall_std = np.std(overall_scores)

        config_scores[config_name] = {
            'overall': {'mean': overall_mean, 'std': overall_std},
            'metrics': metric_scores
        }

        report_lines.append(f"{config_name}:")
        report_lines.append(f"  Overall Score: {overall_mean:.3f} ± {overall_std:.3f}")

        for metric in metrics:
            mean_val = metric_scores[metric]['mean']
            std_val = metric_scores[metric]['std']
            report_lines.append(f"  {metric.replace('_', ' ').title()}: {mean_val:.3f} ± {std_val:.3f}")
        report_lines.append("")

    # Ranking
    report_lines.append("PERFORMANCE RANKING (BY OVERALL SCORE):")
    report_lines.append("-" * 40)

    # Sort by overall score
    sorted_configs = sorted(config_scores.items(), key=lambda x: x[1]['overall']['mean'], reverse=True)

    for i, (config_name, scores) in enumerate(sorted_configs, 1):
        overall_score = scores['overall']['mean']
        report_lines.append(f"{i}. {config_name}: {overall_score:.3f}")

    report_lines.append("")

    # Improvement analysis
    report_lines.append("IMPROVEMENT ANALYSIS:")
    report_lines.append("-" * 40)

    # Check if all expected models are in the results before doing calculations
    if BASE_MODEL_NAME in config_scores:
        base_score = config_scores[BASE_MODEL_NAME]['overall']['mean']
        report_lines.append(f"Baseline Score ({BASE_MODEL_NAME}): {base_score:.3f}")

        for config_name, scores in config_scores.items():
            if config_name != BASE_MODEL_NAME:
                improvement = ((scores['overall']['mean'] - base_score) / base_score) * 100 if base_score != 0 else float('inf')
                report_lines.append(f"  {config_name} vs Base Model: {improvement:+.1f}%")
    else:
        report_lines.append("Could not find base model results to perform improvement analysis.")

    report_lines.append("")

    # Key Findings
    report_lines.append("KEY FINDINGS:")
    report_lines.append("-" * 40)

    if sorted_configs:
        best_config = sorted_configs[0][0]
        best_score = sorted_configs[0][1]['overall']['mean']
        report_lines.append(f"• Best performing configuration: {best_config} (Overall Score: {best_score:.3f})")

    if BASE_MODEL_NAME in config_scores and FINETUNED_MODEL_NAME in config_scores:
        base_score = config_scores[BASE_MODEL_NAME]['overall']['mean']
        ft_score = config_scores[FINETUNED_MODEL_NAME]['overall']['mean']
        ft_improvement = ((ft_score - base_score) / base_score) * 100 if base_score != 0 else float('inf')
        report_lines.append(f"• Fine-tuning improvement (no RAG): {ft_improvement:+.1f}%")

    if BASE_MODEL_NAME in config_scores and BASE_RAG_NAME in config_scores:
        base_score = config_scores[BASE_MODEL_NAME]['overall']['mean']
        base_rag_score = config_scores[BASE_RAG_NAME]['overall']['mean']
        rag_improvement = ((base_rag_score - base_score) / base_score) * 100 if base_score != 0 else float('inf')
        report_lines.append(f"• RAG improvement (on base model): {rag_improvement:+.1f}%")

    if BASE_MODEL_NAME in config_scores and FINETUNED_RAG_NAME in config_scores:
        base_score = config_scores[BASE_MODEL_NAME]['overall']['mean']
        combined_score = config_scores[FINETUNED_RAG_NAME]['overall']['mean']
        combined_improvement = ((combined_score - base_score) / base_score) * 100 if base_score != 0 else float('inf')
        report_lines.append(f"• Combined improvement (Fine-tuning + RAG): {combined_improvement:+.1f}%")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"Evaluation report saved to: {output_file}")

    # Print summary to console
    print("\nQUICK SUMMARY:")
    print("-" * 30)
    for i, (config_name, scores) in enumerate(sorted_configs, 1):
        print(f"{i}. {config_name}: {scores['overall']['mean']:.3f}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_evaluation_report.py <results_file> [output_file]")
        sys.exit(1)

    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    generate_report(results_file, output_file)
