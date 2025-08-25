#!/usr/bin/env python3
"""
Script to create negative examples by randomly pairing paragraphs and questions
from a cleaned QA dataset. Creates mismatched pairs where the answer is always
"The answer cannot be found in the paragraph."
"""

import json
import random
import sys
import math
from pathlib import Path

def create_negative_examples(input_file, output_file=None, percentage=5, seed=42):
    """
    Create negative examples by randomly pairing paragraphs and questions.
    
    Args:
        input_file (str): Path to input JSON file (cleaned dataset)
        output_file (str, optional): Path to output JSON file
        percentage (float): Percentage of data to sample (default: 5)
        seed (int, optional): Random seed for reproducibility
    """
    
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Read the input JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{input_file}': {e}")
        return
    
    # Extract qa_pairs from the dataset
    if 'qa_pairs' not in data:
        print("Error: 'qa_pairs' key not found in the dataset.")
        return
    
    qa_pairs = data['qa_pairs']
    total_entries = len(qa_pairs)
    
    if total_entries == 0:
        print("Error: No QA pairs found in the dataset.")
        return
    
    # Calculate sample size (5% of total entries)
    sample_size = max(1, math.ceil(total_entries * percentage / 100))
    
    print(f"Total entries in dataset: {total_entries}")
    print(f"Sampling {percentage}% = {sample_size} entries")
    
    # Extract all paragraphs and questions separately
    paragraphs = [(entry['id'], entry['paragraph']) for entry in qa_pairs if entry.get('paragraph')]
    questions = [(entry['id'], entry['question']) for entry in qa_pairs if entry.get('question')]
    
    # Randomly sample paragraphs and questions
    sampled_paragraphs = random.sample(paragraphs, min(sample_size, len(paragraphs)))
    sampled_questions = random.sample(questions, min(sample_size, len(questions)))
    
    # Shuffle both lists to ensure random pairing
    random.shuffle(sampled_paragraphs)
    random.shuffle(sampled_questions)
    
    # Create negative examples by pairing paragraphs and questions
    negative_examples = []
    
    # Pair them up (if different sample sizes, use the smaller one)
    pairs_to_create = min(len(sampled_paragraphs), len(sampled_questions))
    
    for i in range(pairs_to_create):
        paragraph_id, paragraph = sampled_paragraphs[i]
        question_id, question = sampled_questions[i]
        
        # Ensure we don't accidentally pair a paragraph with its original question
        # If they match, swap with the next question (or previous if it's the last one)
        if paragraph_id == question_id:
            if i < pairs_to_create - 1:
                # Swap with next question
                sampled_questions[i], sampled_questions[i + 1] = sampled_questions[i + 1], sampled_questions[i]
                question_id, question = sampled_questions[i]
            elif i > 0:
                # Swap with previous question
                sampled_questions[i], sampled_questions[i - 1] = sampled_questions[i - 1], sampled_questions[i]
                question_id, question = sampled_questions[i]
        
        negative_example = {
            'id': i + 1,  # New sequential ID for negative examples
            'paragraph': paragraph,
            'question': question,
            'answer': 'The answer cannot be found in the paragraph.',
            'original_paragraph_id': paragraph_id,
            'original_question_id': question_id
        }
        
        negative_examples.append(negative_example)
    
    # Create output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_negative_examples{input_path.suffix}"
    
    # Create the output dataset
    output_data = {
        'metadata': {
            'description': f'Negative examples created by randomly pairing {percentage}% of paragraphs and questions',
            'source_dataset': input_file,
            'total_negative_examples': len(negative_examples),
            'percentage_sampled': percentage,
            'original_dataset_size': total_entries,
            'creation_method': 'Random paragraph-question pairing',
            'answer_format': 'The answer cannot be found in the paragraph.',
            'random_seed': seed
        },
        'qa_pairs': negative_examples
    }
    
    # Save the negative examples dataset
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully created {len(negative_examples)} negative examples.")
        print(f"Output saved to: {output_file}")
        
        # Print some statistics
        print(f"\nStatistics:")
        print(f"- Original dataset size: {total_entries}")
        print(f"- Percentage sampled: {percentage}%")
        print(f"- Negative examples created: {len(negative_examples)}")
        print(f"- Sample paragraphs used: {len(sampled_paragraphs)}")
        print(f"- Sample questions used: {len(sampled_questions)}")
        
    except Exception as e:
        print(f"Error saving output file: {e}")

def main():
    """Main function to handle command line arguments."""
    
    if len(sys.argv) < 2:
        print("Usage: python create_negative_examples.py <input_file> [output_file] [percentage] [seed]")
        print("Example: python create_negative_examples.py clean_data_3.json")
        print("Example: python create_negative_examples.py clean_data_3.json negative_examples.json 5 42")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    percentage = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 42
    
    # Validate percentage
    if percentage <= 0 or percentage > 100:
        print("Error: Percentage must be between 0 and 100.")
        sys.exit(1)
    
    create_negative_examples(input_file, output_file, percentage, seed)

if __name__ == "__main__":
    main()
