#!/usr/bin/env python3
"""
Combines a primary QA dataset with a dataset of negative examples into a single file.

This script takes two JSON files, extracts the 'qa_pairs' from each,
merges them, shuffles the result for better training distribution,
and saves them to a new output file.
"""

import json
import random
import sys
from pathlib import Path

def combine_and_shuffle_datasets(positive_file: str, negative_file: str, output_file: str):
    """
    Loads QA pairs from two files, combines them, shuffles them, and saves to a new file.

    Args:
        positive_file (str): Path to the JSON file with correct QA pairs.
        negative_file (str): Path to the JSON file with negative/incorrect QA pairs.
        output_file (str): Path for the combined output JSON file.
    """
    print(f"Loading positive examples from: {positive_file}")
    try:
        with open(positive_file, 'r', encoding='utf-8') as f:
            positive_data = json.load(f)
        positive_pairs = positive_data.get('qa_pairs', [])
        print(f"Found {len(positive_pairs)} positive examples.")
    except FileNotFoundError:
        print(f"Error: Positive data file not found at '{positive_file}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{positive_file}'")
        return

    print(f"Loading negative examples from: {negative_file}")
    try:
        with open(negative_file, 'r', encoding='utf-8') as f:
            negative_data = json.load(f)
        negative_pairs = negative_data.get('qa_pairs', [])
        print(f"Found {len(negative_pairs)} negative examples.")
    except FileNotFoundError:
        print(f"Error: Negative data file not found at '{negative_file}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{negative_file}'")
        return

    # Combine the two lists of QA pairs
    combined_pairs = positive_pairs + negative_pairs
    print(f"Total combined examples: {len(combined_pairs)}")

    # Shuffle the combined list to ensure negative examples are distributed randomly
    # This is crucial for stable training.
    print("Shuffling the combined dataset...")
    random.seed(42)  # Use a fixed seed for reproducibility
    random.shuffle(combined_pairs)

    # Re-assign sequential IDs to the shuffled data
    for i, pair in enumerate(combined_pairs, 1):
        pair['id'] = i

    # Create the final output data structure
    output_data = {
        'metadata': {
            'description': 'Combined dataset including both positive and negative examples.',
            'source_positive': positive_file,
            'source_negative': negative_file,
            'total_positive_examples': len(positive_pairs),
            'total_negative_examples': len(negative_pairs),
            'total_combined_examples': len(combined_pairs),
            'shuffled': True,
            'random_seed': 42
        },
        'qa_pairs': combined_pairs
    }

    # Save the combined dataset to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully combined datasets!")
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")

def main():
    """Main function to handle command-line arguments."""
    # Assumes the script is run from the `validated_data` directory
    # where the source files are located.
    default_positive = 'clean_data_3.json'
    default_negative = 'clean_data_3_negative_examples.json'
    default_output = 'combined_dataset_with_negatives.json'

    # You can extend this with argparse for more flexibility if needed
    print("--- Dataset Combination Script ---")
    combine_and_shuffle_datasets(default_positive, default_negative, default_output)

if __name__ == "__main__":
    main()
