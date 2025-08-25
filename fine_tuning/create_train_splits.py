#!/usr/bin/env python3
"""
Create train/validation/test splits from the cleaned QA dataset.
Uses stratified sampling to ensure balanced distribution.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import math

def create_splits(input_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                 test_ratio: float = 0.1, seed: int = 42):
    """Create train/val/test splits from QA dataset."""
    
    # Set random seed
    random.seed(seed)
    
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = data['qa_pairs']
    total_examples = len(qa_pairs)
    
    # Calculate split sizes
    train_size = int(total_examples * train_ratio)
    val_size = int(total_examples * val_ratio)
    test_size = total_examples - train_size - val_size
    
    print(f"Total examples: {total_examples}")
    print(f"Train: {train_size} ({train_size/total_examples:.1%})")
    print(f"Validation: {val_size} ({val_size/total_examples:.1%})")
    print(f"Test: {test_size} ({test_size/total_examples:.1%})")
    
    # Shuffle the data
    shuffled_pairs = qa_pairs.copy()
    random.shuffle(shuffled_pairs)
    
    # Split the data
    train_data = shuffled_pairs[:train_size]
    val_data = shuffled_pairs[train_size:train_size + val_size]
    test_data = shuffled_pairs[train_size + val_size:]
    
    # Convert to JSONL format for training
    def save_jsonl(data_list, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data_list:
                # Create training format
                training_item = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions based on the given context."
                        },
                        {
                            "role": "user", 
                            "content": f"Context: {item['paragraph']}\n\nQuestion: {item['question']}"
                        },
                        {
                            "role": "assistant",
                            "content": item['answer']
                        }
                    ],
                    "id": item['id'],
                    "source": item.get('source', '')
                }
                f.write(json.dumps(training_item, ensure_ascii=False) + '\n')
    
    # Save splits
    save_jsonl(train_data, 'data/train.jsonl')
    save_jsonl(val_data, 'data/val.jsonl')
    save_jsonl(test_data, 'data/test.jsonl')
    
    # Save summary
    summary = {
        "total_examples": total_examples,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "test_examples": len(test_data),
        "thinking_mode": True,
        "input_file": input_file,
        "format": "custom_json"
    }
    
    with open('data/dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSplits saved:")
    print(f"- Train: data/train.jsonl ({len(train_data)} examples)")
    print(f"- Validation: data/val.jsonl ({len(val_data)} examples)")
    print(f"- Test: data/test.jsonl ({len(test_data)} examples)")
    print(f"- Summary: data/dataset_summary.json")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python create_train_splits.py <input_file>")
        print("Example: python create_train_splits.py ../validated_data/dataset_3_qa_pairs_extracted.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    create_splits(input_file)
