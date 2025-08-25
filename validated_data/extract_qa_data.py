#!/usr/bin/env python3
"""
Script to extract specific fields (id, paragraph, question, answer, and source) 
from a JSON dataset containing question-answer pairs.
"""

import json
import sys
from pathlib import Path

def extract_qa_fields(input_file, output_file=None):
    """
    Extract id, paragraph, question, answer, and source fields from JSON dataset.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str, optional): Path to output JSON file. 
                                   If None, creates output file with '_extracted' suffix
    """
    
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
    
    # Extract only the required fields
    extracted_data = []
    for entry in qa_pairs:
        extracted_entry = {
            'id': entry.get('id'),
            'paragraph': entry.get('paragraph'),
            'question': entry.get('question'),
            'answer': entry.get('answer'),
            'source': entry.get('source_article')  # Extract source_article and rename to source
        }
        extracted_data.append(extracted_entry)
    
    # Create output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_extracted{input_path.suffix}"
    
    # Save the extracted data
    output_data = {
        'metadata': {
            'description': 'Extracted QA pairs with id, paragraph, question, answer, and source',
            'total_entries': len(extracted_data),
            'extracted_from': input_file
        },
        'qa_pairs': extracted_data
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully extracted {len(extracted_data)} entries.")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving output file: {e}")

def main():
    """Main function to handle command line arguments."""
    
    if len(sys.argv) < 2:
        print("Usage: python extract_qa_data_fixed.py <input_file> [output_file]")
        print("Example: python extract_qa_data_fixed.py clean_data_3.json")
        print("Example: python extract_qa_data_fixed.py clean_data_3.json dataset_3_qa_pairs_extracted.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_qa_fields(input_file, output_file)

if __name__ == "__main__":
    main()
