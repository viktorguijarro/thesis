#!/usr/bin/env python3
"""
Automated cleaning script for Q&A dataset.
Filters out low-quality entries based on quality scores and content analysis.
"""

import json
import re
from typing import List, Dict, Any

def clean_dataset(input_file: str, output_file: str = "clean_data_3.json", 
                 min_cqs: float = 0.1, min_qqs: float = 0.1, min_aqs: float = 0.1):
    """Clean dataset by removing low-quality entries."""
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs = data.get('qa_pairs', [])
    cleaned_pairs = []
    
    for pair in qa_pairs:
        # Check quality scores
        cqs = pair.get('content_quality_score', 0)
        qqs = pair.get('question_quality_score', 0)
        aqs = pair.get('answer_quality_score', 0)
        
        if cqs < min_cqs or qqs < min_qqs or aqs < min_aqs:
            continue
        
        # Check content quality
        paragraph = pair.get('paragraph', '')
        question = pair.get('question', '')
        answer = pair.get('answer', '')
        
        # Skip if any field is empty
        if not paragraph or not question or not answer:
            continue
        
        # Skip very short answers
        if len(answer.split()) < 2:
            continue
        
        # Skip answers that indicate uncertainty
        uncertainty_phrases = ['not mentioned', 'unclear', 'not specified', 'not provided', 'unknown']
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            continue
        
        # Skip questions that are too generic
        generic_questions = ['what is this about', 'what does this mean', 'tell me more']
        if any(generic in question.lower() for generic in generic_questions):
            continue
        
        cleaned_pairs.append(pair)
    
    # Update IDs to be sequential
    for i, pair in enumerate(cleaned_pairs, 1):
        pair['id'] = i
    
    # Create cleaned dataset
    cleaned_data = {
        'metadata': {
            'description': 'Cleaned Q&A pairs dataset',
            'original_count': len(qa_pairs),
            'cleaned_count': len(cleaned_pairs),
            'removal_rate': (len(qa_pairs) - len(cleaned_pairs)) / len(qa_pairs) if qa_pairs else 0,
            'quality_thresholds': {
                'min_content_quality_score': min_cqs,
                'min_question_quality_score': min_qqs,
                'min_answer_quality_score': min_aqs
            }
        },
        'qa_pairs': cleaned_pairs
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    print(f"Cleaned dataset saved to {output_file}")
    print(f"Original entries: {len(qa_pairs)}")
    print(f"Cleaned entries: {len(cleaned_pairs)}")
    print(f"Removal rate: {cleaned_data['metadata']['removal_rate']:.1%}")

if __name__ == "__main__":
    clean_dataset('dataset_3_qa_pairs.json')
