#!/usr/bin/env python3
"""
Question Generator for Wikipedia Paragraphs
Uses Mistral model via Ollama to generate questions for each paragraph.
"""

import json
import requests
import time
import argparse
import logging
import re
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuestionGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "mistral:latest"):
        self.ollama_url = ollama_url
        self.model = model
        self.session = requests.Session()
        
    def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = self.session.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if self.model not in model_names:
                logger.error(f"Model '{self.model}' not found. Available models: {model_names}")
                return False
            
            logger.info(f"Successfully connected to Ollama. Model '{self.model}' is available.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.ollama_url}: {e}")
            return False
    
    def generate_questions(self, paragraph: str, target_count: int = 7) -> List[str]:
        """Generate questions for a given paragraph using Mistral."""
        
        prompt = f"""Given the following paragraph, generate exactly {target_count} specific, natural-language questions that can be answered using information from this paragraph.

Requirements:
- Questions must be specific to the content of this paragraph
- No yes/no questions
- No multiple choice questions  
- No questions requiring multiple answers
- Questions should test understanding of facts, concepts, and details mentioned
- Each question should have a clear, specific answer found in the paragraph
- Vary the question types (what, who, when, where, why, how, which)
- Make questions natural and conversational

Paragraph:
{paragraph}

Generate exactly {target_count} questions, numbered 1-{target_count}:"""

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '')
            
            # Parse questions from response
            questions = self._parse_questions(generated_text)
            
            # Validate questions
            valid_questions = []
            for question in questions:
                if self._is_valid_question(question, paragraph):
                    valid_questions.append(question)
            
            return valid_questions
            
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return []
    
    def _parse_questions(self, text: str) -> List[str]:
        """Parse questions from generated text."""
        questions = []
        
        # Try numbered format first (1. Question?)
        numbered_pattern = r'^\d+\.\s*(.+\?)\s*$'
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            match = re.match(numbered_pattern, line)
            if match:
                question = match.group(1).strip()
                if question and question.endswith('?'):
                    questions.append(question)
        
        # If no numbered questions found, try question mark delimited
        if not questions:
            question_parts = text.split('?')
            for part in question_parts[:-1]:  # Last part after final ? is usually empty
                # Clean up the question
                question = part.strip()
                # Remove leading numbers or bullets
                question = re.sub(r'^[\d\.\-\*\s]+', '', question).strip()
                if question and len(question.split()) >= 3:  # At least 3 words
                    questions.append(question + '?')
        
        return questions[:10]  # Limit to 10 questions max
    
    def _is_valid_question(self, question: str, paragraph: str) -> bool:
        """Validate question quality."""
        # Must end with question mark
        if not question.endswith('?'):
            return False
        
        # Must be reasonable length
        words = question.split()
        if len(words) < 3 or len(words) > 25:
            return False
        
        # Must start with question word or be properly formatted
        question_starters = ['what', 'who', 'when', 'where', 'why', 'how', 'which', 'whose', 'whom']
        first_word = words[0].lower()
        if first_word not in question_starters and not question.startswith(('Is ', 'Are ', 'Do ', 'Does ', 'Did ', 'Can ', 'Could ', 'Would ', 'Should ')):
            return False
        
        # Avoid yes/no questions (basic check)
        if first_word in ['is', 'are', 'do', 'does', 'did', 'can', 'could', 'would', 'should']:
            return False
        
        # Check for some relevance to paragraph (basic keyword overlap)
        question_words = set(word.lower().strip('.,!?;:') for word in words)
        paragraph_words = set(word.lower().strip('.,!?;:') for word in paragraph.split())
        
        # Remove common words for better relevance check
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can'}
        question_content_words = question_words - common_words
        paragraph_content_words = paragraph_words - common_words
        
        overlap = len(question_content_words & paragraph_content_words)
        if overlap == 0:
            return False
        
        return True
    
    def calculate_question_quality_score(self, question: str, paragraph: str) -> float:
        """Calculate Question Quality Score (QQS)."""
        question_words = question.lower().split()
        paragraph_words = paragraph.lower().split()
        
        # Specificity score (keyword overlap)
        question_keywords = set(word.strip('.,!?;:') for word in question_words)
        paragraph_keywords = set(word.strip('.,!?;:') for word in paragraph_words)
        
        # Remove common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'who', 'when', 'where', 'why', 'how', 'which'}
        question_keywords -= common_words
        paragraph_keywords -= common_words
        
        if len(question_keywords) == 0:
            specificity_score = 0
        else:
            specificity_score = len(question_keywords & paragraph_keywords) / len(question_keywords)
        
        # Length appropriateness (optimal length around 8-12 words)
        question_length = len(question_words)
        optimal_length = 10
        length_appropriateness = min(question_length, optimal_length) / max(question_length, optimal_length)
        
        # Format compliance (proper question format)
        format_compliance = 1 if question.endswith('?') and len(question_words) >= 3 else 0
        
        # Calculate QQS
        qqs = (specificity_score + length_appropriateness + format_compliance) / 3
        return qqs

def main():
    parser = argparse.ArgumentParser(description='Generate questions for Wikipedia paragraphs')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file with paragraphs')
    parser.add_argument('--output', type=str, default='dataset_2_questions.json', help='Output file path')
    parser.add_argument('--model', type=str, default='mistral:latest', help='Ollama model to use')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='Ollama server URL')
    parser.add_argument('--questions-per-paragraph', type=int, default=7, help='Target questions per paragraph')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = QuestionGenerator(ollama_url=args.ollama_url, model=args.model)
    
    # Test connection
    if not generator.test_connection():
        logger.error("Cannot connect to Ollama. Please ensure Ollama is running and the model is available.")
        return
    
    # Load input data
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return
    
    paragraphs = input_data.get('paragraphs', [])
    logger.info(f"Loaded {len(paragraphs)} paragraphs from {args.input}")
    
    # Generate questions for each paragraph
    processed_entries = []
    total_questions = 0
    
    for i, paragraph_data in enumerate(paragraphs, 1):
        paragraph = paragraph_data.get('paragraph', '')
        if not paragraph:
            continue
        
        logger.info(f"Processing paragraph {i}/{len(paragraphs)}: {paragraph[:100]}...")
        
        # Generate questions
        questions = generator.generate_questions(paragraph, args.questions_per_paragraph)
        
        if questions:
            # Calculate quality scores for questions
            question_entries = []
            for j, question in enumerate(questions, 1):
                qqs = generator.calculate_question_quality_score(question, paragraph)
                question_entry = {
                    'question_id': j,
                    'question': question,
                    'question_quality_score': qqs
                }
                question_entries.append(question_entry)
            
            # Create entry with questions
            entry = {
                'id': paragraph_data.get('id', i),
                'source_article': paragraph_data.get('source_article', ''),
                'source_url': paragraph_data.get('source_url', ''),
                'paragraph': paragraph,
                'paragraph_word_count': paragraph_data.get('word_count', len(paragraph.split())),
                'content_quality_score': paragraph_data.get('content_quality_score', 0),
                'questions': question_entries,
                'questions_generated': len(questions),
                'average_question_quality_score': sum(q['question_quality_score'] for q in question_entries) / len(question_entries) if question_entries else 0
            }
            
            processed_entries.append(entry)
            total_questions += len(questions)
            
            logger.info(f"Generated {len(questions)} questions (avg QQS: {entry['average_question_quality_score']:.3f})")
        else:
            logger.warning(f"No valid questions generated for paragraph {i}")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    # Create output dataset
    output_data = {
        'metadata': {
            'description': 'Wikipedia paragraphs with generated questions',
            'total_entries': len(processed_entries),
            'total_questions': total_questions,
            'average_questions_per_paragraph': total_questions / len(processed_entries) if processed_entries else 0,
            'model_used': args.model,
            'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_file': args.input
        },
        'entries': processed_entries
    }
    
    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Successfully generated {total_questions} questions for {len(processed_entries)} paragraphs")
    logger.info(f"Average questions per paragraph: {output_data['metadata']['average_questions_per_paragraph']:.1f}")
    logger.info(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
