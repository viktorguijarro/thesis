#!/usr/bin/env python3
"""
Answer Generator for Question-Paragraph Pairs
Uses Mistral model via Ollama to generate concise answers for questions.
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

class AnswerGenerator:
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
    
    def generate_answer(self, paragraph: str, question: str) -> Optional[str]:
        """Generate answer for a question based on the paragraph."""
        
        prompt = f"""Based on the following paragraph, provide a direct, concise answer to the question. The answer must be found in the paragraph and should be straight to the point without lengthy explanations.

Requirements:
- Answer must be based only on information in the paragraph
- Keep the answer concise and direct
- Do not add explanations or elaborations
- If the answer is a name, date, place, or specific fact, state it clearly
- Do not start with "According to the paragraph" or similar phrases

Paragraph:
{paragraph}

Question: {question}

Answer:"""

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused answers
                    "top_p": 0.9,
                    "max_tokens": 200   # Shorter answers
                }
            }
            
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=45
            )
            response.raise_for_status()
            
            result = response.json()
            generated_answer = result.get('response', '').strip()
            
            # Clean the answer
            cleaned_answer = self._clean_answer(generated_answer)
            
            # Validate answer
            if self._is_valid_answer(cleaned_answer, question, paragraph):
                return cleaned_answer
            else:
                logger.warning(f"Generated invalid answer for question: {question[:50]}...")
                return None
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return None
    
    def _clean_answer(self, answer: str) -> str:
        """Clean generated answer."""
        # Remove common prefixes
        prefixes_to_remove = [
            "According to the paragraph,",
            "Based on the paragraph,",
            "The paragraph states that",
            "The paragraph mentions that",
            "From the paragraph,",
            "The text says",
            "The answer is",
            "Answer:"
        ]
        
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
        
        # Remove excessive punctuation
        answer = re.sub(r'\.+$', '.', answer)
        answer = re.sub(r'\s+', ' ', answer)
        
        return answer.strip()
    
    def _is_valid_answer(self, answer: str, question: str, paragraph: str) -> bool:
        """Validate answer quality."""
        if not answer or len(answer.strip()) == 0:
            return False
        
        # Check length (not too short, not too long)
        words = answer.split()
        if len(words) < 1 or len(words) > 50:
            return False
        
        # Should not be a question
        if answer.strip().endswith('?'):
            return False
        
        # Should not contain phrases indicating uncertainty
        uncertainty_phrases = [
            "i don't know",
            "not mentioned",
            "unclear",
            "cannot determine",
            "not specified",
            "not provided"
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            return False
        
        return True
    
    def calculate_answer_quality_score(self, answer: str, question: str, paragraph: str) -> float:
        """Calculate Answer Quality Score (AQS)."""
        answer_words = answer.lower().split()
        question_words = question.lower().split()
        paragraph_words = paragraph.lower().split()
        
        # Conciseness score (optimal length around 8-15 words)
        answer_length = len(answer_words)
        optimal_length = 12
        conciseness_score = min(answer_length, optimal_length) / max(answer_length, optimal_length)
        
        # Relevance score (overlap with question keywords)
        question_keywords = set(word.strip('.,!?;:') for word in question_words)
        answer_keywords = set(word.strip('.,!?;:') for word in answer_words)
        
        # Remove common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'who', 'when', 'where', 'why', 'how', 'which'}
        question_content = question_keywords - common_words
        answer_content = answer_keywords - common_words
        
        total_unique = len(question_content | answer_content)
        if total_unique == 0:
            relevance_score = 0
        else:
            relevance_score = len(question_content & answer_content) / total_unique
        
        # Grounding score (overlap with paragraph)
        paragraph_keywords = set(word.strip('.,!?;:') for word in paragraph_words)
        paragraph_content = paragraph_keywords - common_words
        
        if len(answer_content) == 0:
            grounding_score = 0
        else:
            grounding_score = len(paragraph_content & answer_content) / len(answer_content)
        
        # Calculate AQS
        aqs = (conciseness_score + relevance_score + grounding_score) / 3
        return aqs

def main():
    parser = argparse.ArgumentParser(description='Generate answers for questions')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file with questions')
    parser.add_argument('--output', type=str, default='dataset_3_qa_pairs.json', help='Output file path')
    parser.add_argument('--model', type=str, default='mistral:latest', help='Ollama model to use')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434', help='Ollama server URL')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = AnswerGenerator(ollama_url=args.ollama_url, model=args.model)
    
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
    
    entries = input_data.get('entries', [])
    logger.info(f"Loaded {len(entries)} entries from {args.input}")
    
    # Generate answers for each question
    qa_pairs = []
    total_questions = 0
    successful_answers = 0
    
    for entry_idx, entry in enumerate(entries, 1):
        paragraph = entry.get('paragraph', '')
        questions = entry.get('questions', [])
        
        if not paragraph or not questions:
            continue
        
        logger.info(f"Processing entry {entry_idx}/{len(entries)} with {len(questions)} questions")
        
        for question_data in questions:
            question = question_data.get('question', '')
            if not question:
                continue
            
            total_questions += 1
            
            # Generate answer
            answer = generator.generate_answer(paragraph, question)
            
            if answer:
                # Calculate answer quality score
                aqs = generator.calculate_answer_quality_score(answer, question, paragraph)
                
                # Create QA pair
                qa_pair = {
                    'id': len(qa_pairs) + 1,
                    'source_article': entry.get('source_article', ''),
                    'source_url': entry.get('source_url', ''),
                    'paragraph': paragraph,
                    'paragraph_word_count': entry.get('paragraph_word_count', len(paragraph.split())),
                    'question': question,
                    'answer': answer,
                    'answer_word_count': len(answer.split()),
                    'content_quality_score': entry.get('content_quality_score', 0),
                    'question_quality_score': question_data.get('question_quality_score', 0),
                    'answer_quality_score': aqs
                }
                
                qa_pairs.append(qa_pair)
                successful_answers += 1
                
                logger.debug(f"Generated answer (AQS: {aqs:.3f}): {answer[:100]}...")
            else:
                logger.warning(f"Failed to generate answer for: {question[:50]}...")
        
        # Small delay between entries
        time.sleep(0.3)
    
    # Create output dataset
    output_data = {
        'metadata': {
            'description': 'Question-Answer pairs generated from Wikipedia paragraphs',
            'total_qa_pairs': len(qa_pairs),
            'total_questions_attempted': total_questions,
            'success_rate': successful_answers / total_questions if total_questions > 0 else 0,
            'average_answer_length': sum(qa['answer_word_count'] for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
            'average_answer_quality_score': sum(qa['answer_quality_score'] for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
            'model_used': args.model,
            'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'source_file': args.input
        },
        'qa_pairs': qa_pairs
    }
    
    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Successfully generated {len(qa_pairs)} QA pairs from {total_questions} questions")
    logger.info(f"Success rate: {output_data['metadata']['success_rate']:.1%}")
    logger.info(f"Average answer length: {output_data['metadata']['average_answer_length']:.1f} words")
    logger.info(f"Average AQS: {output_data['metadata']['average_answer_quality_score']:.3f}")
    logger.info(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
