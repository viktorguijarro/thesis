#!/usr/bin/env python3
"""
Dataset 3: Question-Answer Pair Generator using Mistral
Takes Dataset 2 and generates specific answers for each question
Creates individual entries with paragraph, question, and answer
"""

import json
import requests
import time
from typing import List, Dict, Tuple
import argparse

class MistralAnswerGenerator:
    def __init__(self, model_name: str = "mistral:latest", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.session = requests.Session()
    
    def generate_answer(self, paragraph: str, question: str) -> str:
        """Generate a specific answer for a question based on the paragraph"""
        
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
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more focused answers
                        "top_p": 0.8,
                        "max_tokens": 200  # Shorter answers
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('response', '').strip()
                
                # Clean up the answer
                answer = self.clean_answer(answer)
                return answer
            else:
                print(f"Error calling Ollama API: {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"Error generating answer: {e}")
            return ""
    
    def clean_answer(self, answer: str) -> str:
        """Clean and format the generated answer"""
        # Remove common prefixes
        prefixes_to_remove = [
            "According to the paragraph,",
            "Based on the paragraph,",
            "The paragraph states that",
            "The text mentions that",
            "From the paragraph,",
            "The answer is",
            "Answer:"
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
                break
        
        # Remove quotes if the entire answer is quoted
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        
        # Ensure proper capitalization
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        # Remove trailing periods if it's a short factual answer
        if len(answer.split()) <= 5 and answer.endswith('.'):
            answer = answer[:-1]
        
        return answer.strip()
    
    def validate_answer(self, paragraph: str, question: str, answer: str) -> bool:
        """Basic validation to check if answer seems reasonable"""
        if not answer or len(answer.strip()) == 0:
            return False
        
        # Check if answer is too long (likely an explanation rather than direct answer)
        if len(answer.split()) > 50:
            return False
        
        # Check if answer contains key terms from the question (basic relevance check)
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'who', 'when', 'where', 'why', 'how', 'which'}
        question_words -= common_words
        answer_words -= common_words
        
        return True  # Basic validation passed
    
    def test_ollama_connection(self) -> bool:
        """Test if Ollama is running and the model is available"""
        try:
            response = self.session.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model_name in model_names:
                    print(f"✓ Ollama is running and {self.model_name} is available")
                    return True
                else:
                    print(f"✗ Model {self.model_name} not found. Available models: {model_names}")
                    return False
            else:
                print(f"✗ Ollama API not responding: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Cannot connect to Ollama: {e}")
            print("Make sure Ollama is running with: ollama serve")
            return False
    
    def process_dataset(self, input_file: str, output_file: str = "dataset_3_qa_pairs.json") -> None:
        """Process Dataset 2 and generate answers for each question"""
        
        # Test Ollama connection first
        if not self.test_ollama_connection():
            print("Please ensure Ollama is running and the Mistral model is available.")
            return
        
        # Load Dataset 2
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset_2 = json.load(f)
        except FileNotFoundError:
            print(f"Input file {input_file} not found. Please run the second script first.")
            return
        except json.JSONDecodeError:
            print(f"Invalid JSON in {input_file}")
            return
        
        entries = dataset_2.get('entries', [])
        if not entries:
            print("No entries found in the input dataset")
            return
        
        print(f"Processing {len(entries)} entries...")
        
        qa_pairs = []
        total_questions = sum(len(entry.get('questions', [])) for entry in entries)
        processed_questions = 0
        
        for entry_idx, entry in enumerate(entries):
            paragraph = entry.get('paragraph', '')
            questions = entry.get('questions', [])
            source_article = entry.get('source_article', 'Unknown')
            
            if not paragraph or not questions:
                continue
            
            print(f"Processing entry {entry_idx + 1}/{len(entries)} - {len(questions)} questions")
            
            for question_idx, question in enumerate(questions):
                processed_questions += 1
                print(f"  Question {processed_questions}/{total_questions}: Generating answer...")
                
                answer = self.generate_answer(paragraph, question)
                
                if answer and self.validate_answer(paragraph, question, answer):
                    qa_pair = {
                        "id": len(qa_pairs) + 1,
                        "source_article": source_article,
                        "paragraph": paragraph,
                        "question": question,
                        "answer": answer,
                        "paragraph_word_count": len(paragraph.split()),
                        "answer_word_count": len(answer.split())
                    }
                    qa_pairs.append(qa_pair)
                    print(f"    ✓ Answer generated: {answer[:50]}{'...' if len(answer) > 50 else ''}")
                else:
                    print(f"    ✗ Failed to generate valid answer")
                
                # Small delay between requests
                time.sleep(0.5)
        
        # Save Dataset 3
        dataset_3 = {
            "metadata": {
                "description": "Question-Answer pairs from Wikipedia paragraphs",
                "source_dataset": input_file,
                "total_qa_pairs": len(qa_pairs),
                "model_used": self.model_name,
                "format": "Each entry contains paragraph, question, and answer"
            },
            "qa_pairs": qa_pairs
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_3, f, indent=2, ensure_ascii=False)
        
        print(f"\nDataset 3 saved to {output_file}")
        print(f"Total Q&A pairs created: {len(qa_pairs)}")
        
        # Print statistics
        if qa_pairs:
            avg_answer_length = sum(pair['answer_word_count'] for pair in qa_pairs) / len(qa_pairs)
            print(f"Average answer length: {avg_answer_length:.1f} words")
            
            # Show some examples
            print("\nSample Q&A pairs:")
            for i, pair in enumerate(qa_pairs[:3]):
                print(f"\n{i+1}. Q: {pair['question']}")
                print(f"   A: {pair['answer']}")

def main():
    parser = argparse.ArgumentParser(description='Generate answers for questions using Mistral')
    parser.add_argument('--input', type=str, default='dataset_2_questions.json', 
                       help='Input JSON file from Dataset 2')
    parser.add_argument('--output', type=str, default='dataset_3_qa_pairs.json', 
                       help='Output JSON file for Dataset 3')
    parser.add_argument('--model', type=str, default='mistral:latest', 
                       help='Mistral model name in Ollama')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                       help='Ollama API URL')
    
    args = parser.parse_args()
    
    generator = MistralAnswerGenerator(args.model, args.ollama_url)
    generator.process_dataset(args.input, args.output)

if __name__ == "__main__":
    main()

