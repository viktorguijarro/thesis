#!/usr/bin/env python3
"""
Dataset 2: Question Generator using Mistral
Takes paragraphs from Dataset 1 and generates 5-10 specific questions per paragraph
"""

import json
import requests
import time
from typing import List, Dict
import argparse
import random

class MistralQuestionGenerator:
    def __init__(self, model_name: str = "mistral:latest", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.session = requests.Session()
    
    def generate_questions(self, paragraph: str) -> List[str]:
        """Generate 5-10 specific questions for a given paragraph using Mistral"""
        
        # Create a detailed prompt for question generation
        prompt = f"""Given the following paragraph, generate exactly {random.randint(5, 10)} specific, natural-language questions that can be answered using information from this paragraph.

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

Generate the questions as a numbered list (1., 2., 3., etc.):"""

        try:
            # Call Ollama API
            response = self.session.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                
                # Parse the questions from the response
                questions = self.parse_questions(generated_text)
                return questions
            else:
                print(f"Error calling Ollama API: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []
    
    def parse_questions(self, text: str) -> List[str]:
        """Parse questions from the generated text"""
        questions = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered questions
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                # Remove numbering and clean up
                question = line
                # Remove various numbering formats
                question = question.lstrip('0123456789.-•) \t')
                
                if question and question.endswith('?'):
                    questions.append(question)
        
        # If parsing failed, try alternative approach
        if not questions:
            # Look for question marks
            sentences = text.split('?')
            for sentence in sentences[:-1]:  # Exclude last empty part
                sentence = sentence.strip()
                if sentence:
                    # Clean up any numbering at the start
                    sentence = sentence.lstrip('0123456789.-•) \t')
                    if sentence:
                        questions.append(sentence + '?')
        
        return questions[:10]  # Limit to 10 questions max
    
    def test_ollama_connection(self) -> bool:
        """Test if Ollama is running and the model is available"""
        try:
            # Check if Ollama is running
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
    
    def process_dataset(self, input_file: str, output_file: str = "dataset_2_questions.json") -> None:
        """Process the first dataset and generate questions for each paragraph"""
        
        # Test Ollama connection first
        if not self.test_ollama_connection():
            print("Please ensure Ollama is running and the Mistral model is available.")
            print("Run: ollama pull mistral:latest")
            print("Then: ollama serve")
            return
        
        # Load the first dataset
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset_1 = json.load(f)
        except FileNotFoundError:
            print(f"Input file {input_file} not found. Please run the first script first.")
            return
        except json.JSONDecodeError:
            print(f"Invalid JSON in {input_file}")
            return
        
        paragraphs = dataset_1.get('paragraphs', [])
        if not paragraphs:
            print("No paragraphs found in the input dataset")
            return
        
        print(f"Processing {len(paragraphs)} paragraphs...")
        
        dataset_2_entries = []
        
        for i, paragraph_data in enumerate(paragraphs):
            paragraph_text = paragraph_data.get('paragraph', '')
            if not paragraph_text:
                continue
            
            print(f"Generating questions for paragraph {i+1}/{len(paragraphs)}")
            
            questions = self.generate_questions(paragraph_text)
            
            if questions:
                entry = {
                    "id": paragraph_data.get('id', i+1),
                    "source_article": paragraph_data.get('source_article', 'Unknown'),
                    "paragraph": paragraph_text,
                    "word_count": paragraph_data.get('word_count', len(paragraph_text.split())),
                    "questions": questions,
                    "question_count": len(questions)
                }
                dataset_2_entries.append(entry)
                print(f"  Generated {len(questions)} questions")
            else:
                print(f"  Failed to generate questions for paragraph {i+1}")
            
            # Small delay to be respectful to the local model
            time.sleep(1)
        
        # Save the second dataset
        dataset_2 = {
            "metadata": {
                "description": "Wikipedia paragraphs with generated questions",
                "source_dataset": input_file,
                "total_entries": len(dataset_2_entries),
                "model_used": self.model_name,
                "questions_per_paragraph": "5-10"
            },
            "entries": dataset_2_entries
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_2, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset 2 saved to {output_file}")
        print(f"Total entries with questions: {len(dataset_2_entries)}")
        
        # Print statistics
        total_questions = sum(entry['question_count'] for entry in dataset_2_entries)
        avg_questions = total_questions / len(dataset_2_entries) if dataset_2_entries else 0
        print(f"Total questions generated: {total_questions}")
        print(f"Average questions per paragraph: {avg_questions:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Generate questions for Wikipedia paragraphs using Mistral')
    parser.add_argument('--input', type=str, default='dataset_1_paragraphs.json', 
                       help='Input JSON file from Dataset 1')
    parser.add_argument('--output', type=str, default='dataset_2_questions.json', 
                       help='Output JSON file for Dataset 2')
    parser.add_argument('--model', type=str, default='mistral:latest', 
                       help='Mistral model name in Ollama')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                       help='Ollama API URL')
    
    args = parser.parse_args()
    
    generator = MistralQuestionGenerator(args.model, args.ollama_url)
    generator.process_dataset(args.input, args.output)

if __name__ == "__main__":
    main()

