#!/usr/bin/env python3
"""
Qwen3-0.6B Model Evaluation Script
=================================

This script evaluates the fine-tuned Qwen3-0.6B model on test datasets
and provides various metrics and qualitative analysis.

Author: Manus AI
Date: July 2, 2025
"""

import os
import json
import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QwenEvaluator:
    """Evaluator for fine-tuned Qwen3 models"""
    
    def __init__(self, model_path: str, base_model_path: str = "Qwen/Qwen3-0.6B", 
                 is_lora: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the fine-tuned model
            base_model_path: Path to the base model
            is_lora: Whether the model uses LoRA
        """
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.is_lora = is_lora
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.device = self._setup_device()
        
        # Load model
        self.load_model()
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def _setup_device(self) -> str:
        """Setup device for evaluation"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if self.is_lora:
            # Load base model first
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                trust_remote_code=True
            )
            
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            # Load the full fine-tuned model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                trust_remote_code=True
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def generate_response(self, question: str, context: str = "", 
                         max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate response for a given question.
        
        Args:
            question: The question to answer
            context: Optional context
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        # Prepare messages
        messages = []
        
        # System message
        system_message = "You are a helpful AI assistant that provides accurate and informative answers to questions."
        if context:
            system_message += f" Use the following context to help answer questions: {context}"
        
        messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": question})
        
        # Format using chat template
        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def load_test_data(self, test_data_path: str) -> List[Dict]:
        """Load test data from file"""
        test_data = []
        
        if test_data_path.endswith('.jsonl'):
            with open(test_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    # Extract question and answer from messages
                    messages = data.get('messages', [])
                    question = ""
                    answer = ""
                    context = ""
                    
                    for msg in messages:
                        if msg['role'] == 'user':
                            question = msg['content']
                        elif msg['role'] == 'assistant':
                            answer = msg['content']
                        elif msg['role'] == 'system' and 'context' in msg['content'].lower():
                            context = msg['content']
                    
                    if question and answer:
                        test_data.append({
                            'question': question,
                            'answer': answer,
                            'context': context
                        })
        
        elif test_data_path.endswith('.json'):
            with open(test_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    test_data.append({
                        'question': item.get('question', ''),
                        'answer': item.get('answer', ''),
                        'context': item.get('context', '')
                    })
        
        logger.info(f"Loaded {len(test_data)} test examples")
        return test_data
    
    def calculate_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        # Calculate averages
        avg_scores = {
            'rouge1': np.mean(rouge_scores['rouge1']),
            'rouge2': np.mean(rouge_scores['rouge2']),
            'rougeL': np.mean(rouge_scores['rougeL'])
        }
        
        return avg_scores
    
    def calculate_bert_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate BERTScore"""
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            return {
                'bert_precision': P.mean().item(),
                'bert_recall': R.mean().item(),
                'bert_f1': F1.mean().item()
            }
        except Exception as e:
            logger.warning(f"Could not calculate BERTScore: {e}")
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}
    
    def evaluate_on_dataset(self, test_data: List[Dict], output_file: str = None) -> Dict[str, Any]:
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_data: List of test examples
            output_file: Optional file to save detailed results
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating on {len(test_data)} examples")
        
        predictions = []
        references = []
        detailed_results = []
        
        for i, example in enumerate(test_data):
            if i % 10 == 0:
                logger.info(f"Processing example {i+1}/{len(test_data)}")
            
            question = example['question']
            reference = example['answer']
            context = example.get('context', '')
            
            # Generate prediction
            try:
                prediction = self.generate_response(question, context)
                
                # Clean up prediction (remove thinking tags if present)
                if '<think>' in prediction and '</think>' in prediction:
                    # Extract content after </think>
                    prediction = prediction.split('</think>')[-1].strip()
                
                predictions.append(prediction)
                references.append(reference)
                
                detailed_results.append({
                    'question': question,
                    'reference': reference,
                    'prediction': prediction,
                    'context': context
                })
                
            except Exception as e:
                logger.error(f"Error processing example {i}: {e}")
                predictions.append("")
                references.append(reference)
                detailed_results.append({
                    'question': question,
                    'reference': reference,
                    'prediction': f"ERROR: {str(e)}",
                    'context': context
                })
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics...")
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge_scores(predictions, references)
        
        # BERT scores
        bert_scores = self.calculate_bert_score(predictions, references)
        
        # Combine all metrics
        metrics = {
            'num_examples': len(test_data),
            'rouge_scores': rouge_scores,
            'bert_scores': bert_scores,
            'evaluation_date': datetime.now().isoformat()
        }
        
        # Save detailed results if requested
        if output_file:
            results_data = {
                'metrics': metrics,
                'detailed_results': detailed_results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Detailed results saved to {output_file}")
        
        return metrics
    
    def interactive_evaluation(self):
        """Interactive evaluation mode"""
        logger.info("Starting interactive evaluation mode")
        logger.info("Type 'quit' to exit")
        
        while True:
            try:
                question = input("\nEnter your question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                context = input("Enter context (optional): ").strip()
                
                print("\nGenerating response...")
                response = self.generate_response(question, context)
                
                print(f"\nResponse: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        logger.info("Interactive evaluation ended")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Qwen3-0.6B model")
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--base_model", default="Qwen/Qwen3-0.6B", help="Base model path")
    parser.add_argument("--test_data", help="Test data file path")
    parser.add_argument("--output_file", help="Output file for detailed results")
    parser.add_argument("--is_lora", action="store_true", help="Model uses LoRA")
    parser.add_argument("--interactive", action="store_true", help="Interactive evaluation mode")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = QwenEvaluator(
        model_path=args.model_path,
        base_model_path=args.base_model,
        is_lora=args.is_lora
    )
    
    if args.interactive:
        # Interactive mode
        evaluator.interactive_evaluation()
    elif args.test_data:
        # Evaluate on test dataset
        test_data = evaluator.load_test_data(args.test_data)
        metrics = evaluator.evaluate_on_dataset(test_data, args.output_file)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Number of examples: {metrics['num_examples']}")
        print("\nROUGE Scores:")
        for metric, score in metrics['rouge_scores'].items():
            print(f"  {metric.upper()}: {score:.4f}")
        
        print("\nBERT Scores:")
        for metric, score in metrics['bert_scores'].items():
            print(f"  {metric}: {score:.4f}")
        
    else:
        print("Please specify either --test_data for dataset evaluation or --interactive for interactive mode")

if __name__ == "__main__":
    main()

