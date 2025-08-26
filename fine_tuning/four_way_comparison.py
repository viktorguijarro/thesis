#!/usr/bin/env python3
"""
Four-Way Model Comparison System
==========================================

Compares four different approaches using the PROPER test set:
1. Base Model (Original Qwen2-0.5B)
2. Fine-tuned Model (Trained Qwen2-0.5B)
3. Base Model + RAG (Original + Retrieval)
4. Fine-tuned Model + RAG (Trained + Retrieval)

"""

import json
import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import requests
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrectedFourWayComparison:
    """
    Corrected 4-way model comparison system using proper test set
    """

    def __init__(self, knowledge_base_file: str = "dataset_3_qa_pairs_extracted.json",
                 test_file: str = "data/test.jsonl",
                 ollama_url: str = "http://localhost:11434" ):
        self.knowledge_base_file = knowledge_base_file
        self.test_file = test_file
        self.ollama_url = ollama_url
        self.knowledge_base = []
        self.test_questions = []
        self.vectorizer = None
        self.tfidf_matrix = None

        # Load spaCy for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Some features may be limited.")
            self.nlp = None

        # Initialize components
        self.load_knowledge_base()
        self.load_test_questions()
        self.build_rag_index()

    def load_knowledge_base(self):
        """Load knowledge base for RAG system."""
        try:
            with open(self.knowledge_base_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.knowledge_base = data.get('qa_pairs', [])
            logger.info(f"Loaded {len(self.knowledge_base)} entries into knowledge base")

        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise

    def load_test_questions(self):
        """Load test questions from JSONL file."""
        try:
            with open(self.test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())

                    # Extract question from user message
                    user_message = item['messages'][1]['content']
                    if "Question: " in user_message:
                        question = user_message.split("Question: ")[1].strip()
                    else:
                        question = user_message.strip()

                    expected_answer = item['messages'][2]['content']

                    test_item = {
                        'id': item.get('id', len(self.test_questions) + 1),
                        'question': question,
                        'expected_answer': expected_answer,
                        'source': item.get('source', ''),
                        'context': user_message.split("Context: ")[1].split("\n\nQuestion:")[0] if "Context: " in user_message else ""
                    }

                    self.test_questions.append(test_item)

            logger.info(f"Loaded {len(self.test_questions)} test questions from {self.test_file}")

        except Exception as e:
            logger.error(f"Error loading test questions: {e}")
            raise

    def build_rag_index(self):
        """Build TF-IDF index for RAG retrieval."""
        if not self.knowledge_base:
            logger.error("Knowledge base is empty")
            return

        # Combine paragraph and question for better retrieval
        documents = []
        for entry in self.knowledge_base:
            combined_text = f"{entry.get('paragraph', '')} {entry.get('question', '')}"
            documents.append(combined_text)

        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(documents)
        logger.info(f"Built RAG index with {self.tfidf_matrix.shape[1]} features")

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant context for RAG."""
        if not self.vectorizer or self.tfidf_matrix is None:
            return []

        # Vectorize query
        query_vector = self.vectorizer.transform([query])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        relevant_docs = []
        for idx in top_indices:
            if similarities[idx] >= 0.1:  # Minimum similarity threshold
                doc = self.knowledge_base[idx].copy()
                doc['similarity_score'] = float(similarities[idx])
                relevant_docs.append(doc)

        return relevant_docs

    def call_ollama_model(self, prompt: str, model_name: str) -> str:
        """Call Ollama API to generate response."""
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 200
                }
            }

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result.get('response', '').strip()

        except Exception as e:
            logger.error(f"Error calling Ollama for model {model_name}: {e}")
            return "Error generating response"

    def evaluate_model_configuration(self, model_name: str, use_rag: bool = False) -> List[Dict[str, Any]]:
        """Evaluate a specific model configuration on all test questions."""
        results = []

        logger.info(f"Evaluating {model_name} (RAG: {use_rag}) on {len(self.test_questions)} questions")

        for i, test_item in enumerate(self.test_questions, 1):
            question = test_item['question']
            expected_answer = test_item['expected_answer']

            logger.info(f"Processing question {i}/{len(self.test_questions)}: {question[:50]}...")

            # Prepare prompt
            context_docs = []
            if use_rag:
                context_docs = self.retrieve_context(question, top_k=3)

            if context_docs:
                context_text = "\n\n".join([
                    f"Context: {doc.get('paragraph', '')}"
                    for j, doc in enumerate(context_docs)
                ])
                prompt = f"""System: You are a helpful assistant that answers questions based on the given context.

User: {context_text}

Question: {question}

Assistant:"""
            else:
                # Use the improved prompt structure for no-context queries
                prompt = f"""System: You are a helpful assistant that answers questions.

User: Question: {question}

Assistant:"""

            # Generate response
            generated_answer = self.call_ollama_model(prompt, model_name)

            # Calculate evaluation metrics
            metrics = self.calculate_evaluation_metrics(
                question, expected_answer, generated_answer, test_item.get('context', '')
            )

            result = {
                'id': test_item['id'],
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': generated_answer,
                'context_used': len(context_docs),
                'model': model_name,
                'use_rag': use_rag,
                **metrics
            }

            results.append(result)

            # Small delay to avoid overwhelming the API
            time.sleep(0.5)

        return results

    def calculate_evaluation_metrics(self, question: str, expected: str, generated: str, context: str = "") -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        
        # Tokenize texts
        question_words = set(word.lower().strip('.,!?;:') for word in question.split())
        expected_words = set(word.lower().strip('.,!?;:') for word in expected.split())
        generated_words = set(word.lower().strip('.,!?;:') for word in generated.split())
        context_words = set(word.lower().strip('.,!?;:') for word in context.split()) if context else set()
        
        # Remove common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were'}
        
        question_content = question_words - stop_words
        expected_content = expected_words - stop_words
        generated_content = generated_words - stop_words
        context_content = context_words - stop_words
        
        # 1. Keyword Relevance (KR)
        if len(question_content | expected_content) == 0:
            keyword_relevance = 0
        else:
            keyword_relevance = len(generated_content & (question_content | expected_content)) / len(question_content | expected_content)
        
        # 2. Domain Specificity (DS)
        domain_terms = 0
        if self.nlp:
            doc = self.nlp(generated)
            domain_terms = len([ent for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY']])
        
        domain_specificity = min(1.0, domain_terms / max(1, len(generated.split())))
        
        # 3. Contextual Accuracy (CA)
        if context_content:
            contextual_accuracy = len(generated_content & context_content) / max(1, len(context_content))
        else:
            contextual_accuracy = 0.5
        
        # 4. Response Completeness (RC)
        if len(expected_content) == 0:
            response_completeness = 0
        else:
            response_completeness = len(generated_content & expected_content) / len(expected_content)
        
        # 5. Informativeness (INF)
        unique_content_words = len(generated_content)
        total_words = len(generated.split())
        entity_density_bonus = 1 + (domain_terms / max(1, total_words))
        
        if total_words == 0:
            informativeness = 0
        else:
            informativeness = min(1.0, (unique_content_words / total_words) * entity_density_bonus)
        
        # 6. Linguistic Quality (LQ)
        sentence_count = len([s for s in generated.split('.') if s.strip()])
        avg_sentence_length = total_words / max(1, sentence_count)
        length_score = min(avg_sentence_length, 15) / max(avg_sentence_length, 15)
        grammar_score = 1.0 if generated.endswith('.') or generated.endswith('!') or generated.endswith('?') else 0.8
        linguistic_quality = (length_score + grammar_score) / 2
        
        # 7. Factual Consistency (FC)
        consistency_score = response_completeness
        length_penalty = 1.0
        if total_words < 3: length_penalty = 0.5
        elif total_words > 50: length_penalty = 0.8
        factual_consistency = consistency_score * length_penalty
        
        return {
            'keyword_relevance': keyword_relevance,
            'domain_specificity': domain_specificity,
            'contextual_accuracy': contextual_accuracy,
            'response_completeness': response_completeness,
            'informativeness': informativeness,
            'linguistic_quality': linguistic_quality,
            'factual_consistency': factual_consistency
        }

    def run_four_way_comparison(self) -> Dict[str, Any]:
        """Run complete four-way comparison."""

        logger.info("Starting corrected four-way model comparison...")

        # --- UPDATED MODEL CONFIGURATIONS ---
        configurations = [
            {"name": "Base Model (Qwen2-0.5B)", "model": "qwen2:0.5b", "use_rag": False},
            {"name": "Fine-tuned Model (qwen2finetuned)", "model": "qwen2finetuned", "use_rag": False},
            {"name": "Base Model + RAG", "model": "qwen2:0.5b", "use_rag": True},
            {"name": "Fine-tuned Model + RAG", "model": "qwen2finetuned", "use_rag": True}
        ]

        all_results = {}

        for config in configurations:
            config_name = config["name"]
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating: {config_name}")
            logger.info(f"{'='*50}")

            results = self.evaluate_model_configuration(
                model_name=config["model"],
                use_rag=config["use_rag"]
            )

            all_results[config_name] = results

            # Calculate summary statistics
            metrics = ['keyword_relevance', 'domain_specificity', 'contextual_accuracy',
                      'response_completeness', 'informativeness', 'linguistic_quality', 'factual_consistency']

            summary = {}
            for metric in metrics:
                values = [r[metric] for r in results]
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }

            logger.info(f"\nSummary for {config_name}:")
            for metric, stats in summary.items():
                logger.info(f"  {metric.replace('_', ' ').title()}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        # Create comprehensive results
        comparison_results = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'test_questions_count': len(self.test_questions),
                'knowledge_base_size': len(self.knowledge_base),
                'test_file_used': self.test_file,
                'knowledge_base_file': self.knowledge_base_file,
                'configurations_tested': len(configurations)
            },
            'configurations': configurations,
            'results': all_results,
            'test_questions': self.test_questions
        }

        return comparison_results

    def save_results(self, results: Dict[str, Any], output_file: str = None):
        """Save comparison results to file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"four_way_comparison_results_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_file}")
        return output_file

def main():
    """Main function to run the corrected comparison."""
    import argparse

    parser = argparse.ArgumentParser(description='Corrected Four-Way Model Comparison')
    parser.add_argument('--knowledge-base', type=str, default='dataset_3_qa_pairs_extracted.json',
                       help='Path to knowledge base JSON file')
    parser.add_argument('--test-file', type=str, default='data/test.jsonl',
                       help='Path to test JSONL file')
    parser.add_argument('--output', type=str,
                       help='Output file for results')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                       help='Ollama server URL' )

    args = parser.parse_args()

    # Check if test file exists
    if not os.path.exists(args.test_file):
        logger.error(f"Test file not found: {args.test_file}")
        logger.error("Please ensure you have created the train/test splits using create_train_splits.py")
        return

    # Initialize comparison system
    comparison = CorrectedFourWayComparison(
        knowledge_base_file=args.knowledge_base,
        test_file=args.test_file,
        ollama_url=args.ollama_url
    )

    # Run comparison
    results = comparison.run_four_way_comparison()

    # Save results
    output_file = comparison.save_results(results, args.output)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("FOUR-WAY COMPARISON COMPLETED")
    logger.info(f"{'='*60}")
    logger.info(f"Test questions evaluated: {results['metadata']['test_questions_count']}")
    logger.info(f"Knowledge base size: {results['metadata']['knowledge_base_size']}")
    logger.info(f"Results saved to: {output_file}")

    # Print performance summary
    configurations = results['configurations']
    for config in configurations:
        config_name = config['name']
        config_results = results['results'][config_name]

        # Calculate overall score
        metrics = ['keyword_relevance', 'domain_specificity', 'contextual_accuracy',
                  'response_completeness', 'informativeness', 'linguistic_quality', 'factual_consistency']

        overall_scores = []
        for result in config_results:
            score = np.mean([result[metric] for metric in metrics])
            overall_scores.append(score)

        mean_score = np.mean(overall_scores)
        logger.info(f"{config_name}: {mean_score:.3f}")

if __name__ == "__main__":
    main()
