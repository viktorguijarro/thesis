#!/usr/bin/env python3
"""
Simple RAG (Retrieval-Augmented Generation) system.
Implements TF-IDF based retrieval with Ollama model integration.
Updated to use a fine-tuning-aware prompt structure.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import logging
from typing import List, Dict, Any, Optional, Tuple
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleRAGSystem:
    def __init__(self, knowledge_base_file: str, model_name: str = "qwen3:0.6b",
                 ollama_url: str = "http://localhost:11434" ):
        """
        Initialize RAG system.

        Args:
            knowledge_base_file: Path to JSON file containing QA pairs
            model_name: Name of the Ollama model to use
            ollama_url: URL of Ollama server
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.knowledge_base = []
        self.vectorizer = None
        self.tfidf_matrix = None

        # Load spaCy model for text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Load knowledge base
        self.load_knowledge_base(knowledge_base_file)
        self.build_index()

    def load_knowledge_base(self, file_path: str):
        """Load knowledge base from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            qa_pairs = data.get('qa_pairs', [])

            for pair in qa_pairs:
                self.knowledge_base.append({
                    'id': pair.get('id', ''),
                    'paragraph': pair.get('paragraph', ''),
                    'question': pair.get('question', ''),
                    'answer': pair.get('answer', ''),
                    'source': pair.get('source', '')
                })

            logger.info(f"Loaded {len(self.knowledge_base)} entries into knowledge base")

        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise

    def build_index(self):
        """Build TF-IDF index for retrieval."""
        if not self.knowledge_base:
            logger.error("Knowledge base is empty")
            return

        # Combine paragraph and question for better retrieval
        documents = []
        for entry in self.knowledge_base:
            combined_text = f"{entry['paragraph']} {entry['question']}"
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
        logger.info(f"Built TF-IDF index with {self.tfidf_matrix.shape[1]} features")

    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.

        Args:
            query: User query
            top_k: Number of top results to return

        Returns:
            List of relevant knowledge base entries
        """
        if not self.vectorizer or self.tfidf_matrix is None:
            logger.error("Index not built")
            return []

        # Vectorize query
        query_vector = self.vectorizer.transform([query])

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Filter out very low similarity scores
        min_similarity = 0.1
        relevant_entries = []

        for idx in top_indices:
            if similarities[idx] >= min_similarity:
                entry = self.knowledge_base[idx].copy()
                entry['similarity_score'] = float(similarities[idx])
                relevant_entries.append(entry)

        logger.debug(f"Retrieved {len(relevant_entries)} relevant entries for query: {query[:50]}...")
        return relevant_entries

    def generate_response(self, query: str, use_rag: bool = True,
                         model_path: str = None) -> Dict[str, Any]:
        """
        Generate response using the model with optional RAG.

        Args:
            query: User query
            use_rag: Whether to use RAG or just the base model
            model_path: Path to fine-tuned model (if using local model)

        Returns:
            Dictionary containing response and metadata
        """
        context_docs = []

        if use_rag:
            # Retrieve relevant context
            context_docs = self.retrieve_relevant_context(query, top_k=3)

        # Prepare prompt
        if context_docs:
            # RAG prompt with context - this format is good because it's similar to fine-tuning data
            context_text = "\n\n".join([
                f"Context: {doc['paragraph']}"
                for doc in context_docs
            ])

            prompt = f"""System: You are a helpful assistant that answers questions based on the given context.

User: {context_text}

Question: {query}

Assistant:"""
        else:
            # *** UPDATED PROMPT FOR NO-CONTEXT QUERIES ***
            # This new structure mimics the fine-tuning format to prevent "format bleeding"
            prompt = f"""System: You are a helpful assistant that answers questions.

User: Question: {query}

Assistant:"""

        # Generate response using Ollama
        try:
            # Use the model name passed from the main function, not the one from __init__
            model_to_use = model_path if model_path else self.model_name
            
            response = self._call_ollama(prompt, model_to_use)

            return {
                'query': query,
                'response': response,
                'use_rag': use_rag,
                'context_used': len(context_docs),
                'retrieved_docs': context_docs if use_rag else [],
                'model': model_to_use
            }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'query': query,
                'response': "Error generating response",
                'use_rag': use_rag,
                'context_used': 0,
                'retrieved_docs': [],
                'error': str(e)
            }

    def _call_ollama(self, prompt: str, model_name: str) -> str:
        """Call Ollama API to generate response."""
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

    def evaluate_on_test_set(self, test_file: str, use_rag: bool = True,
                           model_path: str = None) -> Dict[str, Any]:
        """
        Evaluate the system on a test set.

        Args:
            test_file: Path to test JSONL file
            use_rag: Whether to use RAG
            model_path: Path to fine-tuned model

        Returns:
            Evaluation results
        """
        # Load test data
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                test_data.append(item)

        results = []

        for i, item in enumerate(test_data):
            # Extract question from user message
            user_message = item['messages'][1]['content']
            if "Question: " in user_message:
                question = user_message.split("Question: ")[1]
            else:
                question = user_message

            expected_answer = item['messages'][2]['content']

            logger.info(f"Evaluating {i+1}/{len(test_data)}: {question[:50]}...")

            # Generate response
            response_data = self.generate_response(question, use_rag, model_path)

            result = {
                'id': item.get('id', i+1),
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': response_data['response'],
                'use_rag': use_rag,
                'context_used': response_data['context_used'],
                'retrieved_docs': response_data.get('retrieved_docs', [])
            }

            results.append(result)

        return {
            'total_questions': len(test_data),
            'use_rag': use_rag,
            'model': model_path if model_path else self.model_name,
            'results': results
        }

def main():
    """Example usage of the RAG system."""
    import argparse

    parser = argparse.ArgumentParser(description='Simple RAG System')
    parser.add_argument('--knowledge-base', type=str, required=True,
                       help='Path to knowledge base JSON file')
    parser.add_argument('--model', type=str, default='qwen3:0.6b',
                       help='Ollama model name')
    parser.add_argument('--query', type=str,
                       help='Query to test')
    parser.add_argument('--test-file', type=str,
                       help='Test file for evaluation')
    parser.add_argument('--no-rag', action='store_true',
                       help='Disable RAG (use base model only)')
    parser.add_argument('--model-path', type=str,
                       help='Path to fine-tuned model (used as model name in Ollama)')

    args = parser.parse_args()

    # Determine the model to use
    # If --model-path is provided, it overrides --model
    model_to_use = args.model_path if args.model_path else args.model

    # Initialize RAG system with the determined model
    rag_system = SimpleRAGSystem(args.knowledge_base, model_to_use)

    if args.query:
        # Single query test
        result = rag_system.generate_response(
            args.query,
            use_rag=not args.no_rag
        )

        print(f"Query: {result['query']}")
        print(f"Response: {result['response']}")
        print(f"RAG used: {result['use_rag']}")
        print(f"Context documents: {result['context_used']}")

    elif args.test_file:
        # Evaluation on test set
        results = rag_system.evaluate_on_test_set(
            args.test_file,
            use_rag=not args.no_rag
        )

        # Save results
        model_name_for_file = model_to_use.replace(":", "_")
        output_file = f"evaluation_results_{model_name_for_file}_{'rag' if not args.no_rag else 'no_rag'}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Evaluation completed. Results saved to {output_file}")
        print(f"Total questions: {results['total_questions']}")
        print(f"RAG used: {results['use_rag']}")

if __name__ == "__main__":
    main()
