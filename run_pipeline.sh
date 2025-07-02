#!/bin/bash

# Dataset Generation Pipeline
# Automated script to run all three dataset generation scripts in sequence

set -e  # Exit on any error

echo "=== Wikipedia Dataset Generation Pipeline ==="
echo "This script will generate three progressive datasets:"
echo "1. Wikipedia paragraphs (Dataset 1)"
echo "2. Questions for each paragraph (Dataset 2)"
echo "3. Question-Answer pairs (Dataset 3)"
echo ""

# Configuration
ARTICLES=${1:-50}  # Default to 50 articles if not specified
OLLAMA_URL="http://localhost:11434"
MODEL="mistral:latest"

echo "Configuration:"
echo "- Articles to process: $ARTICLES"
echo "- Ollama URL: $OLLAMA_URL"
echo "- Model: $MODEL"
echo ""

# Check if Ollama is running
echo "Checking Ollama connection..."
if ! curl -s "$OLLAMA_URL/api/tags" > /dev/null; then
    echo "❌ Error: Cannot connect to Ollama at $OLLAMA_URL"
    echo "Please start Ollama with: ollama serve"
    exit 1
fi

# Check if Mistral model is available
if ! curl -s "$OLLAMA_URL/api/tags" | grep -q "mistral:latest"; then
    echo "❌ Error: Mistral model not found"
    echo "Please install with: ollama pull mistral:latest"
    exit 1
fi

echo "✅ Ollama is running and Mistral model is available"
echo ""

# Step 1: Generate Wikipedia paragraphs
echo "=== Step 1: Extracting Wikipedia Paragraphs ==="
echo "Processing $ARTICLES articles..."
python3 dataset_1_wikipedia_extractor_fixed.py --articles "$ARTICLES" --output dataset_1_paragraphs.json

if [ ! -f "dataset_1_paragraphs.json" ]; then
    echo "❌ Error: Failed to generate Dataset 1"
    exit 1
fi

echo "✅ Dataset 1 completed: dataset_1_paragraphs.json"
echo ""

# Step 2: Generate questions
echo "=== Step 2: Generating Questions ==="
echo "Using Mistral to generate questions for each paragraph..."
python3 dataset_2_question_generator.py --input dataset_1_paragraphs.json --output dataset_2_questions.json --model "$MODEL" --ollama-url "$OLLAMA_URL"

if [ ! -f "dataset_2_questions.json" ]; then
    echo "❌ Error: Failed to generate Dataset 2"
    exit 1
fi

echo "✅ Dataset 2 completed: dataset_2_questions.json"
echo ""

# Step 3: Generate answers
echo "=== Step 3: Generating Answers ==="
echo "Using Mistral to generate answers for each question..."
python3 dataset_3_answer_generator.py --input dataset_2_questions.json --output dataset_3_qa_pairs.json --model "$MODEL" --ollama-url "$OLLAMA_URL"

if [ ! -f "dataset_3_qa_pairs.json" ]; then
    echo "❌ Error: Failed to generate Dataset 3"
    exit 1
fi

echo "✅ Dataset 3 completed: dataset_3_qa_pairs.json"
echo ""

# Summary
echo "=== Pipeline Completed Successfully! ==="
echo ""
echo "Generated files:"
echo "📄 dataset_1_paragraphs.json - Wikipedia paragraphs"
echo "📄 dataset_2_questions.json - Paragraphs with questions"
echo "📄 dataset_3_qa_pairs.json - Question-Answer pairs"
echo ""

# Show statistics
if command -v jq > /dev/null; then
    echo "Dataset Statistics:"
    echo "- Paragraphs: $(jq '.metadata.total_paragraphs' dataset_1_paragraphs.json 2>/dev/null || echo 'N/A')"
    echo "- Entries with questions: $(jq '.metadata.total_entries' dataset_2_questions.json 2>/dev/null || echo 'N/A')"
    echo "- Q&A pairs: $(jq '.metadata.total_qa_pairs' dataset_3_qa_pairs.json 2>/dev/null || echo 'N/A')"
else
    echo "Install 'jq' to see detailed statistics: brew install jq"
fi

echo ""
echo "🎉 All datasets generated successfully!"
echo "You can now use these datasets for your machine learning projects."

