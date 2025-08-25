# README

Welcome to the execution steps for these repository for IOS usage.

Before starting the project, please create 3 different folders. One for the creation of the data, one for the validation and error of the data, and lastly one folder for the fine-tuning. On top of this you need to create 3 folders inside of the fine-tuning folder: data, models, working_qwen3_model.

- From step 1 to 4 we work in the data creation folder.
- From step 5 to 9 we work in the data error and validation folder.
- From step 10 to 15 we work in the fine-tuning folder.

## Step 1

Install Homebrew

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required system packages
brew install python@3.11 jq curl wget
```

## Step 2

Install Ollama and pull the 2 needed models mistral latest and qwen2 0.5b

```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve &

# Install Mistral model (used for dataset generation)
ollama pull mistral:latest

# Install Qwen3 model (the model we'll fine-tune)
ollama pull qwen2:0.5b

# Verify installations
ollama list
```

## Step 3

Set up dataset venv

```bash
# Create virtual environment for dataset creation
python3.11 -m venv dataset_env
source dataset_env/bin/activate

# Install required packages
pip install --upgrade pip
pip install requests wikipedia-api tqdm python-dotenv
```

## Step 4

Run the data creation pipeline

```bash
# Make script executable
chmod +x run_pipeline.sh

# Run the pipeline (this will take 30-60 minutes depending on your internet connection)
./run_pipeline.sh 50 # remember that 50 is the number of articles the program goes through (it can be changed to any number)
```

## Step 5

Copy the final dataset to the data error and validation folder

## Step 6

The data can either be cleaned manually or automatically

Here is the automated way of doing it

```bash
python3 automated_cleaning.py
```

## Step 7

Create the negative examples

```bash
python3 create_negative_examples.py clean_data_3.json
```

## Step 8

Combine negative and positive data

```bash
python3 combine_datasets.py <positive_file> <negative_file> <output_file>
```

## Step 9

Extract only the necessary information

```bash
python3 extract_qa_data.py <combined_data> <output_file>
```

## Step 10

Set up venv for the fine-tuning

```bash
# Create virtual environment for fine-tuning
python3.11 -m venv qwen3_env
source qwen3_env/bin/activate

# Install required packages
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install scikit-learn sentence-transformers
pip install spacy nltk tqdm requests
pip install ollama-python

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Step 11

Copy the extracted dataset into the fine-tuning folder

## Step 12

Split the dataset in the 3 different sets

```bash
# Create splits
python3 create_train_splits.py dataset_3_qa_pairs_extracted.json
```

## Step 13

Run the training of the model

```bash
# Run fine-tuning (this will take 1-3 hours depending on your hardware)
python3 simple_train.py
```

## Step 14

Run the evaluation of all 4 models

```bash
python3 four_way_comparison.py \
--knowledge-base dataset_3_qa_pairs_extracted.json \
--test-file data/test.jsonl \
--output comparison_results.json
```

## Step 15

Generate the final report

```bash
python3 generate_evaluation_report.py comparison_results.json
```

