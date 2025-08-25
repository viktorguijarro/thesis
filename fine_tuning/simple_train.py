#!/usr/bin/env python3
"""
Simple fine-tuning script for Qwen2-0.5B model.
Fine-tunes the model on the QA dataset using LoRA for efficiency.
"""

import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QADataset:
    def __init__(self, jsonl_file: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(jsonl_file)

    def load_data(self, jsonl_file: str):
        """Load data from JSONL file."""
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item['messages']

        # Format conversation
        conversation = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == "system":
                conversation += f"System: {content}\n"
            elif role == "user":
                conversation += f"User: {content}\n"
            elif role == "assistant":
                conversation += f"Assistant: {content}"

        # Tokenize
        encoding = self.tokenizer(
            conversation,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

def main():
    # Configuration
    # IMPORTANT: Use the model name you intend to use.
    # For example: "Qwen/Qwen2-0.5B"
    model_name = "Qwen/Qwen2-0.5B"
    output_dir = "./models/qwen2_0.5B_finetuned"

    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Prepare datasets
    logger.info("Loading datasets...")
    train_dataset = QADataset("data/train.jsonl", tokenizer)
    val_dataset = QADataset("data/val.jsonl", tokenizer)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Convert to HuggingFace datasets
    train_hf = Dataset.from_list([train_dataset[i] for i in range(len(train_dataset))])
    val_hf = Dataset.from_list([val_dataset[i] for i in range(len(val_dataset))])


    # Training arguments - CORRECTED FOR APPLE SILICON (MPS)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        learning_rate=2e-5,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        # Use this for Apple Silicon Macs
        use_mps_device=True
    )



    #     # Training arguments - CORRECTED FOR APPLE SILICON (MPS)
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     overwrite_output_dir=True,
    #     num_train_epochs=3,
    #     per_device_train_batch_size=4,  # Keep this low for M1/M2 chips
    #     per_device_eval_batch_size=4,
    #     gradient_accumulation_steps=2,
    #     warmup_steps=100,
    #     learning_rate=2e-5,
    #     logging_steps=10,
    #     evaluation_strategy="steps",    # This will now be recognized
    #     eval_steps=50,
    #     save_steps=100,
    #     save_total_limit=3,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="eval_loss",
    #     greater_is_better=False,
    #     report_to=None,                 # Disable wandb
    #     # --- KEY CHANGES FOR MACOS ---
    #     # 1. Remove fp16, it's for CUDA GPUs
    #     # fp16=torch.cuda.is_available(),
    #     # 2. Explicitly tell the trainer to use the Metal Performance Shaders (MPS)
    #     use_mps_device=True
    # )


    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save the model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    logger.info("Training completed!")

if __name__ == "__main__":
    main()
