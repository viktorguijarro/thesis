#!/usr/bin/env python3
"""
Simple Qwen3 Fine-tuning Script
Bypasses cache issues and compatibility problems
"""

import torch
import json
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def format_conversation(messages):
    """Format messages into a single conversation string"""
    conversation = ""
    for message in messages:
        role = message['role']
        content = message['content']
        
        if role == "system":
            conversation += f"System: {content}\n"
        elif role == "user":
            conversation += f"User: {content}\n"
        elif role == "assistant":
            conversation += f"Assistant: {content}\n"
    
    return conversation.strip()

def prepare_dataset(data, tokenizer, max_length=1024):
    """Prepare dataset for training"""
    texts = []
    
    for example in data:
        conversation = format_conversation(example['messages'])
        texts.append(conversation)
    
    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # For causal LM, labels are the same as input_ids
    encodings["labels"] = encodings["input_ids"].clone()
    
    return Dataset.from_dict(encodings)

def main():
    logger.info("🚀 Starting Qwen3 Fine-tuning")
    
    # Model configuration
    model_name = "Qwen/Qwen3-0.6B"
    
    # Clear any problematic environment variables
    os.environ.pop("TOKENIZERS_PARALLELISM", None)
    
    try:
        # Load tokenizer with error handling
        logger.info("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            force_download=False,  # Try cache first
            resume_download=True
        )
        
        # Ensure we have a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("✅ Set pad_token to eos_token")
        
        logger.info("✅ Tokenizer loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Tokenizer loading failed: {e}")
        logger.info("🔄 Trying with force_download=True...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            force_download=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("✅ Tokenizer loaded with force download")
    
    # Load model
    logger.info("📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None
    )
    logger.info("✅ Model loaded successfully")
    
    # Setup LoRA
    logger.info("🔧 Setting up LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("✅ LoRA applied successfully")
    
    # Load and prepare data
    logger.info("📊 Loading training data...")
    train_data = load_jsonl("data/train.jsonl")
    val_data = load_jsonl("data/val.jsonl")
    
    logger.info(f"📈 Training examples: {len(train_data)}")
    logger.info(f"📉 Validation examples: {len(val_data)}")
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_data, tokenizer)
    val_dataset = prepare_dataset(val_data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./qwen3_finetuned",
        num_train_epochs=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
        report_to=None,  # Disable wandb
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("🎯 Starting training...")
    trainer.train()
    
    # Save model
    logger.info("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("./qwen3_finetuned")
    
    logger.info("🎉 Training completed successfully!")
    logger.info("📁 Model saved to: ./qwen3_finetuned")

if __name__ == "__main__":
    main()
