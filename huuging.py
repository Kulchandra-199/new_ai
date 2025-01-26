import os
from typing import Dict, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling

def load_model_and_tokenizer(
    model_name: str, 
    load_in_4bit: bool = True
) -> tuple:
    """Load model and tokenizer with robust configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure padding token exists
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        trust_remote_code=True
    )
    
    return model, tokenizer

def configure_lora(model, rank: int = 16) -> PeftModel:
    """Configure and apply LoRA to the model."""
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    return get_peft_model(model, lora_config)

def tokenize_chat_dataset(
    dataset: Dataset, 
    tokenizer, 
    max_length: int = 512
) -> Dataset:
    """Tokenize chat-formatted dataset."""
    def tokenize_function(examples: Dict[str, List]) -> Dict[str, torch.Tensor]:
        texts = [
            tokenizer.apply_chat_template(
                msgs, 
                tokenize=False, 
                add_generation_prompt=False
            ) for msgs in examples['messages']
        ]
        
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    return dataset.map(tokenize_function, batched=True)

def setup_training_args(
    output_dir: str = "./lora_output",
    epochs: int = 3
) -> TrainingArguments:
    """Configure comprehensive training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=epochs,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        evaluation_strategy="steps",
        load_best_model_at_end=True
    )

def main():
    # Model configuration
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Configure LoRA
    model = configure_lora(model)
    model.print_trainable_parameters()
    
    # Load dataset (replace with your actual dataset)
    dataset = load_dataset('json', data_files='your_dataset.json')['train']
    
    # Tokenize dataset
    tokenized_dataset = tokenize_chat_dataset(dataset, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # Training arguments
    training_args = setup_training_args()
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Train model
    trainer.train()
    
    # Save model
    model.save_pretrained("lora_adapters")
    tokenizer.save_pretrained("lora_adapters")

if __name__ == "__main__":
    main()