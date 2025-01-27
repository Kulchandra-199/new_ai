import os
from typing import Dict, List

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    PeftModel
)
from datasets import Dataset, load_dataset

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        trust_remote_code=True,
        torch_dtype=torch.float32,
        offload_folder="offload"
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
    
    dataset = load_dataset('json', data_files='dataset.json')['train']
    
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




# ## Todos

#     1. **GPU Utilization**: The current code uses device_map="cpu", which isn't efficient. They should use GPU if available. Maybe the user didn't set it because of constraints, but in practice, you'd want to use CUDA.

# 2. **Quantization**: The code doesn't mention 4-bit or 8-bit quantization, which can reduce memory usage. Using bitsandbytes for quantization could help with GPU constraints.

# 3. **Dataset Splitting**: The code loads the dataset but doesn't split it into training and validation sets. Adding a validation split would help in evaluating the model's performance during training.

# 4. **Gradient Checkpointing**: To save memory, enabling gradient checkpointing could be beneficial, especially on smaller GPUs.

# 5. **Hyperparameter Tuning**: The current hyperparameters are set, but in practice, tuning them based on the model's performance is necessary. For example, adjusting learning rate, batch size, or LoRA rank.

# 6. **Monitoring and Logging**: The code uses report_to="none", but using tools like Weights & Biases or TensorBoard for logging would provide better insights.

# 7. **Evaluation**: The setup includes evaluation_strategy but doesn't provide an eval_dataset. The user should split the dataset and pass the evaluation data to the Trainer.

# 8. **Safety and Model Checks**: Adding steps like checkpoint saving, model validation, and testing after training to ensure the model works as expected.

# 9. **Error Handling**: The code lacks error handling for cases like missing files, incompatible models, or tokenization errors.

# 10. **Deployment Considerations**: After saving the adapters, instructions on how to merge them with the base model or use them for inference might be needed.

# 11. **Memory Optimization Techniques**: Besides LoRA, using methods like flash attention, mixed precision training (though fp16 is already there), or offloading parameters could help with GPU constraints.

# 12. **Dataset Preprocessing**: More detailed preprocessing steps, like filtering, balancing, or augmenting data, might be necessary depending on the dataset quality.