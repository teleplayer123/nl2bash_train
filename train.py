from datasets import load_from_disk
import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# --- Load the pre-trained model and tokenizer ---
#model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_id = "HuggingFaceTB/SmolLM2-135M"

# Load tokenizer with fast inference settings (optimized for Picolm/Raspberry Pi)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
    truncation_side="left",  # Left padding for causal LM compatibility with llama.cpp
    padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set

# Load model with 4-bit quantization for memory efficiency on Pi Zero
# Reduces memory usage by ~75% while maintaining accuracy
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,  # Nested quantization for further savings
    bnb_4bit_quant_type="nf4",  # Normalized float 4 for better accuracy
    bnb_4bit_compute_dtype=torch.float16  # Match BF16 for training
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map={"": 0},  # Single GPU/CPU device
    trust_remote_code=True,
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"  # Use flash attention if available
)

# --- Load the dataset ---

train_save_path = os.path.join(os.getcwd(), "assets", "bash_command_data_6K")
dataset = load_from_disk(train_save_path)
print(dataset)

# --- Format the dataset for training ---

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text

# format structured output
def format_json(example):
    prompt = clean_text(example["prompt"])
    completion = clean_text(example["completion"])

    return {
        "text": f"""Convert the request into a JSON object.

Request: {prompt}
Output: {{"command": "{completion}"}}"""
    }

def format_ml(example):
    prompt = clean_text(example["prompt"])
    completion = clean_text(example["completion"])

    return {
        "text": f"""Convert the request to a Linux command.

Request: {prompt}
Command: <cmd>{completion}</cmd>"""
    }

ds = dataset["train"].map(format_ml)

# Remove bad rows
def is_valid(example):
    cmd = example["text"]
    return (
        "<cmd>" in cmd and
        "</cmd>" in cmd and
        len(cmd) < 300  # avoid weird long outputs
    )

ds = ds.filter(is_valid)

# Limit length - reduced to 128 for Pi Zero memory constraints
def short_enough(example):
    return len(example["text"]) < 128

ds = ds.filter(short_enough)

for i in range(3):
    print(ds[i]["text"])
    print("-----")

# --- Tokenize the dataset ---

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=128,  # Reduced for Pi Zero memory efficiency
        padding="max_length"
    )
    # Proper loss masking: ignore padding tokens in loss calculation
    tokens["labels"] = tokens["input_ids"].copy()
    tokens["attention_mask"] = tokens["attention_mask"].bool()
    # Set padding tokens to -100 to exclude from loss
    tokens["labels"] = torch.where(
        tokens["attention_mask"] == 0,
        torch.tensor(-100, dtype=torch.long),
        tokens["labels"]
    )
    return tokens

tokenized_ds = ds.map(tokenize, batched=True)

# Create validation split for monitoring
split = tokenized_ds.train_test_split(test_size=0.05)
train_ds = split["train"]
val_ds = split["test"]

# --- Fine-tune the model using LoRA ---

# Optimized LoRA config for small models and Pi Zero deployment
# Target only critical modules to reduce trainable params and memory
lora_config = LoraConfig(
    r=8,                          # Lower rank for smaller model capacity
    lora_alpha=16,                # Keep 2x ratio for stable scaling
    target_modules=[              # Focus on attention layers (most impactful)
        "q_proj", "v_proj",
        "k_proj", "o_proj"
    ],
    lora_dropout=0.05,            # Lower dropout to prevent underfitting on small dataset
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "lm_head"]  # Train embedding layers for better command mapping
)

model = get_peft_model(model, lora_config)

# --- Train ---
training_args = TrainingArguments(
    output_dir="./smollm2-bash",
    per_device_train_batch_size=8,        # Smaller batch for memory efficiency
    gradient_accumulation_steps=4,        # Effective batch size = 32
    num_train_epochs=5,                   # More epochs for better convergence
    learning_rate=2e-4,                   # Slightly higher LR for faster convergence
    weight_decay=0.01,                    # Prevent overfitting
    warmup_steps=50,                      # Gradual learning rate increase
    logging_steps=10,
    save_steps=50,
    evaluation_strategy="steps",          # Evaluate during training
    eval_steps=50,
    save_total_limit=2,                   # Keep only last 2 checkpoints
    load_best_model_at_end=True,          # Load best model at end
    metric_for_best_model="loss",
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,          # Save memory during training
    report_to="none"                      # Disable external logging
)

class CustomTrainer(Trainer):
    """Custom trainer with proper loss computation using attention mask"""
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Use attention_mask to compute loss only on non-padded tokens
        attention_mask = inputs.get("attention_mask", None)
        labels = inputs.get("labels")

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask
        )
        logits = outputs.logits

        # Shift logits and labels for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Mask padding tokens in loss calculation
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            # Apply attention mask to ignore padding tokens
            loss = loss * shift_attention_mask.view(-1).float()
            loss = loss.sum() / shift_attention_mask.view(-1).float().sum()
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

trainer.train()

# --- Merge LoRA weights into the base model and save ---

model = model.merge_and_unload()
model.save_pretrained("./final-model")

# Save original tokenizer in attempt to fix potential tokenization issues with llama.cpp (may need to modify tokenizer settings or use a custom tokenizer for best results)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True
)
tokenizer.save_pretrained("./final-model")

# Save additional config for Picolm compatibility
config = {
    "model_id": model_id,
    "max_length": 128,
    "tokenizer_settings": {
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "bos_token": tokenizer.bos_token,
        "unk_token": tokenizer.unk_token
    }
}
with open("./final-model/picolm_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\n" + "="*50)
print("Training complete! Model saved to ./final-model")
print("="*50)
print("\nNext steps for Picolm on Raspberry Pi Zero 2 W:")
print("1. Convert to GGUF: python3 convert_hf_to_gguf.py ./final-model --outfile model.gguf")
print("2. Quantize for Pi Zero (Q4_K_M is good balance): ./quantize model.gguf model_Q4_K_M.gguf Q4_K_M")
print("   Alternative (smaller): ./quantize model.gguf model_Q3_K_M.gguf Q3_K_M")
print("3. Load with Picolm using the quantized model")

# Use llama.cpp to convert to GGUF and quantize
# Note: In tokenizer_config.json change "extra_special_tokens" to "special_tokens" to avoid issues with llama.cpp conversion.
# python3 convert_hf_to_gguf.py ./final-model --outfile model.gguf
# ./quantize model.gguf model_Q3_K_M.gguf Q3_K_M