from datasets import load_from_disk
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# --- Load the pre-trained model and tokenizer ---

# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_id = "HuggingFaceTB/SmolLM2-135M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# --- Load the dataset ---

train_save_path = os.path.join(os.getcwd(), "assets", "bash_command_data_6K")
dataset = load_from_disk(train_save_path)
print(dataset)

# --- Format the dataset for training ---

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text

# consider structured output
def format_structured(example):
    prompt = clean_text(example["prompt"])
    completion = clean_text(example["completion"])

    return {
        "text": f"""Convert the request into a JSON object.

Request: {prompt}
Output: {{"command": "{completion}"}}"""
    }

def format_example(example):
    prompt = clean_text(example["prompt"])
    completion = clean_text(example["completion"])

    return {
        "text": f"""Convert the request to a Linux command.

Request: {prompt}
Command: <cmd>{completion}</cmd>"""
    }

ds = dataset["train"].map(format_example)

# Remove bad rows
def is_valid(example):
    cmd = example["text"]
    return (
        "<cmd>" in cmd and
        "</cmd>" in cmd and
        len(cmd) < 300  # avoid weird long outputs
    )

ds = ds.filter(is_valid)

# Limit length
def short_enough(example):
    return len(example["text"]) < 200

ds = ds.filter(short_enough)

for i in range(3):
    print(ds[i]["text"])
    print("-----")

# --- Tokenize the dataset ---

def tokenize(example):
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=160,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_ds = ds.map(tokenize, batched=True)

# Optional validation split
# split = tokenized_ds.train_test_split(test_size=0.05)
# train_ds = split["train"]
# val_ds = split["test"]

# --- Fine-tune the model using LoRA ---

lora_config = LoraConfig(
    r=16,                     # Higher rank helps models with small capacity actually learn the task
    lora_alpha=32,            # keep ~2xr for stable scaling, which prevents updates from being too aggressive
    target_modules=[
        "q_proj", "v_proj",   # attention (most important)
        "k_proj",             # helps stability on small models
        "o_proj",              # improves output quality
        "gate_proj", "up_proj", "down_proj"  # MLP layers can help, but be careful of overfitting
    ],
    lora_dropout=0.1,         # higher than usual (prevents overfit)
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# --- Train ---
training_args = TrainingArguments(
    output_dir="./smollm2-bash",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    learning_rate=1e-4, 
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds
)

trainer.train()

# --- Merge LoRA weights into the base model and save ---

model = model.merge_and_unload()
model.save_pretrained("./final-model")

# Save original tokenizer in attempt to fix potential tokenization issues with llama.cpp (may need to modify tokenizer settings or use a custom tokenizer for best results)
tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M",
    use_fast=True
)
tokenizer.save_pretrained("./final-model")

# Use llama.cpp to convert to GGUF and quantize
# python3 convert_hf_to_gguf.py ./final-model --outfile model.gguf
# ./quantize model.gguf model_Q3_K_M.gguf Q3_K_M