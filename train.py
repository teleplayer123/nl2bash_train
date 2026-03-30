from datasets import load_from_disk
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# --- Load the pre-trained model and tokenizer ---

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# --- Load the dataset ---

train_save_path = os.path.join(os.getcwd(), "assets", "bash_command_data_6K")
dataset = load_from_disk(train_save_path)
print(dataset["train"][0])

# --- Format the dataset for training ---

def format_example(example):
    return {
        "text": f"""Convert the request to a Linux command.

Request: {example['input']}
Command: {example['output']}"""
    }

dataset = dataset["train"].map(format_example)

# --- Tokenize the dataset ---

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized = dataset.map(tokenize, batched=True)

# --- Fine-tune the model using LoRA ---

# For small models, it's often best to focus on the attention layers, as they have a significant impact on performance and are more likely to benefit from fine-tuning. The MLP layers can be less effective for small models and may lead to overfitting, especially with a limited dataset.
# target_modules=[
#     "q_proj", "k_proj", "v_proj", "o_proj",
#     "gate_proj", "up_proj", "down_proj"   # MLP layers
# ]

lora_config = LoraConfig(
    r=16,                     # Higher rank helps models with small capacity actually learn the task
    lora_alpha=32,            # keep ~2xr for stable scaling, which prevents updates from being too aggressive
    target_modules=[
        "q_proj", "v_proj",   # attention (most important)
        "k_proj",             # helps stability on small models
        "o_proj"              # improves output quality
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
    train_dataset=tokenized
)

trainer.train()

# --- Merge LoRA weights into the base model and save ---

model = model.merge_and_unload()
model.save_pretrained("./final-model")
tokenizer.save_pretrained("./final-model")

# Use llama.cpp to convert to GGUF and quantize
# python3 convert_hf_to_gguf.py ./final-model --outfile model.gguf
# ./quantize model.gguf model_Q3_K_M.gguf Q3_K_M