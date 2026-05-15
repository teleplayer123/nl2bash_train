"""
Fine-tunes TinyLlama-1.1B-Chat to use tools: read, edit, and bash.

Training format uses multi-turn chat with structured tool calls:
  <tool_call>{"tool": "...", "input": {...}}</tool_call>
  <tool_result>...</tool_result>

Loss is computed only on assistant-generated tokens (tool calls and
final answers), not on system/user/tool-result tokens.
"""

import json
import os
import random
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-tools"
FINAL_MODEL_DIR = "./tinyllama-tools-final"
MAX_LENGTH = 768
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Tool definitions and system prompt
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "read",
        "description": "Read the contents of a file from disk.",
        "parameters": {"path": "string — absolute or relative file path"},
    },
    {
        "name": "edit",
        "description": "Replace a substring in a file with new content.",
        "parameters": {
            "path": "string — file path",
            "old_string": "string — exact text to find and replace",
            "new_string": "string — replacement text",
        },
    },
    {
        "name": "bash",
        "description": "Execute a shell command and return its output.",
        "parameters": {"command": "string — the shell command to run"},
    },
]

SYSTEM_PROMPT = f"""You are a capable coding assistant with access to tools.

Available tools:
{json.dumps(TOOLS, indent=2)}

To call a tool, emit exactly:
<tool_call>
{{"tool": "<name>", "input": {{...}}}}
</tool_call>

The environment will reply with:
<tool_result>
...output...
</tool_result>

Use as many tool calls as needed, then give a final plain-text answer."""

# ---------------------------------------------------------------------------
# Synthetic training data
# ---------------------------------------------------------------------------

def _tc(tool: str, **kwargs) -> str:
    return json.dumps({"tool": tool, "input": kwargs})


def make_read_examples():
    files = {
        "main.py": "def main():\n    print('Hello, world!')\n\nif __name__ == '__main__':\n    main()",
        "config.json": '{\n  "host": "localhost",\n  "port": 8080,\n  "debug": false\n}',
        "requirements.txt": "requests==2.28.0\nflask==2.2.0\nnumpy==1.24.0",
        "README.md": "# My Project\n\nA sample project.\n\n## Install\n\n```bash\npip install -r requirements.txt\n```",
        "utils.py": "import os\nimport json\n\ndef load_config(path):\n    with open(path) as f:\n        return json.load(f)",
        ".env": "DATABASE_URL=postgresql://localhost/mydb\nSECRET_KEY=supersecretkey\nDEBUG=True",
        "Makefile": "test:\n\tpython -m pytest tests/\n\nclean:\n\trm -rf build/",
        "setup.py": "from setuptools import setup, find_packages\nsetup(name='myproject', packages=find_packages())",
    }

    prompts = [
        "Read {f} and show me its contents.",
        "What's inside {f}?",
        "Can you open {f} for me?",
        "Show me {f}.",
        "I need to see the contents of {f}.",
    ]

    examples = []
    for path, content in files.items():
        for tmpl in random.sample(prompts, min(3, len(prompts))):
            user = tmpl.format(f=path)
            examples.append({
                "user": user,
                "turns": [
                    {"role": "tool_call",   "content": _tc("read", path=path)},
                    {"role": "tool_result", "content": content},
                    {"role": "assistant",   "content": f"Here are the contents of `{path}`:\n\n```\n{content}\n```"},
                ],
            })
    return examples


def make_edit_examples():
    edits = [
        {
            "file": "config.json",
            "request": "Change the port from 8080 to 9000 in config.json.",
            "old": '"port": 8080',
            "new": '"port": 9000',
            "summary": "Changed the port from `8080` to `9000` in `config.json`.",
        },
        {
            "file": "main.py",
            "request": "Update the greeting in main.py to say 'Hello, Python!' instead of 'Hello, world!'.",
            "old": "print('Hello, world!')",
            "new": "print('Hello, Python!')",
            "summary": "Updated the greeting in `main.py`.",
        },
        {
            "file": "requirements.txt",
            "request": "Upgrade requests to 2.31.0 in requirements.txt.",
            "old": "requests==2.28.0",
            "new": "requests==2.31.0",
            "summary": "Upgraded `requests` to `2.31.0` in `requirements.txt`.",
        },
        {
            "file": "config.json",
            "request": "Enable debug mode in config.json.",
            "old": '"debug": false',
            "new": '"debug": true',
            "summary": "Enabled debug mode in `config.json`.",
        },
        {
            "file": "utils.py",
            "request": "Add error handling to load_config: catch FileNotFoundError and return an empty dict.",
            "old": "def load_config(path):\n    with open(path) as f:\n        return json.load(f)",
            "new": "def load_config(path):\n    try:\n        with open(path) as f:\n            return json.load(f)\n    except FileNotFoundError:\n        return {}",
            "summary": "Added `FileNotFoundError` handling to `load_config` in `utils.py`.",
        },
        {
            "file": "setup.py",
            "request": "Set version to 1.0.0 in setup.py.",
            "old": "setup(name='myproject', packages=find_packages())",
            "new": "setup(name='myproject', version='1.0.0', packages=find_packages())",
            "summary": "Set `version='1.0.0'` in `setup.py`.",
        },
    ]

    request_variants = [
        lambda e: e["request"],
        lambda e: f"In {e['file']}, replace `{e['old']}` with `{e['new']}`.",
        lambda e: f"Please edit {e['file']}: {e['request'].lower()}",
    ]

    examples = []
    for edit in edits:
        for variant in request_variants:
            tool_call = _tc("edit", path=edit["file"], old_string=edit["old"], new_string=edit["new"])
            result = f"Replaced 1 occurrence in `{edit['file']}`."
            examples.append({
                "user": variant(edit),
                "turns": [
                    {"role": "tool_call",   "content": tool_call},
                    {"role": "tool_result", "content": result},
                    {"role": "assistant",   "content": edit["summary"]},
                ],
            })
    return examples


def make_bash_examples():
    tasks = [
        {
            "user": "List all Python files in this project.",
            "cmd": "find . -name '*.py' -type f",
            "out": "./main.py\n./utils.py\n./setup.py\n./tests/test_main.py",
            "ans": "Python files in the project:\n- `./main.py`\n- `./utils.py`\n- `./setup.py`\n- `./tests/test_main.py`",
        },
        {
            "user": "How much disk space is available?",
            "cmd": "df -h .",
            "out": "Filesystem  Size  Used Avail Use% Mounted on\n/dev/sda1    50G   12G   35G  26% /",
            "ans": "50G total, 12G used (26%), 35G available.",
        },
        {
            "user": "What's listening on port 8080?",
            "cmd": "lsof -i :8080",
            "out": "COMMAND   PID USER   FD   TYPE\npython3  1234 user    5u  IPv4",
            "ans": "Port 8080 is used by `python3` (PID 1234).",
        },
        {
            "user": "Show me the last 5 git commits.",
            "cmd": "git log --oneline -5",
            "out": "a1b2c3d Fix auth bug\ne4f5g6h Add profile page\ni7j8k9l Update deps\nm0n1o2p DB schema\nq3r4s5t Init",
            "ans": "Last 5 commits:\n1. `a1b2c3d` — Fix auth bug\n2. `e4f5g6h` — Add profile page\n3. `i7j8k9l` — Update deps\n4. `m0n1o2p` — DB schema\n5. `q3r4s5t` — Init",
        },
        {
            "user": "Install the requests library.",
            "cmd": "pip install requests",
            "out": "Successfully installed requests-2.31.0",
            "ans": "Installed `requests-2.31.0` successfully.",
        },
        {
            "user": "Run the test suite.",
            "cmd": "python -m pytest tests/ -v",
            "out": "12 passed in 1.23s",
            "ans": "All 12 tests passed in 1.23 seconds.",
        },
        {
            "user": "Create a directory called 'output'.",
            "cmd": "mkdir -p output",
            "out": "",
            "ans": "Created the `output` directory.",
        },
        {
            "user": "Count the lines in main.py.",
            "cmd": "wc -l main.py",
            "out": "      47 main.py",
            "ans": "`main.py` has 47 lines.",
        },
        {
            "user": "What Python version is installed?",
            "cmd": "python --version",
            "out": "Python 3.11.4",
            "ans": "Python 3.11.4 is installed.",
        },
        {
            "user": "Show the current directory.",
            "cmd": "pwd",
            "out": "/home/user/my-project",
            "ans": "Current directory: `/home/user/my-project`.",
        },
        {
            "user": "Show git status.",
            "cmd": "git status",
            "out": "On branch main\nnothing to commit, working tree clean",
            "ans": "Working tree is clean on `main`.",
        },
        {
            "user": "List all branches.",
            "cmd": "git branch -a",
            "out": "* main\n  dev\n  remotes/origin/main",
            "ans": "Branches: `main` (current), `dev`, `remotes/origin/main`.",
        },
        {
            "user": "Search for TODO comments in the codebase.",
            "cmd": "grep -rn 'TODO' --include='*.py' .",
            "out": "./utils.py:12:    # TODO: add caching\n./main.py:5:    # TODO: handle errors",
            "ans": "Found 2 TODOs:\n- `utils.py:12` — add caching\n- `main.py:5` — handle errors",
        },
        {
            "user": "Show the size of each file in the project.",
            "cmd": "du -sh * 2>/dev/null | sort -h",
            "out": "4.0K\tREADME.md\n8.0K\tmain.py\n12K\tutils.py\n48K\tsetup.py",
            "ans": "File sizes (smallest to largest):\n- README.md: 4 KB\n- main.py: 8 KB\n- utils.py: 12 KB\n- setup.py: 48 KB",
        },
        {
            "user": "Check if Docker is running.",
            "cmd": "docker ps 2>&1 | head -1",
            "out": "CONTAINER ID   IMAGE   COMMAND   CREATED   STATUS   PORTS   NAMES",
            "ans": "Docker is running. No containers are currently active.",
        },
    ]

    examples = []
    for t in tasks:
        examples.append({
            "user": t["user"],
            "turns": [
                {"role": "tool_call",   "content": _tc("bash", command=t["cmd"])},
                {"role": "tool_result", "content": t["out"]},
                {"role": "assistant",   "content": t["ans"]},
            ],
        })
    return examples


def make_multi_tool_examples():
    return [
        {
            "user": "Read requirements.txt, then install everything in it.",
            "turns": [
                {"role": "tool_call",   "content": _tc("read", path="requirements.txt")},
                {"role": "tool_result", "content": "requests==2.28.0\nflask==2.2.0\nnumpy==1.24.0"},
                {"role": "assistant",   "content": "Got the requirements. Installing now."},
                {"role": "tool_call",   "content": _tc("bash", command="pip install -r requirements.txt")},
                {"role": "tool_result", "content": "Successfully installed requests-2.28.0 flask-2.2.0 numpy-1.24.0"},
                {"role": "assistant",   "content": "All three packages installed: `requests`, `flask`, `numpy`."},
            ],
        },
        {
            "user": "Check config.json, then enable debug mode.",
            "turns": [
                {"role": "tool_call",   "content": _tc("read", path="config.json")},
                {"role": "tool_result", "content": '{\n  "host": "localhost",\n  "port": 8080,\n  "debug": false\n}'},
                {"role": "assistant",   "content": "`debug` is currently `false`. Enabling it now."},
                {"role": "tool_call",   "content": _tc("edit", path="config.json", old_string='"debug": false', new_string='"debug": true')},
                {"role": "tool_result", "content": "Replaced 1 occurrence in `config.json`."},
                {"role": "assistant",   "content": "Done — debug mode is now enabled in `config.json`."},
            ],
        },
        {
            "user": "Find test files and run them.",
            "turns": [
                {"role": "tool_call",   "content": _tc("bash", command="find . -name 'test_*.py' -o -name '*_test.py'")},
                {"role": "tool_result", "content": "./tests/test_main.py\n./tests/test_utils.py"},
                {"role": "assistant",   "content": "Found 2 test files. Running them now."},
                {"role": "tool_call",   "content": _tc("bash", command="python -m pytest tests/ -v")},
                {"role": "tool_result", "content": "8 passed in 0.87s"},
                {"role": "assistant",   "content": "All 8 tests passed."},
            ],
        },
        {
            "user": "Read main.py, add a shebang line at the top, then verify the change.",
            "turns": [
                {"role": "tool_call",   "content": _tc("read", path="main.py")},
                {"role": "tool_result", "content": "def main():\n    print('Hello, world!')\n\nif __name__ == '__main__':\n    main()"},
                {"role": "assistant",   "content": "No shebang yet. Adding it now."},
                {"role": "tool_call",   "content": _tc("edit", path="main.py", old_string="def main():", new_string="#!/usr/bin/env python3\n\ndef main():")},
                {"role": "tool_result", "content": "Replaced 1 occurrence in `main.py`."},
                {"role": "assistant",   "content": "Added the shebang. Let me verify."},
                {"role": "tool_call",   "content": _tc("bash", command="head -1 main.py")},
                {"role": "tool_result", "content": "#!/usr/bin/env python3"},
                {"role": "assistant",   "content": "Confirmed — `main.py` now starts with `#!/usr/bin/env python3`."},
            ],
        },
        {
            "user": "Check if there's a virtual environment, and create one if not.",
            "turns": [
                {"role": "tool_call",   "content": _tc("bash", command="test -d venv && echo exists || echo missing")},
                {"role": "tool_result", "content": "missing"},
                {"role": "assistant",   "content": "No virtual environment found. Creating one."},
                {"role": "tool_call",   "content": _tc("bash", command="python -m venv venv")},
                {"role": "tool_result", "content": ""},
                {"role": "assistant",   "content": "Virtual environment created at `./venv/`. Activate with `source venv/bin/activate`."},
            ],
        },
        {
            "user": "Show git status, commit any staged changes with the message 'chore: update config'.",
            "turns": [
                {"role": "tool_call",   "content": _tc("bash", command="git status --short")},
                {"role": "tool_result", "content": "M  config.json"},
                {"role": "assistant",   "content": "`config.json` is staged. Committing now."},
                {"role": "tool_call",   "content": _tc("bash", command="git commit -m 'chore: update config'")},
                {"role": "tool_result", "content": "[main a1b2c3d] chore: update config\n 1 file changed, 1 insertion(+), 1 deletion(-)"},
                {"role": "assistant",   "content": "Committed as `a1b2c3d — chore: update config`."},
            ],
        },
    ]


def build_dataset():
    examples = (
        make_read_examples()
        + make_edit_examples()
        + make_bash_examples()
        + make_multi_tool_examples()
    )
    random.shuffle(examples)
    return examples


# ---------------------------------------------------------------------------
# Tokenization with per-turn loss masking
# ---------------------------------------------------------------------------

def encode_conversation(example: dict, tokenizer, max_length: int) -> dict:
    """
    Converts a structured example into input_ids + labels.
    Labels are -100 for system/user/tool_result turns so the loss is
    computed only on tokens the assistant must generate.
    """
    # Build a flat list of (role, text) segments
    segments = []
    segments.append(("system", f"<|system|>\n{SYSTEM_PROMPT}</s>\n"))
    segments.append(("user",   f"<|user|>\n{example['user']}</s>\n"))

    pending_assistant = ""
    for turn in example["turns"]:
        role = turn["role"]
        content = turn["content"]

        if role == "tool_call":
            pending_assistant += f"<tool_call>\n{content}\n</tool_call>\n"

        elif role == "tool_result":
            if pending_assistant:
                segments.append(("assistant", f"<|assistant|>\n{pending_assistant}</s>\n"))
                pending_assistant = ""
            segments.append(("tool_result", f"<|user|>\n<tool_result>\n{content}\n</tool_result></s>\n"))

        elif role == "assistant":
            pending_assistant += content

    if pending_assistant:
        segments.append(("assistant", f"<|assistant|>\n{pending_assistant}</s>\n"))

    # Tokenize each segment separately and build masked labels
    input_ids, labels = [], []
    for role, text in segments:
        ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(ids)
        if role == "assistant":
            labels.extend(ids)
        else:
            labels.extend([-100] * len(ids))

    # Truncate
    input_ids = input_ids[:max_length]
    labels    = labels[:max_length]

    return {
        "input_ids":      input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels":         labels,
    }


# ---------------------------------------------------------------------------
# Model and tokenizer
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    cuda_available = torch.cuda.is_available()

    if cuda_available:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    else:
        # MPS (Apple Silicon) or CPU — no bitsandbytes quantization
        dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

def apply_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Padding collator
# ---------------------------------------------------------------------------

class PaddingCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_masks, labels = [], [], []
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad_len)
            attention_masks.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels":         torch.tensor(labels,          dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading model and tokenizer…")
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    model = apply_lora(model)

    print("Building synthetic dataset…")
    raw_examples = build_dataset()

    tokenized = [
        encode_conversation(ex, tokenizer, MAX_LENGTH)
        for ex in raw_examples
    ]
    # Drop examples where no assistant token has a real label
    tokenized = [t for t in tokenized if any(l != -100 for l in t["labels"])]

    dataset = Dataset.from_list(tokenized)
    split   = dataset.train_test_split(test_size=0.1, seed=SEED)
    train_ds, eval_ds = split["train"], split["test"]

    print(f"Train: {len(train_ds)} examples  |  Eval: {len(eval_ds)} examples")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,      # effective batch = 16
        num_train_epochs=5,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PaddingCollator(tokenizer),
    )

    print("Training…")
    trainer.train()

    print("Merging LoRA weights and saving…")
    model = model.merge_and_unload()
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print(f"\nDone. Model saved to {FINAL_MODEL_DIR}")
    print("\nTo run inference:")
    print(f"  from transformers import pipeline")
    print(f"  pipe = pipeline('text-generation', model='{FINAL_MODEL_DIR}')")


if __name__ == "__main__":
    main()
