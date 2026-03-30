from datasets import load_dataset

dataset = load_dataset("emirkaanozdemr/bash_command_data_6K")

save_path = "bash_command_data_6K"
dataset.save_to_disk(save_path)
print(f"Dataset saved to {save_path}")
