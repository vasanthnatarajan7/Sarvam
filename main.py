# main.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model & tokenizer
model_name = "sarvamai/sarvam-1"  # Example; replace with correct repo if needed
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

print(f"Loaded {model_name} on {device}")

from benchmark_runner import run_prompt_test

#prompt_file = "my_prompts.txt"
prompt_file = "my_prompts1.txt"

run_prompt_test(prompt_file)