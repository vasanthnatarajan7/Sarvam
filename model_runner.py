import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#MODEL_NAME = "sarvamai/sarvam-30b"
MODEL_NAME = "sarvamai/sarvam-1"  # Example; replace with correct repo if needed

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

print("Model loaded")

def generate_response(prompt, max_tokens=200):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)