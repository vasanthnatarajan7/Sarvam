import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

models = {
    "Sarvam-1": "sarvamai/sarvam-1",
    "Gemma-2-2B": "google/gemma-2-2b",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B"
}

prompt = "Translate to Hindi: Artificial intelligence will transform healthcare."

results = []

HF_TOKEN = "hf_xxxhf_HQAjmkxDeEzjZDEchSeUJVmIfrTYmafQbB"

for name, model_id in models.items():

    print(f"\nLoading {name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=HF_TOKEN  # important for gated models
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,  # important for gated models
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"\n{name} Response:\n{response}")

    results.append({
        "Model": name,
        "Prompt": prompt,
        "Response": response
    })

    # free memory before loading next model
    del model
    torch.cuda.empty_cache()

df = pd.DataFrame(results)

df.to_csv("model_comparison.csv", index=False)

print("\nResults saved to model_comparison.csv")