import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate

# Models to compare
models = {
    "Sarvam-1": {"id": "sarvamai/sarvam-1", "type": "auto"},
    "Qwen-1.8B": {"id": "Qwen/Qwen-1_8B", "type": "auto"},
    "Phi-2": {"id": "microsoft/phi-2", "type": "auto"}
}

#prompt = "Translate to Hindi: Artificial intelligence will transform healthcare."
#prompt = "English: Artificial intelligence will transform healthcare. Translate into Bengali (bn), Gujarati (gu), Hindi (hi), Kannada (kn), Malayalam (ml), Marathi (mr), Odia (or), Punjabi (pa), Tamil (ta), Telugu (te). Output only the translation."
prompt = """
Translate the following sentence into Hindi.

Sentence:
Artificial intelligence will transform healthcare.

Hindi translation:
"""

reference = ["कृत्रिम बुद्धिमत्ता स्वास्थ्य सेवा को बदल देगी।"]  # Reference translation

HF_TOKEN = ""  # Replace with your token

# Load evaluation metrics
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
bertscore = evaluate.load("bertscore")

results = []

for name, info in models.items():
    model_id = info["id"]
    print(f"\nLoading {name}...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=HF_TOKEN,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Generate translation
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=120, 
        pad_token_id=tokenizer.eos_token_id)
    
    #response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"{name} Response:\n{response}")

    # Compute metrics
    #bleu_score = bleu.compute(predictions=[response], references=[reference])["score"]
    bleu_score = bleu.compute(predictions=[response], references=[[reference]])["score"]
    chrf_score = chrf.compute(predictions=[response], references=[reference])["score"]
    bert_f1 = bertscore.compute(predictions=[response], references=[reference], lang="hi")["f1"][0] * 100

    # Subjective evaluation
    fluency = "High" if len(response.split()) > 5 else "Low"
    completeness = "High" if response.strip()[-1] in ["।", "."] else "Medium"
    factual_correctness = "Manual check required"

    results.append({
        "Model": name,
        "Prompt": prompt,
        "Response": response,
        "BLEU": bleu_score,
        "ChrF": chrf_score,
        "BERTScore_F1": bert_f1,
        "Fluency": fluency,
        "Completeness": completeness,
        "Factual_Correctness": factual_correctness
    })

    del model
    torch.cuda.empty_cache()

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("model_comparison.csv", index=False)
print("Results saved to model_comparison.csv")