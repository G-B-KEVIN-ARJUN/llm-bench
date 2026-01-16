import torch
import time
import json
import os
import evaluate
import numpy as np
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

# 1. Setup
load_dotenv()
# model_path = "./models/llama3-8b"
# output_file = "benchmark_results.json"

# model_path = "./models/mistral-7b-v0.3"
# output_file = "benchmark_results_mistral_pytorch.json"

# model_path = "./models/gemma-2-9b" 
# output_file = "benchmark_results_gemma_pytorch.json"

model_path = "./models/qwen-2.5-7b"
output_file = "benchmark_results_qwen2.5_pytorch.json"

# Load Metrics
print("Loading Metrics (ROUGE, BERTScore, BLEU)...")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")

# 2. Load Model
print(f"Loading Llama 3 from {model_path}...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True
)

# 3. Load Dataset
print("Loading Test Data...")
# Running 5 samples
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:5]")

print("\n--- Starting Benchmark with Streaming Metrics ---")
results = []
references = []
predictions = []

for item in tqdm(dataset):
    article = item['article']
    true_summary = item['highlights']
    
    # Prompt
    messages = [
        {"role": "system", "content": "Summarize the following article in 3 sentences."},
        {"role": "user", "content": article}
    ]

    # # NEW CODE (Correct for Gemma)
    # messages = [
    #     {"role": "user", "content": f"Summarize the following article in 3 sentences:\n\n{article}"}
    # ]
    
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
    
    # --- STREAMING SETUP ---
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=150,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )
    
    # Run generation in background thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    
    # Start Timer
    start_time = time.time()
    thread.start()
    
    generated_text = ""
    first_token_received = False
    ttft = 0.0
    
    # Consume Stream
    for new_text in streamer:
        if not first_token_received:
            ttft = time.time() - start_time
            first_token_received = True
        generated_text += new_text

    end_time = time.time()
    
    # Calculations
    latency = end_time - start_time
    output_tokens = len(tokenizer.encode(generated_text))
    
    if output_tokens > 1:
        tpot = (latency - ttft) / (output_tokens - 1)
    else:
        tpot = 0.0
        
    speed_tps = output_tokens / latency if latency > 0 else 0

    # Store Data (FLATTENED STRUCTURE)
    predictions.append(generated_text)
    references.append(true_summary)
    
    results.append({
        "generated": generated_text,
        "reference": true_summary,
        "latency": latency,     # <--- Top Level
        "ttft": ttft,           # <--- Top Level
        "tpot": tpot,           # <--- Top Level
        "speed_tps": speed_tps,
        "total_tokens": output_tokens
    })

# 4. Compute Scores
print("\nComputing Final Scores...")
rouge_res = rouge.compute(predictions=predictions, references=references)
bert_res = bertscore.compute(predictions=predictions, references=references, lang="en", device="cpu")
bleu_references = [[ref] for ref in references]
bleu_res = bleu.compute(predictions=predictions, references=bleu_references)

# 5. Final Averages
avg_speed = np.mean([r['speed_tps'] for r in results])
avg_ttft = np.mean([r['ttft'] for r in results])
avg_tpot = np.mean([r['tpot'] for r in results])
avg_latency = np.mean([r['latency'] for r in results])
avg_bert = np.mean(bert_res['f1'])

print("\n" + "="*50)
print(f"  BENCHMARK REPORT: Llama 3 (8B) - RTX 4060")
print("="*50)
print(f"Avg Latency:    {avg_latency:.4f} s")
print(f"Avg TTFT:       {avg_ttft:.4f} s")
print(f"Avg TPOT:       {avg_tpot:.4f} s")
print(f"Throughput:     {avg_speed:.2f} tokens/sec")
print("-" * 50)
print(f"ROUGE-L:        {rouge_res['rougeL']:.4f}")
print(f"BERTScore F1:   {avg_bert:.4f}")
print(f"BLEU:           {bleu_res['bleu']:.4f}")
print("="*50)

# Save to file
output_data = {
    "summary": {
        "model": "Llama-3-8B-Instruct", #update according to model
        "latency": avg_latency,    # <--- Renamed to simple keys
        "ttft": avg_ttft,
        "tpot": avg_tpot,
        "throughput": avg_speed,
        "rouge": rouge_res,
        "bleu": bleu_res,
        "bert_score_f1": avg_bert
    },
    "detailed_runs": results
}

with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)
print(f"Detailed results saved to {output_file}")