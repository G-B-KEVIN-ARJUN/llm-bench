from huggingface_hub import snapshot_download, login
import os 

hf_token = os.getenv("HF_TOKEN")

login(token = hf_token)

# model_id = "meta-llama/Meta-Llama-3-8b-Instruct"
# local_folder = "./models/llama3-8b"

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
local_folder = "./models/mistral-7b-v0.3"

print(f"Downloading model {model_id} to {local_folder}... ")

snapshot_download(
    repo_id = model_id,
    local_dir = local_folder,
    local_dir_use_symlinks=False,
    ignore_patterns=["*.pth","*.msgpack"],
    token=hf_token
)

print("Download complete!")