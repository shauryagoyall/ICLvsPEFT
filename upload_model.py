from huggingface_hub import login
from config import hf_token
from huggingface_hub import HfApi

login(token=hf_token)

local_folder = "./llama-3.2-1b-french-5000samples"  
repo_name = "shauryagoyall/llama-3.2-1b-french-5000samples" 

api = HfApi()

api.create_repo(repo_id=repo_name, exist_ok=True)

api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_name,
    repo_type="model"
)

print(f"Uploaded to https://huggingface.co/{repo_name}")