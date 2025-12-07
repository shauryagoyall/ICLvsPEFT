import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_TOKEN")

if hf_token:
    print("Hugging Face Token successfully loaded.")
else:
    raise ValueError("Hugging Face TOKEN is not set in environment variables")