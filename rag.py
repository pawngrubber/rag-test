import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb

# Path to the downloaded model
model_path = './Meta-Llama-3-8B-Instruct.Q4_K_S.gguf'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, device_map='auto')
model.to('cuda')
