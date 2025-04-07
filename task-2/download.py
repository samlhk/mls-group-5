from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
import os

base_dir = "./offline_models"
os.makedirs(base_dir, exist_ok=True)

embed_dir = os.path.join(base_dir, "multilingual-e5-large-instruct")
AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct", cache_dir=embed_dir)
AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct", cache_dir=embed_dir)

chat_dir = os.path.join(base_dir, "facebook-opt-125m")
AutoTokenizer.from_pretrained("facebook/opt-125m", cache_dir=chat_dir)
AutoModelForCausalLM.from_pretrained("facebook/opt-125m", cache_dir=chat_dir)

print("All models and tokenizers downloaded to:", base_dir)
