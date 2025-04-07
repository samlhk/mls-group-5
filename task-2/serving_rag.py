import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import threading
import time
from torch.utils.data import Dataset
import math
import uuid

app = FastAPI()

# Example documents in memory
documents = [
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings."
]

# change the hash to match your device
embed_model_path = "offline_models/multilingual-e5-large-instruct/models--intfloat--multilingual-e5-large-instruct/snapshots/84344a23ee1820ac951bc365f1e91d094a911763"
chat_model_path = "offline_models/facebook-opt-125m/models--facebook--opt-125m//snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6/"

# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
embed_model = AutoModel.from_pretrained(embed_model_path)

# Basic Chat LLM
chat_pipeline = pipeline("text-generation", model=chat_model_path)
# Note: try this 1.5B model if you got enough GPU memory
# chat_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")



## Hints:

### Step 3.1:
# 1. Initialize a request queue
# 2. Initialize a background thread to process the request (via calling the rag_pipeline function)
# 3. Modify the predict function to put the request in the queue, instead of processing it immediately

MAX_BATCH_SIZE = 5
MAX_WAITING_TIME = 2
request_queue = []
response_queue = []

class QuestionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

def consume_request():
    start = math.inf
    freeze_start = False
    global request_queue
    while True:
        queue_length = len(request_queue)
        if queue_length > 0:
            if not freeze_start:
                start = time.time()
                freeze_start = True
            if queue_length >= MAX_BATCH_SIZE or time.time() - start > MAX_WAITING_TIME:
                batch_size = min(queue_length, MAX_BATCH_SIZE)
                requests = request_queue[:batch_size]
                results = rag_pipeline_batched(requests)
                request_queue = request_queue[batch_size:]
                for id, result in enumerate(results):
                    response_queue.append({"id": requests[id]["id"], "response": result[0]["generated_text"]})
                start = math.inf
                freeze_start = False
                print(f'Completed a batch of: {batch_size} requests')
        time.sleep(0.5)


consume_thread = threading.Thread(target=consume_request)
consume_thread.daemon = True
consume_thread.start()

### Step 3.2:
# 1. Take up to MAX_BATCH_SIZE requests from the queue or wait until MAX_WAITING_TIME
# 2. Process the batched requests

def get_embedding(text: str) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

### You may want to use your own top-k retrieval method (task 1)
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity."""
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

def rag_pipeline(query: str, k: int = 2) -> str:
    # Step 1: Input embedding
    query_emb = get_embedding(query)
    
    # Step 2: Retrieval
    retrieved_docs = retrieve_top_k(query_emb, k)
    
    # Construct the prompt from query + retrieved docs
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    
    # Step 3: LLM Output
    # max_length modified to increase the duration of the processing for performance testings
    generated = chat_pipeline(prompt, max_length=50, do_sample=True)[0]["generated_text"]
    return generated

def rag_pipeline_batched(requests):
    prompts = []
    for request in requests:
        query = request["payload"].query

        # Step 1: Input embedding
        query_emb = get_embedding(query)
        
        # Step 2: Retrieval
        retrieved_docs = retrieve_top_k(query_emb, request["payload"].k)
        
        # Construct the prompt from query + retrieved docs
        context = "\n".join(retrieved_docs)
        prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"

        prompts.append(prompt)
    
    dataset = QuestionDataset(prompts)
    
    # Step 3: LLM Output
    # max_length modified to increase the duration of the processing for performance testings
    generated = chat_pipeline(dataset, batch_size=len(requests), max_length=50, do_sample=True)
    return generated

# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag")
def predict(payload: QueryRequest):
    id = uuid.uuid4().hex
    request_queue.append({"id": id, "payload": payload})
    while True:
        response = [response for response in response_queue if response["id"] == id]
        if len(response) > 0:
            return response[0]
        time.sleep(0.5)

    # result = rag_pipeline(payload.query, payload.k)
    
    # return {
    #     "query": payload.query,
    #     "result": result,
    # }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
