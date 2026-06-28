import os
import json
import time
import random
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src import data_loader, config

def create_sandbox(num_samples=10000):
    sandbox_dir = Path("sandbox_data")
    sandbox_dir.mkdir(exist_ok=True)
    
    sandbox_jsonl = sandbox_dir / "candidates.jsonl"
    sandbox_npy = sandbox_dir / "candidate_embeddings.npy"
    sandbox_ids = sandbox_dir / "candidate_ids.npy"
    
    print(f"Loading all candidates to randomly sample {num_samples}...")
    
    all_candidates = []
    with open("data/raw/candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            all_candidates.append(json.loads(line))
            
    # Randomly sample so we get a good mix of all professions (ML, SWE, Marketing, etc)
    random.seed(42)
    candidates = random.sample(all_candidates, num_samples)
            
    with open(sandbox_jsonl, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
            
    print(f"Saved {len(candidates)} candidates to {sandbox_jsonl}")
    
    print("Loading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    candidate_ids = [c["candidate_id"] for c in candidates]
    texts = [data_loader.get_full_text(c) for c in candidates]
    
    print("Computing embeddings...")
    start = time.time()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    print(f"Computed embeddings in {time.time() - start:.1f}s")
    
    np.save(sandbox_npy, embeddings)
    np.save(sandbox_ids, np.array(candidate_ids))
    print(f"Saved embeddings to {sandbox_npy}")
    
if __name__ == "__main__":
    create_sandbox()
