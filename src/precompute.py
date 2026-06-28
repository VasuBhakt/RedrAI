"""
Precomputes embeddings for a given candidates JSONL file.
This must be run before `src.rank` if a new candidates file is provided.
"""

import argparse
import time
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
from pathlib import Path

from src import config, data_loader

def main(candidates_file: str, out_embeddings: str, out_ids: str):
    start = time.time()
    
    print(f"Loading candidates from {candidates_file}...")
    candidates = data_loader.load_candidates(Path(candidates_file))
    print(f"Loaded {len(candidates)} candidates.")
    
    # Extract candidate IDs and text for embedding
    candidate_ids = [c["candidate_id"] for c in candidates]
    texts = [data_loader.get_full_text(c) for c in candidates]
    
    print(f"Loading embedding model ({config.EMBEDDING_MODEL})...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    print("Computing embeddings (this may take a while depending on candidate volume)...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    out_emb_path = Path(out_embeddings)
    out_ids_path = Path(out_ids)
    
    out_emb_path.parent.mkdir(parents=True, exist_ok=True)
    out_ids_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving embeddings to {out_emb_path}...")
    np.save(out_emb_path, embeddings)
    
    print(f"Saving candidate IDs to {out_ids_path}...")
    np.save(out_ids_path, np.array(candidate_ids))
    
    print(f"Precomputation complete in {time.time() - start:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings.")
    parser.add_argument("--candidates_file", default=str(config.CANDIDATES_FILE))
    parser.add_argument("--out_embeddings", default=str(config.CANDIDATE_EMBEDDINGS_FILE))
    parser.add_argument("--out_ids", default=str(config.CANDIDATE_IDS_FILE))
    args = parser.parse_args()
    
    main(args.candidates_file, args.out_embeddings, args.out_ids)
