"""
One-time, untimed precomputation step.

Generates sentence embeddings for every candidate's narrative text and for
the job description, then saves them to disk as .npy files. The timed
rank.py script only LOADS these; it never calls the embedding model
itself, which is what keeps rank.py inside the 5-minute CPU budget.

Run this manually whenever candidates.jsonl or the JD changes:
    python -m src.precompute_embeddings --jd_file data/raw/job_description.docx
"""

import argparse
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from src import config, data_loader


def precompute(jd_file: str, batch_size: int = 64) -> None:
    print(f"Loading model: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print("Loading candidates...")
    candidates = data_loader.load_candidates()
    print(f"Loaded {len(candidates)} candidates")

    candidate_ids = [c["candidate_id"] for c in candidates]
    texts = [data_loader.get_full_text(c) for c in candidates]

    print("Embedding candidates (this is the slow part, runs once)...")
    start = time.time()
    candidate_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"Done in {time.time() - start:.1f}s")

    print("Embedding JD...")
    jd_text = data_loader.load_jd_text(jd_file)
    jd_embedding = model.encode(jd_text, convert_to_numpy=True)

    config.DATA_PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Saving to disk...")
    np.save(config.CANDIDATE_EMBEDDINGS_FILE, candidate_embeddings)
    np.save(config.CANDIDATE_IDS_FILE, np.array(candidate_ids))
    np.save(config.JD_EMBEDDING_FILE, jd_embedding)

    print(f"Saved candidate_embeddings: {candidate_embeddings.shape}")
    print(f"Saved jd_embedding: {jd_embedding.shape}")
    print("Precomputation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd_file", required=True, help="Path to job description file")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    precompute(args.jd_file, args.batch_size)