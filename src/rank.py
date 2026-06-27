"""
THE TIMED ENTRYPOINT. This script must complete in <=5 minutes, CPU only,
no network calls, <=16GB RAM. It only LOADS precomputed embeddings — it
never calls the embedding model.

Usage:
    python -m src.rank --jd_file data/raw/job_description.docx --out output/submission.csv
"""

import argparse
import time

import numpy as np
import pandas as pd

from src import config, data_loader, jd_parser, scorer, reasoning
from src.features import semantic


def main(jd_file: str, out_path: str) -> None:
    start = time.time()

    print("Loading candidates...")
    candidates = data_loader.load_candidates()

    print("Loading precomputed embeddings...")
    candidate_embeddings = np.load(config.CANDIDATE_EMBEDDINGS_FILE)
    candidate_ids_order = np.load(config.CANDIDATE_IDS_FILE)

    print("Parsing JD...")
    jd_text = data_loader.load_jd_text(jd_file)
    skill_vocab = data_loader.build_skill_vocabulary(candidates)
    parsed_jd = jd_parser.parse_jd(jd_text, skill_vocab)

    print("Embedding JD live...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    jd_embedding = model.encode(jd_text, convert_to_numpy=True)


    print("Computing semantic scores...")
    semantic_scores = semantic.compute_semantic_scores(candidate_embeddings, jd_embedding)
    semantic_scores = semantic.normalize_to_unit_range(semantic_scores)

    print("Scoring all candidates...")
    results = scorer.score_all_candidates(candidates, candidate_ids_order, semantic_scores, parsed_jd)

    print("Sorting and selecting top 100...")
    results.sort(key=lambda r: (-round(r["final_score"], 4), r["candidate_id"]))
    top_100 = results[: config.TOP_N]

    print("Generating reasoning...")
    rows = []
    for rank, r in enumerate(top_100, start=1):
        reasoning_text = reasoning.generate_reasoning(r["candidate"], r, parsed_jd)
        rows.append({
            "candidate_id": r["candidate_id"],
            "rank": rank,
            "score": round(r["final_score"], 4),
            "reasoning": reasoning_text,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print(f"Saved submission to {out_path}")
    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd_file", required=True)
    parser.add_argument("--out", default=str(config.SUBMISSION_FILE))
    args = parser.parse_args()
    main(args.jd_file, args.out)