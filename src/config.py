"""
Central configuration for the RedrAI ranking pipeline.
Every tunable constant lives here.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PREPROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "output"

CANDIDATES_FILE = DATA_RAW_DIR / "candidates.jsonl"
SCHEMA_FILE = DATA_RAW_DIR / "candidate_schema.json"

CANDIDATE_EMBEDDINGS_FILE = DATA_PREPROCESSED_DIR / "candidate_embeddings.npy"
CANDIDATE_IDS_FILE = DATA_PREPROCESSED_DIR / "candidate_ids.npy"
JD_EMBEDDING_FILE = DATA_PREPROCESSED_DIR / "jd_embedding.npy"

SUBMISSION_FILE = OUTPUT_DIR / "submission.csv"

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SERVICES_INDUSTRIES = {"it services", "consulting"}

COMMON_LOCATIONS = {
    "pune", "noida", "mumbai", "delhi", "ncr", "hyderabad", "bangalore",
    "bengaluru", "chennai", "kolkata", "gurugram", "gurgaon", "ahmedabad",
}

# Flags to prevent wrong negative marking
NO_GITHUB_SENTINEL = -1
NO_OFFER_HISTORY_SENTINEL = -1

# Scoring weights — skill match weighted higher than semantic, since
# semantic similarity alone let irrelevant-domain candidates (e.g.
# Mechanical Engineers) rank too highly purely on generic professional
# language overlap.
WEIGHTS = {
    "semantic_score": 0.20,
    "skill_depth_score": 0.35,
    "experience_fit_score": 0.20,
    "behavioral_score": 0.15,
    "services_only_penalty": 0.10,
}

# Minimum number of JD-required skills a candidate must match before
# they're eligible for top-tier scoring. Candidates below this get an
# additional multiplicative penalty — this is the fix for low-skill-match
# candidates outranking genuine domain matches via semantic score alone.
MIN_SKILL_MATCHES_FOR_FULL_SCORE = 3
LOW_SKILL_MATCH_PENALTY = 0.5  # multiplier applied if below threshold
HIGH_SKILL_MATCH_THRESHOLD = 8  # out of ~23 typical required skills
HIGH_SKILL_MATCH_BOOST = 1.15   # multiplier

RELEVANT_TITLE_KEYWORDS = {
    "ai", "ml", "machine learning", "nlp", "data scientist", "data engineer",
    "search engineer", "recommendation", "research engineer", "applied ml",
    "software engineer", "backend engineer",
}

TOP_N = 100
RANDOM_SEED = 42