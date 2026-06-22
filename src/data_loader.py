"""
Loads candidates.jsonl and exposes them as plain Python dicts, with some
helper methods to normalize / extract text.
"""

import json 
from pathlib import Path
from typing import Iterator


from src import config


def load_candidates(path: Path = config.CANDIDATES_FILE) -> list[dict]:
    """Load every candidate record from the JSONL file into memory"""
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            candidates.append(json.loads(line))
    return candidates

def iter_candidates(path: Path = config.CANDIDATES_FILE) -> Iterator[dict]:
    """Stream candidates one at a time instead of loading the whole file"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def get_full_text(candidate: dict) -> str:
    """Concatenate the free-text fields used for semantic embedding:
    headline + summary + every career history title/description."""
    profile = candidate.get("profile", {})
    parts = [profile.get("headline",""), profile.get("summary","")]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " ".join(p for p in parts if p)

def get_skill_names(candidate: dict) -> set[str]:
    """Lowercased set of skill names listed on the profile."""
    return {s["name"].strip().lower() for s in candidate.get("skills", []) if s.get("name")}

def get_all_industries(candidate: dict) -> set[str]:
    """Every industry the candidate has worked in (current + past), lowercased."""
    industries = set()
    profile = candidate.get("profile", {})
    if profile.get("current_industry"):
        industries.add(profile["current_industry"].strip().lower())
    for job in candidate.get("career_history", []):
        if job.get("industry"):
            industries.add(job["industry"].strip().lower())
    return industries

def build_skill_vocabulary(candidates: list[dict]) -> set[str]:
    """Returns the full set of distinct skill names (lowercased) seen
    across the dataset. Used by jd_parser.py to detect required skills
    from real data instead of a hand-typed list."""
    vocabulary = set()
    for candidate in candidates:
        vocabulary.update(get_skill_names(candidate))
    return vocabulary

def load_jd_text(path: Path) -> str:
    """Reads a job description from disk. Supports .txt and .docx —
    extend here if other formats need support later."""
    path = Path(path)
    if path.suffix == ".docx":
        import docx
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        return path.read_text(encoding="utf-8")