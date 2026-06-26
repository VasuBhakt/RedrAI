"""
Combines all feature scores into a single final_score per candidate.
This is what rank.py calls directly during the timed ranking step.
"""

from src import config
from src.features import semantic, skill_match, experience_fit, behavioral, disqualifiers, honeypot
from src.jd_parser import ParsedJD
from src.config import WEIGHTS, MIN_SKILL_MATCHES_FOR_FULL_SCORE, LOW_SKILL_MATCH_PENALTY

import numpy as np


def score_candidate(candidate: dict, parsed_jd: ParsedJD, semantic_score: float) -> dict:
    """Computes every component score for one candidate and combines them.
    Returns a dict of all sub-scores plus the final_score, so reasoning.py
    can use the same breakdown to generate explanations.
    """
    if honeypot.is_honeypot(candidate):
        return {
            "final_score": 0.0,
            "semantic_score": semantic_score,
            "skill_overlap": 0.0,
            "skill_depth": 0.0,
            "experience_fit": 0.0,
            "behavioral": 0.0,
            "disqualifier_penalty": 0.0,
            "n_skill_matches": 0,
            "title_relevant": False,
            "is_honeypot": True,
        }

    candidate_skills = {s["name"].strip().lower() for s in candidate.get("skills", []) if s.get("name")}
    n_skill_matches = len(candidate_skills & parsed_jd.required_skills)

    skill_overlap = skill_match.skill_overlap_score(candidate, parsed_jd.required_skills)
    skill_depth = skill_match.skill_depth_score(candidate, parsed_jd.required_skills)
    skill_combined = (skill_overlap * 0.5) + (skill_depth * 0.5)

    exp_fit = experience_fit.experience_years_fit(candidate, parsed_jd.min_years, parsed_jd.max_years)
    coding_fit = experience_fit.recent_coding_score(candidate)
    experience_combined = (exp_fit * 0.6) + (coding_fit * 0.4)

    behavioral_combined = behavioral.behavioral_score(
        candidate, parsed_jd.preferred_locations, parsed_jd.max_notice_days
    )

    penalty = disqualifiers.disqualifier_penalty(candidate)

    raw_score = (
        semantic_score * WEIGHTS["semantic_score"]
        + skill_combined * WEIGHTS["skill_depth_score"]
        + experience_combined * WEIGHTS["experience_fit_score"]
        + behavioral_combined * WEIGHTS["behavioral_score"]
    )

    # Title-relevance check: a high skill-match count only means something
    # if the candidate's actual role is in a relevant domain. This directly
    # encodes the JD's explicit warning: "a candidate who has all the AI
    # keywords listed as skills but whose title is 'Marketing Manager' is
    # not a fit, no matter how perfect their skill list looks."
    current_title_lower = candidate.get("profile", {}).get("current_title", "").lower()
    title_is_relevant = any(kw in current_title_lower for kw in config.RELEVANT_TITLE_KEYWORDS)

    if n_skill_matches < MIN_SKILL_MATCHES_FOR_FULL_SCORE:
        raw_score *= LOW_SKILL_MATCH_PENALTY
    elif n_skill_matches >= config.HIGH_SKILL_MATCH_THRESHOLD:
        if title_is_relevant:
            raw_score *= config.HIGH_SKILL_MATCH_BOOST
        else:
            # High skill count but irrelevant title = classic keyword-
            # stuffing trap candidate. Penalize harder than a normal
            # low-match candidate since this pattern is worse, not better.
            raw_score *= 0.4

    final_score = raw_score * penalty

    return {
        "final_score": final_score,
        "semantic_score": semantic_score,
        "skill_overlap": skill_overlap,
        "skill_depth": skill_depth,
        "experience_fit": experience_combined,
        "behavioral": behavioral_combined,
        "disqualifier_penalty": penalty,
        "n_skill_matches": n_skill_matches,
        "title_relevant": title_is_relevant,
        "is_honeypot": False,
    }


def score_all_candidates(
    candidates: list[dict],
    candidate_ids_order: np.ndarray,
    semantic_scores: np.ndarray,
    parsed_jd: ParsedJD,
) -> list[dict]:
    """Scores every candidate, returning a list of result dicts each
    including candidate_id and all sub-scores. semantic_scores must be
    aligned with candidate_ids_order (same order as saved embeddings).
    """
    id_to_semantic = dict(zip(candidate_ids_order, semantic_scores))
    results = []
    for candidate in candidates:
        cid = candidate["candidate_id"]
        sem_score = float(id_to_semantic.get(cid, 0.0))
        scores = score_candidate(candidate, parsed_jd, sem_score)
        scores["candidate_id"] = cid
        scores["candidate"] = candidate
        results.append(scores)
    return results