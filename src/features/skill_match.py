"""
Skill overlap and depth scoring between a candidate and a parsed JD.
"""

from src.data_loader import get_skill_names


def skill_overlap_score(candidate: dict, required_skills: set[str]) -> float:
    """Fraction of JD-required skills the candidate actually has listed."""
    if not required_skills:
        return 0.0
    candidate_skills = get_skill_names(candidate)
    overlap = candidate_skills & required_skills
    return len(overlap) / len(required_skills)


def skill_depth_score(candidate: dict, required_skills: set[str]) -> float:
    """Rewards skills the candidate has used for longer and gotten endorsed
    for, not just listed. Skills outside the JD's required set don't count.
    """
    if not required_skills:
        return 0.0

    matched = [
        s for s in candidate.get("skills", [])
        if s.get("name", "").strip().lower() in required_skills
    ]
    if not matched:
        return 0.0

    proficiency_map = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}

    scores = []
    for s in matched:
        prof = proficiency_map.get(s.get("proficiency", "beginner"), 0.25)
        duration_factor = min(s.get("duration_months", 0) / 36, 1.0)  # cap at 3 years
        endorsement_factor = min(s.get("endorsements", 0) / 30, 1.0)  # cap at 30 endorsements
        scores.append((prof + duration_factor + endorsement_factor) / 3)

    return sum(scores) / len(scores)