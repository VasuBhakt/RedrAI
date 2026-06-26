"""
Honeypot detection: catches candidates with internally impossible or
inconsistent profiles (e.g. years of experience that don't add up,
"expert" skills with zero duration, career dates that don't add up).

These are forced toward the bottom of the ranking — submissions with
>10% honeypot rate in top 100 get disqualified at Stage 3.
"""

from datetime import date


def has_impossible_skill_claims(candidate: dict) -> bool:
    """Flags 'expert' or 'advanced' proficiency claimed with near-zero
    actual duration_months — a classic honeypot pattern."""
    for skill in candidate.get("skills", []):
        proficiency = skill.get("proficiency", "")
        duration = skill.get("duration_months", 0)
        if proficiency in ("expert", "advanced") and duration < 3:
            return True
    return False


def has_inconsistent_experience(candidate: dict) -> bool:
    """Flags candidates whose stated years_of_experience doesn't roughly
    match the sum of their career_history durations."""
    stated_years = candidate.get("profile", {}).get("years_of_experience", 0)
    history = candidate.get("career_history", [])
    if not history:
        return False

    total_months = sum(j.get("duration_months", 0) for j in history)
    total_years = total_months / 12

    # allow generous tolerance for overlapping roles / gaps
    if abs(stated_years - total_years) > max(stated_years * 0.6, 3):
        return True
    return False


def has_too_many_expert_skills(candidate: dict, threshold: int = 8) -> bool:
    """Flags candidates claiming 'expert' proficiency in an implausibly
    large number of skills — breadth-of-expertise honeypot pattern."""
    expert_count = sum(
        1 for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert"
    )
    return expert_count >= threshold


def is_honeypot(candidate: dict) -> bool:
    """True if ANY honeypot pattern is detected."""
    return (
        has_impossible_skill_claims(candidate)
        or has_inconsistent_experience(candidate)
        or has_too_many_expert_skills(candidate)
    )