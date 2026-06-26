"""
Hard disqualifier checks from the JD. These don't produce a 0-1 score —
they return a boolean "is this candidate disqualified" plus a penalty
multiplier the scorer applies. Encodes the JD's explicit "we will not
move forward" rules.
"""

from src.data_loader import get_all_industries
from src.config import SERVICES_INDUSTRIES


def is_services_only_career(candidate: dict) -> bool:
    """True if every industry in the candidate's career history (current +
    past) falls into a generic services/consulting category — i.e. no
    product company experience anywhere."""
    industries = get_all_industries(candidate)
    if not industries:
        return False
    return industries.issubset(SERVICES_INDUSTRIES)


def is_pure_research_no_production(candidate: dict) -> bool:
    """Flags candidates whose career history reads as academic/research-only
    with no signal of shipping to production. Heuristic: checks for research
    keywords in titles/descriptions with no 'production'/'deployed'/'shipped'
    language anywhere in their history."""
    research_markers = ["research scientist", "phd", "postdoc", "academic",
                          "research fellow", "research associate"]
    production_markers = ["production", "deployed", "shipped", "scale",
                            "real-time", "live system"]

    full_text = " ".join([
        candidate.get("profile", {}).get("current_title", ""),
        candidate.get("profile", {}).get("summary", ""),
    ] + [j.get("description", "") for j in candidate.get("career_history", [])]).lower()

    has_research = any(marker in full_text for marker in research_markers)
    has_production = any(marker in full_text for marker in production_markers)

    return has_research and not has_production


def disqualifier_penalty(candidate: dict, penalize_services_only: bool = True) -> float:
    """Returns a multiplier in [0, 1] to apply to the final score.
    1.0 = no penalty. Lower = harder disqualifier triggered.
    """
    penalty = 1.0

    if penalize_services_only and is_services_only_career(candidate):
        penalty *= 0.5  # soft penalty, not a hard zero — JD says "case by case"

    if is_pure_research_no_production(candidate):
        penalty *= 0.2  # JD explicitly says "we will not move forward"

    return penalty