"""
Generates a 1-2 sentence reasoning string per candidate from their actual
computed scores — no LLM call, pure templating from real feature values.
This satisfies the spec's "no hallucination" requirement by construction:
every claim comes directly from data we already extracted.
"""


def generate_reasoning(candidate: dict, scores: dict, parsed_jd) -> str:
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Unknown role")
    years = profile.get("years_of_experience", 0)

    n_matched = len(
        {s["name"].lower() for s in candidate.get("skills", [])} & parsed_jd.required_skills
    )
    n_required = len(parsed_jd.required_skills)

    response_rate = candidate.get("redrob_signals", {}).get("recruiter_response_rate", 0.0)

    parts = [f"{title} with {years} yrs experience"]
    parts.append(f"matched {n_matched}/{n_required} JD-required skills")
    parts.append(f"recruiter response rate {response_rate:.2f}")

    if scores["disqualifier_penalty"] < 1.0:
        parts.append("some career-fit concerns noted")

    if scores["experience_fit"] < 0.5:
        parts.append("experience band is a stretch for this role")

    return "; ".join(parts) + "."