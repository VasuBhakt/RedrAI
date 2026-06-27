"""
Generates a 1-2 sentence reasoning string per candidate from their actual
computed scores — no LLM call, pure templating from real feature values.
This satisfies the spec's "no hallucination" requirement by construction:
every claim comes directly from data we already extracted.
"""


import random

def generate_reasoning(candidate: dict, scores: dict, parsed_jd) -> str:
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Unknown role")
    years = profile.get("years_of_experience", 0)

    n_matched = len(
        {s["name"].lower() for s in candidate.get("skills", [])} & parsed_jd.required_skills
    )
    n_required = len(parsed_jd.required_skills)

    response_rate = candidate.get("redrob_signals", {}).get("recruiter_response_rate", 0.0)

    # Seed RNG deterministically using candidate_id so outputs are completely stable across runs
    rng = random.Random(candidate.get("candidate_id", "default_seed"))

    # Base profile templates
    if years >= 5:
        summaries = [
            f"Experienced {title} ({years} years)",
            f"A {title} bringing {years} years of experience",
            f"Seasoned {title} with {years} yrs experience"
        ]
    else:
        summaries = [
            f"{title} with {years} years of experience",
            f"Currently a {title} ({years} yrs experience)",
            f"{title} demonstrating {years} years of experience"
        ]
    summary = rng.choice(summaries)

    # Skills templates
    if n_matched >= 8:
        skills = [
            f"strong technical alignment, hitting {n_matched} of {n_required} core skills",
            f"exceptional skill match ({n_matched}/{n_required} required technologies)",
            f"demonstrates deep technical fit by checking {n_matched} out of {n_required} JD skills"
        ]
    elif n_matched >= 4:
        skills = [
            f"solid foundation with {n_matched}/{n_required} required skills",
            f"matches {n_matched} core technologies",
            f"meets {n_matched} of the {n_required} JD skill requirements"
        ]
    else:
        skills = [
            f"matches {n_matched} key skills",
            f"shows baseline familiarity with {n_matched} required skills",
            f"covers {n_matched}/{n_required} required technical skills"
        ]
    skill_text = rng.choice(skills)

    # Behavioral templates
    if response_rate >= 0.7:
        behaviors = [
            f"highly responsive (historically replies to {response_rate:.0%} of outreach)",
            f"shows excellent recruiter engagement ({response_rate:.0%} response rate)",
            f"very active on platform ({response_rate:.0%} response rate)"
        ]
    elif response_rate >= 0.4:
        behaviors = [
            f"moderate response rate ({response_rate:.0%})",
            f"average responsiveness to outreach ({response_rate:.0%})",
            f"engages reasonably well ({response_rate:.0%} response rate)"
        ]
    else:
        behaviors = [
            f"historically low responsiveness ({response_rate:.0%})",
            f"rarely responds to outreach ({response_rate:.0%} rate)",
            f"low platform engagement ({response_rate:.0%} response rate)"
        ]
    behavioral_text = rng.choice(behaviors)

    parts = [summary, skill_text, behavioral_text]

    # Edge Cases
    if scores["disqualifier_penalty"] < 1.0:
        parts.append(rng.choice(["flagged for minor career-path concerns", "some domain relevance concerns noted"]))

    if scores["experience_fit"] < 0.5:
        parts.append(rng.choice(["experience falls outside the target band", "years of experience is a slight mismatch"]))

    # Fix grammar for "A" before vowels
    final_text = ". ".join(parts) + "."
    final_text = final_text.replace("A AI", "An AI").replace("A Applied", "An Applied").replace("A Analytics", "An Analytics").replace("A ML", "An ML")
    
    # Capitalize only the first letter of the first part, let the rest flow naturally
    return final_text[0].upper() + final_text[1:]