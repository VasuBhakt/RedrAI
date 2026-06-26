"""
Experience-fit scoring: years of experience vs JD's stated band, plus a
basic "active production coder" check from career history recency.
"""

from datetime import date


def experience_years_fit(candidate: dict, min_years: float | None, max_years: float | None) -> float:
    """1.0 if squarely within the JD's band, tapering off outside it.
    If JD gives no band, returns a neutral 0.5 (no penalty either way).
    """
    years = candidate.get("profile", {}).get("years_of_experience", 0)
    if min_years is None:
        return 0.5

    if max_years is None:
        max_years = min_years + 5  # loose upper fallback

    if min_years <= years <= max_years:
        return 1.0

    if years < min_years:
        gap = min_years - years
        return max(0.0, 1.0 - gap / min_years)

    gap = years - max_years
    return max(0.0, 1.0 - gap / max_years)


def recent_coding_score(candidate: dict, stale_months_threshold: int = 18) -> float:
    """Penalizes candidates whose current role title suggests they've
    moved away from hands-on coding (architect/lead/manager titles) AND
    have held that role for longer than the staleness threshold.

    This directly encodes the JD's disqualifier: 'senior engineer who
    hasn't written production code in 18 months because they moved into
    architecture/tech lead roles.'
    """
    current_title = candidate.get("profile", {}).get("current_title", "").lower()
    non_coding_markers = ["architect", "tech lead", "engineering manager",
                            "director", "vp", "head of"]

    if not any(marker in current_title for marker in non_coding_markers):
        return 1.0  # title suggests still hands-on

    career_history = candidate.get("career_history", [])
    if not career_history:
        return 1.0

    current_job = next((j for j in career_history if j.get("is_current")), None)
    if not current_job:
        return 1.0

    duration = current_job.get("duration_months", 0)
    if duration > stale_months_threshold:
        return 0.3  # likely stale on hands-on coding
    return 0.7  # recently moved, some benefit of the doubt