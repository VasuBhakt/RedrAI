"""
Behavioral/availability scoring from redrob_signals — activity, response
rate, notice period, location/relocation fit. Handles sentinel values
(-1) correctly so missing data isn't punished as if it were the worst
possible score.
"""

from src import config


def activity_score(candidate: dict) -> float:
    """Combines recruiter response rate and recent platform activity into
    one availability signal. A perfect-on-paper candidate who's gone dark
    should score low here regardless of skills."""
    signals = candidate.get("redrob_signals", {})

    response_rate = signals.get("recruiter_response_rate", 0.0)
    open_to_work = config.OPEN_TO_WORK_SCORE if signals.get("open_to_work_flag", False) else config.NOT_OPEN_TO_WORK_SCORE

    return (response_rate * config.RESPONSE_RATE_WEIGHT) + (open_to_work * config.OPEN_TO_WORK_WEIGHT)


def github_score(candidate: dict) -> float:
    """0.5 neutral if no GitHub linked (sentinel -1), not a penalty —
    plenty of strong candidates simply don't link GitHub."""
    signals = candidate.get("redrob_signals", {})
    score = signals.get("github_activity_score", config.NO_GITHUB_SENTINEL)
    if score == config.NO_GITHUB_SENTINEL:
        return config.GITHUB_NO_LINK_SCORE
    return min(score / 100, 1.0)


def location_notice_fit(candidate: dict, preferred_locations: set[str],
                          max_notice_days: int | None) -> float:
    """Checks location match (or willingness to relocate) and notice
    period against the JD's stated preferences."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = profile.get("location", "").lower()
    location_match = any(loc in location for loc in preferred_locations)
    willing_to_relocate = signals.get("willing_to_relocate", False)

    location_score = config.LOCATION_MATCH_SCORE if (location_match or willing_to_relocate) else config.LOCATION_NO_MATCH_SCORE

    notice_days = signals.get("notice_period_days", 60)
    if max_notice_days is None:
        notice_score = config.NOTICE_NEUTRAL_SCORE  # neutral if JD didn't specify
    elif notice_days <= max_notice_days:
        notice_score = 1.0
    else:
        gap = notice_days - max_notice_days
        notice_score = max(0.0, 1.0 - gap / 90)

    return (location_score * 0.5) + (notice_score * 0.5)


def behavioral_score(candidate: dict, preferred_locations: set[str],
                       max_notice_days: int | None) -> float:
    """Combined behavioral signal: activity + github + location/notice."""
    return (
        activity_score(candidate) * config.BEHAVIORAL_ACTIVITY_WEIGHT
        + github_score(candidate) * config.BEHAVIORAL_GITHUB_WEIGHT
        + location_notice_fit(candidate, preferred_locations, max_notice_days) * config.BEHAVIORAL_LOCATION_WEIGHT
    )