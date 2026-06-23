"""
Parses raw job-description text into structured requirements.
Designed to be JD-format-agnostic: looks for patterns (years ranges,
known city names, skill terms from the dataset's own vocabulary) rather
than assuming a fixed template of headers/sections.
"""

import re
from dataclasses import dataclass, field
from src.config import COMMON_LOCATIONS

@dataclass
class ParsedJD:
    raw_text: str
    min_years: float | None = None
    max_years: float | None = None
    required_skills: set[str] = field(default_factory=set)
    preferred_locations: set[str] = field(default_factory=set)
    max_notice_days: int | None = None
    requires_production_experience: bool = False

def extract_years_range(text: str) -> tuple[float | None, float | None]:
    """Finds patterns like '5-9 years', '5–9 yrs', '5 to 9 years of experience'."""
    pattern = r'(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*(?:years|yrs)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    single = re.search(r'(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)', text, re.IGNORECASE)
    if single:
        return float(single.group(1)), None
    return None, None

def extract_locations(text: str, known_cities: set[str] = COMMON_LOCATIONS) -> set[str]:
    text_lower = text.lower()
    return {city for city in known_cities if city in text_lower}

def extract_notice_period(text: str) -> int | None:
    """Finds patterns like 'sub-30-day notice', '30 day notice period'."""
    match = re.search(r'(\d+)[\s-]*day\s*notice', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_required_skills(text: str, skill_vocabulary: set[str]) -> set[str]:
    """Matches against the skill vocabulary built from the candidate dataset
    (data_loader.build_skill_vocabulary), not a hardcoded list 
    """
    text_lower = text.lower()
    return {skill for skill in skill_vocabulary if skill in text_lower}

def parse_jd(jd_text: str, skill_vocabulary: set[str]) -> ParsedJD:
    min_years, max_years = extract_years_range(jd_text)
    return ParsedJD(
        raw_text=jd_text,
        min_years=min_years,
        max_years=max_years,
        required_skills=extract_required_skills(jd_text, skill_vocabulary),
        preferred_locations=extract_locations(jd_text),
        max_notice_days=extract_notice_period(jd_text),
        requires_production_experience="production" in jd_text.lower(),
    )