import pytest
from src.scorer import score_candidate
from src.jd_parser import ParsedJD

def test_ghost_candidate_null_fields():
    # Candidate missing entirely optional or all fields
    ghost = {
        "candidate_id": "GHOST",
        # no profile
        # no skills
        # no career_history
        # no years_experience
    }
    
    jd = ParsedJD(
        raw_text="Looking for someone",
        min_years=0.0,
        max_years=None,
        required_skills=set(),
        preferred_locations=set(),
        max_notice_days=None,
        requires_production_experience=False
    )
    
    # Engine should not crash. It should just score them terribly.
    result = score_candidate(ghost, jd, semantic_score=0.1)
    
    assert result["final_score"] >= 0.0
    assert result["final_score"] < 0.5 # Should be very low since they have nothing
