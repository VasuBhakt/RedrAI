import pytest
from src.scorer import score_candidate
from src.jd_parser import ParsedJD
from src import config

def get_mock_jd():
    # JD needing 8 skills minimum for HIGH_SKILL_MATCH_THRESHOLD
    required_skills = {"python", "pytorch", "nlp", "machine learning", "deep learning", "aws", "docker", "fastapi"}
    return ParsedJD(
        raw_text="looking for ml engineer...",
        min_years=3.0,
        max_years=8.0,
        required_skills=required_skills,
        preferred_locations={"pune", "bangalore"},
        max_notice_days=30,
        requires_production_experience=False
    )

def test_ultimate_keyword_stuffer():
    # A Barista who stuffed 8+ AI skills into their profile
    candidate = {
        "candidate_id": "STUFFER",
        "profile": {"current_title": "barista"},
        "career_history": [{"title": "barista"}],
        "skills": [{"name": s} for s in ["python", "pytorch", "nlp", "machine learning", "deep learning", "aws", "docker", "fastapi"]],
        "years_experience": 2.0
    }
    jd = get_mock_jd()
    
    # Give them a high semantic score of 1.0 (they copied the JD)
    result = score_candidate(candidate, jd, semantic_score=1.0)
    
    # They should have the KEYWORD_STUFFER_PENALTY applied because they have 8 skills but an irrelevant title
    assert result["title_relevant"] == False
    
    # Check that penalty is actually applied to the final score
    assert result["final_score"] < 0.5  # With a 0.4 penalty, it should be severely reduced

def test_ml_engineer_boost():
    # True ML Engineer
    candidate_ml = {
        "candidate_id": "ML_ENG",
        "profile": {"current_title": "ml engineer"},
        "career_history": [{"title": "ml engineer"}],
        "skills": [{"name": "python"}, {"name": "pytorch"}, {"name": "machine learning"}],
        "years_experience": 4.0
    }
    
    # Generic SWE
    candidate_swe = {
        "candidate_id": "SWE",
        "profile": {"current_title": "software engineer"},
        "career_history": [{"title": "software engineer"}],
        "skills": [{"name": "python"}, {"name": "pytorch"}, {"name": "machine learning"}],
        "years_experience": 4.0
    }
    
    jd = get_mock_jd()
    
    res_ml = score_candidate(candidate_ml, jd, semantic_score=0.8)
    res_swe = score_candidate(candidate_swe, jd, semantic_score=0.8)
    
    # Both are relevant titles, so title_relevant should be True for both
    assert res_ml["title_relevant"] == True
    assert res_swe["title_relevant"] == True
    
    # But ML Engineer should get the HIGHLY_RELEVANT_TITLE_BOOST (1.2x)
    assert res_ml["final_score"] > res_swe["final_score"]
    
def test_stale_senior():
    from src.features import experience_fit
    
    candidate_recent = {
        "candidate_id": "REC",
        "profile": {"current_title": "director of engineering"},
        "career_history": [
            {"title": "director of engineering", "duration_months": 12, "is_current": True},
            {"title": "software engineer", "duration_months": 120}
        ]
    }
    
    # Stale senior who transitioned to Director
    candidate_stale = {
        "candidate_id": "DIR",
        "profile": {"current_title": "director of engineering"},
        "career_history": [
            {"title": "director of engineering", "duration_months": 24, "is_current": True},
            {"title": "software engineer", "duration_months": 120}
        ]
    }
    
    score_recent = experience_fit.recent_coding_score(candidate_recent)
    score_stale = experience_fit.recent_coding_score(candidate_stale)
    
    assert score_stale == config.CODING_STALE_SCORE
    assert score_recent == config.CODING_RECENT_SCORE
