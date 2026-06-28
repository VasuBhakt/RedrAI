import pytest
from src.features import honeypot

def test_temporal_anomaly_honeypot():
    # A candidate who claims a skill duration much longer than their total career
    candidate_lying = {
        "candidate_id": "LIAR",
        "profile": {"years_of_experience": 2.0}, # 24 months total experience
        "skills": [
            {
                "name": "pytorch",
                "duration_months": 60 # Claims 5 years experience with pytorch
            }
        ]
    }
    
    # Should flag as honeypot because 60 months > 24 months + 12 (buffer)
    assert honeypot.is_honeypot(candidate_lying) == True
    
def test_honest_candidate_honeypot():
    candidate_honest = {
        "candidate_id": "HONEST",
        "profile": {"years_of_experience": 5.0}, # 60 months
        "skills": [
            {
                "name": "pytorch",
                "duration_months": 40
            }
        ]
    }
    
    assert honeypot.is_honeypot(candidate_honest) == False
