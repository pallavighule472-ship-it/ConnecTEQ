import pytest
from HR_backend import (
    _parse_llm_json,
    _regex_email,
    _regex_phone,
    ScoringWeights,
    shortlisting_node,
    route_after_shortlist,
    _build_evaluation_dimensions,
    JobProfile,
    Skills,
)


# ─── _parse_llm_json ─────────────────────────────────────────────────────────

def test_parse_llm_json_plain():
    assert _parse_llm_json('{"score": 7}') == {"score": 7}

def test_parse_llm_json_strips_markdown():
    raw = "```json\n{\"score\": 7}\n```"
    assert _parse_llm_json(raw) == {"score": 7}

def test_parse_llm_json_strips_plain_fence():
    raw = "```\n{\"score\": 7}\n```"
    assert _parse_llm_json(raw) == {"score": 7}

def test_parse_llm_json_invalid_raises():
    with pytest.raises(Exception):
        _parse_llm_json("not json at all")


# ─── _regex_email ─────────────────────────────────────────────────────────────

def test_regex_email_found():
    assert _regex_email("Contact me at john.doe@example.com please") == "john.doe@example.com"

def test_regex_email_not_found():
    assert _regex_email("No email here") is None

def test_regex_email_with_plus():
    assert _regex_email("john+tag@gmail.com") == "john+tag@gmail.com"


# ─── _regex_phone ─────────────────────────────────────────────────────────────

def test_regex_phone_found():
    assert _regex_phone("Call me at +91 9876543210") is not None

def test_regex_phone_not_found():
    assert _regex_phone("No phone number here") is None


# ─── ScoringWeights.normalized ───────────────────────────────────────────────

def test_scoring_weights_default_sum_to_one():
    weights = ScoringWeights()
    total = sum(weights.normalized().values())
    assert abs(total - 1.0) < 0.001

def test_scoring_weights_custom_sum_to_one():
    weights = ScoringWeights(skills=0.5, experience=0.3, education=0.1, culture_fit=0.1)
    total = sum(weights.normalized().values())
    assert abs(total - 1.0) < 0.001

def test_scoring_weights_uneven_normalizes():
    weights = ScoringWeights(skills=2.0, experience=1.0, education=0.5, culture_fit=0.5)
    normalized = weights.normalized()
    assert abs(sum(normalized.values()) - 1.0) < 0.001
    assert normalized["skills"] > normalized["experience"]


# ─── shortlisting_node ───────────────────────────────────────────────────────

def test_shortlisting_above_threshold():
    result = shortlisting_node({"match_score": 7.0, "candidate_id": "x"})
    assert result["shortlisted"] is True

def test_shortlisting_at_threshold():
    result = shortlisting_node({"match_score": 6.0, "candidate_id": "x"})
    assert result["shortlisted"] is True

def test_shortlisting_below_threshold():
    result = shortlisting_node({"match_score": 5.9, "candidate_id": "x"})
    assert result["shortlisted"] is False

def test_shortlisting_zero_score():
    result = shortlisting_node({"match_score": 0.0, "candidate_id": "x"})
    assert result["shortlisted"] is False


# ─── route_after_shortlist ───────────────────────────────────────────────────

def test_route_shortlisted_goes_to_interview():
    assert route_after_shortlist({"shortlisted": True}) == "Interview_scheduling"

def test_route_not_shortlisted_goes_to_rejection():
    assert route_after_shortlist({"shortlisted": False}) == "Shortlist_rejection"


# ─── _build_evaluation_dimensions ────────────────────────────────────────────

def test_build_dimensions_has_core():
    job = JobProfile(skills=Skills(must_have=["Python", "Django"]))
    dims = _build_evaluation_dimensions(job)
    names = [d["name"] for d in dims]
    assert "Communication" in names
    assert "Problem Solving" in names
    assert "Culture Fit" in names

def test_build_dimensions_includes_skills():
    job = JobProfile(skills=Skills(must_have=["Python", "FastAPI"]))
    dims = _build_evaluation_dimensions(job)
    names = [d["name"] for d in dims]
    assert "Python" in names
    assert "FastAPI" in names

def test_build_dimensions_caps_at_five_skills():
    job = JobProfile(skills=Skills(must_have=["A", "B", "C", "D", "E", "F", "G"]))
    dims = _build_evaluation_dimensions(job)
    technical = [d for d in dims if d["name"] not in ("Communication", "Problem Solving", "Culture Fit")]
    assert len(technical) == 5

def test_build_dimensions_weights_are_positive():
    job = JobProfile(skills=Skills(must_have=["Python"]))
    dims = _build_evaluation_dimensions(job)
    assert all(d["weight"] > 0 for d in dims)
