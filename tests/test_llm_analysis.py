from src.scoring.fraud_patterns import KOREAN_FRAUD_PATTERNS
from src.scoring.llm_analysis import NarrativeAnalysis, analyze_narrative, generate_synthetic_narrative


def test_synthetic_narrative_is_korean_text_without_api_key(monkeypatch):
    monkeypatch.setattr("src.scoring.llm_analysis.ANTHROPIC_API_KEY", "")
    row = {"FraudFound_P": 1, "Days_Policy_Claim": "more than 30", "AccidentArea": "Urban"}
    text = generate_synthetic_narrative(row)
    assert isinstance(text, str)
    assert text
    assert KOREAN_FRAUD_PATTERNS["jiyeon_singo"]["narrative_hint"] in text


def test_non_fraud_narrative_has_no_fraud_pattern_language(monkeypatch):
    monkeypatch.setattr("src.scoring.llm_analysis.ANTHROPIC_API_KEY", "")
    row = {"FraudFound_P": 0, "AccidentArea": "Rural"}
    text = generate_synthetic_narrative(row)
    assert "한방병원" not in text
    assert "블랙박스" not in text


def test_narrative_deterministic_for_same_row(monkeypatch):
    monkeypatch.setattr("src.scoring.llm_analysis.ANTHROPIC_API_KEY", "")
    row = {"FraudFound_P": 1, "case_id": 7}
    assert generate_synthetic_narrative(row) == generate_synthetic_narrative(row)


def test_analyze_narrative_round_trips_generated_fraud_narrative(monkeypatch):
    monkeypatch.setattr("src.scoring.llm_analysis.ANTHROPIC_API_KEY", "")
    row = {"FraudFound_P": 1, "Days_Policy_Claim": "more than 30", "AccidentArea": "Urban"}
    narrative = generate_synthetic_narrative(row)

    result = analyze_narrative(narrative)

    assert isinstance(result, NarrativeAnalysis)
    assert result.keywords
    assert any("지연신고" in kw for kw in result.keywords)
    assert result.suspicion_adjustment > 0
    assert result.explanation


def test_analyze_narrative_round_trips_generated_normal_narrative(monkeypatch):
    monkeypatch.setattr("src.scoring.llm_analysis.ANTHROPIC_API_KEY", "")
    row = {"FraudFound_P": 0, "AccidentArea": "Rural"}
    narrative = generate_synthetic_narrative(row)

    result = analyze_narrative(narrative)

    assert result.keywords == []
    assert result.suspicion_adjustment < 0
    assert result.explanation


def test_analyze_narrative_empty_text_short_circuits(monkeypatch):
    monkeypatch.setattr("src.scoring.llm_analysis.ANTHROPIC_API_KEY", "")
    result = analyze_narrative("   ")
    assert result.keywords == []
    assert result.suspicion_adjustment == 0.0


def test_analyze_narrative_suspicion_adjustment_bounded(monkeypatch):
    monkeypatch.setattr("src.scoring.llm_analysis.ANTHROPIC_API_KEY", "")
    text = (
        "경미한 접촉사고였음에도 목·허리 통증을 호소하며 장기간 통원치료를 이어가고 있다는 정황이 확인됨. "
        "인근 한방병원에 입원해 수 주간 입원치료를 이어가고 있다는 정황이 확인됨. "
        "사고 발생일로부터 상당 기간이 지난 뒤에야 보험사에 사고 접수를 했다는 정황이 확인됨. "
        "사고 당시 차량에 블랙박스가 장착되어 있지 않았고 목격자도 없어 사고 경위를 객관적으로 확인하기 어렵다는 정황이 확인됨."
    )
    result = analyze_narrative(text)
    assert result.suspicion_adjustment <= 1.0
