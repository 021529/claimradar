from src.scoring.fraud_patterns import KOREAN_FRAUD_PATTERNS, select_patterns, select_patterns_structural


def test_non_fraud_row_has_no_patterns():
    row = {"FraudFound_P": 0, "Days_Policy_Claim": "more than 30"}
    assert select_patterns(row) == []


def test_delayed_claim_triggers_jiyeon_singo():
    row = {"FraudFound_P": 1, "Days_Policy_Claim": "more than 30"}
    assert "jiyeon_singo" in select_patterns(row)


def test_no_police_report_and_no_witness_triggers_blackbox():
    row = {"FraudFound_P": 1, "PoliceReportFiled": "No", "WitnessPresent": "No"}
    assert "blackbox_mijangchak" in select_patterns(row)


def test_high_past_claims_triggers_naeilong_hwanja():
    row = {"FraudFound_P": 1, "PastNumberOfClaims": "more than 4"}
    assert "naeilong_hwanja" in select_patterns(row)


def test_simplified_schema_high_past_claims_triggers_naeilong_hwanja():
    row = {"FraudFound_P": 1, "past_number_of_claims": 3}
    assert "naeilong_hwanja" in select_patterns(row)


def test_naeilong_with_cheap_vehicle_escalates_to_hanbang():
    row = {
        "FraudFound_P": 1,
        "PastNumberOfClaims": "more than 4",
        "VehiclePrice": 8000,
    }
    patterns = select_patterns(row)
    assert "naeilong_hwanja" in patterns
    assert "hanbang_jangi_ipwon" in patterns


def test_naeilong_with_categorical_kaggle_vehicle_price_escalates_to_hanbang():
    row = {
        "FraudFound_P": 1,
        "PastNumberOfClaims": "more than 4",
        "VehiclePrice": "less than 20000",
    }
    patterns = select_patterns(row)
    assert "naeilong_hwanja" in patterns
    assert "hanbang_jangi_ipwon" in patterns


def test_fraud_row_without_triggers_falls_back_deterministically():
    row = {"FraudFound_P": 1, "case_id": 42}
    first = select_patterns(row, seed_key=42)
    second = select_patterns(row, seed_key=42)
    assert first == second
    assert 1 <= len(first) <= 2
    assert all(key in KOREAN_FRAUD_PATTERNS for key in first)


def test_patterns_capped_at_two():
    row = {
        "FraudFound_P": 1,
        "Days_Policy_Claim": "more than 30",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "PastNumberOfClaims": "more than 4",
        "VehiclePrice": 8000,
    }
    assert len(select_patterns(row)) <= 2


def test_structural_ignores_label_triggers_for_non_fraud_row():
    """select_patterns()와 정반대로, 라벨이 0이어도 구조적 조건만 맞으면 정황이 붙어야 한다."""
    row = {"FraudFound_P": 0, "Days_Policy_Claim": "more than 30"}
    assert "jiyeon_singo" in select_patterns_structural(row)


def test_structural_gives_no_patterns_for_fraud_row_without_triggers():
    """select_patterns()는 라벨 전용 랜덤 폴백으로 뭔가 채워주지만, 구조적 버전은
    조건이 하나도 안 맞으면 사기 건이라도 빈 리스트를 반환해야 한다(라벨 무관 폴백 없음)."""
    row = {"FraudFound_P": 1, "case_id": 42}
    assert select_patterns_structural(row) == []


def test_structural_same_conditions_as_select_patterns_for_fraud_row():
    row = {
        "FraudFound_P": 1,
        "Days_Policy_Claim": "more than 30",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
    }
    assert select_patterns_structural(row) == select_patterns(row)
