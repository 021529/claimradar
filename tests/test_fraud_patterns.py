from src.scoring.fraud_patterns import KOREAN_FRAUD_PATTERNS, select_patterns


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
