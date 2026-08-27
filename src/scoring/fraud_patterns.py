"""국내 자동차보험 특유 사기 패턴 정의 및 정형 레코드 기반 트리거 로직.

Kaggle "Vehicle Insurance Claim Fraud Detection" 컬럼(Days_Policy_Claim,
PoliceReportFiled, WitnessPresent, PastNumberOfClaims, NumberOfSuppliments,
BasePolicy, VehiclePrice 등)과, 데모용 sample_claims.csv의 축약 컬럼(
past_number_of_claims, vehicle_price 등) 양쪽에서 동작하도록 컬럼명 후보를
순서대로 조회한다.
"""

import random
from typing import Any, Mapping

KOREAN_FRAUD_PATTERNS: dict[str, dict[str, str]] = {
    "naeilong_hwanja": {
        "label": "나이롱환자",
        "description": "경미한 사고임에도 장기 통원치료 및 증상을 과장하는 정황",
        "narrative_hint": "경미한 접촉사고였음에도 목·허리 통증을 호소하며 장기간 통원치료를 이어가고 있다는 정황",
    },
    "hanbang_jangi_ipwon": {
        "label": "한방병원 장기입원",
        "description": "한방병원(한의원)에 장기간 입원하며 치료비를 청구하는 정황",
        "narrative_hint": "인근 한방병원에 입원해 수 주간 입원치료를 이어가고 있다는 정황",
    },
    "jiyeon_singo": {
        "label": "지연신고",
        "description": "사고 발생 후 상당 기간이 지난 뒤에야 보험금을 청구하는 정황",
        "narrative_hint": "사고 발생일로부터 상당 기간이 지난 뒤에야 보험사에 사고 접수를 했다는 정황",
    },
    "blackbox_mijangchak": {
        "label": "블랙박스 미장착",
        "description": "차량에 블랙박스가 없어 사고 상황을 객관적으로 확인할 목격자·영상 자료가 없는 정황",
        "narrative_hint": "사고 당시 차량에 블랙박스가 장착되어 있지 않았고 목격자도 없어 사고 경위를 객관적으로 확인하기 어렵다는 정황",
    },
}

_DELAYED_BINS = {"15 to 30", "more than 30"}
_HIGH_PAST_CLAIMS = {"2 to 4", "more than 4"}
_HIGH_SUPPLEMENTS = {"3 to 5", "more than 5"}
_MAX_PATTERNS = 2


def _get(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        try:
            value = row[name]
        except (KeyError, IndexError):
            continue
        if value is None:
            continue
        try:
            import math

            if isinstance(value, float) and math.isnan(value):
                continue
        except TypeError:
            pass
        return value
    return None


def _is_fraud(row: Mapping[str, Any]) -> bool:
    value = _get(row, "FraudFound_P")
    try:
        return int(value) == 1
    except (TypeError, ValueError):
        return False


def select_patterns(row: Mapping[str, Any], seed_key: Any = None) -> list[str]:
    """정형 레코드의 신호를 바탕으로 반영할 국내 사기 패턴 키 목록을 선택.

    FraudFound_P == 1 인 레코드에 한해 최대 _MAX_PATTERNS개까지 선택한다.
    트리거될 신호가 전혀 없는 사기 건은 seed_key로 재현 가능한 난수 선택을 폴백으로 사용한다.
    """
    if not _is_fraud(row):
        return []

    triggered: list[str] = []

    days_claim = _get(row, "Days_Policy_Claim")
    days_accident = _get(row, "Days_Policy_Accident")
    if days_claim in _DELAYED_BINS or days_accident in _DELAYED_BINS:
        triggered.append("jiyeon_singo")

    police_report = _get(row, "PoliceReportFiled")
    witness = _get(row, "WitnessPresent")
    if police_report == "No" and witness == "No":
        triggered.append("blackbox_mijangchak")

    past_claims = _get(row, "PastNumberOfClaims", "past_number_of_claims")
    supplements = _get(row, "NumberOfSuppliments")
    high_past_claims = past_claims in _HIGH_PAST_CLAIMS or (
        isinstance(past_claims, (int, float)) and past_claims >= 2
    )
    if high_past_claims or supplements in _HIGH_SUPPLEMENTS:
        triggered.append("naeilong_hwanja")

    if "naeilong_hwanja" in triggered:
        base_policy = _get(row, "BasePolicy")
        vehicle_price = _get(row, "VehiclePrice", "vehicle_price")
        if isinstance(vehicle_price, (int, float)):
            cheap_vehicle = vehicle_price < 15000
        else:
            cheap_vehicle = vehicle_price in {"less than 20000", "20000 to 29000"}
        if base_policy == "All Perils" or cheap_vehicle:
            triggered.append("hanbang_jangi_ipwon")

    # 중복 제거, 순서 유지
    triggered = list(dict.fromkeys(triggered))

    if not triggered:
        fallback_seed = seed_key if seed_key is not None else _get(row, "PolicyNumber", "case_id")
        rng = random.Random(str(fallback_seed) if fallback_seed is not None else None)
        triggered = rng.sample(list(KOREAN_FRAUD_PATTERNS), k=rng.randint(1, 2))

    return triggered[:_MAX_PATTERNS]
