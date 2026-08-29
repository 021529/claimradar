"""app.py와 오프라인 분석 스크립트(scripts/lambda_sweep_24case.py 등)가 공유하는
스코어링 피처 목록. scripts/prepare_app_dataset.py가 만드는 컬럼과 정확히
일치해야 한다.

CSV 업로드 등 옛 5피처 스키마와도 호환되도록, 범주형은 "존재하는 것만" 쓰는
선택적 목록으로 둔다 — 없어도 스코어링은 CORE_NUMERIC_FEATURE_COLS만으로 동작한다.
범주형 보강 효과는 scripts/ml_model_comparison_results.md에서 검증됨
(AUC-ROC 0.53→0.80, PR-AUC 0.07→0.17).
"""

CORE_NUMERIC_FEATURE_COLS = [
    "driver_age",
    "vehicle_price",
    "deductible",
    "driver_rating",
    "past_number_of_claims",
]

OPTIONAL_CATEGORICAL_FEATURE_COLS = [
    "age_of_vehicle_rank",
    "age_of_policyholder_rank",
    "days_policy_accident_rank",
    "days_policy_claim_rank",
    "number_of_suppliments_rank",
    "address_change_claim_rank",
    "number_of_cars_rank",
    "Make",
    "Sex",
    "MaritalStatus",
    "Fault",
    "PolicyType",
    "VehicleCategory",
    "AccidentArea",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgentType",
    "BasePolicy",
]


def feature_cols_for(df) -> list[str]:
    """데이터에 실제로 있는 컬럼만 골라 사용 (옛 5피처 스키마와 호환)."""
    return CORE_NUMERIC_FEATURE_COLS + [c for c in OPTIONAL_CATEGORICAL_FEATURE_COLS if c in df.columns]
