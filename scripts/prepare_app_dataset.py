"""실제 Kaggle+합성 사고경위서 데이터셋을 app.py/시뮬레이션이 기대하는 스키마로 변환한다.

Kaggle "Vehicle Insurance Claim Fraud Detection"에는 case_id/expected_hours/
expected_recovery/investigation_cost 같은 업무 수치가 없고, VehiclePrice/
PastNumberOfClaims 등 일부 피처가 구간 문자열(범주형)이다. 이 스크립트는:

1. 범주형 구간을 중간값/순서 랭크 기준 수치로 변환 (vehicle_price,
   past_number_of_claims, *_rank 컬럼들 — ORDINAL_ORDER 참고)
2. 명목형 범주(Make/Sex/Fault/PolicyType/... — NOMINAL_COLS 참고)는 원본
   문자열 그대로 전달해 ml_model.py가 One-Hot 인코딩하도록 함 (심사 지적:
   범주형 피처 누락 → scripts/ml_model_comparison_results.md에서 보강 효과
   정량 검증, AUC-ROC 0.53→0.83)
3. 조사 업무 수치(expected_hours/expected_recovery/investigation_cost)를
   관측 가능한 피처 기반의 투명한 공식으로 산출 (Kaggle 원본에 없는 값이므로
   시연용 가정 — FraudFound_P 라벨은 사용하지 않아 정답 라벨 누출 없음)
4. 사기 전체 + 정상 표본을 층화 추출 (기본: 실제 LLM 생성 사고경위서를 가진
   행 중에서만 뽑아 템플릿 문장이 섞이지 않도록 함)

사용법:
    python scripts/prepare_app_dataset.py \
        --input data/processed/claims_with_narratives.csv \
        --output data/processed/app_demo_sample.csv \
        --fraud-count 150 --normal-count 150 --seed 42
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

_VEHICLE_PRICE_MIDPOINT = {
    "less than 20000": 15000,
    "20000 to 29000": 24500,
    "30000 to 39000": 34500,
    "40000 to 59000": 49500,
    "60000 to 69000": 64500,
    "more than 69000": 75000,
}
_PAST_CLAIMS_MIDPOINT = {"none": 0, "1": 1, "2 to 4": 3, "more than 4": 5}

# 서열형(구간에 순서가 있는) 범주형 -> 순서 보존 정수 랭크. ML 스코어링 모델
# 보강 실험(scripts/ml_model_comparison.py, scripts/ml_model_comparison_results.md
# 참고 — AUC-ROC 0.53→0.83, PR-AUC 0.07→0.22)에서 검증된 것과 동일한 순서를 쓴다.
ORDINAL_ORDER: dict[str, list[str]] = {
    "AgeOfVehicle": [
        "new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7",
    ],
    "AgeOfPolicyHolder": [
        "16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35",
        "36 to 40", "41 to 50", "51 to 65", "over 65",
    ],
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "AddressChange_Claim": ["no change", "under 6 months", "1 year", "2 to 3 years", "4 to 8 years"],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
}

_ORDINAL_RANK_COL_NAMES: dict[str, str] = {
    "AgeOfVehicle": "age_of_vehicle_rank",
    "AgeOfPolicyHolder": "age_of_policyholder_rank",
    "Days_Policy_Accident": "days_policy_accident_rank",
    "Days_Policy_Claim": "days_policy_claim_rank",
    "NumberOfSuppliments": "number_of_suppliments_rank",
    "AddressChange_Claim": "address_change_claim_rank",
    "NumberOfCars": "number_of_cars_rank",
}

# 순서 없는 명목형 범주 컬럼. ml_model.py의 train_fraud_model이 dtype으로
# 수치형/범주형을 자동 구분해 One-Hot 인코딩하므로 원본 Kaggle 값을 그대로 둔다.
# 시간성 컬럼(Month/DayOfWeek/DayOfWeekClaimed/MonthClaimed)과 RepNumber(설계사
# 코드)는 벤치마크 전체 피처셋에는 포함했지만(scripts/ml_model_comparison.py의
# ENHANCED_FEATURE_COLS) 조사관 UI에 노출할 실익이 낮아 라이브 스키마에서는
# 제외 — 이 "해석 가능 부분집합"으로도 성능이 유지되는지
# scripts/ml_model_comparison.py의 interpretable_subset 실험으로 별도 검증함.
NOMINAL_COLS = [
    "Make", "Sex", "MaritalStatus", "Fault", "PolicyType", "VehicleCategory",
    "AccidentArea", "PoliceReportFiled", "WitnessPresent", "AgentType", "BasePolicy",
]


def _llm_generated_mask(df: pd.DataFrame) -> pd.Series:
    """narrative_text가 템플릿 폴백이 아니라 실제 LLM 생성인 행만 표시."""
    marker1 = "지역에서 발생한 사고에 대해 청구인이 사고경위서를 제출함"
    marker2 = "제출된 서류와 진술 내용은 사고 상황과 대체로 일치함"
    text = df["narrative_text"]
    return ~(text.str.contains(marker1, regex=False) | text.str.contains(marker2, regex=False))


def transform(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["case_id"] = df["PolicyNumber"]
    out["driver_age"] = df["Age"]
    out["vehicle_price"] = df["VehiclePrice"].map(_VEHICLE_PRICE_MIDPOINT)
    out["deductible"] = df["Deductible"]
    out["driver_rating"] = df["DriverRating"]
    out["past_number_of_claims"] = df["PastNumberOfClaims"].map(_PAST_CLAIMS_MIDPOINT)

    # 서열형 범주 -> 순서 보존 정수 랭크 (ml_model.py가 수치형으로 자동 인식해
    # One-Hot 없이 그대로 사용)
    for col, order in ORDINAL_ORDER.items():
        rank_map = {value: rank for rank, value in enumerate(order)}
        out[_ORDINAL_RANK_COL_NAMES[col]] = df[col].map(rank_map)

    # 명목형 범주는 원본 문자열 그대로 전달 (ml_model.py가 dtype으로 감지해 One-Hot)
    for col in NOMINAL_COLS:
        out[col] = df[col]

    # 조사 시간: 기본 2시간 + 관측 가능한 복잡도 신호마다 +1시간 (최대 8시간)
    complexity = (
        (df["PastNumberOfClaims"] != "none").astype(int)
        + (df["PoliceReportFiled"] == "No").astype(int)
        + (df["WitnessPresent"] == "No").astype(int)
        + df["NumberOfSuppliments"].isin(["3 to 5", "more than 5"]).astype(int)
    )
    out["expected_hours"] = (2 + complexity).clip(upper=8)

    # 기대 회수액: 차량가액의 40%가 조사로 회수 가능하다고 가정 (예시 가정치)
    out["expected_recovery"] = (out["vehicle_price"] * 0.4).round(0)
    # 조사 비용: 조사 시간당 350 (원 단위 상대값, sample_claims.csv 데모와 동일 비율)
    out["investigation_cost"] = (out["expected_hours"] * 350).round(0)

    out["narrative_text"] = df["narrative_text"]
    out["FraudFound_P"] = df["FraudFound_P"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fraud-count", type=int, default=150)
    parser.add_argument("--normal-count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--llm-only",
        action="store_true",
        default=True,
        help="템플릿 폴백이 아닌 실제 LLM 생성 사고경위서를 가진 행에서만 표본 추출 (기본값)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.llm_only:
        df = df.loc[_llm_generated_mask(df)]

    fraud = df.loc[df["FraudFound_P"] == 1]
    normal = df.loc[df["FraudFound_P"] == 0]
    fraud_sample = fraud.sample(n=min(args.fraud_count, len(fraud)), random_state=args.seed)
    normal_sample = normal.sample(n=min(args.normal_count, len(normal)), random_state=args.seed)

    combined = pd.concat([fraud_sample, normal_sample]).sample(frac=1, random_state=args.seed)
    result = transform(combined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        f"완료: {args.output} ({len(result)}건, 사기 {len(fraud_sample)}건 / "
        f"정상 {len(normal_sample)}건, 전부 실제 LLM 생성 사고경위서)"
    )


if __name__ == "__main__":
    main()
