"""실제 Kaggle+합성 사고경위서 데이터셋을 app.py/시뮬레이션이 기대하는 스키마로 변환한다.

Kaggle "Vehicle Insurance Claim Fraud Detection"에는 case_id/expected_hours/
expected_recovery/investigation_cost 같은 업무 수치가 없고, VehiclePrice/
PastNumberOfClaims 등 일부 피처가 구간 문자열(범주형)이다. 이 스크립트는:

1. 범주형 구간을 중간값 기준 수치로 변환 (vehicle_price, past_number_of_claims)
2. 조사 업무 수치(expected_hours/expected_recovery/investigation_cost)를
   관측 가능한 피처 기반의 투명한 공식으로 산출 (Kaggle 원본에 없는 값이므로
   시연용 가정 — FraudFound_P 라벨은 사용하지 않아 정답 라벨 누출 없음)
3. 사기 전체 + 정상 표본을 층화 추출 (기본: 실제 LLM 생성 사고경위서를 가진
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
