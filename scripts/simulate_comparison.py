"""baseline vs OR-Tools 최적화 배정 비교 시뮬레이션.

샘플 청구 데이터에 ML 스코어 + LLM 이상징후 보정을 결합한 뒤,
baseline_assignment(단순 스코어 내림차순)과 optimize_assignment(OR-Tools)를
동일한 조사 리소스 제약 하에서 비교해 기대 회수액 개선율과 미조사 고위험 건
감소율을 계산한다.

사용법:
    python scripts/simulate_comparison.py [--input path.csv] [--investigators 3] [--hours 10] [--risk-threshold 0.5]

--input을 지정하지 않으면 번들 샘플 데이터를 사용한다. 입력 CSV에 llm_suspicion_adjustment
컬럼이 이미 있으면(사전 캐싱된 분석 결과) 이를 그대로 재사용하고 analyze_narrative를
다시 호출하지 않는다 — 재실행 시 추가 API 호출/비용이 발생하지 않는다.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.data.loader import load_sample_claims, normalize_columns  # noqa: E402
from src.optimization.assignment import optimize_assignment  # noqa: E402
from src.optimization.baseline import baseline_assignment  # noqa: E402
from src.scoring.combine import combine_scores  # noqa: E402
from src.scoring.llm_analysis import analyze_narrative  # noqa: E402
from src.scoring.ml_model import predict_fraud_score, train_fraud_model  # noqa: E402

NUMERIC_FEATURE_COLS = [
    "driver_age",
    "vehicle_price",
    "deductible",
    "driver_rating",
    "past_number_of_claims",
]
LABEL_COL = "FraudFound_P"


def score_claims(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    model, _ = train_fraud_model(df[NUMERIC_FEATURE_COLS + [LABEL_COL]])
    df["ml_score"] = predict_fraud_score(model, df[NUMERIC_FEATURE_COLS])

    if "llm_suspicion_adjustment" in df.columns:
        print("사전 계산된 LLM 분석 결과를 재사용합니다 (추가 API 호출 없음).", file=sys.stderr)
        llm_adjustment = df["llm_suspicion_adjustment"].tolist()
    else:
        llm_adjustment = [analyze_narrative(text).suspicion_adjustment for text in df["narrative_text"]]
    df["llm_adjustment"] = llm_adjustment
    df["combined_score"] = combine_scores(df["ml_score"], pd.Series(llm_adjustment, index=df.index))
    return df


def net_value(df: pd.DataFrame) -> float:
    assigned = df.dropna(subset=["assigned_investigator"])
    return float((assigned["expected_recovery"] - assigned["investigation_cost"]).sum())


def gross_recovery(df: pd.DataFrame) -> float:
    assigned = df.dropna(subset=["assigned_investigator"])
    return float(assigned["expected_recovery"].sum())


def uninvestigated_high_risk(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    high_risk = df[df["combined_score"] >= threshold]
    return high_risk[high_risk["assigned_investigator"].isna()]


def pct_change(before: float, after: float) -> float:
    return (after - before) / before * 100 if before else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--investigators", type=int, default=3)
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    args = parser.parse_args()

    claims = normalize_columns(pd.read_csv(args.input)) if args.input else normalize_columns(load_sample_claims())
    scored = score_claims(claims)

    baseline = baseline_assignment(scored, args.investigators, args.hours)
    optimized = optimize_assignment(scored, args.investigators, args.hours)

    baseline_net, optimized_net = net_value(baseline), net_value(optimized)
    baseline_gross, optimized_gross = gross_recovery(baseline), gross_recovery(optimized)

    baseline_missed = uninvestigated_high_risk(baseline, args.risk_threshold)
    optimized_missed = uninvestigated_high_risk(optimized, args.risk_threshold)
    n_baseline_missed, n_optimized_missed = len(baseline_missed), len(optimized_missed)

    total_capacity = args.investigators * args.hours
    total_demand = float(scored["expected_hours"].sum())
    n_high_risk = int((scored["combined_score"] >= args.risk_threshold).sum())

    print(
        f"조사관 {args.investigators}명 x 1인당 {args.hours:.0f}시간 "
        f"(총 가용 {total_capacity:.0f}시간 / 총 수요 {total_demand:.0f}시간)"
    )
    print(
        f"총 사건 {len(scored)}건, 실제 사기(FraudFound_P=1) {int(scored[LABEL_COL].sum())}건, "
        f"고위험(combined_score>={args.risk_threshold}) {n_high_risk}건\n"
    )

    print("[기대 회수액]")
    print(f"  baseline 순회수액(회수액-조사비용): {baseline_net:,.0f}")
    print(f"  최적화   순회수액(회수액-조사비용): {optimized_net:,.0f}")
    print(f"  순회수액 개선율: {pct_change(baseline_net, optimized_net):+.1f}%\n")
    print(f"  baseline 총회수액(조사비용 반영 전): {baseline_gross:,.0f}")
    print(f"  최적화   총회수액(조사비용 반영 전): {optimized_gross:,.0f}")
    print(f"  총회수액 개선율: {pct_change(baseline_gross, optimized_gross):+.1f}%\n")

    missed_reduction_pct = pct_change(n_baseline_missed, n_optimized_missed) * -1

    print("[미조사 고위험 건]")
    print(f"  baseline 미조사 고위험 건: {n_baseline_missed}건 {sorted(baseline_missed['case_id'].tolist())}")
    print(f"  최적화   미조사 고위험 건: {n_optimized_missed}건 {sorted(optimized_missed['case_id'].tolist())}")
    print(f"  감소율(baseline 대비): {missed_reduction_pct:+.1f}%")


if __name__ == "__main__":
    main()
