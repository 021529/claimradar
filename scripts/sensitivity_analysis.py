"""목적함수 재무 파라미터(expected_recovery 배율, expected_hours 복잡도 가중치,
investigation_cost 단가) 민감도 분석.

"expected_recovery/expected_hours가 가상의 가정치인데, 이걸로 나온 개선율이
의미 있냐"는 지적에 대응하기 위해 세 파라미터를 각각 ±30% 흔들고 baseline
(그리디, src/optimization/baseline.py) vs OR-Tools 최적화(src/optimization/
assignment.py)의 순회수액 개선율이 파라미터 값에 관계없이 항상 baseline보다
우위인지를 확인한다.

데이터: data/processed/app_demo_sample.csv (300건, 사기 150/정상 150).
- ml_score, llm_suspicion_adjustment는 이미 계산/캐싱돼 있어 API 호출 없음.
  (llm_suspicion_adjustment는 재무 파라미터와 무관하므로 캐시값을 그대로 재사용)
- expected_hours 재계산에 필요한 원본 신호(PastNumberOfClaims, PoliceReportFiled,
  WitnessPresent, NumberOfSuppliments)는 data/processed/claims_with_narratives.csv
  에서 case_id(=PolicyNumber) 기준으로 조인해 가져온다.

공식 (scripts/prepare_app_dataset.py 기준, 파라미터화):
    complexity = (과거청구 있음) + (경찰 미신고) + (목격자 없음) + (부속서류 3건 이상)
    expected_hours = clip(2 + hours_weight * complexity, upper=8)
    expected_recovery = vehicle_price * recovery_mult
    investigation_cost = expected_hours * cost_per_hour

기준값: recovery_mult=0.4, hours_weight=1.0, cost_per_hour=350 (원 공식과 동일)

사용법:
    python scripts/sensitivity_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.optimization.assignment import optimize_assignment  # noqa: E402
from src.optimization.baseline import baseline_assignment  # noqa: E402
from src.scoring.combine import combine_scores  # noqa: E402
from src.scoring.ml_model import predict_fraud_score, train_fraud_model  # noqa: E402

DEMO_PATH = Path("data/processed/app_demo_sample.csv")
RAW_PATH = Path("data/processed/claims_with_narratives.csv")

NUMERIC_FEATURE_COLS = [
    "driver_age",
    "vehicle_price",
    "deductible",
    "driver_rating",
    "past_number_of_claims",
]
LABEL_COL = "FraudFound_P"

NUM_INVESTIGATORS = 3
HOURS_PER_INVESTIGATOR = 10

# 기준값과 ±30% 실험값
BASE_RECOVERY_MULT = 0.4
BASE_HOURS_WEIGHT = 1.0
BASE_COST_PER_HOUR = 350.0

RECOVERY_MULTS = [0.28, 0.34, 0.40, 0.46, 0.52]  # 0.4 기준 ±30%
HOURS_WEIGHTS = [0.7, 0.85, 1.0, 1.15, 1.3]  # 1.0 기준 ±30%
COST_PER_HOURS = [245.0, 297.5, 350.0, 402.5, 455.0]  # 350 기준 ±30%


def _load_base_data() -> pd.DataFrame:
    demo = pd.read_csv(DEMO_PATH)
    raw = pd.read_csv(RAW_PATH)[
        ["PolicyNumber", "PastNumberOfClaims", "PoliceReportFiled", "WitnessPresent", "NumberOfSuppliments"]
    ]
    merged = demo.merge(raw, left_on="case_id", right_on="PolicyNumber", how="left")
    assert merged["PastNumberOfClaims"].isna().sum() == 0, "case_id-PolicyNumber 조인 실패 건 존재"
    return merged


def _complexity(df: pd.DataFrame) -> pd.Series:
    return (
        (df["PastNumberOfClaims"] != "none").astype(int)
        + (df["PoliceReportFiled"] == "No").astype(int)
        + (df["WitnessPresent"] == "No").astype(int)
        + df["NumberOfSuppliments"].isin(["3 to 5", "more than 5"]).astype(int)
    )


def _apply_financial_params(
    df: pd.DataFrame, recovery_mult: float, hours_weight: float, cost_per_hour: float
) -> pd.DataFrame:
    df = df.copy()
    complexity = _complexity(df)
    df["expected_hours"] = (2 + hours_weight * complexity).clip(upper=8)
    df["expected_recovery"] = (df["vehicle_price"] * recovery_mult).round(0)
    df["investigation_cost"] = (df["expected_hours"] * cost_per_hour).round(0)
    return df


def net_value(df: pd.DataFrame) -> float:
    assigned = df.dropna(subset=["assigned_investigator"])
    return float((assigned["expected_recovery"] - assigned["investigation_cost"]).sum())


def high_risk_coverage_pct(df: pd.DataFrame, high_risk_case_ids: set, n_high_risk_total: int) -> float:
    if n_high_risk_total == 0:
        return 0.0
    assigned = df.dropna(subset=["assigned_investigator"])
    covered = assigned["case_id"].isin(high_risk_case_ids).sum()
    return 100 * covered / n_high_risk_total


def run_scenario(
    base_df: pd.DataFrame,
    combined_score: pd.Series,
    label: str,
    recovery_mult: float,
    hours_weight: float,
    cost_per_hour: float,
) -> dict:
    df = _apply_financial_params(base_df, recovery_mult, hours_weight, cost_per_hour)
    df["combined_score"] = combined_score

    high_risk_threshold = df["combined_score"].quantile(0.80)
    high_risk_ids = set(df.loc[df["combined_score"] >= high_risk_threshold, "case_id"])
    n_high_risk = len(high_risk_ids)

    baseline = baseline_assignment(df, NUM_INVESTIGATORS, HOURS_PER_INVESTIGATOR)
    optimized = optimize_assignment(df, NUM_INVESTIGATORS, HOURS_PER_INVESTIGATOR)

    baseline_net = net_value(baseline)
    optimized_net = net_value(optimized)
    improvement_pct = (optimized_net - baseline_net) / baseline_net * 100 if baseline_net else float("nan")

    baseline_cov = high_risk_coverage_pct(baseline, high_risk_ids, n_high_risk)
    optimized_cov = high_risk_coverage_pct(optimized, high_risk_ids, n_high_risk)

    return {
        "label": label,
        "recovery_mult": recovery_mult,
        "hours_weight": hours_weight,
        "cost_per_hour": cost_per_hour,
        "baseline_net": baseline_net,
        "optimized_net": optimized_net,
        "improvement_pct": improvement_pct,
        "baseline_cov": baseline_cov,
        "optimized_cov": optimized_cov,
        "cov_delta": optimized_cov - baseline_cov,
    }


def main() -> None:
    merged = _load_base_data()

    with_hours = _apply_financial_params(merged, BASE_RECOVERY_MULT, BASE_HOURS_WEIGHT, BASE_COST_PER_HOUR)
    model, _ = train_fraud_model(with_hours[NUMERIC_FEATURE_COLS + [LABEL_COL]])
    ml_score = predict_fraud_score(model, with_hours[NUMERIC_FEATURE_COLS])
    llm_adjustment = with_hours["llm_suspicion_adjustment"]
    combined_score = combine_scores(ml_score, llm_adjustment)

    results = []
    results.append(
        run_scenario(merged, combined_score, "기준값(0.4/1.0/350)", BASE_RECOVERY_MULT, BASE_HOURS_WEIGHT, BASE_COST_PER_HOUR)
    )

    for v in RECOVERY_MULTS:
        if v == BASE_RECOVERY_MULT:
            continue
        results.append(run_scenario(merged, combined_score, f"recovery_mult={v}", v, BASE_HOURS_WEIGHT, BASE_COST_PER_HOUR))

    for v in HOURS_WEIGHTS:
        if v == BASE_HOURS_WEIGHT:
            continue
        results.append(run_scenario(merged, combined_score, f"hours_weight={v}", BASE_RECOVERY_MULT, v, BASE_COST_PER_HOUR))

    for v in COST_PER_HOURS:
        if v == BASE_COST_PER_HOUR:
            continue
        results.append(run_scenario(merged, combined_score, f"cost_per_hour={v}", BASE_RECOVERY_MULT, BASE_HOURS_WEIGHT, v))

    # 세 파라미터를 동시에 흔든 최악/최선 조합 (스트레스 테스트)
    results.append(
        run_scenario(merged, combined_score, "최악조합(0.28/1.3/455)", 0.28, 1.3, 455.0)
    )
    results.append(
        run_scenario(merged, combined_score, "최선조합(0.52/0.7/245)", 0.52, 0.7, 245.0)
    )

    print(
        f"{'시나리오':26}{'recovery':10}{'hours_w':10}{'cost/h':10}"
        f"{'baseline순':14}{'최적화순':14}{'개선율':10}{'커버리지Δ':10}"
    )
    for r in results:
        print(
            f"{r['label']:<26}{r['recovery_mult']:<10.2f}{r['hours_weight']:<10.2f}{r['cost_per_hour']:<10.1f}"
            f"{r['baseline_net']:<14,.0f}{r['optimized_net']:<14,.0f}{r['improvement_pct']:<+10.1f}{r['cov_delta']:<+10.1f}"
        )

    all_positive = all(r["improvement_pct"] > 0 for r in results)
    min_r = min(results, key=lambda r: r["improvement_pct"])
    max_r = max(results, key=lambda r: r["improvement_pct"])
    print(
        f"\n모든 시나리오에서 개선율 > 0: {all_positive}\n"
        f"개선율 범위: 최소 {min_r['improvement_pct']:+.1f}% ({min_r['label']}) "
        f"~ 최대 {max_r['improvement_pct']:+.1f}% ({max_r['label']})"
    )


if __name__ == "__main__":
    main()
