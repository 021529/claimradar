"""scripts/lambda_sweep_300case.py와 동일한 λ(risk_weight) 스윕을, 실제
"샘플로 체험하기" 버튼이 로드하는 24건 data/sample/sample_claims.csv 기준으로
돌려 300건 결과와 나란히 비교할 수 있게 한다.

sample_claims.csv에는 llm_suspicion_adjustment 캐시가 없어(API 비용 없이
순수 배정 로직만 보기 위해) 휴리스틱 폴백(_heuristic_analysis, API 호출 0건)
으로 1회 계산해 고정 재사용한다. expected_hours/expected_recovery/
investigation_cost는 이미 공식에 맞게 재생성된 값을 그대로 쓴다
(scripts/generate_sample_claims.py 참고).

이 24건 데이터에서는 λ=30,000~500,000 구간에서 회수액↓/고위험 커버리지↑가
40%→60%→80%→100%로 뚜렷한 4단계 계단을 그리며 나타난다(옛 슬라이더 범위
0~25,100에서는 전혀 안 보임). app.py 슬라이더 최댓값을 500,000으로 확장한
근거가 이 결과다.

사용법:
    python scripts/lambda_sweep_24case.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.optimization.assignment import optimize_assignment  # noqa: E402
from src.optimization.baseline import baseline_assignment  # noqa: E402
from src.scoring.combine import combine_scores  # noqa: E402
from src.scoring.llm_analysis import _heuristic_analysis  # noqa: E402
from src.scoring.ml_model import predict_fraud_score, train_fraud_model  # noqa: E402

DATA_PATH = Path("data/sample/sample_claims.csv")

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
HIGH_RISK_QUANTILE = 0.80

# 24건 데이터는 0~25,100(옛 슬라이더 범위)에서는 전혀 안 움직이다가, 30,000
# 부근부터 500,000 부근까지 4단계로 뚜렷한 회수액↓/고위험 커버리지↑ 트레이드오프
# 계단을 그린다(40%→60%→80%→100%). 480,000 이상에서는 baseline과 완전히
# 동일한 배정(커버리지 100%, 개선율 0%)으로 수렴한다.
LAMBDAS = [0, 25100, 30000, 50000, 80000, 90000, 100000, 200000, 300000, 450000, 480000, 500000]


def net_value(df: pd.DataFrame) -> float:
    assigned = df.dropna(subset=["assigned_investigator"])
    return float((assigned["expected_recovery"] - assigned["investigation_cost"]).sum())


def coverage_pct(df: pd.DataFrame, high_risk_ids: set, n_high_risk: int) -> float:
    if n_high_risk == 0:
        return 0.0
    assigned = df.dropna(subset=["assigned_investigator"])
    covered = assigned["case_id"].isin(high_risk_ids).sum()
    return 100 * covered / n_high_risk


def build_combined_score(df: pd.DataFrame) -> pd.Series:
    model, _ = train_fraud_model(df[NUMERIC_FEATURE_COLS + [LABEL_COL]])
    ml_score = predict_fraud_score(model, df[NUMERIC_FEATURE_COLS])
    llm_adjustment = df["narrative_text"].apply(lambda t: _heuristic_analysis(t).suspicion_adjustment)
    return combine_scores(ml_score, llm_adjustment)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print("주의: llm_suspicion_adjustment 캐시가 없어 휴리스틱 폴백으로 1회 계산 (API 호출 0건).\n")
    df["combined_score"] = build_combined_score(df)

    high_risk_threshold = df["combined_score"].quantile(HIGH_RISK_QUANTILE)
    high_risk_ids = set(df.loc[df["combined_score"] >= high_risk_threshold, "case_id"])
    n_high_risk = len(high_risk_ids)

    baseline = baseline_assignment(df, NUM_INVESTIGATORS, HOURS_PER_INVESTIGATOR)
    baseline_net = net_value(baseline)
    baseline_cov = coverage_pct(baseline, high_risk_ids, n_high_risk)

    print(
        f"총 사건 {len(df)}건, 고위험(상위 {100 * (1 - HIGH_RISK_QUANTILE):.0f}%) {n_high_risk}건, "
        f"조사관 {NUM_INVESTIGATORS}명 x {HOURS_PER_INVESTIGATOR}시간\n"
        f"baseline: 순회수액 {baseline_net:,.0f} / 고위험 커버리지 {baseline_cov:.1f}%\n"
    )

    print(f"{'lambda':<10}{'최적화 순회수액':<16}{'개선율':<10}{'커버리지':<10}{'커버리지Δ':<10}")
    for lam in LAMBDAS:
        optimized = optimize_assignment(df, NUM_INVESTIGATORS, HOURS_PER_INVESTIGATOR, risk_weight=lam)
        opt_net = net_value(optimized)
        opt_cov = coverage_pct(optimized, high_risk_ids, n_high_risk)
        improvement_pct = (opt_net - baseline_net) / baseline_net * 100 if baseline_net else float("nan")
        print(
            f"{lam:<10}{opt_net:<16,.0f}{improvement_pct:<+10.1f}{opt_cov:<10.1f}{opt_cov - baseline_cov:<+10.1f}"
        )


if __name__ == "__main__":
    main()
