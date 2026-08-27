"""위험도 가중치 λ(risk_weight)를 0~25,100(앱 슬라이더 범위와 동일)까지 쓸어보며
OR-Tools 최적화의 baseline 대비 순회수액 개선율과 고위험(상위 20%) 커버리지가
λ에 따라 어떻게 변하는지 재현 가능하게 기록한다.

배경: 예전 세션에서 "λ=6,300 공짜개선 / λ=25,100 트레이드오프, 개선율
26.2%→36.1%→62.3%" 식의 수치가 발표 준비 중 언급됐으나, 이 숫자를 뒷받침하는
스크립트나 로그가 저장소 어디에도 없고, 코드 커밋 이력(1f51a8b)에는 오히려
"λ 임계값에 따른 결과를 일반 법칙처럼 하드코딩했다가 다른 조건(24건 샘플)에서
틀린 것으로 확인돼 제거했다"는 기록만 남아있다. 이 스크립트는 그 옛 숫자를
폐기하고, 현재 코드 기준으로 실제 λ 스윕 결과를 재현 가능하게 남기기 위한 것.

데이터: data/processed/app_demo_sample.csv (300건). ml_score, llm_suspicion_adjustment
는 이미 계산/캐싱돼 있어 API 호출 없음.

사용법:
    python scripts/lambda_sweep_300case.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.optimization.assignment import optimize_assignment  # noqa: E402
from src.optimization.baseline import baseline_assignment  # noqa: E402
from src.scoring.combine import combine_scores  # noqa: E402
from src.scoring.ml_model import predict_fraud_score, train_fraud_model  # noqa: E402

DATA_PATH = Path("data/processed/app_demo_sample.csv")

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

# app.py 슬라이더 범위(0~25,100, step 100)를 대표하는 스윕 포인트.
# 1f51a8b 이전 코드의 임계값(12,550 = 슬라이더 최댓값의 절반)과, 예전에 언급된
# λ=6,300 / λ=25,100도 포함시켜 직접 비교할 수 있게 함.
LAMBDAS = [0, 1000, 3000, 6300, 9000, 12550, 15000, 18000, 21000, 25100]


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
    return combine_scores(ml_score, df["llm_suspicion_adjustment"])


def main() -> None:
    df = pd.read_csv(DATA_PATH)
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
