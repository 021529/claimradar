"""위험도 가중치 λ(risk_weight)를 0~500,000까지 쓸어보며 OR-Tools 최적화의
baseline 대비 순회수액 개선율과 고위험(상위 20%) 커버리지가 λ에 따라 어떻게
변하는지 재현 가능하게 기록한다.

배경: 예전 세션에서 "λ=6,300 공짜개선 / λ=25,100 트레이드오프, 개선율
26.2%→36.1%→62.3%" 식의 수치가 발표 준비 중 언급됐으나, 이 숫자를 뒷받침하는
스크립트나 로그가 저장소 어디에도 없어 폐기했다. 옛 슬라이더 범위(0~25,100)로
다시 스윕해봐도 이 데이터에서는 트레이드오프가 거의 나타나지 않았고(λ=1,000
이후 완전히 평평), 0~500,000으로 넓혀서야 실제 계단(약 150,000, 약 400,000
부근)을 확인했다. app.py 슬라이더 최댓값도 이 결과를 반영해 500,000으로
확장했다.

데이터: data/processed/app_demo_sample.csv (300건). llm_suspicion_adjustment
는 이미 계산/캐싱돼 있어 API 호출 없음 (ml_score는 매 실행 시 재학습).

2026-08-30 ML 백본 보강(범주형 One-Hot + class_weight='balanced') 이후 재실행됨.
옛 5피처 모델 기준 수치는 scripts/ml_backbone_reexperiment_2026-08-30.md 참고.

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
from src.scoring.features import feature_cols_for  # noqa: E402
from src.scoring.ml_model import predict_fraud_score, train_fraud_model  # noqa: E402

DATA_PATH = Path("data/processed/app_demo_sample.csv")

LABEL_COL = "FraudFound_P"

NUM_INVESTIGATORS = 3
HOURS_PER_INVESTIGATOR = 10
HIGH_RISK_QUANTILE = 0.80

# 0~25,100(옛 슬라이더 범위)에서는 트레이드오프가 거의 안 보여 0~500,000까지
# 넓혀 재탐색했다. 300건 데이터는 계단이 25,100 이후에도 한참 뒤(약 150,000,
# 약 400,000)에나 나타나고, 500,000을 넘어서면 CBC 1.5초 시간제한 내에서
# 진짜 최적해를 못 찾는 것으로 보이는 비단조 구간이 나타나 500,000까지만 본다.
LAMBDAS = [0, 25100, 50000, 100000, 125000, 150000, 175000, 250000, 350000, 400000, 500000]


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
    feature_cols = feature_cols_for(df)
    model, _ = train_fraud_model(df[feature_cols + [LABEL_COL]], class_weight="balanced")
    ml_score = predict_fraud_score(model, df[feature_cols])
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
