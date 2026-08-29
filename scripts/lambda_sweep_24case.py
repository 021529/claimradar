"""scripts/lambda_sweep_300case.py와 동일한 λ(risk_weight) 스윕을, 실제
"샘플로 체험하기" 버튼이 로드하는 24건 data/sample/sample_claims.csv 기준으로
돌려 300건 결과와 나란히 비교할 수 있게 한다.

2026-08-30: sample_claims.csv에 실제 LLM 캐시(sample_claims_24case_llm_cache.csv)가
영구 병합되어(scripts/generate_sample_claims.py 참고), 배포 앱("샘플로 체험하기")과
이 스크립트가 정확히 같은 llm_suspicion_adjustment 값을 쓴다 — 문서 수치와 배포
화면 수치가 일치하도록 하기 위함. API 호출 0건(캐시 재사용). expected_hours/
expected_recovery/investigation_cost도 이미 공식에 맞게 재생성된 값 그대로 쓴다.

2026-08-30 ML 백본 보강(범주형 One-Hot + class_weight='balanced', src/scoring/features.py)
이후 재실행됨 — 이 스윕에서 나온 λ 구간·계단 형태가 바뀌었을 수 있다. 옛 5피처
모델 기준 수치는 scripts/ml_backbone_reexperiment_2026-08-30.md에 나란히 남겨뒀다.

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
from src.scoring.features import feature_cols_for  # noqa: E402
from src.scoring.ml_model import predict_fraud_score, train_fraud_model  # noqa: E402

DATA_PATH = Path("data/sample/sample_claims.csv")

LABEL_COL = "FraudFound_P"

NUM_INVESTIGATORS = 3
HOURS_PER_INVESTIGATOR = 10
HIGH_RISK_QUANTILE = 0.80

# 실제 LLM 캐시 기준 계단 50%→66.7%→83.3%→100%가 0~80,000 구간에 몰려 있어
# 그 구간을 촘촘히, 나머지는 성긴 그리드로 확인한다.
LAMBDAS = [
    0, 20000, 40000, 42000, 44000, 50000, 60000, 64000, 66000, 70000, 78000,
    80000, 90000, 100000, 150000, 200000, 250000, 300000, 400000, 480000, 500000,
]


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
    print("sample_claims.csv에 병합된 실제 LLM 캐시 재사용 (API 호출 0건, 배포 앱과 동일 값).\n")
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
