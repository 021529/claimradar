"""OR-Tools 최적화 vs risk_weighted_greedy("강한" 그리디) 정면 비교.

scripts/lambda_sweep_24case.py와 동일한 24건 데모 데이터(data/sample/sample_claims.csv),
동일한 λ(risk_weight) 값들에서 세 가지 배정 방식을 나란히 비교한다:

  - baseline (약한 그리디): 사기의심점수 내림차순, risk_weight 무시
  - risk_weighted_greedy (강한 그리디): OR-Tools와 동일한 목적함수
    (기대회수액 - 조사비용) + λ*위험도점수 를 예상조사소요시간으로 나눈
    "단위시간당 효율" 기준 내림차순 배정 (multiple-knapsack greedy)
  - OR-Tools (optimize_assignment): 동일 목적함수의 정수계획 최적해

핵심 질문: "그리디로도 어느 정도 잘 되는거 아니냐"는 반박에 답하기 위해,
가장 똑똑한 그리디(risk_weighted_greedy)조차 OR-Tools 대비 얼마나 손해를
보는지(optimality gap %)를 λ별로 측정한다. 특히 λ가 중간값이라 회수액과
고위험 커버리지 사이 실제 트레이드오프가 발생하는 구간에서 gap이 커지는지가
관건이다 — 다목적 목적함수 + multiple-knapsack 제약(조사관별 가용시간)
조합에서는, 단위시간당 효율이 가장 높은 사건을 먼저 집어먹는 그리디가
"나중에 더 잘 맞는 조합"을 놓치는 배낭문제형 손실이 구조적으로 발생하기
때문이다.

2026-08-30: sample_claims.csv에 실제 LLM 캐시가 영구 병합되어(scripts/generate_sample_claims.py
참고) 배포 앱과 정확히 같은 llm_suspicion_adjustment 값을 쓴다 (API 호출 0건).
같은 날 ML 백본 보강(범주형 One-Hot + class_weight='balanced')도 반영됨 — 옛
5피처+휴리스틱 모델 기준 "최대 gap 6.22%" 등은 scripts/ml_backbone_reexperiment_2026-08-30.md
참고.

사용법:
    python scripts/greedy_vs_ortools_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.optimization.assignment import optimize_assignment  # noqa: E402
from src.optimization.baseline import baseline_assignment  # noqa: E402
from src.optimization.greedy_strong import risk_weighted_greedy  # noqa: E402
from src.scoring.combine import combine_scores  # noqa: E402
from src.scoring.features import feature_cols_for  # noqa: E402
from src.scoring.ml_model import predict_fraud_score, train_fraud_model  # noqa: E402

DATA_PATH = Path("data/sample/sample_claims.csv")

LABEL_COL = "FraudFound_P"

NUM_INVESTIGATORS = 3
HOURS_PER_INVESTIGATOR = 10
HIGH_RISK_QUANTILE = 0.80

# scripts/lambda_sweep_24case.py와 동일한 스윕 그리드 사용 (실제 LLM 캐시 기준
# 계단 50%→66.7%→83.3%→100%가 나타나는 0~80,000 구간을 촘촘히 포함)
LAMBDAS = [
    0, 20000, 40000, 42000, 44000, 50000, 60000, 64000, 66000, 70000, 78000,
    80000, 90000, 100000, 150000, 200000, 250000, 300000, 400000, 480000, 500000,
]


def net_value(df: pd.DataFrame) -> float:
    assigned = df.dropna(subset=["assigned_investigator"])
    return float((assigned["expected_recovery"] - assigned["investigation_cost"]).sum())


def objective_value(df: pd.DataFrame, risk_weight: float) -> float:
    """OR-Tools 목적함수와 동일한 값: (기대회수액 - 조사비용) + λ*위험도점수 합."""
    assigned = df.dropna(subset=["assigned_investigator"])
    net = assigned["expected_recovery"] - assigned["investigation_cost"]
    return float((net + risk_weight * assigned["combined_score"]).sum())


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
        f"약한 그리디(baseline, λ 무시): 순회수액 {baseline_net:,.0f} / 고위험 커버리지 {baseline_cov:.1f}%\n"
    )

    rows = []
    header = (
        f"{'lambda':<10}{'OR-Tools 순회수액':<18}{'그리디 순회수액':<18}"
        f"{'OR-Tools 커버리지':<18}{'그리디 커버리지':<16}{'gap(목적함수 %)':<18}"
    )
    print(header)
    print("-" * len(header))

    for lam in LAMBDAS:
        opt = optimize_assignment(df, NUM_INVESTIGATORS, HOURS_PER_INVESTIGATOR, risk_weight=lam)
        greedy = risk_weighted_greedy(df, NUM_INVESTIGATORS, HOURS_PER_INVESTIGATOR, risk_weight=lam)

        opt_net = net_value(opt)
        greedy_net = net_value(greedy)
        opt_cov = coverage_pct(opt, high_risk_ids, n_high_risk)
        greedy_cov = coverage_pct(greedy, high_risk_ids, n_high_risk)

        opt_obj = objective_value(opt, lam)
        greedy_obj = objective_value(greedy, lam)
        gap_pct = (opt_obj - greedy_obj) / opt_obj * 100 if opt_obj else float("nan")

        rows.append(
            {
                "lambda": lam,
                "or_tools_net": opt_net,
                "greedy_net": greedy_net,
                "or_tools_coverage": opt_cov,
                "greedy_coverage": greedy_cov,
                "gap_pct": gap_pct,
            }
        )
        print(
            f"{lam:<10}{opt_net:<18,.0f}{greedy_net:<18,.0f}"
            f"{opt_cov:<18.1f}{greedy_cov:<16.1f}{gap_pct:<+18.2f}"
        )

    print("\n### 마크다운 표 (QA 문서용)\n")
    print("| λ | OR-Tools 순회수액 | 그리디 순회수액 | OR-Tools 고위험 커버리지 | 그리디 고위험 커버리지 | gap (목적함수 기준, %) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['lambda']:,} | {r['or_tools_net']:,.0f} | {r['greedy_net']:,.0f} | "
            f"{r['or_tools_coverage']:.1f}% | {r['greedy_coverage']:.1f}% | {r['gap_pct']:+.2f}% |"
        )


if __name__ == "__main__":
    main()
