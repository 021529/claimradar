"""OR-Tools CBC의 1.5초 타임리밋이 실제로 얼마나 최적해에 가까운 해를 주는지
현재 ML 백본(범주형 피처 + class_weight='balanced', 2026-08-30 재작업분) 기준으로
재측정한다.

배경: README.md와 src/optimization/assignment.py의 주석에 있던 "200ms 해와
1.5초/10초 해의 목적함수 차이는 0.05%~1% 수준"이라는 수치는 2026-08-27(커밋
61e3876)에 옛 5피처 ML 백본으로 측정된 것으로, scripts/final_numbers_for_docs.md에
이미 기록된 다른 지표들과 마찬가지로 8/30 백본 재작업 이후 재검증되지 않은 채
남아 있었다. 이 스크립트는 그 수치를 현재 백본으로 재측정해 문서 인용 근거를
갱신하기 위한 것이다.

측정 방법: src/optimization/assignment.py의 optimize_assignment()은 시간제한이
1,500ms로 고정돼 있어 다른 시간제한과 비교할 수 없다. 그래서 이 스크립트는 그
함수와 동일한 모델(변수/제약/목적함수)을 시간제한만 주입 가능하게 로컬에 재구성해
200ms(빠른 해)와 1,500ms(현재 앱이 실제로 쓰는 시간제한) 각각의 목적함수 값을
비교한다 — optimize_assignment() 자체는 건드리지 않는다. assignment.py의 모델이
바뀌면 이 스크립트도 함께 갱신해야 한다.

데이터: data/sample/sample_claims.csv(24건, 앱 데모 규모), data/processed/
app_demo_sample.csv(300건, "실 서비스 규모" 검증에 쓰이는 데이터 — 없으면
scripts/prepare_app_dataset.py로 생성).

사용법:
    python scripts/solver_gap_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from ortools.linear_solver import pywraplp  # noqa: E402

from src.scoring.combine import combine_scores  # noqa: E402
from src.scoring.features import feature_cols_for  # noqa: E402
from src.scoring.ml_model import predict_fraud_score, train_fraud_model  # noqa: E402

LABEL_COL = "FraudFound_P"
HOURS_PER_INVESTIGATOR = 10
FAST_TIME_LIMIT_MS = 200
SLOW_TIME_LIMIT_MS = 1_500  # src/optimization/assignment.py가 실제로 쓰는 값

REPORT_PATH = Path("scripts/solver_gap_check_results.md")

SAMPLE_24_PATH = Path("data/sample/sample_claims.csv")
SAMPLE_300_PATH = Path("data/processed/app_demo_sample.csv")

_STATUS_NAMES = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
}

# (데이터 라벨, 조사관 수) x (λ) 조합 — 24건은 앱 정책 프리셋 3종
# (회수액 우선/균형/고위험 차단 우선), 300건은 이 규모의 대표 λ 스윕에 이미
# 쓰인 값(scripts/lambda_sweep_300case.py, scripts/final_numbers_for_docs.md의
# "계단 위치 약 125,000" 참고)과 동일하게 맞춰 다른 문서 수치와 어긋나지 않게 한다.
LAMBDAS_24 = [0, 50_000, 80_000]
LAMBDAS_300 = [0, 50_000, 125_000]
INVESTIGATORS_24 = [3, 5]
INVESTIGATORS_300 = [3, 5, 10]


def add_combined_score(df: pd.DataFrame) -> pd.DataFrame:
    """현재 앱과 동일한 방식(범주형 포함 피처 + class_weight='balanced')으로
    ml_score를 학습하고, 캐시된 llm_suspicion_adjustment와 결합한다."""
    df = df.copy()
    feature_cols = feature_cols_for(df)
    model, _ = train_fraud_model(df[feature_cols + [LABEL_COL]], class_weight="balanced")
    ml_score = predict_fraud_score(model, df[feature_cols])
    df["combined_score"] = combine_scores(ml_score, df["llm_suspicion_adjustment"])
    return df


def solve_with_time_limit(
    cases_df: pd.DataFrame, num_investigators: int, hours_per_investigator: float, risk_weight: float, time_limit_ms: int
):
    """src/optimization/assignment.py의 optimize_assignment()과 동일한 모델을
    시간제한만 바꿔 풀 수 있게 재구성한 버전. 반환값: (status_name, objective_value, wall_time_ms)."""
    solver = pywraplp.Solver.CreateSolver("CBC")
    solver.SetTimeLimit(time_limit_ms)

    cases = cases_df.reset_index(drop=True)
    n_cases = len(cases)

    x = {(c, i): solver.BoolVar(f"x_{c}_{i}") for c in range(n_cases) for i in range(num_investigators)}
    for c in range(n_cases):
        solver.Add(sum(x[c, i] for i in range(num_investigators)) <= 1)
    for i in range(num_investigators):
        solver.Add(
            sum(x[c, i] * cases.loc[c, "expected_hours"] for c in range(n_cases)) <= hours_per_investigator
        )

    objective = solver.Objective()
    for c in range(n_cases):
        net_value = cases.loc[c, "expected_recovery"] - cases.loc[c, "investigation_cost"]
        risk_bonus = risk_weight * cases.loc[c, "combined_score"]
        for i in range(num_investigators):
            objective.SetCoefficient(x[c, i], float(net_value + risk_bonus))
    objective.SetMaximization()

    status = solver.Solve()
    return _STATUS_NAMES.get(status, str(status)), objective.Value(), solver.wall_time()


def main() -> None:
    df24 = add_combined_score(pd.read_csv(SAMPLE_24_PATH))
    df300 = add_combined_score(pd.read_csv(SAMPLE_300_PATH))

    combos = [
        (f"{len(df24)}건", df24, inv, lam) for inv in INVESTIGATORS_24 for lam in LAMBDAS_24
    ] + [
        (f"{len(df300)}건", df300, inv, lam) for inv in INVESTIGATORS_300 for lam in LAMBDAS_300
    ]

    rows = []
    for label, df, num_inv, lam in combos:
        status_fast, obj_fast, wt_fast = solve_with_time_limit(
            df, num_inv, HOURS_PER_INVESTIGATOR, lam, FAST_TIME_LIMIT_MS
        )
        status_slow, obj_slow, wt_slow = solve_with_time_limit(
            df, num_inv, HOURS_PER_INVESTIGATOR, lam, SLOW_TIME_LIMIT_MS
        )
        gap_pct = (obj_slow - obj_fast) / obj_slow * 100 if obj_slow else 0.0
        rows.append(
            {
                "data": label,
                "investigators": num_inv,
                "lambda": lam,
                "status_fast": status_fast,
                "obj_fast": obj_fast,
                "wt_fast": wt_fast,
                "status_slow": status_slow,
                "obj_slow": obj_slow,
                "wt_slow": wt_slow,
                "gap_pct": gap_pct,
            }
        )
        print(
            f"[{label} inv={num_inv} λ={lam:,}] "
            f"{FAST_TIME_LIMIT_MS}ms={status_fast}({obj_fast:,.1f}) vs "
            f"{SLOW_TIME_LIMIT_MS}ms={status_slow}({obj_slow:,.1f}) gap={gap_pct:.3f}%"
        )

    lines = [
        "# Solver 시간제한별 목적함수 격차 재측정",
        "",
        f"- 데이터: `{SAMPLE_24_PATH.as_posix()}`({len(df24)}건), `{SAMPLE_300_PATH.as_posix()}`({len(df300)}건)",
        f"- ML 백본: 범주형 피처 포함 + class_weight='balanced' (2026-08-30 재작업분, 현재 app.py와 동일)",
        f"- 비교: {FAST_TIME_LIMIT_MS}ms 해 vs {SLOW_TIME_LIMIT_MS}ms 해(현재 assignment.py가 실제 쓰는 시간제한)의 목적함수 값 차이",
        "",
        "이전 수치(README.md \"0.05%\"/\"1% 미만\")는 2026-08-27(커밋 61e3876)에 옛 5피처 "
        "백본으로 측정된 것으로, 아래는 현재 백본 기준 재측정 결과다.",
        "",
        "| 데이터 | 조사관 | λ | 200ms 상태(목적함수) | 1,500ms 상태(목적함수) | 격차 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['data']} | {r['investigators']} | {r['lambda']:,} | "
            f"{r['status_fast']}({r['obj_fast']:,.0f}) | {r['status_slow']}({r['obj_slow']:,.0f}) | "
            f"{r['gap_pct']:.2f}% |"
        )

    max_gap_24 = max((r["gap_pct"] for r in rows if r["data"] == f"{len(df24)}건"), default=0.0)
    max_gap_300 = max((r["gap_pct"] for r in rows if r["data"] == f"{len(df300)}건"), default=0.0)
    all_optimal_24 = all(r["status_slow"] == "OPTIMAL" for r in rows if r["data"] == f"{len(df24)}건")
    any_feasible_300 = any(r["status_slow"] == "FEASIBLE" for r in rows if r["data"] == f"{len(df300)}건")

    lines.extend(
        [
            "",
            "## 요약",
            "",
            f"- {len(df24)}건(앱 데모 규모): 조사관 {INVESTIGATORS_24} x λ {LAMBDAS_24} 전 조합에서 "
            f"1,500ms 결과가 {'항상 OPTIMAL' if all_optimal_24 else '일부 FEASIBLE 포함'}, "
            f"최대 격차 {max_gap_24:.2f}%.",
            f"- {len(df300)}건: 조사관 {INVESTIGATORS_300} x λ {LAMBDAS_300} 조합 중 "
            f"{'일부는 1,500ms로도 FEASIBLE(미증명)' if any_feasible_300 else '모두 OPTIMAL'}, "
            f"최대 격차 {max_gap_300:.2f}%.",
            "- 문서에 인용 시 데이터 규모·조사관 수·λ 조건을 함께 명시할 것 "
            "(조건에 따라 격차가 0%~약 8%까지 크게 달라짐).",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n결과 저장 완료: {REPORT_PATH}")


if __name__ == "__main__":
    main()
