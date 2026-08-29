import pandas as pd


def risk_weighted_greedy(
    cases_df: pd.DataFrame,
    num_investigators: int,
    hours_per_investigator: float,
    risk_weight: float = 0.0,
) -> pd.DataFrame:
    """비교군: OR-Tools(optimize_assignment)와 동일한 목적함수 값을 쓰는 "강한" 그리디.

    baseline_assignment(단순 스코어 내림차순)와 달리, 각 사건의 가치를
    OR-Tools 목적함수와 동일하게 (기대회수액 - 조사비용) + risk_weight * 위험도스코어
    로 계산하고, 이를 예상조사소요시간으로 나눈 "단위시간당 효율"을 내림차순
    정렬해 가장 여유 있는 조사관에게 순차 배정한다 (multiple-knapsack greedy).

    가치가 0 이하인 사건은 배정하지 않는다. OR-Tools도 사건별 제약이 용량
    제약(조사관 가용시간) 하나뿐이므로, 그런 사건을 배정하는 것은 항상
    목적함수를 손해 보게 만들어 최적해가 스스로 배제하기 때문이다.

    cases_df required columns: case_id, combined_score, expected_hours, expected_recovery, investigation_cost
    """
    cases = cases_df.copy()
    value = (cases["expected_recovery"] - cases["investigation_cost"]) + risk_weight * cases["combined_score"]
    efficiency = value / cases["expected_hours"]

    remaining = {i: hours_per_investigator for i in range(num_investigators)}
    order = efficiency[value > 0].sort_values(ascending=False).index

    cases["assigned_investigator"] = None
    for idx in order:
        row = cases.loc[idx]
        candidates = [i for i, h in remaining.items() if h >= row["expected_hours"]]
        if not candidates:
            continue
        target = max(candidates, key=lambda i: remaining[i])
        remaining[target] -= row["expected_hours"]
        cases.at[idx, "assigned_investigator"] = target

    return cases
