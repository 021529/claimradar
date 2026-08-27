import pandas as pd


def baseline_assignment(cases_df: pd.DataFrame, num_investigators: int, hours_per_investigator: float) -> pd.DataFrame:
    """비교군: 스코어 내림차순으로 가용시간이 소진될 때까지 순차 배정.

    cases_df required columns: case_id, combined_score, expected_hours, expected_recovery
    """
    remaining = {i: hours_per_investigator for i in range(num_investigators)}
    sorted_cases = cases_df.sort_values("combined_score", ascending=False).copy()
    sorted_cases["assigned_investigator"] = None

    inv_idx = 0
    for idx, row in sorted_cases.iterrows():
        # 가장 여유 있는 조사관에게 배정 (라운드로빈 fallback)
        candidates = [i for i, h in remaining.items() if h >= row["expected_hours"]]
        if not candidates:
            continue
        target = max(candidates, key=lambda i: remaining[i])
        remaining[target] -= row["expected_hours"]
        sorted_cases.at[idx, "assigned_investigator"] = target

    return sorted_cases
