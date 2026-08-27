import pandas as pd

from src.optimization.assignment import optimize_assignment
from src.optimization.baseline import baseline_assignment


def _sample_cases() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [1, 2, 3, 4],
            "combined_score": [0.9, 0.8, 0.5, 0.3],
            "expected_hours": [3, 5, 2, 4],
            "expected_recovery": [10000, 8000, 3000, 2000],
            "investigation_cost": [1000, 1500, 500, 800],
        }
    )


def test_baseline_assignment_respects_capacity():
    cases = _sample_cases()
    result = baseline_assignment(cases, num_investigators=1, hours_per_investigator=5)
    assigned = result.dropna(subset=["assigned_investigator"])
    assert assigned["expected_hours"].sum() <= 5


def test_optimize_assignment_respects_capacity():
    cases = _sample_cases()
    result = optimize_assignment(cases, num_investigators=1, hours_per_investigator=5)
    assigned = result.dropna(subset=["assigned_investigator"])
    assert assigned["expected_hours"].sum() <= 5
