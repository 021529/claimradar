import pandas as pd

from src.optimization.assignment import optimize_assignment
from src.optimization.baseline import baseline_assignment
from src.optimization.greedy_strong import risk_weighted_greedy


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


def _objective_value(result: pd.DataFrame, risk_weight: float) -> float:
    assigned = result.dropna(subset=["assigned_investigator"])
    net = assigned["expected_recovery"] - assigned["investigation_cost"]
    return float((net + risk_weight * assigned["combined_score"]).sum())


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


def test_risk_weighted_greedy_respects_capacity():
    cases = _sample_cases()
    result = risk_weighted_greedy(cases, num_investigators=1, hours_per_investigator=5)
    assigned = result.dropna(subset=["assigned_investigator"])
    assert assigned["expected_hours"].sum() <= 5


def test_risk_weighted_greedy_never_beats_or_tools_objective():
    cases = _sample_cases()
    for risk_weight in (0, 500, 2000, 10000):
        greedy = risk_weighted_greedy(cases, num_investigators=1, hours_per_investigator=5, risk_weight=risk_weight)
        optimal = optimize_assignment(cases, num_investigators=1, hours_per_investigator=5, risk_weight=risk_weight)
        assert _objective_value(greedy, risk_weight) <= _objective_value(optimal, risk_weight) + 1e-6


def test_risk_weighted_greedy_skips_non_positive_value_cases():
    cases = _sample_cases()
    cases.loc[cases["case_id"] == 4, "expected_recovery"] = 0  # net value < 0
    result = risk_weighted_greedy(cases, num_investigators=1, hours_per_investigator=20, risk_weight=0)
    assert pd.isna(result.loc[result["case_id"] == 4, "assigned_investigator"].iloc[0])
