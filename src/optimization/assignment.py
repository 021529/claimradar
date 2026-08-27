import pandas as pd
from ortools.linear_solver import pywraplp


def optimize_assignment(cases_df: pd.DataFrame, num_investigators: int, hours_per_investigator: float) -> pd.DataFrame:
    """OR-Tools 정수계획으로 조사 우선순위 배정 최적화.

    목적함수: sum(배정된 사건의 기대회수액 - 조사비용) 최대화
    제약: 조사관별 총 배정시간 <= 가용시간

    cases_df required columns: case_id, expected_hours, expected_recovery, investigation_cost
    """
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("OR-Tools CBC solver 를 생성할 수 없습니다.")

    # 조사관이 서로 동일(가용시간 동일)해서 배정 순열마다 목적함수 값이 같아지는
    # 대칭성 때문에 조사관 수가 늘면 CBC 탐색이 기하급수적으로 느려질 수 있다.
    # 최적성 증명에 시간을 쓰기보다 합리적 시간 내 feasible 해를 얻도록 시간 제한을 둔다.
    solver.SetTimeLimit(10_000)

    cases = cases_df.reset_index(drop=True)
    n_cases = len(cases)

    # x[c, i] = 1 이면 사건 c 를 조사관 i 에게 배정
    x = {
        (c, i): solver.BoolVar(f"x_{c}_{i}")
        for c in range(n_cases)
        for i in range(num_investigators)
    }

    # 사건은 최대 1명에게만 배정
    for c in range(n_cases):
        solver.Add(sum(x[c, i] for i in range(num_investigators)) <= 1)

    # 조사관별 가용시간 제약
    for i in range(num_investigators):
        solver.Add(
            sum(x[c, i] * cases.loc[c, "expected_hours"] for c in range(n_cases))
            <= hours_per_investigator
        )

    objective = solver.Objective()
    for c in range(n_cases):
        net_value = cases.loc[c, "expected_recovery"] - cases.loc[c, "investigation_cost"]
        for i in range(num_investigators):
            objective.SetCoefficient(x[c, i], float(net_value))
    objective.SetMaximization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("최적화 문제를 풀 수 없습니다 (infeasible).")

    cases["assigned_investigator"] = None
    for c in range(n_cases):
        for i in range(num_investigators):
            if x[c, i].solution_value() > 0.5:
                cases.at[c, "assigned_investigator"] = i

    return cases
