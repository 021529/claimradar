import pandas as pd
from ortools.linear_solver import pywraplp


def optimize_assignment(
    cases_df: pd.DataFrame,
    num_investigators: int,
    hours_per_investigator: float,
    risk_weight: float = 0.0,
) -> pd.DataFrame:
    """OR-Tools 정수계획으로 조사 우선순위 배정 최적화.

    목적함수: sum(배정된 사건의 (기대회수액 - 조사비용 + risk_weight * 위험도스코어)) 최대화
    제약: 조사관별 총 배정시간 <= 가용시간

    risk_weight(λ)는 특정 선행연구의 수식을 인용한 것이 아니라, 다중목적 최적화의
    표준 기법인 가중합 스칼라화(weighted sum scalarization)를 회수액 극대화와
    고위험 사건 커버리지라는 두 목적에 직접 적용해 우리가 설계한 것이다.
    λ=0이면 기존 목적함수(회수액 극대화)와 완전히 동일하다.

    cases_df required columns: case_id, combined_score, expected_hours, expected_recovery, investigation_cost
    """
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:
        raise RuntimeError("OR-Tools CBC solver 를 생성할 수 없습니다.")

    # 실측(scripts/solver_gap_check.py, 2026-08-30 ML 백본 기준) 결과 24건 앱
    # 데모 규모는 조사관 수·λ와 무관하게 200ms 안에 항상 OPTIMAL(격차 0%)이다.
    # 300건 이상·조사관 5명 이상에서는 1.5초로도 최적성이 증명 안 된 FEASIBLE로
    # 끝나는 경우가 있고, 그 시점 목적함수 값은 200ms 해 대비 최대 약 8% 더
    # 크다(즉 200ms는 이 규모에서 부족하지만, 1.5초 해도 "진짜 최적"이라고
    # 단정할 수는 없다 — scripts/solver_gap_check_results.md 참고). 그래도 응답
    # 속도를 위해 1.5초를 유지한다.
    solver.SetTimeLimit(1_500)

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
        risk_bonus = risk_weight * cases.loc[c, "combined_score"]
        for i in range(num_investigators):
            objective.SetCoefficient(x[c, i], float(net_value + risk_bonus))
    objective.SetMaximization()

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("최적화 문제를 풀 수 없습니다 (infeasible).")

    cases["assigned_investigator"] = None
    for c in range(n_cases):
        for i in range(num_investigators):
            if x[c, i].solution_value() > 0.5:
                cases.at[c, "assigned_investigator"] = i

    # DataFrame.attrs는 컬럼/값과 무관한 부가 메타데이터라 기존 호출부(테스트 포함)의
    # 동작에 영향을 주지 않는다 — app.py가 "최적해(OPTIMAL)"인지 "1.5초 제한시간 내
    # 실행가능해(FEASIBLE)"인지 화면에 구분 표시하기 위해 추가.
    cases.attrs["solver_status"] = "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE"

    return cases
