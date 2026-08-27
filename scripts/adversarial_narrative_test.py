"""analyze_narrative가 키워드 문자열 매칭이 아니라 의미론적으로 이상징후를
잡아내는지 검증하는 적대적 테스트.

합성 데이터 생성 프롬프트(src/scoring/fraud_patterns.py)와 폴백 휴리스틱
(src/scoring/llm_analysis.py의 _HEURISTIC_KEYWORDS)이 실제로 매칭하는 정확한
문자열("나이롱환자", "한방병원 장기입원", "지연신고", "블랙박스 미장착" 등)을
절대 쓰지 않고, 완전히 다른 표현으로 같은 의미의 상황을 서술한 10건을 만들어
LLM 경로(analyze_narrative, 실제 API 호출)와 휴리스틱 폴백(_heuristic_analysis,
API 미호출)에 동일하게 돌려 비교한다.

- API 호출은 최대 10건(케이스당 1회)으로 제한됨.
- 각 케이스에는 필자가 판단한 ground_truth("fraud"/"normal")와 난이도
  ("clear"/"ambiguous" — ambiguous는 표면적으로는 평범해 보이지만 자세히 보면
  의심스러운 케이스)를 라벨로 붙여 채점 기준으로 삼는다. 이 라벨은 데이터셋에
  내장된 것이 아니라 이 스크립트 작성 시점에 사람이 직접 부여한 것이므로,
  "정답"이 아니라 "적대적 테스트 설계자의 기대값"으로 해석해야 한다.

사용법:
    python scripts/adversarial_narrative_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring.fraud_patterns import KOREAN_FRAUD_PATTERNS  # noqa: E402
from src.scoring.llm_analysis import _HEURISTIC_KEYWORDS, _heuristic_analysis, analyze_narrative  # noqa: E402

CASES = [
    # --- 명확한 사기 의심 (4건), 각기 다른 국내 패턴을 완전히 다른 어휘로 재서술 ---
    {
        "id": "F1",
        "ground_truth": "fraud",
        "difficulty": "clear",
        "note": "나이롱환자 패턴을 다른 어휘로",
        "text": (
            "가벼운 범퍼 접촉 수준의 사고였음에도 청구인은 두 달 넘게 재활치료를 "
            "지속하고 있으며, 초진 당시 진단서상 소견은 단순 타박상 정도에 그친다. "
            "치료 기간과 초기 진단 내용 사이의 격차가 커 보인다."
        ),
    },
    {
        "id": "F2",
        "ground_truth": "fraud",
        "difficulty": "clear",
        "note": "한방병원 장기입원 패턴을 다른 어휘로",
        "text": (
            "접촉사고 이후 청구인은 인근 동양의학 전문 요양기관에 입실해 삼 주 넘게 "
            "숙식하며 치료비를 청구하고 있다. 외래 통원으로도 충분해 보이는 상해 "
            "정도에 비해 입실 기간이 이례적으로 길다."
        ),
    },
    {
        "id": "F3",
        "ground_truth": "fraud",
        "difficulty": "clear",
        "note": "지연신고 패턴을 다른 어휘로",
        "text": (
            "청구인은 사고가 있었던 날짜로부터 한 달 반이 지나서야 보험사 창구를 "
            "찾아 접수를 진행했다. 그동안 별도의 병원 진료나 수리 이력도 확인되지 "
            "않아 접수를 미룬 이유가 불분명하다."
        ),
    },
    {
        "id": "F4",
        "ground_truth": "fraud",
        "difficulty": "clear",
        "note": "블랙박스 미장착 패턴을 다른 어휘로",
        "text": (
            "사고 당시 차량에는 영상 기록 장치가 설치돼 있지 않았고, 주변에 이를 "
            "지켜본 사람도 없어 청구인의 진술 외에는 사고 경위를 뒷받침할 객관적인 "
            "자료가 전혀 없는 상태다."
        ),
    },
    # --- 평범한 정상 사고 (3건) ---
    {
        "id": "N1",
        "ground_truth": "normal",
        "difficulty": "clear",
        "text": (
            "출근길 교차로에서 신호 대기 중 뒤차에 가볍게 추돌당해 범퍼에 흠집이 "
            "생겼다. 사고 당일 오후 바로 보험사에 접수했고 상대 차량 운전자도 "
            "과실을 인정했다."
        ),
    },
    {
        "id": "N2",
        "ground_truth": "normal",
        "difficulty": "clear",
        "text": (
            "지방 국도를 주행하던 중 노면에 떨어져 있던 낙하물을 밟아 타이어가 "
            "파손됐다. 사고 직후 인근 정비소로 견인해 수리를 맡겼고 다친 사람은 "
            "없었다."
        ),
    },
    {
        "id": "N3",
        "ground_truth": "normal",
        "difficulty": "clear",
        "text": (
            "주차장에 세워둔 차량이 알 수 없는 사이 옆면이 긁힌 채 발견되어 접수한 "
            "건이다. 인적 피해는 없고 도장 수리만 필요한 경미한 손상이다."
        ),
    },
    # --- 애매한 케이스 (3건): 표면적으로는 평범해 보이지만 세부에 의심스러운 신호가 섞여있음 ---
    {
        "id": "A1",
        "ground_truth": "fraud",
        "difficulty": "ambiguous",
        "note": "표면상 평범한 도난 서술 + 계약 시점 근접이라는 미묘한 신호",
        "text": (
            "주차해 둔 차량이 사라져 도난 신고를 접수한 건이다. 인근 CCTV 사각지대라 "
            "촬영된 영상은 없었다고 한다. 다만 해당 보험 계약이 체결된 지 채 일주일이 "
            "지나지 않은 시점에 발생한 사고다."
        ),
    },
    {
        "id": "A2",
        "ground_truth": "fraud",
        "difficulty": "ambiguous",
        "note": "정상적으로 들리는 화재 서술 + 계약자 진술 간 사소한 시점 어긋남",
        "text": (
            "정차 중이던 차량 엔진룸에서 원인 모를 화재가 발생해 전소된 사고다. "
            "다만 청구인이 최초 접수 시 밝힌 발화 시각과, 이후 손해사정 방문 때 "
            "진술한 시각이 한 시간 가량 차이가 나서 정리 중이다."
        ),
    },
    {
        "id": "A3",
        "ground_truth": "fraud",
        "difficulty": "ambiguous",
        "note": "정상적 표현이지만 동일 명의로 짧은 간격의 유사 청구 이력",
        "text": (
            "야간 주행 중 노루가 튀어나와 급제동하다 가드레일을 긁은 사고다. 사고 "
            "경위 자체는 흔한 편이나, 동일 계약자 명의로 이와 유사한 형태의 청구가 "
            "지난 넉 달 사이 이미 접수된 이력이 있어 함께 검토가 필요하다."
        ),
    },
]


def _find_leaked_substrings(text: str) -> list[str]:
    """휴리스틱/합성데이터 키워드 사전에 있는 문자열이 실수로 섞여 있는지 검사."""
    leaked = []
    for label, substrings in _HEURISTIC_KEYWORDS.items():
        for s in substrings:
            if s and s in text:
                leaked.append(f"heuristic:{label}:{s!r}")
    for pattern in KOREAN_FRAUD_PATTERNS.values():
        if pattern["label"] in text:
            leaked.append(f"pattern_label:{pattern['label']!r}")
    return leaked


def main() -> None:
    print("=== 0. 키워드 오염 검사 (금지된 정확 문자열이 섞여있는지) ===")
    contaminated = False
    for case in CASES:
        leaked = _find_leaked_substrings(case["text"])
        if leaked:
            contaminated = True
            print(f"[{case['id']}] 오염 발견: {leaked}")
    if not contaminated:
        print("오염 없음 — 모든 케이스가 사전 키워드와 무관한 표현으로 작성됨.\n")
    else:
        print("\n경고: 위 케이스는 휴리스틱이 '진짜로' 매칭한 게 아니라 우연히 문자열이 겹친 것일 수 있음.\n")

    rows = []
    print("=== 1. 휴리스틱 폴백 분석 (API 호출 없음) ===")
    for case in CASES:
        h = _heuristic_analysis(case["text"])
        rows.append({"case": case, "heuristic": h})
        print(f"[{case['id']}] adj={h.suspicion_adjustment:+.2f} keywords={h.keywords}")

    print("\n=== 2. LLM 분석 (analyze_narrative, 실제 API 호출 최대 10건) ===")
    for row in rows:
        case = row["case"]
        llm = analyze_narrative(case["text"])
        row["llm"] = llm
        print(f"[{case['id']}] adj={llm.suspicion_adjustment:+.2f} keywords={llm.keywords}")
        print(f"      설명: {llm.explanation}")

    print("\n=== 3. 비교 표 (판정 기준: adjustment > 0 이면 '사기 의심'으로 예측한 것으로 간주) ===")
    print(f"{'ID':4}{'정답(GT)':10}{'난이도':10}{'휴리스틱 adj':14}{'LLM adj':10}{'휴리스틱':8}{'LLM':8}")
    h_correct_n = llm_correct_n = 0
    for row in rows:
        case, h, llm = row["case"], row["heuristic"], row["llm"]
        gt_is_fraud = case["ground_truth"] == "fraud"
        h_correct = (h.suspicion_adjustment > 0) == gt_is_fraud
        llm_correct = (llm.suspicion_adjustment > 0) == gt_is_fraud
        h_correct_n += h_correct
        llm_correct_n += llm_correct
        print(
            f"{case['id']:<4}{case['ground_truth']:<10}{case['difficulty']:<10}"
            f"{h.suspicion_adjustment:<+14.2f}{llm.suspicion_adjustment:<+10.2f}"
            f"{'O' if h_correct else 'X':<8}{'O' if llm_correct else 'X':<8}"
        )
    print(f"\n정확도: 휴리스틱 {h_correct_n}/{len(rows)}, LLM {llm_correct_n}/{len(rows)}")


if __name__ == "__main__":
    main()
