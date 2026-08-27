import anthropic

from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

# 체크리스트 단계 구조 근거: 선행연구 수식을 인용한 것이 아니라, 송윤아(2011),
# 「사기성 클레임에 대한 최적 조사방안」, 보험연구원 경영보고서 2011-5,
# <표 Ⅱ-14> "보험사기의심건 조사 절차"에 실린 실제 SIU(보험사기특별조사팀)
# 조사 절차 7단계(인지→조회→자료취합→분석→서류작성→수사의뢰→수사지원)를
# 참고해 우리 제품 흐름에 맞게 적용한 것이다(표 하단에는 저자 자신의 2010년
# 선행 자료가 해당 표의 원출처로 별도 표기되어 있으나, 보고서 자체의 공식
# 발행연도는 2011년이다). 이 체크리스트는 "이미 배정된 사건"을
# 다루므로 7단계 중 인지(=스코어링 단계에서 이미 완료)와 서류작성/수사의뢰/수사지원
# (=이 체크리스트 이후의 후속 조치)은 범위 밖이며, 조사관이 실제 수행하는 조회→
# 자료취합→분석 3단계만 체크리스트 항목의 단계 태그로 사용한다.
# ※ 위 보고서의 보험사기 발생 통계 자체는 2011년 기준으로 낡았지만, 조사 프로세스의
# 기본 구조(인지→조회→자료취합→분석→...)는 최신(2026년) 업계 동향으로도 유사하게
# 유지되고 있는 것으로 재확인함.
_MIN_ITEMS = 3
_MAX_ITEMS = 5
_PROCESS_STAGES = ("조회", "자료취합", "분석")

# 이상징후 키워드 -> 체크리스트 항목 템플릿 (ANTHROPIC_API_KEY 미설정 시 폴백용, 30자 내외)
# 각 항목의 [단계] 태그는 위 3단계 프레임에 우리가 매핑한 것으로, 보고서 원문에
# 키워드별 단계 매핑이 있는 것은 아니다.
_CHECKLIST_TEMPLATES: dict[str, str] = {
    "나이롱환자": "[분석] 통원치료 기간이 진단 대비 과도한지 확인한다.",
    "한방병원 장기입원": "[자료취합] 한방병원 입원 필요성과 실제 재원 여부를 확인한다.",
    "지연신고": "[조회] 신고 지연 사유와 목격자 유무를 확인한다.",
    "블랙박스 미장착": "[자료취합] 블랙박스 미장착 경위와 제3자 영상자료를 확인한다.",
    "진술 불일치": "[분석] 진술 간 불일치 지점을 확인하고 대질한다.",
    "반복 청구": "[조회] 과거 청구 이력의 반복 패턴을 조회한다.",
    "경보장치 미작동": "[분석] 경보장치 미작동 경위와 조작 가능성을 점검한다.",
    "과다 청구": "[자료취합] 청구액과 시세·견적을 비교해 소명자료를 요청한다.",
}

# 매칭되는 키워드가 없거나 항목 수가 부족할 때 채우는 일반 확인 항목 (30자 내외)
_GENERIC_ITEMS: list[str] = [
    "[분석] 사고 서류와 경위 진술의 정합성을 확인한다.",
    "[조회] 과거 청구·계약 변동 이력을 조회한다.",
    "[분석] 손상 사진과 진술 내용의 부합 여부를 확인한다.",
]


def _heuristic_guide(suspicion_keywords: list[str]) -> list[str]:
    """LLM 미사용(또는 실패) 시 키워드 -> 템플릿 매핑으로 동작하는 폴백 체크리스트."""
    items = [
        _CHECKLIST_TEMPLATES[kw] for kw in dict.fromkeys(suspicion_keywords) if kw in _CHECKLIST_TEMPLATES
    ]

    for generic in _GENERIC_ITEMS:
        if len(items) >= _MIN_ITEMS:
            break
        if generic not in items:
            items.append(generic)

    return items[:_MAX_ITEMS]


_GUIDE_TOOL = {
    "name": "report_investigation_checklist",
    "description": "배정된 사건에 대해 조사관이 현장/서류 조사 시 확인해야 할 체크리스트 항목을 보고한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "checklist_items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": _MIN_ITEMS,
                "maxItems": _MAX_ITEMS,
                "description": (
                    "조사관이 '확인할 것' 위주로 작성한 체크리스트 항목 3~5개. "
                    "각 항목은 실제 SIU 조사 절차의 조회/자료취합/분석 단계 중 하나를 "
                    "'[조회]', '[자료취합]', '[분석]' 형태로 맨 앞에 태그로 붙이고, "
                    "그 뒤에 30자 내외의 짧은 한 문장으로 확인 행위를 작성한다(예: "
                    "'[조회] 신고 지연 사유와 목격자 유무를 확인한다.'). 대화 스크립트나 "
                    "질문 형식이 아닌 확인/대조/조회 행위를 명시하는 간결한 지시문으로 작성한다. "
                    "이상징후 키워드와 사고경위서 내용에 실제로 근거가 있는 확인 포인트만 "
                    "포함하고, 키워드가 있다면 각 키워드가 조회/자료취합/분석 중 어느 단계에서 "
                    "확인되어야 하는지 판단해 해당 단계 태그로 우선 포함한다."
                ),
            },
        },
        "required": ["checklist_items"],
    },
}


def generate_investigation_checklist(case_summary: str, suspicion_keywords: list[str]) -> list[str]:
    """배정된 사건 정보 + 이상징후 키워드 -> 실무자용 조사 체크리스트 3~5개 생성.

    각 항목은 "[조회]", "[자료취합]", "[분석]" 중 하나의 단계 태그로 시작하는
    30자 내외 짧은 문장(체크리스트형)이며, 대화 스크립트가 아니다. 이 3단계는
    특정 선행연구의 수식을 인용한 것이 아니라, 송윤아(2011), 「사기성 클레임에
    대한 최적 조사방안」, 보험연구원 경영보고서 2011-5, <표 Ⅱ-14> "보험사기의심건
    조사 절차"에 실린 실제 SIU 조사 절차 7단계(인지→조회→자료취합→분석→
    서류작성→수사의뢰→수사지원) 중 이미 배정된 사건의 조사관이 실제
    수행하는 가운데 3단계만 우리가 이 제품의 체크리스트 구조로 가져와 적용한
    것이다(인지는 스코어링 단계에서, 서류작성/수사의뢰/수사지원은 이 체크리스트
    이후 단계에서 이뤄지므로 범위 밖).

    ANTHROPIC_API_KEY가 설정되어 있으면 tool use로 생성하고, 미설정이거나 호출 실패 시
    이상징후 키워드 -> 확인 포인트 템플릿 매핑으로 폴백한다.
    """
    if not ANTHROPIC_API_KEY:
        return _heuristic_guide(suspicion_keywords)

    keywords_block = ", ".join(suspicion_keywords) if suspicion_keywords else "(감지된 이상징후 키워드 없음)"
    stages_block = " / ".join(_PROCESS_STAGES)
    prompt = (
        "당신은 국내 자동차보험사의 보험사기 조사관을 보조하는 어시스턴트입니다.\n"
        "이 회사는 실제 SIU(보험사기특별조사팀) 조사 절차인 "
        "인지→조회→자료취합→분석→서류작성→수사의뢰→수사지원 중, 이미 사건이 배정된 "
        f"조사관이 직접 수행하는 {stages_block} 3단계에 맞춰 체크리스트를 구성합니다.\n"
        "아래 사건 요약과 이상징후 키워드를 참고하여, 조사관이 현장/서류 조사 시 바로 활용할 "
        "실무용 체크리스트를 report_investigation_checklist 도구로 작성하세요.\n"
        f"- 항목은 3~5개, 각 항목은 '[조회]', '[자료취합]', '[분석]' 중 이 확인 행위에 "
        "가장 맞는 단계 하나를 앞에 태그로 붙이고, 뒤에 30자 내외의 짧은 한 문장을 씁니다. "
        "부연 설명이나 괄호를 붙이지 말고 핵심 확인 대상 하나만 짧게 씁니다.\n"
        "  · [조회]: 과거 사고력·청구이력·계약사항 등 기존 시스템/자료 대조\n"
        "  · [자료취합]: 진단서·정비내역·영상자료 등 추가 서류·증거 확보\n"
        "  · [분석]: 진술 간 정합성, 사고 정황과 손상 내용의 부합 여부 등 판단\n"
        "- '~에게 물어본다' 식의 대화 스크립트가 아니라, '~을 확인한다/대조한다/조회한다'처럼 "
        "확인 행위 위주의 지시문으로 작성합니다.\n"
        "- 사고경위서 본문과 이상징후 키워드에 실제로 근거가 있는 확인 포인트만 포함하고, "
        "본문에 없는 내용을 추측해서 만들어내지 마세요. 키워드가 있다면 그 키워드가 "
        "조회/자료취합/분석 중 어느 단계에서 확인되어야 하는지 판단해 우선 포함하세요. "
        '예: "지연신고" 감지 시 -> "[조회] 신고 지연 사유와 목격자 유무를 확인한다."\n\n'
        f"[사건 요약]\n{case_summary}\n\n[이상징후 키워드]\n{keywords_block}"
    )

    try:
        response = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=600,
            tools=[_GUIDE_TOOL],
            tool_choice={"type": "tool", "name": "report_investigation_checklist"},
            messages=[{"role": "user", "content": prompt}],
        )
        tool_use = next(
            block for block in response.content if getattr(block, "type", None) == "tool_use"
        )
        items = [str(item).strip() for item in tool_use.input.get("checklist_items", []) if str(item).strip()]
        if len(items) < _MIN_ITEMS:
            items = (items + _heuristic_guide(suspicion_keywords))[:_MAX_ITEMS]
        return items[:_MAX_ITEMS]
    except Exception:
        return _heuristic_guide(suspicion_keywords)
