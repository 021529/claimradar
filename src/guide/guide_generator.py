import anthropic

from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

_MIN_ITEMS = 3
_MAX_ITEMS = 5

# 이상징후 키워드 -> 체크리스트 항목 템플릿 (ANTHROPIC_API_KEY 미설정 시 폴백용, 30자 내외)
_CHECKLIST_TEMPLATES: dict[str, str] = {
    "나이롱환자": "통원치료 기간이 진단 대비 과도한지 확인한다.",
    "한방병원 장기입원": "한방병원 입원 필요성과 실제 재원 여부를 확인한다.",
    "지연신고": "신고 지연 사유와 목격자 유무를 확인한다.",
    "블랙박스 미장착": "블랙박스 미장착 경위와 제3자 영상자료를 확인한다.",
    "진술 불일치": "진술 간 불일치 지점을 확인하고 대질한다.",
    "반복 청구": "과거 청구 이력의 반복 패턴을 조회한다.",
    "경보장치 미작동": "경보장치 미작동 경위와 조작 가능성을 점검한다.",
    "과다 청구": "청구액과 시세·견적을 비교해 소명자료를 요청한다.",
}

# 매칭되는 키워드가 없거나 항목 수가 부족할 때 채우는 일반 확인 항목 (30자 내외)
_GENERIC_ITEMS: list[str] = [
    "사고 서류와 경위 진술의 정합성을 확인한다.",
    "과거 청구·계약 변동 이력을 조회한다.",
    "손상 사진과 진술 내용의 부합 여부를 확인한다.",
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
                    "각 항목은 30자 내외의 짧은 한 문장으로 작성하고, 대화 스크립트나 질문 형식이 아닌 "
                    "확인 행위/대조 대상을 명시하는 간결한 지시문으로 작성한다. "
                    "이상징후 키워드가 주어졌다면 각 키워드에 대응하는 구체적 확인 포인트를 우선 포함한다."
                ),
            },
        },
        "required": ["checklist_items"],
    },
}


def generate_investigation_checklist(case_summary: str, suspicion_keywords: list[str]) -> list[str]:
    """배정된 사건 정보 + 이상징후 키워드 -> 실무자용 조사 체크리스트 3~5개 생성.

    각 항목은 "확인할 것" 위주의 30자 내외 짧은 문장(체크리스트형)이며, 대화 스크립트가 아니다.
    ANTHROPIC_API_KEY가 설정되어 있으면 tool use로 생성하고, 미설정이거나 호출 실패 시
    이상징후 키워드 -> 확인 포인트 템플릿 매핑으로 폴백한다.
    """
    if not ANTHROPIC_API_KEY:
        return _heuristic_guide(suspicion_keywords)

    keywords_block = ", ".join(suspicion_keywords) if suspicion_keywords else "(감지된 이상징후 키워드 없음)"
    prompt = (
        "당신은 국내 자동차보험사의 보험사기 조사관을 보조하는 어시스턴트입니다.\n"
        "아래 사건 요약과 이상징후 키워드를 참고하여, 조사관이 현장/서류 조사 시 바로 활용할 "
        "실무용 체크리스트를 report_investigation_checklist 도구로 작성하세요.\n"
        "- 항목은 3~5개, 각 항목은 30자 내외의 짧은 한 문장으로 작성합니다. 부연 설명이나 "
        "괄호를 붙이지 말고 핵심 확인 대상 하나만 짧게 씁니다.\n"
        "- '~에게 물어본다' 식의 대화 스크립트가 아니라, '~을 확인한다/대조한다/조회한다'처럼 "
        "확인 행위 위주의 지시문으로 작성합니다.\n"
        "- 이상징후 키워드가 있다면 각 키워드에 대응하는 구체적 확인 포인트를 우선 포함하세요. "
        '예: "지연신고" 감지 시 -> "신고 지연 사유와 목격자 유무를 확인한다."\n\n'
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
