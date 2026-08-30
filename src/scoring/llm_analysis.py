import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

import anthropic

from src.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from src.scoring.fraud_patterns import KOREAN_FRAUD_PATTERNS, select_patterns


@dataclass
class NarrativeAnalysis:
    keywords: list[str]
    suspicion_adjustment: float  # -1.0 ~ 1.0, ML 스코어 보정치
    explanation: str  # "왜 이 점수인지" 1~2문장


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


_usage_lock = threading.Lock()
_usage_totals = {"input_tokens": 0, "output_tokens": 0, "requests": 0}


def _record_usage(response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    with _usage_lock:
        _usage_totals["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        _usage_totals["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        _usage_totals["requests"] += 1


def get_usage_totals() -> dict:
    """이 프로세스에서 실제 API 호출로 누적된 토큰 사용량 (비용 모니터링용)."""
    with _usage_lock:
        return dict(_usage_totals)


def reset_usage_totals() -> None:
    with _usage_lock:
        _usage_totals["input_tokens"] = 0
        _usage_totals["output_tokens"] = 0
        _usage_totals["requests"] = 0


_ANALYSIS_TOOL = {
    "name": "report_narrative_analysis",
    "description": "사고경위서 본문에서 발견한 이상징후 키워드와 의심도 보정치, 근거 설명을 보고한다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "본문에 실제로 근거가 있는 이상징후 키워드 목록. "
                    "국내 특유 사기 정황(나이롱환자, 한방병원 장기입원, 지연신고, 블랙박스 미장착)이나 "
                    "그 외 본문에서 확인되는 의심 정황(진술 불일치, 반복 청구 등)을 짧은 한국어 명사구로 작성. "
                    "특이사항이 없으면 빈 배열."
                ),
            },
            "suspicion_adjustment": {
                "type": "number",
                "description": "ML 스코어에 더할 의심도 보정치. -1.0(정상 정황이 뚜렷함) ~ 1.0(사기 의심 정황이 뚜렷함).",
            },
            "explanation": {
                "type": "string",
                "description": "왜 이 보정치를 부여했는지 근거를 1~2문장 한국어로 요약.",
            },
        },
        "required": ["keywords", "suspicion_adjustment", "explanation"],
    },
}

# 폴백(휴리스틱) 분석용 키워드 사전: 라벨 -> 본문에서 찾을 부분 문자열들
_HEURISTIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    pattern["label"]: (pattern["label"], pattern["narrative_hint"][:10])
    for pattern in KOREAN_FRAUD_PATTERNS.values()
} | {
    "진술 불일치": ("일치하지 않", "진술이 다르", "일관되지 않"),
    "반복 청구": ("반복", "두 번째", "세 번째"),
    "경보장치 미작동": ("경보장치가 작동하지 않",),
    "과다 청구": ("시세를 크게 상회", "청구 금액이 과다"),
}


def _heuristic_analysis(narrative_text: str) -> NarrativeAnalysis:
    """LLM 미사용(또는 실패) 시 키워드 사전 매칭으로 동작하는 폴백 분석."""
    matched = [
        label
        for label, substrings in _HEURISTIC_KEYWORDS.items()
        if any(s in narrative_text for s in substrings)
    ]

    if matched:
        adjustment = min(1.0, 0.35 * len(matched))
        explanation = f"{', '.join(matched)} 등 이상징후가 본문에서 확인되어 의심도를 상향 조정함."
    elif "일치함" in narrative_text or "일치" in narrative_text:
        adjustment = -0.3
        explanation = "제출 내용이 사고 상황과 일치해 의심도를 하향 조정함."
    else:
        adjustment = 0.0
        explanation = "본문에서 뚜렷한 이상징후를 확인하지 못해 의심도를 조정하지 않음."

    return NarrativeAnalysis(keywords=matched, suspicion_adjustment=adjustment, explanation=explanation)


def analyze_narrative(narrative_text: str) -> NarrativeAnalysis:
    """사고경위서 텍스트에서 이상징후 키워드 + 의심도 보정치 + 근거 설명 추출.

    ANTHROPIC_API_KEY가 설정되어 있으면 tool use로 구조화된 JSON을 받아 분석하고,
    미설정이거나 호출 실패 시 키워드 사전 기반 휴리스틱으로 폴백한다.
    """
    if not narrative_text or not narrative_text.strip():
        return NarrativeAnalysis(keywords=[], suspicion_adjustment=0.0, explanation="사고경위서 내용이 없어 분석하지 못함.")

    if not ANTHROPIC_API_KEY:
        return _heuristic_analysis(narrative_text)

    prompt = (
        "당신은 국내 자동차보험사의 보험사기 조사를 보조하는 어시스턴트입니다.\n"
        "아래 사고경위서를 읽고, 나이롱환자·한방병원 장기입원·지연신고·블랙박스 미장착 등 "
        "국내 자동차보험에서 흔한 사기 의심 정황을 포함해 본문에 실제로 근거가 있는 이상징후만 "
        "report_narrative_analysis 도구로 보고하세요. 본문에 없는 내용을 추측해서 만들어내지 마세요.\n\n"
        f"[사고경위서]\n{narrative_text}"
    )

    try:
        response = _client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=500,
            tools=[_ANALYSIS_TOOL],
            tool_choice={"type": "tool", "name": "report_narrative_analysis"},
            messages=[{"role": "user", "content": prompt}],
        )
        _record_usage(response)
        tool_use = next(
            block for block in response.content if getattr(block, "type", None) == "tool_use"
        )
        payload = tool_use.input
        adjustment = max(-1.0, min(1.0, float(payload["suspicion_adjustment"])))
        return NarrativeAnalysis(
            keywords=list(payload.get("keywords", [])),
            suspicion_adjustment=adjustment,
            explanation=str(payload.get("explanation", "")),
        )
    except Exception:
        return _heuristic_analysis(narrative_text)


_FIELD_LABELS: list[tuple[str, tuple[str, ...]]] = [
    ("사고 지역", ("AccidentArea",)),
    ("과실 여부", ("Fault",)),
    ("차종 구분", ("VehicleCategory",)),
    ("차량가액", ("VehiclePrice", "vehicle_price")),
    ("기본 담보", ("BasePolicy", "PolicyType")),
    ("계약자 나이", ("AgeOfPolicyHolder", "Age", "driver_age")),
    ("과거 청구 건수", ("PastNumberOfClaims", "past_number_of_claims")),
    ("사고 후 청구 지연기간", ("Days_Policy_Claim",)),
    ("경찰신고 여부", ("PoliceReportFiled",)),
    ("목격자 유무", ("WitnessPresent",)),
]


def _get(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        try:
            value = row[name]
        except (KeyError, IndexError):
            continue
        if value is None:
            continue
        try:
            import math

            if isinstance(value, float) and math.isnan(value):
                continue
        except TypeError:
            pass
        return value
    return None


def _row_context_lines(row: Mapping[str, Any]) -> list[str]:
    lines = []
    for label, candidates in _FIELD_LABELS:
        value = _get(row, *candidates)
        if value is not None:
            lines.append(f"- {label}: {value}")
    return lines


def _seed_key(row: Mapping[str, Any]) -> Any:
    value = _get(row, "PolicyNumber", "case_id")
    return str(value) if value is not None else id(row)


def _template_narrative(row: Mapping[str, Any], patterns: list[str]) -> str:
    """LLM 미사용(또는 실패) 시 결정론적으로 사고경위서를 조립하는 폴백."""
    rng = random.Random(_seed_key(row))

    area = _get(row, "AccidentArea")
    opening_choices = [
        "차량 운행 중 경미한 접촉사고가 발생했다는 사고경위서가 접수됨.",
        "야간 시간대 발생한 사고에 대한 청구가 접수됨.",
        "일상 주행 중 발생한 사고에 대해 청구인이 사고경위서를 제출함.",
    ]
    if area:
        opening = f"{area} 지역에서 발생한 사고에 대해 청구인이 사고경위서를 제출함."
    else:
        opening = rng.choice(opening_choices)

    sentences = [opening]
    for key in patterns:
        hint = KOREAN_FRAUD_PATTERNS[key]["narrative_hint"]
        sentences.append(f"{hint}이 확인됨.")

    if not patterns:
        sentences.append("제출된 서류와 진술 내용은 사고 상황과 대체로 일치함.")

    return " ".join(sentences)


def generate_synthetic_narrative(
    claim_row, style_reference_text: str = "", patterns_override: list[str] | None = None
) -> str:
    """Kaggle 정형 라벨 + 국내 특유 사기패턴을 결합해 한글 합성 사고경위서를 생성.

    FraudFound_P == 1 인 레코드는 나이롱환자/한방병원 장기입원/지연신고/블랙박스
    미장착 등 국내 자동차보험 특유 사기 패턴 중 일부를 확률적으로 반영하고,
    정상 건은 이러한 정황 없이 사실관계 위주로 서술한다.

    style_reference_text가 주어지면(NHTSA NMVCCS 등 실제 사고경위서 원문) 문체·구조
    참고용 few-shot으로 프롬프트에 포함하되, 언어와 내용은 완전히 새로 생성한다.

    patterns_override가 주어지면 select_patterns() 대신 그 리스트를 그대로 쓴다 —
    scripts/leak_free_contribution_test.py가 라벨 누출 없는 select_patterns_structural()
    결과를 주입하는 데 쓴다. 기본값 None이면 기존과 완전히 동일하게 동작해 기존
    호출부(scripts/generate_synthetic_dataset.py 등)는 영향받지 않는다.
    """
    patterns = patterns_override if patterns_override is not None else select_patterns(claim_row, seed_key=_seed_key(claim_row))

    if not ANTHROPIC_API_KEY:
        return _template_narrative(claim_row, patterns)

    context_lines = "\n".join(_row_context_lines(claim_row))
    pattern_instruction = (
        "다음 국내 자동차보험 특유 사기 정황을 사고경위서에 자연스럽게 녹여 서술하세요:\n"
        + "\n".join(f"- {KOREAN_FRAUD_PATTERNS[key]['narrative_hint']}" for key in patterns)
        if patterns
        else "사기를 의심할 만한 특이 정황 없이, 사실관계 위주로 담백하게 서술하세요."
    )
    style_block = (
        f"\n\n[문체/구조 참고용 원문 발췌 — 내용은 참고하지 말고 사실 기술 방식만 참고]\n{style_reference_text}"
        if style_reference_text
        else ""
    )

    prompt = (
        "당신은 국내 자동차보험사의 사고경위서 작성을 보조하는 어시스턴트입니다.\n"
        "아래 청구 정보를 바탕으로, 실제 접수될 법한 사고경위서를 한국어로 3~5문장, "
        "전체 400자 이내로 간결하게, 조사 보고서체(개조식이 아닌 서술형, 존댓말 미사용)로 작성하세요.\n"
        "글자 수 제한을 넘기지 않도록 마지막 문장을 반드시 끝까지 완결하세요.\n"
        "인물의 실명, 구체적 주소, 차량번호 등 식별정보는 만들어내지 마세요.\n\n"
        f"[청구 정보]\n{context_lines}\n\n{pattern_instruction}{style_block}"
    )

    _MAX_ATTEMPTS = 3
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = _client().messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}],
            )
            _record_usage(response)
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
            if response.stop_reason == "max_tokens":
                text = _trim_to_last_complete_sentence(text)
            if _looks_complete(text):
                return text
        except Exception:
            pass
        if attempt < _MAX_ATTEMPTS - 1:
            time.sleep(2 * (attempt + 1))

    return _template_narrative(claim_row, patterns)


def _looks_complete(text: str) -> bool:
    """네트워크 중단 등으로 응답이 중간에 끊긴 경우를 감지 (짧고 문장부호로 안 끝남)."""
    return len(text) >= 50 and text.rstrip().endswith(("다.", "음.", "함.", "됨.", "임."))


def _trim_to_last_complete_sentence(text: str) -> str:
    """max_tokens 도달로 문장이 잘렸을 때, 마지막으로 완결된 문장까지만 남긴다."""
    last_end = max(text.rfind(e) for e in ("다.", "음.", "함.", "됨.", "임."))
    return text[: last_end + 2] if last_end != -1 else text
