"""사고경위서에 "표면(표기/문체)만 훼손하고 의미는 보존"하는 현실적 노이즈를
주입하는 함수 모음 — scripts/noise_robustness_test.py 전용 유틸리티.

실제 보험사 접수 메모/조사노트의 특성을 반영한 6개 기법을 하나의 파이프라인으로
합성한다:
  1. 오타·자모 오류 — 한글 음절을 초성/중성/종성으로 분해해 인접한(키보드/발음상
     혼동되는) 자모로 바꿔치기. 숫자는 건드리지 않는다.
  2. 띄어쓰기 파괴 — 공백을 확률적으로 삭제(전체 삭제 아님).
  3. 구어체·비문·메모체 — 격식체 어미를 메모체로 변환, 반복되는 주어 생략,
     문장 경계 마침표 제거(단어 손실 없음).
  4. 은어·업계 약어 — 고정 치환 사전(예: 블랙박스→블박).
  5. 감정적 삽입구 — 조사관 주관 서술을 추가(기존 내용 삭제 없음, 순수 부가).
  6. 분석과 무관한 정보 혼입 — 가짜 접수번호/연락처 스니펫 삽입.

모든 함수는 case_id로 시드된 random.Random 인스턴스를 받아 완전히 재현 가능하다.
적용 순서 주의: destroy_spacing()은 반드시 마지막에 적용해야 한다 — 그 전 단계들
(주어 생략, 어미 치환, 은어 치환)이 공백을 기준으로 패턴 매칭을 하기 때문이다.
"""

import random
import re

_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3
_JUNG_COUNT = 21
_JONG_COUNT = 28

# 흔히 혼동되는(키보드 인접/발음 유사) 자모 쌍 — 실제 예시("한방병원"→"한방병웜"의
# ㄴ(4)→ㅁ(16), "블랙박스"→"블랙박수"의 ㅡ(18)→ㅜ(13))를 우선 반영하고, 나머지는
# ±1 인덱스 폴백으로 커버한다.
_JONG_CONFUSE: dict[int, list[int]] = {
    4: [16, 8],  # ㄴ -> ㅁ, ㄹ
    16: [4],  # ㅁ -> ㄴ
    8: [4, 7],  # ㄹ -> ㄴ, ㄷ
    19: [20],  # ㅅ -> ㅆ
    20: [19],  # ㅆ -> ㅅ
    7: [19],  # ㄷ -> ㅅ
    1: [24],  # ㄱ -> ㅋ
    24: [1],  # ㅋ -> ㄱ
    21: [4],  # ㅇ -> ㄴ
}
_JUNG_CONFUSE: dict[int, list[int]] = {
    18: [13],  # ㅡ -> ㅜ
    13: [18],  # ㅜ -> ㅡ
    1: [5],  # ㅐ -> ㅔ
    5: [1],  # ㅔ -> ㅐ
    4: [8],  # ㅓ -> ㅗ
    8: [4],  # ㅗ -> ㅓ
    2: [6],  # ㅑ -> ㅕ
    6: [2],  # ㅕ -> ㅑ
    9: [14],  # ㅘ -> ㅝ
    14: [9],  # ㅝ -> ㅘ
}

_FRAUD_SIGNAL_WORDS = [
    "블랙박스", "지연", "목격자", "경찰", "청구", "과거", "접수", "미장착", "미신고", "한방병원", "장기입원",
]

_SLANG_MAP = {
    "블랙박스": "블박",
    "지연신고": "늦신고",
    "한방병원": "한방",
    "장기입원": "장입원",
    "종합담보": "올담보",
}

_EMOTIONAL_ASIDES = [
    "본인 억울하다고 계속 주장함.",
    "설명 중 언성 높임.",
    "조사 협조에 다소 소극적인 태도 보임.",
    "같은 내용 반복 진술함.",
    "말 바꾸는 부분 있어 재확인 필요해 보임.",
]

_SUBJECT_MARKERS = ("계약자는", "계약자가")


def _is_hangul_syllable(ch: str) -> bool:
    return _HANGUL_BASE <= ord(ch) <= _HANGUL_LAST


def _decompose(syllable: str) -> tuple[int, int, int]:
    code = ord(syllable) - _HANGUL_BASE
    cho = code // (_JUNG_COUNT * _JONG_COUNT)
    jung = (code % (_JUNG_COUNT * _JONG_COUNT)) // _JONG_COUNT
    jong = code % _JONG_COUNT
    return cho, jung, jong


def _compose(cho: int, jung: int, jong: int) -> str:
    return chr(_HANGUL_BASE + (cho * _JUNG_COUNT + jung) * _JONG_COUNT + jong)


def _mutate_syllable(ch: str, rng: random.Random) -> str:
    """음절 하나의 중성 또는 종성을 인접 자모로 바꿔치기(오타 시뮬레이션)."""
    cho, jung, jong = _decompose(ch)
    if rng.random() < 0.5:
        candidates = _JUNG_CONFUSE.get(jung, [(jung + 1) % _JUNG_COUNT, (jung - 1) % _JUNG_COUNT])
        jung = rng.choice(candidates)
    else:
        candidates = _JONG_CONFUSE.get(jong, [(jong + 1) % _JONG_COUNT, (jong - 1) % _JONG_COUNT])
        jong = rng.choice(candidates)
    return _compose(cho, jung, jong)


def corrupt_jamo(text: str, rng: random.Random, n_targeted: int = 4, n_random: int = 2) -> str:
    """사기 신호 단어(블랙박스/지연/목격자/경찰 등) 근처를 우선 타깃으로, 나머지
    본문에서도 일부 랜덤하게 음절을 오타로 바꾼다. 숫자는 건드리지 않는다.

    문장 맨 끝(마침표 직전) 음절은 타깃에서 제외한다 — informalize_endings()의
    어미 변환 규칙("...다." 매칭)과 겹치면 "확인된다."가 "확인된닥."처럼 이도저도
    아닌 문자열이 돼 표면 노이즈가 아니라 판독 불가능한 훼손이 될 수 있어서다."""
    chars = list(text)

    def _before_period(i: int) -> bool:
        return i + 1 < len(chars) and chars[i + 1] == "."

    occurrences: list[tuple[int, int]] = []
    for word in _FRAUD_SIGNAL_WORDS:
        start = 0
        while True:
            idx = text.find(word, start)
            if idx == -1:
                break
            occurrences.append((idx, idx + len(word)))
            start = idx + len(word)
    rng.shuffle(occurrences)

    mutated_positions: set[int] = set()
    for start, end in occurrences[:n_targeted]:
        positions = [
            i for i in range(start, end) if _is_hangul_syllable(chars[i]) and not _before_period(i)
        ]
        if positions:
            idx = positions[-1]
            chars[idx] = _mutate_syllable(chars[idx], rng)
            mutated_positions.add(idx)

    hangul_positions = [
        i
        for i, c in enumerate(chars)
        if _is_hangul_syllable(c) and i not in mutated_positions and not _before_period(i)
    ]
    if hangul_positions:
        extra = rng.sample(hangul_positions, k=min(n_random, len(hangul_positions)))
        for i in extra:
            chars[i] = _mutate_syllable(chars[i], rng)

    return "".join(chars)


def destroy_spacing(text: str, rng: random.Random, removal_prob: float = 0.35) -> str:
    """공백을 확률적으로 삭제 — 전체 삭제가 아니라 실제 메모처럼 부분적으로만 붙여쓴다."""
    return "".join(ch for ch in text if not (ch == " " and rng.random() < removal_prob))


def _informalize_sentence(s: str) -> str:
    """격식체 문장 종결을 메모체로 변환. sample_claims.csv 실제 종결 어미
    37종을 확인해 만든 일반 규칙(이다./하다./았다·었다·였다./ㄴ다. 4갈래)이라
    이 문체 범위를 벗어나는 입력에는 그대로(무변화) 반환될 수 있다."""
    if s.endswith("이다."):
        return s[:-3] + "임."
    if s.endswith("하다."):
        return s[:-3] + "함."
    m = re.match(r"^(.*)([가-힣])다\.$", s)
    if m:
        prefix, last = m.group(1), m.group(2)
        cho, jung, jong = _decompose(last)
        if jong == 4:  # 현재형 'ㄴ다.' -> 'ㅁ.' (예: 확인된다->확인됨)
            return prefix + _compose(cho, jung, 16) + "."
    if s.endswith("다."):  # 그 외 과거형 등 -> '음.' (예: 있다->있음, 였다->였음)
        return s[:-2] + "음."
    return s


def informalize_endings(text: str, rng: random.Random, apply_prob: float = 0.6) -> str:
    sentences = re.split(r"(?<=\.)\s+", text)
    out = [_informalize_sentence(s) if rng.random() < apply_prob else s for s in sentences]
    return " ".join(out)


def drop_repeated_subject(text: str, rng: random.Random, drop_prob: float = 0.6) -> str:
    """두 번째 이후 등장하는 '계약자는/계약자가'를 확률적으로 생략(첫 등장은 유지)
    — 문맥상 주어가 이미 확정돼 있어 의미 손실이 없다."""
    pattern = re.compile("(" + "|".join(_SUBJECT_MARKERS) + ")\\s*")
    matches = list(pattern.finditer(text))
    if len(matches) <= 1:
        return text
    to_drop = [m for m in matches[1:] if rng.random() < drop_prob]
    if not to_drop:
        return text
    result, last = [], 0
    for m in to_drop:
        result.append(text[last : m.start()])
        last = m.end()
    result.append(text[last:])
    return "".join(result)


def merge_sentences(text: str, rng: random.Random, merge_prob: float = 0.3) -> str:
    """일부 문장 경계의 마침표를 지워 이어붙인다 — 단어 손실 없는 순수 구두점 노이즈."""
    parts = text.split(". ")
    if len(parts) <= 1:
        return text
    merged = [parts[0]]
    for p in parts[1:]:
        if rng.random() < merge_prob:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return ". ".join(merged)


def apply_slang(text: str) -> str:
    for full, slang in _SLANG_MAP.items():
        text = text.replace(full, slang)
    return text


def insert_emotional_aside(text: str, rng: random.Random) -> str:
    aside = rng.choice(_EMOTIONAL_ASIDES)
    return f"{text} {aside}"


def insert_admin_noise(text: str, rng: random.Random) -> str:
    phone = f"010-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
    receipt = f"2024-{rng.randint(1000, 9999)}"
    officer = rng.choice(["김", "이", "박", "최", "정"]) + "OO"
    return f"{text} (접수번호 {receipt}, 담당 {officer}, 연락처 {phone})"


def inject_noise(text: str, case_id) -> str:
    """6개 노이즈 기법을 case_id로 시드된 순서로 합성 적용해 재현 가능한 노이즈
    버전 사고경위서를 만든다. destroy_spacing은 공백 의존 규칙들 뒤에 마지막으로
    적용한다."""
    rng = random.Random(f"noise-{case_id}")
    t = text
    t = apply_slang(t)
    t = corrupt_jamo(t, rng)
    t = informalize_endings(t, rng)
    t = drop_repeated_subject(t, rng)
    t = merge_sentences(t, rng)
    t = insert_emotional_aside(t, rng)
    t = insert_admin_noise(t, rng)
    t = destroy_spacing(t, rng)
    return t
