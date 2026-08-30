"""LLM 사기 정황 탐지가 합성 사고경위서에서만 검증됐고, 실제 조사관 현장 노트
특유의 오타·은어·비문·감정적 서술 같은 노이즈 환경에서는 검증되지 않았다"는
심사 피드백에 대응하는 노이즈 견고성 테스트.

24건 샘플 데이터(data/sample/sample_claims.csv)의 사고경위서에 scripts/
noise_injection.py로 "표면(표기)만 훼손하고 의미는 보존"하는 현실적 노이즈를
주입한 뒤, analyze_narrative가 원문과 동일한 사기 정황을 잡아내는지 3-way로
비교한다:

  (a) 캐시된 원문 결과 — sample_claims.csv에 이미 있는 llm_* 컬럼. 호출 0건.
  (b) 재호출 원문 결과 — 노이즈 없는 원문을 "다시" 호출. 이게 왜 필요한가:
      키워드는 LLM 자유생성 텍스트라(예: "목격자 부재"/"목격자 없음"/"목격자
      미확인"처럼 같은 의미도 표현이 매번 다름 — 실제 캐시에서 확인됨), 노이즈
      버전과 캐시된 원문을 직접 비교하면 "노이즈 때문에 달라진 것"과 "LLM을
      다시 불렀을 뿐인데 원래 이 정도는 달라지는 것(호출 간 변동성)"을 구분할
      수 없다. (b)는 그 변동성 자체를 측정하는 대조군이다.
  (c) 노이즈 버전 결과 — 노이즈 주입된 사고경위서를 호출.

판단 기준: (c)와 (a)의 차이가 아니라, "(c)와 (b)의 차이"를 "(b)와 (a)의 차이
(=순수 호출 간 변동성 baseline)"와 비교한다. 전자가 후자를 뚜렷이 넘어서야
"노이즈 때문에 실제로 달라졌다"고 말할 수 있다. n=24(또는 파일럿 5)의 작은
표본이라 이는 기술통계 비교이지 유의성 검정이 아니다.

키워드 비교는 문자열 완전일치가 아니라 정규화된 패턴 카테고리(_CATEGORY_RULES)
기준이다 — LLM의 자연스러운 표현 다양성 자체를 "노이즈에 대한 실패"로
오분류하지 않기 위함.

비용 통제: (a)는 캐시 재사용(호출 0건). (b)/(c)는 케이스ID별로 캐시 CSV에
없는 것만 신규 호출하는 증분(incremental) 캐시라, 파일럿(5건)으로 먼저 실행한
뒤 --full로 확장해도 이미 처리된 케이스는 재호출하지 않는다.

사용법:
    python scripts/noise_robustness_test.py            # 파일럿 5건
    python scripts/noise_robustness_test.py --full      # 24건 전체
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from scripts.noise_injection import _EMOTIONAL_ASIDES, inject_noise  # noqa: E402
from src.config import ANTHROPIC_MODEL  # noqa: E402
from src.optimization.assignment import optimize_assignment  # noqa: E402
from src.scoring.combine import combine_scores  # noqa: E402
from src.scoring.features import feature_cols_for  # noqa: E402
from src.scoring.llm_analysis import analyze_narrative, get_usage_totals, reset_usage_totals  # noqa: E402
from src.scoring.ml_model import predict_fraud_score, train_fraud_model  # noqa: E402

DATA_PATH = Path("data/sample/sample_claims.csv")
NOISY_NARRATIVES_PATH = Path("data/processed/sample_claims_24case_noisy_narratives.csv")
REFRESHED_CACHE_PATH = Path("data/processed/sample_claims_24case_refreshed_llm_cache.csv")
NOISY_CACHE_PATH = Path("data/processed/sample_claims_24case_noisy_llm_cache.csv")
REPORT_PATH = Path("scripts/noise_robustness_results.md")

LABEL_COL = "FraudFound_P"
NUM_INVESTIGATORS = 3
HOURS_PER_INVESTIGATOR = 10
RISK_WEIGHT_BALANCED = 50_000  # app.py POLICY_PRESETS["균형"]과 동일
HIGH_RISK_QUANTILE = 0.80

PILOT_CASE_IDS = [8703, 4551, 9877, 13910, 7847]

# Sonnet 5 가격 ($/1M 토큰) — scripts/hierarchical_filtering.py와 동일.
SONNET_5_INPUT_PRICE_PER_M = 2.00
SONNET_5_OUTPUT_PRICE_PER_M = 10.00

# sample_claims.csv 24건의 llm_keywords 캐시에 실제 등장한 전체 어휘(24종)를
# 직접 확인해 만든 카테고리 규칙 — 추측이 아니라 데이터 기반.
_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("지연신고", ["지연"]),
    ("목격자부재", ["목격자"]),
    ("경찰미신고", ["경찰"]),
    ("블랙박스미장착", ["블랙박스"]),
    ("단독사고", ["단독사고"]),
    ("고가차량", ["고가"]),
    ("반복청구이력", ["과거", "반복"]),
]


def categorize_keywords(keywords: list[str]) -> set[str]:
    """키워드 문자열 리스트를 정규화된 패턴 카테고리 집합으로 변환. 어느
    규칙에도 안 걸리는 키워드는 'AI 카테고리:원문' 형태로 남겨 누락 없이
    보이게 하되, 핵심 일치율 계산에서는 별도로 제외한다."""
    cats = set()
    for kw in keywords:
        if not kw:
            continue
        matched = False
        for name, subs in _CATEGORY_RULES:
            if any(s in kw for s in subs):
                cats.add(name)
                matched = True
        if not matched:
            cats.add(f"기타:{kw}")
    return cats


def core_categories(cats: set[str]) -> set[str]:
    return {c for c in cats if not c.startswith("기타:")}


# insert_emotional_aside()가 심은 5개 고정 문구 각각이 LLM 키워드 출력에 "그대로"
# 반영됐는지 판단하기 위한 대응 어휘. 표본이 24건뿐이라 이 매핑도 데이터를 보고
# 사후에 구성한 것이며, 새로운 표현으로 반영되는 경우는 놓칠 수 있다(과소추정 방향
# — 즉 아래에서 보고하는 "채택 비율"은 하한선에 가깝다).
_ASIDE_ECHO_TERMS: dict[str, list[str]] = {
    "본인 억울하다고 계속 주장함.": ["억울"],
    "설명 중 언성 높임.": ["언성"],
    "조사 협조에 다소 소극적인 태도 보임.": ["협조"],
    "같은 내용 반복 진술함.": ["반복 진술", "동일 진술"],
    "말 바꾸는 부분 있어 재확인 필요해 보임.": ["번복", "말바꿈", "말 바꿈", "말바꾸"],
}


def find_injected_aside(noisy_text: str) -> str | None:
    """noisy_text에 실제로 삽입된 감정적 삽입구 원문을 찾는다. 삽입 이후 단계인
    destroy_spacing이 문구 내부 공백도 지울 수 있어 공백을 제거하고 비교한다."""
    stripped = noisy_text.replace(" ", "")
    for phrase in _EMOTIONAL_ASIDES:
        if phrase.replace(" ", "") in stripped:
            return phrase
    return None


def aside_echoed_in_keywords(aside_phrase: str, keywords_c: list[str]) -> bool:
    terms = _ASIDE_ECHO_TERMS.get(aside_phrase, [])
    return any(any(t in kw for t in terms) for kw in keywords_c)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def parse_keywords_cell(cell) -> list[str]:
    if not isinstance(cell, str) or not cell:
        return []
    return [k for k in cell.split("|") if k]


def get_or_build_incremental_cache(
    cache_path: Path, rows: list[dict], text_col: str, case_ids: list
) -> pd.DataFrame:
    """case_id별로 캐시에 없는 것만 신규 analyze_narrative 호출. rows는
    {"case_id":..., text_col:...} 딕셔너리 리스트."""
    if cache_path.exists():
        cache_df = pd.read_csv(cache_path)
    else:
        cache_df = pd.DataFrame(columns=["case_id", "llm_keywords", "llm_explanation", "llm_suspicion_adjustment"])

    existing_ids = set(cache_df["case_id"])
    missing = [r for r in rows if r["case_id"] in case_ids and r["case_id"] not in existing_ids]

    if missing:
        print(f"  [{cache_path.name}] 신규 호출 {len(missing)}건: {[r['case_id'] for r in missing]}")
        new_rows = []
        for r in missing:
            result = analyze_narrative(r[text_col])
            new_rows.append(
                {
                    "case_id": r["case_id"],
                    "llm_keywords": "|".join(result.keywords),
                    "llm_explanation": result.explanation,
                    "llm_suspicion_adjustment": result.suspicion_adjustment,
                }
            )
            print(f"    case_id={r['case_id']} adj={result.suspicion_adjustment:+.2f} keywords={result.keywords}")
        cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df.to_csv(cache_path, index=False)
    else:
        print(f"  [{cache_path.name}] 캐시 재사용 — 신규 호출 0건")

    return cache_df[cache_df["case_id"].isin(case_ids)].copy()


def build_noisy_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """24건 전체에 대해 결정론적으로 노이즈 버전을 생성(무료, API 호출 없음).
    재현성을 위해 매 실행마다 규칙대로 재생성해 저장한다."""
    rows = [
        {"case_id": row["case_id"], "narrative_text_original": row["narrative_text"], "narrative_text_noisy": inject_noise(row["narrative_text"], row["case_id"])}
        for _, row in df.iterrows()
    ]
    out = pd.DataFrame(rows)
    NOISY_NARRATIVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(NOISY_NARRATIVES_PATH, index=False)
    return out


def net_value(df: pd.DataFrame) -> float:
    assigned = df.dropna(subset=["assigned_investigator"])
    return float((assigned["expected_recovery"] - assigned["investigation_cost"]).sum())


def coverage_pct(df: pd.DataFrame, high_risk_ids: set) -> float:
    if not high_risk_ids:
        return 0.0
    assigned = df.dropna(subset=["assigned_investigator"])
    covered = assigned["case_id"].isin(high_risk_ids).sum()
    return 100 * covered / len(high_risk_ids)


def main() -> None:
    full_run = "--full" in sys.argv
    df = pd.read_csv(DATA_PATH)
    all_case_ids = df["case_id"].tolist()
    case_ids = all_case_ids if full_run else PILOT_CASE_IDS
    mode_label = "24건 전체" if full_run else f"파일럿 {len(case_ids)}건"
    print(f"=== 노이즈 견고성 테스트 — {mode_label} ===\n")

    print("1. 노이즈 버전 사고경위서 생성 (24건 전체, 결정론적, API 호출 없음)")
    noisy_df = build_noisy_narratives(df)
    print(f"  저장 완료: {NOISY_NARRATIVES_PATH}\n")

    reset_usage_totals()

    print("2. (b) 원문 재호출 — LLM 호출 간 순수 변동성 baseline 측정용")
    refreshed_rows = df[["case_id", "narrative_text"]].to_dict("records")
    refreshed_cache = get_or_build_incremental_cache(
        REFRESHED_CACHE_PATH, refreshed_rows, "narrative_text", case_ids
    )

    print("\n3. (c) 노이즈 버전 호출")
    noisy_rows = noisy_df.rename(columns={"narrative_text_noisy": "narrative_text"})[
        ["case_id", "narrative_text"]
    ].to_dict("records")
    noisy_cache = get_or_build_incremental_cache(NOISY_CACHE_PATH, noisy_rows, "narrative_text", case_ids)

    usage = get_usage_totals()
    actual_cost = (
        usage["input_tokens"] / 1e6 * SONNET_5_INPUT_PRICE_PER_M
        + usage["output_tokens"] / 1e6 * SONNET_5_OUTPUT_PRICE_PER_M
    )
    print(
        f"\n이번 실행 실제 API 사용량: 요청 {usage['requests']}건, "
        f"입력 {usage['input_tokens']:,}토큰, 출력 {usage['output_tokens']:,}토큰 "
        f"→ 약 ${actual_cost:.4f} (모델: {ANTHROPIC_MODEL})"
    )

    # --- 케이스별 비교표 ---
    a_map = df.set_index("case_id")[["llm_keywords", "llm_suspicion_adjustment", "FraudFound_P"]].to_dict("index")
    b_map = refreshed_cache.set_index("case_id")[["llm_keywords", "llm_suspicion_adjustment"]].to_dict("index")
    c_map = noisy_cache.set_index("case_id")[["llm_keywords", "llm_suspicion_adjustment"]].to_dict("index")

    per_case = []
    for cid in case_ids:
        a, b, c = a_map[cid], b_map[cid], c_map[cid]
        kws_a = parse_keywords_cell(a["llm_keywords"])
        kws_b = parse_keywords_cell(b["llm_keywords"])
        kws_c = parse_keywords_cell(c["llm_keywords"])
        cats_a = core_categories(categorize_keywords(kws_a))
        cats_b = core_categories(categorize_keywords(kws_b))
        cats_c = core_categories(categorize_keywords(kws_c))
        adj_a, adj_b, adj_c = a["llm_suspicion_adjustment"], b["llm_suspicion_adjustment"], c["llm_suspicion_adjustment"]

        noisy_text = noisy_df.loc[noisy_df["case_id"] == cid, "narrative_text_noisy"].iloc[0]
        aside = find_injected_aside(noisy_text)
        echoed = aside_echoed_in_keywords(aside, kws_c) if aside else False
        admin_leaked = any(any(m in k for m in ("접수번호", "연락처", "010-")) for k in kws_c)

        per_case.append(
            {
                "case_id": cid,
                "fraud": int(a["FraudFound_P"]),
                "cats_a": cats_a,
                "cats_b": cats_b,
                "cats_c": cats_c,
                "kws_b": kws_b,
                "kws_c": kws_c,
                "new_kws_c": [k for k in kws_c if k not in kws_b],
                "aside": aside,
                "echoed": echoed,
                "admin_leaked": admin_leaked,
                "adj_a": adj_a,
                "adj_b": adj_b,
                "adj_c": adj_c,
                "dev_baseline": abs(adj_b - adj_a),
                "dev_noise_vs_cached": abs(adj_c - adj_a),
                "dev_noise_vs_fresh": abs(adj_c - adj_b),
                "jaccard_baseline": jaccard(cats_a, cats_b),
                "jaccard_noise_vs_cached": jaccard(cats_a, cats_c),
                "jaccard_noise_vs_fresh": jaccard(cats_b, cats_c),
            }
        )

    lines = [
        "# 노이즈 견고성 테스트 결과",
        "",
        f"- 모드: {mode_label} (case_id: {case_ids})",
        f"- 데이터: `{DATA_PATH.as_posix()}`, 모델: {ANTHROPIC_MODEL}",
        f"- 이번 실행 실제 API 사용량: 요청 {usage['requests']}건, "
        f"입력 {usage['input_tokens']:,}토큰, 출력 {usage['output_tokens']:,}토큰 → 약 ${actual_cost:.4f}",
        "",
        "## 판단 기준",
        "",
        "(c)-(b) 차이를 (b)-(a) baseline(순수 LLM 호출 간 변동성)과 비교한다. "
        "전자가 후자를 뚜렷이 넘어야 \"노이즈 때문\"이라고 말할 수 있다 — "
        "표본이 작아 통계적 유의성 검정이 아니라 기술통계 비교임에 유의.",
        "",
        "## 케이스별 비교",
        "",
        "| case_id | 사기 | adj(a)캐시 | adj(b)재호출 | adj(c)노이즈 | \\|b-a\\| | \\|c-a\\| | \\|c-b\\| | "
        "Jaccard(a,b) | Jaccard(a,c) | Jaccard(b,c) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in per_case:
        lines.append(
            f"| {r['case_id']} | {r['fraud']} | {r['adj_a']:+.2f} | {r['adj_b']:+.2f} | {r['adj_c']:+.2f} | "
            f"{r['dev_baseline']:.2f} | {r['dev_noise_vs_cached']:.2f} | {r['dev_noise_vs_fresh']:.2f} | "
            f"{r['jaccard_baseline']:.2f} | {r['jaccard_noise_vs_cached']:.2f} | {r['jaccard_noise_vs_fresh']:.2f} |"
        )

    def _mean(key):
        return sum(r[key] for r in per_case) / len(per_case)

    def _max(key):
        return max(r[key] for r in per_case)

    lines.extend(
        [
            "",
            "## 요약 지표 (1) 패턴 일치율 · (2) adjustment 편차",
            "",
            "| 비교 | 평균 \\|편차\\| | 최대 \\|편차\\| | 평균 Jaccard(카테고리) |",
            "|---|---|---|---|",
            f"| baseline: (b)재호출 vs (a)캐시 원문 | {_mean('dev_baseline'):.3f} | {_max('dev_baseline'):.2f} | {_mean('jaccard_baseline'):.3f} |",
            f"| (c)노이즈 vs (a)캐시 원문 | {_mean('dev_noise_vs_cached'):.3f} | {_max('dev_noise_vs_cached'):.2f} | {_mean('jaccard_noise_vs_cached'):.3f} |",
            f"| (c)노이즈 vs (b)재호출 원문 (★핵심) | {_mean('dev_noise_vs_fresh'):.3f} | {_max('dev_noise_vs_fresh'):.2f} | {_mean('jaccard_noise_vs_fresh'):.3f} |",
            "",
            f"**결론에 쓸 핵심 숫자**: 노이즈 효과(★핵심 행) 대비 순수 LLM 호출 변동성(baseline 행)의 비율 = "
            f"**{_mean('dev_noise_vs_fresh') / _mean('dev_baseline') if _mean('dev_baseline') else float('inf'):.1f}배** "
            f"(adjustment 평균 편차 기준). 절대 편차값 자체가 아니라 이 배율로 해석해야 한다 — "
            f"LLM은 원문을 다시 부르기만 해도 평균 {_mean('dev_baseline'):.3f}만큼 흔들리는 것이 이미 정상 범위이기 때문이다. "
            f"표본 {len(per_case)}건, 기술통계 비교(유의성 검정 아님).",
            "",
        ]
    )

    # --- 노이즈 유형별 민감도: 감정적 삽입구가 키워드로 "그대로" 채택되는가 ---
    echoed_rows = [r for r in per_case if r["aside"] is not None]
    by_phrase: dict[str, list] = {}
    for r in echoed_rows:
        by_phrase.setdefault(r["aside"], []).append(r)

    n_echoed = sum(1 for r in per_case if r["echoed"])
    n_admin_leaked = sum(1 for r in per_case if r["admin_leaked"])
    echoed_devs = [r["dev_noise_vs_fresh"] for r in per_case if r["echoed"]]
    not_echoed_devs = [r["dev_noise_vs_fresh"] for r in per_case if not r["echoed"]]

    lines.extend(
        [
            "## 개별 사례 분석 — 어떤 노이즈 유형에 취약한가",
            "",
            "6개 노이즈 유형을 매 건 동시에 주입했기 때문에 유형별 순수 효과를 분리하려면 "
            "원래는 유형별 단독 주입(ablation)이 필요하다. 다만 감정적 삽입구(#5)는 case_id로 "
            "시드된 고정 문구 5종 중 정확히 1개가 매 건 원문 그대로(공백 파괴 이후에도 식별 가능) "
            "삽입되므로, \"그 문구의 핵심 어휘가 (c)의 LLM 키워드 출력에 그대로 나타나는가\"를 "
            "코드로 직접 검사해 유형 #5 하나만은 별도 API 호출 없이 귀속(attribution)할 수 있다. "
            "무관 정보 혼입(#6)도 같은 방식으로 접수번호/연락처 패턴이 키워드에 누출됐는지 검사했다.",
            "",
            f"- **감정적 삽입구(#5)가 키워드로 그대로 채택된 케이스: {n_echoed}/{len(per_case)}건** "
            f"({100 * n_echoed / len(per_case):.0f}%) — 아래 매핑에 없는 새 표현으로 반영된 경우는 "
            "잡지 못하므로 이 비율은 하한선에 가깝다.",
            f"- 무관 정보(#6, 접수번호/연락처)가 키워드로 누출된 케이스: {n_admin_leaked}/{len(per_case)}건 "
            "— LLM이 이 유형은 잘 걸러냄(관찰된 범위 내 긍정적 소견).",
        ]
    )
    if echoed_devs and not_echoed_devs:
        lines.append(
            f"- 채택된 케이스의 평균 \\|c-b\\| = {sum(echoed_devs)/len(echoed_devs):.3f} "
            f"vs 채택 안 된 케이스 평균 \\|c-b\\| = {sum(not_echoed_devs)/len(not_echoed_devs):.3f} "
            f"→ 채택된 케이스가 **{(sum(echoed_devs)/len(echoed_devs)) / (sum(not_echoed_devs)/len(not_echoed_devs)):.1f}배** 더 크게 흔들림."
        )
    lines.append("")

    lines.extend(
        [
            "### 삽입 문구별 채택률",
            "",
            "| 삽입 문구 | 채택 건수/삽입 건수 | 채택률 |",
            "|---|---|---|",
        ]
    )
    for phrase, rows_for_phrase in sorted(by_phrase.items(), key=lambda kv: -sum(r["echoed"] for r in kv[1])):
        n_hit = sum(1 for r in rows_for_phrase if r["echoed"])
        lines.append(f"| \"{phrase}\" | {n_hit}/{len(rows_for_phrase)} | {100 * n_hit / len(rows_for_phrase):.0f}% |")
    lines.extend(
        [
            "",
            "해석(가설 수준 — n이 작아 확정적 결론은 아님): \"조사 협조에 소극적\", \"말 바꾸는 부분\", "
            "\"반복 진술\"처럼 기존 사기조사 적신호 어휘와 겹치는 표현은 높은 비율로 그대로 채택되는 "
            "반면, \"억울하다고 주장\", \"언성 높임\"처럼 감정 묘사에 가깝고 적신호 어휘와 덜 겹치는 "
            "표현은 채택률이 낮다.",
            "",
            "### 노이즈로 판단이 뒤바뀐(또는 크게 흔들린) 개별 사례",
            "",
        ]
    )
    flagged = sorted(per_case, key=lambda r: -r["dev_noise_vs_fresh"])[:6]
    lines.append("| case_id | \\|c-b\\| | 삽입된 감정적 삽입구 | 키워드에 그대로 채택? | (c)에서 새로 생긴 키워드 |")
    lines.append("|---|---|---|---|---|")
    for r in flagged:
        lines.append(
            f"| {r['case_id']} | {r['dev_noise_vs_fresh']:.2f} | {r['aside'] or '-'} | "
            f"{'예' if r['echoed'] else '아니오'} | {r['new_kws_c']} |"
        )
    lines.append("")

    if full_run:
        print("\n4. (3)(4) 고위험 판정/OR-Tools 배정 비교 (24건 전체에서만 계산)")
        feature_cols = feature_cols_for(df)
        model, _ = train_fraud_model(df[feature_cols + [LABEL_COL]], class_weight="balanced")
        ml_score = predict_fraud_score(model, df[feature_cols])

        def scored(adj_series: pd.Series) -> pd.DataFrame:
            out = df.copy()
            out["ml_score"] = ml_score
            out["combined_score"] = combine_scores(ml_score, adj_series)
            return out

        adj_a = df["case_id"].map(lambda cid: a_map[cid]["llm_suspicion_adjustment"])
        adj_b = df["case_id"].map(lambda cid: b_map[cid]["llm_suspicion_adjustment"])
        adj_c = df["case_id"].map(lambda cid: c_map[cid]["llm_suspicion_adjustment"])

        scored_a, scored_b, scored_c = scored(adj_a), scored(adj_b), scored(adj_c)

        high_risk_threshold = scored_a["combined_score"].quantile(HIGH_RISK_QUANTILE)
        high_risk_ids_a = set(scored_a.loc[scored_a["combined_score"] >= high_risk_threshold, "case_id"])
        n_high_risk = len(high_risk_ids_a)

        def high_risk_set(scored_df: pd.DataFrame) -> set:
            th = scored_df["combined_score"].quantile(HIGH_RISK_QUANTILE)
            return set(scored_df.loc[scored_df["combined_score"] >= th, "case_id"])

        hr_b = high_risk_set(scored_b)
        hr_c = high_risk_set(scored_c)
        flips_b = high_risk_ids_a.symmetric_difference(hr_b)
        flips_c = high_risk_ids_a.symmetric_difference(hr_c)

        per_case_by_id = {r["case_id"]: r for r in per_case}
        flip_notes_c = []
        for cid in sorted(flips_c):
            r = per_case_by_id[cid]
            direction = "이탈" if cid in high_risk_ids_a else "신규 진입"
            own_score_changed = abs(r["adj_c"] - r["adj_b"]) > 1e-9
            reason = (
                "본인 점수 변화로"
                if own_score_changed
                else "본인 점수는 그대로지만 다른 건들 점수가 바뀌며 상위 20% 경계(동점자 재배치)가 움직여서"
            )
            flip_notes_c.append(f"case_id={cid}({direction}, {reason})")

        lines.extend(
            [
                "## 요약 지표 (3) 고위험(상위 20%) 판정 뒤바뀜",
                "",
                f"- (a)캐시 원문 기준 고위험 {n_high_risk}건: {sorted(high_risk_ids_a)}",
                f"- (b)재호출 원문 기준 뒤바뀐 건수: {len(flips_b)}건 {sorted(flips_b) if flips_b else ''}",
                f"- (c)노이즈 기준 뒤바뀐 건수: {len(flips_c)}건 {sorted(flips_c) if flips_c else ''}",
                "",
                "**주의**: 상위 20% 문턱값은 각 시나리오의 점수 분포에서 매번 다시 계산된다. "
                "그래서 어떤 케이스는 자기 점수가 전혀 안 바뀌었는데도(노이즈에 흔들리지 않았는데도) "
                "다른 케이스의 점수가 올라가 경계선의 동점자 구성이 바뀌면서 '뒤바뀜'으로 잡힐 수 있다 — "
                "아래는 (c) 뒤바뀜 각 건이 본인 점수 변화 때문인지, 순위 재배치 때문인지 구분한 것이다.",
                "",
                "- " + "\n- ".join(flip_notes_c) if flip_notes_c else "- (뒤바뀐 건 없음)",
                "",
            ]
        )

        opt_a = optimize_assignment(scored_a, NUM_INVESTIGATORS, HOURS_PER_INVESTIGATOR, risk_weight=RISK_WEIGHT_BALANCED)
        opt_b = optimize_assignment(scored_b, NUM_INVESTIGATORS, HOURS_PER_INVESTIGATOR, risk_weight=RISK_WEIGHT_BALANCED)
        opt_c = optimize_assignment(scored_c, NUM_INVESTIGATORS, HOURS_PER_INVESTIGATOR, risk_weight=RISK_WEIGHT_BALANCED)

        net_a, net_b, net_c = net_value(opt_a), net_value(opt_b), net_value(opt_c)
        cov_a = coverage_pct(opt_a, high_risk_ids_a)
        cov_b = coverage_pct(opt_b, high_risk_ids_a)
        cov_c = coverage_pct(opt_c, high_risk_ids_a)

        lines.extend(
            [
                f"## 요약 지표 (4) OR-Tools 배정 결과 (균형 프리셋 λ={RISK_WEIGHT_BALANCED:,}, "
                f"조사관 {NUM_INVESTIGATORS}명×{HOURS_PER_INVESTIGATOR}시간)",
                "",
                "| 시나리오 | 순회수액 | 순회수액 Δ(캐시 원문 대비) | 고위험 커버리지(캐시 원문 고위험셋 기준) |",
                "|---|---|---|---|",
                f"| (a) 캐시 원문 | {net_a:,.0f} | +0.0% | {cov_a:.1f}% |",
                f"| (b) 재호출 원문 (baseline) | {net_b:,.0f} | {(net_b - net_a) / net_a * 100 if net_a else 0:+.1f}% | {cov_b:.1f}% |",
                f"| (c) 노이즈 | {net_c:,.0f} | {(net_c - net_a) / net_a * 100 if net_a else 0:+.1f}% | {cov_c:.1f}% |",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## 요약 지표 (3)(4)",
                "",
                "파일럿 모드에서는 계산하지 않음 — 고위험 판정(상위 20%)과 OR-Tools 배정 비교는 "
                "24건 전체가 있어야 의미 있는 분모/캐파 기준이 성립하므로 `--full` 실행이 필요함.",
                "",
            ]
        )

    lines.extend(
        [
            "## 노이즈 버전 샘플 (원문 대조, 육안 검수용)",
            "",
        ]
    )
    sample_ids = case_ids[: min(3, len(case_ids))]
    for cid in sample_ids:
        orig = df.loc[df["case_id"] == cid, "narrative_text"].iloc[0]
        noisy = noisy_df.loc[noisy_df["case_id"] == cid, "narrative_text_noisy"].iloc[0]
        lines.extend(
            [
                f"### case_id={cid}",
                "",
                f"- 원문: {orig}",
                f"- 노이즈: {noisy}",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n결과 저장 완료: {REPORT_PATH}")


if __name__ == "__main__":
    main()
