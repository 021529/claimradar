"""계층적 필터링(Hierarchical Filtering) — "전체 건 LLM 호출" vs "ML 점수로 걸러낸 상위 건만
LLM 호출" 두 파이프라인을 24건 데모 데이터(data/sample/sample_claims.csv)로 스코어링→LLM분석→
OR-Tools 최적배정까지 끝까지 돌려 비교한다.

배경: 실서비스 규모(수천~수만 건)에서 사건 전량에 LLM(analyze_narrative)을 호출하면 API 비용이
선형으로 증가한다. "이 규모에서 비용 감당되냐"는 질문에 답하기 위해, 비용이 거의 0인 1차
ML 스코어링(RandomForest)으로 전체 건을 먼저 스코어링하고, 2차로 ML 점수 기준 상위 건만
LLM 호출, 나머지는 LLM을 스킵(중립값 llm_suspicion_adjustment=0.0 — combine.py/app.py가
이미 쓰는 "LLM 미사용 시 0" 컨벤션과 동일)하는 2단계 필터링이 (a) LLM 호출 수를 얼마나
줄이고 (b) 그 때문에 놓치는 고위험 건이 있는지 (c) 최종 OR-Tools 배정 결과(순회수액/고위험
커버리지)가 실제로 나빠지는지를 측정한다.

API 비용: sample_claims.csv에는 llm_suspicion_adjustment 캐시가 없어 실제 Claude 호출이
필요하다. 동일 사건은 필터링 강도를 몇 개를 스윕하든 같은 응답이므로, 24건 전체에 대해
**1회만** analyze_narrative를 호출해 로컬 캐시(LLM_CACHE_PATH)에 저장하고, 이후 모든
스윕 포인트(상위 N% / 절대 임계값)는 이 캐시에서 "선택된 사건은 실값, 탈락한 사건은
0(스킵)"으로 재구성만 한다 — API 재호출 없음. 캐시 파일이 이미 있으면 그것을 그대로
재사용한다 (API 호출 0건).

결과는 콘솔에 표를 바로 찍지 않고 scripts/hierarchical_filtering_results.md 에 마크다운 표로
저장한다 (한글 레이블이 섞인 고정폭 콘솔 표는 폭이 깨지기 쉬우므로, 파일에는 순수 마크다운
파이프 표만 남긴다).

2026-08-30 ML 백본 보강(범주형 One-Hot + class_weight='balanced') 이후 재실행됨 —
ml_score 분포가 바뀌어 THRESHOLD_LEVELS/RISK_WEIGHTS를 새 분포·새 λ 대표값에
맞게 조정했다. 옛 5피처 모델 기준 수치는 scripts/ml_backbone_reexperiment_2026-08-30.md
참고.

사용법:
    python scripts/hierarchical_filtering.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import ANTHROPIC_MODEL  # noqa: E402
from src.optimization.assignment import optimize_assignment  # noqa: E402
from src.scoring.combine import combine_scores  # noqa: E402
from src.scoring.features import feature_cols_for  # noqa: E402
from src.scoring.llm_analysis import analyze_narrative, get_usage_totals, reset_usage_totals  # noqa: E402
from src.scoring.ml_model import predict_fraud_score, train_fraud_model  # noqa: E402

DATA_PATH = Path("data/sample/sample_claims.csv")
LLM_CACHE_PATH = Path("data/processed/sample_claims_24case_llm_cache.csv")
REPORT_PATH = Path("scripts/hierarchical_filtering_results.md")

LABEL_COL = "FraudFound_P"

NUM_INVESTIGATORS = 3
HOURS_PER_INVESTIGATOR = 10
HIGH_RISK_QUANTILE = 0.80

# Sonnet 5 가격 ($/1M 토큰) — 캐시를 새로 만들 때 실제 소요 비용을 계산해 보여주기 위함.
SONNET_5_INPUT_PRICE_PER_M = 2.00
SONNET_5_OUTPUT_PRICE_PER_M = 10.00

TOP_PCT_LEVELS = [0.10, 0.20, 0.30, 0.50, 1.00]
THRESHOLD_LEVELS = [0.3, 0.5, 0.7, 0.85]

# OR-Tools 목적함수는 (기대회수액-조사비용) + λ*combined_score 이므로 λ=0이면
# combined_score(=LLM 분석 결과)가 목적함수에서 완전히 사라져, 필터링으로 LLM을
# 몇 건 스킵하든 최종 배정이 똑같아진다 — "필터링이 최종 품질에 영향 없다"가
# 아니라 λ=0이라는 조건 자체가 그 결과를 강제한 것이다. 그래서 λ=0(회수액만
# 볼 때)과, app.py POLICY_PRESETS["균형"]=50,000(순회수액이 회수액 우선/고위험
# 차단 우선의 산술 중간값에 가장 가까운 전환 구간 대표값, 위험도가 실제로
# 반영될 때) 둘 다 비교해 이 효과를 명시적으로 보여준다.
RISK_WEIGHTS = [0, 50000]


def get_or_build_llm_cache(df: pd.DataFrame) -> pd.DataFrame:
    """24건 전체에 대한 실제 LLM 분석 결과를 1회만 호출해 로컬 캐시에 저장, 이후 재사용."""
    if LLM_CACHE_PATH.exists():
        print(f"캐시 재사용: {LLM_CACHE_PATH} (API 호출 0건)\n")
        return pd.read_csv(LLM_CACHE_PATH)

    print(f"캐시 없음 — {len(df)}건 전체에 대해 실제 Claude API({ANTHROPIC_MODEL}) 호출 시작 (1회성, 이후 재사용)...")
    reset_usage_totals()
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        result = analyze_narrative(row["narrative_text"])
        rows.append(
            {
                "case_id": row["case_id"],
                "llm_keywords": "|".join(result.keywords),
                "llm_explanation": result.explanation,
                "llm_suspicion_adjustment": result.suspicion_adjustment,
            }
        )
        print(f"  [{i + 1}/{len(df)}] case_id={row['case_id']} adj={result.suspicion_adjustment:+.2f}")

    cache_df = pd.DataFrame(rows)
    LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_csv(LLM_CACHE_PATH, index=False)

    usage = get_usage_totals()
    cost = (
        usage["input_tokens"] / 1e6 * SONNET_5_INPUT_PRICE_PER_M
        + usage["output_tokens"] / 1e6 * SONNET_5_OUTPUT_PRICE_PER_M
    )
    print(
        f"캐시 저장 완료: {LLM_CACHE_PATH}\n"
        f"실제 사용량: 요청 {usage['requests']}건, 입력 {usage['input_tokens']:,}토큰, "
        f"출력 {usage['output_tokens']:,}토큰 → 약 ${cost:.4f}\n"
    )
    return cache_df


def build_ml_scores(df: pd.DataFrame) -> pd.Series:
    feature_cols = feature_cols_for(df)
    model, _ = train_fraud_model(df[feature_cols + [LABEL_COL]], class_weight="balanced")
    return predict_fraud_score(model, df[feature_cols])


def select_mask_top_pct(ml_score: pd.Series, pct: float) -> pd.Series:
    """ml_score 상위 pct 비율(0~1)에 해당하는 사건만 True (LLM 호출 대상)."""
    if pct >= 1.0:
        return pd.Series(True, index=ml_score.index)
    n_selected = max(1, int(len(ml_score) * pct))
    rank_cutoff = ml_score.sort_values(ascending=False).iloc[n_selected - 1]
    return ml_score >= rank_cutoff


def select_mask_threshold(ml_score: pd.Series, threshold: float) -> pd.Series:
    """ml_score >= threshold 인 사건만 True (LLM 호출 대상)."""
    return ml_score >= threshold


def apply_hierarchical_filtering(
    df: pd.DataFrame, ml_score: pd.Series, llm_cache: pd.DataFrame, mask: pd.Series
) -> pd.DataFrame:
    """mask=True인 사건만 캐시된 실제 LLM 결과를 쓰고, 나머지는 LLM을 스킵(중립값 0)한다."""
    result = df.copy()
    result["ml_score"] = ml_score
    merged = result.merge(llm_cache, on="case_id", how="left")
    mask_values = mask.reindex(merged.index).values
    merged["llm_suspicion_adjustment"] = merged["llm_suspicion_adjustment"].where(mask_values, 0.0)
    merged["llm_keywords"] = merged["llm_keywords"].where(mask_values, "")
    merged["llm_explanation"] = merged["llm_explanation"].where(
        mask_values, "LLM 분석 스킵 (계층적 필터링 — ML 점수 기준 하위 우선순위)"
    )
    merged["combined_score"] = combine_scores(merged["ml_score"], merged["llm_suspicion_adjustment"])
    return merged


def net_value(df: pd.DataFrame) -> float:
    assigned = df.dropna(subset=["assigned_investigator"])
    return float((assigned["expected_recovery"] - assigned["investigation_cost"]).sum())


def coverage_pct(df: pd.DataFrame, high_risk_ids: set, n_high_risk: int) -> float:
    if n_high_risk == 0:
        return 0.0
    assigned = df.dropna(subset=["assigned_investigator"])
    covered = assigned["case_id"].isin(high_risk_ids).sum()
    return 100 * covered / n_high_risk


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    # ML 스코어링 모델 보강으로 sample_claims.csv에 이제 llm_* 컬럼이 자체
    # 포함돼 있다 — 이 스크립트는 mask 기준으로 선택적 병합을 직접 하므로,
    # 미리 지워야 아래 merge(llm_cache)에서 _x/_y 접미사 충돌이 안 생긴다.
    df = df.drop(columns=["llm_keywords", "llm_suspicion_adjustment", "llm_explanation"], errors="ignore")
    ml_score = build_ml_scores(df)
    llm_cache = get_or_build_llm_cache(df)

    # 기준선(ground truth): 전체 건 LLM 호출(필터링 미적용) 파이프라인.
    # 고위험 판정과 "회수액/커버리지가 얼마나 나빠지는가"의 비교 기준점을 여기서 고정한다.
    full = apply_hierarchical_filtering(df, ml_score, llm_cache, pd.Series(True, index=df.index))
    high_risk_threshold = full["combined_score"].quantile(HIGH_RISK_QUANTILE)
    high_risk_ids = set(full.loc[full["combined_score"] >= high_risk_threshold, "case_id"])
    n_high_risk = len(high_risk_ids)

    lines = [
        "# 계층적 필터링(Hierarchical Filtering) 비교 결과",
        "",
        f"- 데이터: `{DATA_PATH.as_posix()}` ({len(df)}건)",
        f"- 고위험(상위 20%, 전체-LLM 기준) {n_high_risk}건",
        f"- 조사관 {NUM_INVESTIGATORS}명 x {HOURS_PER_INVESTIGATOR}시간",
        "",
    ]

    for risk_weight in RISK_WEIGHTS:
        lines.extend(build_comparison_section(risk_weight, df, ml_score, llm_cache, high_risk_ids, n_high_risk))

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"결과 저장 완료: {REPORT_PATH}")


def build_comparison_section(
    risk_weight: float,
    df: pd.DataFrame,
    ml_score: pd.Series,
    llm_cache: pd.DataFrame,
    high_risk_ids: set,
    n_high_risk: int,
) -> list[str]:
    full = apply_hierarchical_filtering(df, ml_score, llm_cache, pd.Series(True, index=df.index))
    full_optimized = optimize_assignment(full, NUM_INVESTIGATORS, HOURS_PER_INVESTIGATOR, risk_weight=risk_weight)
    full_net = net_value(full_optimized)
    full_cov = coverage_pct(full_optimized, high_risk_ids, n_high_risk)

    rows = []

    def evaluate(label: str, mask: pd.Series) -> None:
        n_calls = int(mask.sum())
        reduction_pct = 100 * (1 - n_calls / len(df))
        selected_ids = set(df.loc[mask.values, "case_id"])
        recall_pct = 100 * len(high_risk_ids & selected_ids) / n_high_risk if n_high_risk else 0.0

        filtered = apply_hierarchical_filtering(df, ml_score, llm_cache, mask)
        optimized = optimize_assignment(filtered, NUM_INVESTIGATORS, HOURS_PER_INVESTIGATOR, risk_weight=risk_weight)
        opt_net = net_value(optimized)
        opt_cov = coverage_pct(optimized, high_risk_ids, n_high_risk)
        net_delta_pct = (opt_net - full_net) / full_net * 100 if full_net else float("nan")

        rows.append(
            {
                "label": label,
                "llm_calls": n_calls,
                "reduction_pct": reduction_pct,
                "recall_pct": recall_pct,
                "net": opt_net,
                "net_delta_pct": net_delta_pct,
                "coverage_pct": opt_cov,
                "coverage_delta": opt_cov - full_cov,
            }
        )

    for pct in TOP_PCT_LEVELS:
        evaluate(f"상위 {int(pct * 100)}%", select_mask_top_pct(ml_score, pct))

    for th in THRESHOLD_LEVELS:
        evaluate(f"임계값 {th}", select_mask_threshold(ml_score, th))

    section = [
        f"## λ={risk_weight:,}",
        "",
        f"전체 LLM 호출(필터링 없음): 순회수액 **{full_net:,.0f}** / 고위험 커버리지 **{full_cov:.1f}%** "
        f"/ LLM 호출 {len(df)}/{len(df)}건",
        "",
        "| 필터링 강도 | LLM 호출 수 | 호출 절감율 | 고위험 탐지율(재현율) | "
        "순회수액 | 순회수액 Δ(전체호출 대비) | 고위험 커버리지 | 커버리지 Δ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        section.append(
            f"| {r['label']} | {r['llm_calls']}/{len(df)} | {r['reduction_pct']:.1f}% | "
            f"{r['recall_pct']:.1f}% | {r['net']:,.0f} | {r['net_delta_pct']:+.1f}% | "
            f"{r['coverage_pct']:.1f}% | {r['coverage_delta']:+.1f}%p |"
        )
    section.append("")
    return section


if __name__ == "__main__":
    main()
