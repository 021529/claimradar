import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MAX_UPLOAD_ROWS, SCORE_WEIGHT_LLM, SCORE_WEIGHT_ML
from src.data.loader import load_sample_claims, load_uploaded_claims, normalize_columns
from src.guide.guide_generator import generate_investigation_checklist
from src.optimization.assignment import optimize_assignment
from src.optimization.baseline import baseline_assignment
from src.scoring.combine import combine_scores
from src.scoring.features import CORE_NUMERIC_FEATURE_COLS, feature_cols_for
from src.scoring.hierarchical_filtering import select_mask_threshold, select_mask_top_pct
from src.scoring.llm_analysis import analyze_narrative
from src.scoring.ml_model import predict_fraud_score, train_fraud_model

LABEL_COL = "FraudFound_P"
REQUIRED_COLUMNS = CORE_NUMERIC_FEATURE_COLS + [
    LABEL_COL,
    "case_id",
    "narrative_text",
    "expected_hours",
    "expected_recovery",
    "investigation_cost",
]

st.set_page_config(page_title="클레임레이더", page_icon="🔍", layout="wide")

st.title("🔍 클레임레이더 (가제)")
st.caption("AI 스코어링 + 수리적 최적화로 보험사기 조사 우선순위를 배정하는 의사결정 지원 시스템")

for key in [
    "claims_df",
    "scored_df",
    "model",
    "feature_cols",
    "baseline_result",
    "optimized_result",
]:
    st.session_state.setdefault(key, None)
st.session_state.setdefault("guide_checklists", {})
st.session_state.setdefault("guide_feedback_log", [])


def _validate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def _feature_evidence_table(row: pd.Series, all_cases: pd.DataFrame, numeric_feature_cols: list[str]) -> pd.DataFrame:
    """이 사건의 수치형 피처 값 대 전체 평균 비교표."""
    means = all_cases[numeric_feature_cols].mean()
    return pd.DataFrame(
        {
            "피처": numeric_feature_cols,
            "이 사건 값": [row.get(c) for c in numeric_feature_cols],
            "전체 평균": means.round(1).values,
        }
    )


def _top_feature_importance_table(model, top_n: int = 10) -> pd.DataFrame:
    """모델(전처리+RandomForest 파이프라인) 상위 피처 중요도 (범주형은 One-Hot 항목별)."""
    prep = model.named_steps["prep"]
    clf = model.named_steps["clf"]
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    importances = clf.feature_importances_
    table = pd.DataFrame({"피처": names, "모델 중요도": (importances * 100).round(1)})
    table = table.sort_values("모델 중요도", ascending=False).head(top_n)
    table["모델 중요도"] = table["모델 중요도"].astype(str) + "%"
    return table


def _net_value(df: pd.DataFrame) -> float:
    assigned = df.dropna(subset=["assigned_investigator"])
    return float((assigned["expected_recovery"] - assigned["investigation_cost"]).sum())


def _high_risk_ids(scored_df: pd.DataFrame, quantile: float = 0.80) -> set:
    threshold = scored_df["combined_score"].quantile(quantile)
    return set(scored_df.loc[scored_df["combined_score"] >= threshold, "case_id"])


def _coverage_pct(df: pd.DataFrame, high_risk_ids: set) -> float:
    if not high_risk_ids:
        return 0.0
    assigned = df.dropna(subset=["assigned_investigator"])
    covered = assigned["case_id"].isin(high_risk_ids).sum()
    return 100 * covered / len(high_risk_ids)


# 24건 샘플 데이터(조사관 3명x10시간) 기준 lambda_sweep_24case.py 재실행(2026-08-30,
# ML 백본 보강+실제 LLM 캐시 반영) 결과로 정한 대표값. 이 데이터는 4단계 계단을
# 그린다: λ=0~42,800(순회수액 160,000/커버리지 50.0%) → 43,000~65,000(140,150/66.7%)
# → 67,000~79,000(119,950/83.3%, λ=66,000 정확히 한 지점만 130,500으로 튀는 불안정한
# 예외 있어 대표값에서 제외) → 80,000 이상(109,950/100.0%로 수렴, 500,000까지 평평함
# 확인). "균형"은 순회수액이 회수액 우선(160,000)과 고위험 차단 우선(109,950)의 산술
# 중간값(134,975)에 가장 가까운 2단계 전환 구간(140,150)에서 골라 세 프리셋의
# 순회수액이 160,000/140,150/109,950으로 고르게 분포하도록 했다(3단계 83.3%
# 구간을 쓰면 119,950으로 고위험 차단 우선과 거의 붙어 "균형"이라는 이름과
# 어긋난다). 실제 카드에 표시되는 숫자는 이 값이 아니라 현재 업로드된 데이터로
# 매번 다시 계산한 값이다 (아래 _compute_policy_previews).
POLICY_PRESETS = {
    "회수액 우선": 0,
    "균형": 50_000,
    "고위험 차단 우선": 80_000,
}
POLICY_DESCRIPTIONS = {
    "회수액 우선": "순회수액을 최대화합니다. 고위험 건이라도 기대회수액이 낮으면 배정에서 밀릴 수 있습니다.",
    "균형": "순회수액과 고위험 커버리지를 절충합니다.",
    "고위험 차단 우선": "고위험 건을 최대한 빠짐없이 조사합니다. 순회수액은 가장 낮아질 수 있습니다.",
}


# 계층적 필터링 미리보기용 고정 기준 — scripts/hierarchical_filtering.py 검증 조건과
# 동일(조사관 3명×10시간, λ=50,000=균형 프리셋)으로 고정해 검증 문서 수치와 어긋나지
# 않게 한다. 3.에서 사용자가 실제로 고른 조사관 수/λ와는 별개의 참고용 시뮬레이션이다.
HIERARCHICAL_FILTERING_PREVIEW_INVESTIGATORS = 3
HIERARCHICAL_FILTERING_PREVIEW_HOURS = 10
HIERARCHICAL_FILTERING_PREVIEW_RISK_WEIGHT = POLICY_PRESETS["균형"]


# analyze_narrative 프롬프트 구조(고정 지시문 + 사고경위서 본문 평균 270자 + 도구 스키마)
# 기준 대략적 추정치 — 실측이 아니라 프롬프트 길이 기반 추정이므로 범위로 제시한다.
# 실제 요금 상수는 scripts/hierarchical_filtering.py의 Sonnet 5 가격과 동일.
LLM_CALL_EST_INPUT_TOKENS = (450, 700)
LLM_CALL_EST_OUTPUT_TOKENS = (80, 180)
SONNET_5_INPUT_PRICE_PER_M = 2.00
SONNET_5_OUTPUT_PRICE_PER_M = 10.00


def _estimate_llm_cost_usd(n_calls: int) -> tuple[float, float]:
    """n_calls건에 대해 실제 API를 호출할 경우의 대략적 비용 범위(추정치, USD)."""
    low = n_calls * (
        LLM_CALL_EST_INPUT_TOKENS[0] / 1e6 * SONNET_5_INPUT_PRICE_PER_M
        + LLM_CALL_EST_OUTPUT_TOKENS[0] / 1e6 * SONNET_5_OUTPUT_PRICE_PER_M
    )
    high = n_calls * (
        LLM_CALL_EST_INPUT_TOKENS[1] / 1e6 * SONNET_5_INPUT_PRICE_PER_M
        + LLM_CALL_EST_OUTPUT_TOKENS[1] / 1e6 * SONNET_5_OUTPUT_PRICE_PER_M
    )
    return low, high


@st.cache_data(show_spinner=False)
def _hierarchical_filtering_preview(claims_df: pd.DataFrame, mask_kind: str, mask_value: float) -> dict:
    """필터링 적용 시 실제로 LLM 호출될 건수를 계산한다. 데이터에 LLM 분석 캐시가 이미
    있으면(=샘플 데이터) 균형 프리셋(λ=50,000)·조사관 3명×10시간 기준 순회수액/고위험
    커버리지 변화까지 실측해 net_delta_pct/coverage_delta에 채운다. 캐시가 없는 데이터는
    스킵될 행의 LLM 결과를 미리 알 수 없어(=API를 안 부르고는 손실 시뮬레이션 자체가
    불가능해) 그 두 값을 None으로 둔다 — 이 경우에도 호출 건수/비용 추정은 가능하다."""
    df = claims_df.copy()
    feature_cols = feature_cols_for(df)
    model, _ = train_fraud_model(df[feature_cols + [LABEL_COL]], class_weight="balanced")
    ml_score = predict_fraud_score(model, df[feature_cols])
    mask = (
        select_mask_top_pct(ml_score, mask_value)
        if mask_kind == "top_pct"
        else select_mask_threshold(ml_score, mask_value)
    )
    n_total = len(df)
    n_selected = int(mask.sum())
    result = {
        "n_total": n_total,
        "n_selected": n_selected,
        "reduction_pct": 100 * (1 - n_selected / n_total) if n_total else 0.0,
        "net_delta_pct": None,
        "coverage_delta": None,
    }

    if "llm_suspicion_adjustment" not in df.columns:
        return result

    def _build(selected_mask: pd.Series) -> pd.DataFrame:
        out = df.copy()
        out["ml_score"] = ml_score
        adjustment = out["llm_suspicion_adjustment"].where(selected_mask, 0.0)
        out["combined_score"] = combine_scores(ml_score, adjustment)
        return out

    full = _build(pd.Series(True, index=df.index))
    filtered = _build(mask)
    high_risk_ids = _high_risk_ids(full)
    full_opt = optimize_assignment(
        full,
        HIERARCHICAL_FILTERING_PREVIEW_INVESTIGATORS,
        HIERARCHICAL_FILTERING_PREVIEW_HOURS,
        risk_weight=HIERARCHICAL_FILTERING_PREVIEW_RISK_WEIGHT,
    )
    filtered_opt = optimize_assignment(
        filtered,
        HIERARCHICAL_FILTERING_PREVIEW_INVESTIGATORS,
        HIERARCHICAL_FILTERING_PREVIEW_HOURS,
        risk_weight=HIERARCHICAL_FILTERING_PREVIEW_RISK_WEIGHT,
    )
    full_net, filtered_net = _net_value(full_opt), _net_value(filtered_opt)
    full_cov = _coverage_pct(full_opt, high_risk_ids)
    filtered_cov = _coverage_pct(filtered_opt, high_risk_ids)
    result["net_delta_pct"] = (filtered_net - full_net) / full_net * 100 if full_net else 0.0
    result["coverage_delta"] = filtered_cov - full_cov
    return result


@st.cache_data(show_spinner="정책별 예상 결과 계산 중...")
def _compute_policy_previews(
    scored_df: pd.DataFrame, num_investigators: int, hours_per_investigator: int
) -> dict:
    """현재 데이터·캐파 기준으로 3개 프리셋 λ 각각을 실제로 최적화 실행해 미리보기 수치를 만든다."""
    high_risk_ids = _high_risk_ids(scored_df)
    previews = {}
    for label, lam in POLICY_PRESETS.items():
        optimized = optimize_assignment(
            scored_df, num_investigators, hours_per_investigator, risk_weight=lam
        )
        previews[label] = {
            "net": _net_value(optimized),
            "coverage": _coverage_pct(optimized, high_risk_ids),
        }
    return previews


# λ=0~42,800은 baseline과 완전히 동일한 결과(순회수액 160,000/커버리지 50.0%)라
# 아직 "전환"이 시작되지 않은 회수액 우선 구간이다. 43,000부터 실제로 결과가
# 바뀌기 시작해(scripts/lambda_sweep_24case.py 참고) 80,000(고위험 차단 우선
# 프리셋, 커버리지 100% 수렴)까지가 균형(전환) 구간이다.
_RECOVERY_PLATEAU_END = 43_000


def _lambda_regime(risk_weight: int) -> str:
    if risk_weight < _RECOVERY_PLATEAU_END:
        return "회수액 우선 구간"
    if risk_weight < POLICY_PRESETS["고위험 차단 우선"]:
        return "균형(전환) 구간"
    return "고위험 차단 우선 구간"


st.header("1. 청구 데이터")
source = st.radio("데이터 소스", ["샘플로 체험하기", "CSV 업로드"], horizontal=True)

if source == "샘플로 체험하기":
    if st.button("샘플 데이터 불러오기"):
        st.session_state.claims_df = normalize_columns(load_sample_claims())
else:
    uploaded = st.file_uploader("청구 데이터 CSV 업로드", type="csv")
    if uploaded is not None:
        try:
            uploaded_df = normalize_columns(load_uploaded_claims(uploaded))
            if len(uploaded_df) > MAX_UPLOAD_ROWS:
                st.error(
                    f"업로드한 CSV가 {len(uploaded_df)}건으로, 이 데모의 상한({MAX_UPLOAD_ROWS}건)을 "
                    "초과합니다. 이는 버그가 아니라 MVP 데모 환경의 API 비용 통제를 위한 설계상 "
                    "제한입니다 — 이 앱은 퍼블릭 URL로 배포되어 있어 상한이 없으면 대용량 CSV "
                    "하나로 실제 Claude API 크레딧이 소진될 수 있습니다. 실 도입 시에는 계층적 "
                    f"필터링(2.의 토글 참고)과 배치 처리로 대규모 데이터를 처리합니다. {MAX_UPLOAD_ROWS}건 "
                    "이하로 잘라서 다시 업로드해 주세요."
                )
                st.session_state.claims_df = None
            else:
                st.session_state.claims_df = uploaded_df
        except Exception as e:
            st.error(f"CSV 파일을 읽지 못했습니다: {e}")
            st.session_state.claims_df = None

if st.session_state.claims_df is not None:
    missing = _validate_columns(st.session_state.claims_df)
    if missing:
        st.warning(f"다음 필수 컬럼이 없어 이후 단계가 실패할 수 있습니다: {', '.join(missing)}")
    st.dataframe(st.session_state.claims_df, use_container_width=True)

st.header("2. 이상거래 스코어링")
if st.session_state.claims_df is None:
    st.info("먼저 청구 데이터를 불러오세요.")
else:
    use_filtering = st.toggle(
        "계층적 필터링 사용 (ML 점수로 1차 선별 후 상위 건만 LLM 호출)",
        value=False,
        key="use_hierarchical_filtering",
        help=(
            "꺼두면 전체 건에 LLM 분석을 수행합니다(기본 데모 경로). 켜면 비용이 거의 0인 "
            "ML 스코어링으로 먼저 거른 뒤, 선택된 건만 LLM을 호출해 API 비용을 절감합니다."
        ),
    )

    filter_kind = "top_pct"
    filter_value = 1.0  # 필터링을 끄면 전체 호출과 동일 — 아래 preview 계산에 그대로 재사용
    if use_filtering:
        filter_kind_label = st.radio(
            "필터링 기준",
            ["ML 점수 상위 N%", "ML 점수 절대 임계값"],
            horizontal=True,
            key="filter_kind_ui",
        )
        filter_kind = "top_pct" if filter_kind_label == "ML 점수 상위 N%" else "threshold"
        if filter_kind == "top_pct":
            filter_value = (
                st.slider(
                    "상위 몇 %만 LLM 호출",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=10,
                    key="filter_top_pct_ui",
                )
                / 100
            )
        else:
            filter_value = st.slider(
                "ML 점수 임계값 이상만 LLM 호출",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="filter_threshold_ui",
            )

    try:
        preview = _hierarchical_filtering_preview(st.session_state.claims_df, filter_kind, filter_value)
    except Exception:
        preview = None

    has_cache = preview is not None and preview["net_delta_pct"] is not None

    if use_filtering and preview is not None:
        if has_cache:
            net_delta = preview["net_delta_pct"]
            cov_delta = preview["coverage_delta"]
            base_msg = (
                f"균형 프리셋(λ={HIERARCHICAL_FILTERING_PREVIEW_RISK_WEIGHT:,})·조사관 "
                f"{HIERARCHICAL_FILTERING_PREVIEW_INVESTIGATORS}명×{HIERARCHICAL_FILTERING_PREVIEW_HOURS}시간 "
                f"기준 실측: 전체 {preview['n_total']}건 중 {preview['n_selected']}건만 LLM 호출"
                f"(호출 {preview['reduction_pct']:.1f}% 절감)"
            )
            if abs(net_delta) < 0.5 and abs(cov_delta) < 0.5:
                st.success(f"{base_msg} — 전체 LLM 호출 대비 순회수액·고위험 커버리지 변화 없음(무손실 구간).")
            else:
                st.warning(
                    f"{base_msg}이지만, 전체 LLM 호출 대비 순회수액 {net_delta:+.1f}% / "
                    f"고위험 커버리지 {cov_delta:+.1f}%p 변화가 관측된 구간입니다. 결과를 확인한 뒤 "
                    "진행하거나 안전 구간(예: 상위 50%, 임계값 0.3)으로 조정하세요."
                )
        else:
            st.info(
                "이 데이터에는 아직 LLM 분석 캐시가 없어 필터링 손실 여부를 사전에 시뮬레이션할 수 "
                "없습니다(스킵될 건의 LLM 결과를 미리 알 수 없기 때문). 24건 샘플 데이터·균형 프리셋"
                "(λ=50,000) 기준 검증에서는 상위 10/50/100%·임계값 0.3/0.5/0.85가 절감 최대 91.7%까지 "
                "무손실이었고, 상위 20~30%·임계값 0.7 부근은 순회수액 -6.9%/커버리지 +16.7%p로 결과가 "
                "달라졌습니다(scripts/hierarchical_filtering_results.md 참고) — 이 데이터에 그대로 "
                "적용되지는 않으니 참고만 하세요."
            )

    if preview is not None and not has_cache:
        cost_low, cost_high = _estimate_llm_cost_usd(preview["n_selected"])
        st.warning(
            f"⚠️ 이 데이터에는 LLM 분석 캐시가 없어 '스코어링 실행' 시 {preview['n_selected']}건에 대해 "
            f"실제 Claude API 호출이 발생합니다. 예상 비용 약 ${cost_low:.2f}~${cost_high:.2f}"
            "(프롬프트 길이 기반 추정치, 실측 아님). 업로드는 API 비용 통제를 위해 최대 "
            f"{MAX_UPLOAD_ROWS}건으로 제한되어 있습니다."
        )

    if st.button("스코어링 실행"):
        missing = _validate_columns(st.session_state.claims_df)
        if missing:
            st.error(f"필수 컬럼이 없어 스코어링을 실행할 수 없습니다: {', '.join(missing)}")
        else:
            df = st.session_state.claims_df.copy()
            try:
                feature_cols = feature_cols_for(df)
                with st.spinner("ML 모델 학습 중..."):
                    model, _ = train_fraud_model(df[feature_cols + [LABEL_COL]], class_weight="balanced")
                    ml_score = predict_fraud_score(model, df[feature_cols])

                if use_filtering:
                    mask = (
                        select_mask_top_pct(ml_score, filter_value)
                        if filter_kind == "top_pct"
                        else select_mask_threshold(ml_score, filter_value)
                    )
                else:
                    mask = pd.Series(True, index=df.index)
                n_total = len(df)
                n_selected = int(mask.sum())
                reduction_pct = 100 * (1 - n_selected / n_total) if n_total else 0.0
                skip_note = "LLM 분석 스킵 (계층적 필터링 — ML 점수 기준 하위 우선순위)"

                if "llm_suspicion_adjustment" in df.columns:
                    st.info("사전 계산된 LLM 분석 결과를 재사용합니다 (추가 API 호출 없음).")
                    llm_adjustment = df["llm_suspicion_adjustment"].where(mask, 0.0)
                    # 캐시 CSV는 키워드를 "지연신고|블랙박스 미장착" 형태의 파이프 구분
                    # 문자열로 저장한다(scripts/hierarchical_filtering.py의
                    # get_or_build_llm_cache와 동일 포맷) — 리스트로 되돌려야 문자
                    # 단위가 아니라 키워드 단위로 join/순회된다.
                    llm_keywords_cached = (
                        df.get("llm_keywords", pd.Series("", index=df.index))
                        .fillna("")
                        .apply(lambda s: [k for k in s.split("|") if k] if isinstance(s, str) else [])
                    )
                    llm_keywords = pd.Series(
                        [kw if sel else [] for kw, sel in zip(llm_keywords_cached, mask)], index=df.index
                    )
                    llm_explanation = df.get("llm_explanation", pd.Series("", index=df.index)).where(
                        mask, skip_note
                    )
                    if use_filtering:
                        st.caption(
                            f"⚠️ 캐시된 분석 결과를 사용 중이라 실제로는 API가 호출되지 않았습니다. "
                            f"캐시 없이 이 필터링 설정으로 처음 분석했다면 전체 {n_total}건 중 "
                            f"{n_selected}건만 호출되어 {reduction_pct:.1f}% 절감됐을 것입니다."
                        )
                else:
                    llm_adjustment = pd.Series(0.0, index=df.index)
                    llm_keywords = pd.Series([[]] * len(df), index=df.index)
                    llm_explanation = pd.Series("", index=df.index)
                    with st.spinner("사고경위서 이상징후 분석 중..."):
                        selected_indices = df.index[mask]
                        progress = st.progress(0.0)
                        for i, idx in enumerate(selected_indices):
                            result = analyze_narrative(df.loc[idx, "narrative_text"])
                            llm_adjustment.loc[idx] = result.suspicion_adjustment
                            llm_keywords.loc[idx] = result.keywords
                            llm_explanation.loc[idx] = result.explanation
                            progress.progress((i + 1) / max(len(selected_indices), 1))
                        progress.empty()
                    llm_explanation.loc[~mask] = skip_note
                    if use_filtering:
                        st.success(
                            f"전체 {n_total}건 중 {n_selected}건만 LLM 분석 수행 "
                            f"(호출 {reduction_pct:.1f}% 절감)."
                        )

                df["ml_score"] = ml_score
                df["llm_keywords"] = llm_keywords
                df["llm_explanation"] = llm_explanation
                df["llm_suspicion_adjustment"] = llm_adjustment
                df["llm_status"] = mask.map({True: "분석", False: "스킵(필터링)"})
                df["combined_score"] = combine_scores(ml_score, llm_adjustment)
                st.session_state.model = model
                st.session_state.feature_cols = feature_cols
                st.session_state.scored_df = df.sort_values("combined_score", ascending=False)
            except Exception as e:
                st.error(f"스코어링 중 오류가 발생했습니다: {e}")

if st.session_state.scored_df is not None:
    st.dataframe(
        st.session_state.scored_df[
            [
                "case_id",
                "combined_score",
                "llm_status",
                "llm_keywords",
                "llm_explanation",
                "expected_hours",
                "expected_recovery",
            ]
        ],
        use_container_width=True,
    )

st.header("3. 조사 리소스")
num_investigators = st.slider("조사관 수", min_value=1, max_value=10, value=3)
hours_per_investigator = st.slider("1인당 가용 시간", min_value=1, max_value=40, value=10)

st.subheader("조사 정책 선택")
st.session_state.setdefault("risk_weight_slider", POLICY_PRESETS["회수액 우선"])

if st.session_state.scored_df is not None:
    try:
        policy_previews = _compute_policy_previews(
            st.session_state.scored_df, num_investigators, hours_per_investigator
        )
    except Exception:
        policy_previews = None
else:
    policy_previews = None

current_lambda = st.session_state.risk_weight_slider
policy_cols = st.columns(3)
for col, label in zip(policy_cols, POLICY_PRESETS):
    with col:
        selected = current_lambda == POLICY_PRESETS[label]
        st.markdown(f"**{'✅ ' if selected else ''}{label}**")
        st.caption(POLICY_DESCRIPTIONS[label])
        if policy_previews is not None:
            st.metric("예상 순회수액", f"{policy_previews[label]['net']:,.0f}원")
            st.metric("고위험 커버리지", f"{policy_previews[label]['coverage']:.1f}%")
        else:
            st.metric("예상 순회수액", "—")
            st.metric("고위험 커버리지", "—")
            st.caption("스코어링 실행 후 예상 결과가 표시됩니다.")
        if st.button(
            "이 정책 선택",
            key=f"policy_btn_{label}",
            type="primary" if selected else "secondary",
            use_container_width=True,
        ):
            st.session_state.risk_weight_slider = POLICY_PRESETS[label]
            st.rerun()

with st.expander("🔧 고급 설정: λ 직접 조정"):
    st.slider(
        "위험도 가중치 (λ)",
        min_value=0,
        max_value=150_000,
        step=1000,
        key="risk_weight_slider",
        help="목적함수 = 기대회수액 - 조사비용 + λ × 위험도스코어 × 배정여부. "
        "λ=0이면 기존 방식(회수액 극대화)과 동일합니다. 데모 데이터 기준 λ=43,000부터 "
        "회수액을 포기하는 대신 고위험 건 커버리지가 올라가기 시작해 λ=80,000에서 "
        "baseline 수준(100%)까지 도달합니다 — 그 이하(수만 단위 미만)에서는 거의 "
        "변화가 없을 수 있습니다. 상한(150,000)을 넘겨도 결과는 80,000과 동일하게 "
        "평평합니다(0~500,000 전체 스윕으로 확인됨 — scripts/lambda_sweep_24case.py). "
        "위 프리셋 값과 다르게 조정하면 세 정책 카드 중 어느 것도 선택됨(✅) 표시가 "
        "없는 상태가 됩니다.",
    )
    st.caption(
        f"현재 λ={st.session_state.risk_weight_slider:,} → "
        f"{_lambda_regime(st.session_state.risk_weight_slider)}"
    )

risk_weight = st.session_state.risk_weight_slider
if risk_weight == 0:
    risk_desc = "회수액 중심 (λ=0, 기존 방식과 동일)"
else:
    risk_desc = (
        f"회수액과 위험도스코어를 함께 고려합니다 (λ={risk_weight:,}, 클수록 위험도 비중 ↑). "
        "실제 회수액·고위험 커버리지 변화는 사건 구성과 조사 캐파에 따라 달라지므로, "
        "아래 배정 실행 결과에서 직접 확인하세요."
    )
st.caption(f"💡 현재 설정: {risk_desc}")

st.header("4. 최적 배정 산출")
if st.session_state.scored_df is None:
    st.info("먼저 스코어링을 실행하세요.")
elif st.button("배정 실행"):
    cases = st.session_state.scored_df
    try:
        with st.spinner("최적 배정 계산 중..."):
            st.session_state.baseline_result = baseline_assignment(
                cases, num_investigators, hours_per_investigator
            )
            st.session_state.optimized_result = optimize_assignment(
                cases, num_investigators, hours_per_investigator, risk_weight=risk_weight
            )
    except Exception as e:
        st.error(f"배정 계산 중 오류가 발생했습니다: {e}")
        st.session_state.baseline_result = None
        st.session_state.optimized_result = None

if st.session_state.optimized_result is not None:
    baseline = st.session_state.baseline_result
    optimized = st.session_state.optimized_result

    # 고위험 건 = 전체 스코어링된 사건 중 combined_score 상위 20%
    high_risk_case_ids = _high_risk_ids(st.session_state.scored_df)
    n_high_risk_total = len(high_risk_case_ids)

    baseline_net = _net_value(baseline)
    optimized_net = _net_value(optimized)
    improvement_pct = (
        (optimized_net - baseline_net) / baseline_net * 100 if baseline_net else 0.0
    )

    baseline_coverage = _coverage_pct(baseline, high_risk_case_ids)
    optimized_coverage = _coverage_pct(optimized, high_risk_case_ids)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline 순회수액", f"{baseline_net:,.0f}")
    col2.metric("최적화 순회수액", f"{optimized_net:,.0f}")
    col3.metric("효율 개선", f"{improvement_pct:+.1f}%")

    col4, col5, col6 = st.columns(3)
    col4.metric("Baseline 고위험 커버리지", f"{baseline_coverage:.1f}%")
    col5.metric(
        "최적화 고위험 커버리지",
        f"{optimized_coverage:.1f}%",
        delta=f"{optimized_coverage - baseline_coverage:+.1f}%p",
    )
    col6.metric("고위험 건 수(상위 20%)", f"{n_high_risk_total}건")

    # 목적함수는 "combined_score 합"을 최대화할 뿐 "상위 20% 건수 커버리지"를 직접
    # 최대화하지 않는다 - 표본이 작을수록 이 둘이 어긋나 커버리지가 baseline보다
    # 낮게 나올 수 있다. 이 사실을 미리 예측해 문구로 박아두는 대신, 매번 실제
    # 계산된 숫자를 그대로 요약해 보여줘서 안내문과 결과가 어긋나지 않게 한다.
    coverage_delta = optimized_coverage - baseline_coverage
    if optimized_net >= baseline_net and coverage_delta >= 0:
        st.success(
            f"이번 조건(조사관 {num_investigators}명 × {hours_per_investigator}시간, λ={risk_weight:,})에서는 "
            f"최적화가 baseline 대비 회수액 {improvement_pct:+.1f}%, 고위험 커버리지 {coverage_delta:+.1f}%p로 "
            "둘 다 같거나 개선됐습니다."
        )
    elif coverage_delta < 0:
        st.warning(
            f"이번 조건(조사관 {num_investigators}명 × {hours_per_investigator}시간, λ={risk_weight:,})에서는 "
            f"최적화의 고위험 커버리지가 baseline보다 {coverage_delta:.1f}%p 낮습니다. "
            "목적함수는 위험도스코어의 '합'을 최대화할 뿐 '상위 20% 건수'를 직접 최대화하지 않고, "
            "baseline은 위험도스코어만 보는 그리디 방식이라 캐파가 작을수록(=사건 수가 적을수록) "
            "이런 역전이 나타날 수 있습니다. λ를 더 키우거나 조사 캐파를 늘려보세요."
        )
    else:
        st.info(
            f"이번 조건에서는 최적화가 baseline 대비 회수액 {improvement_pct:+.1f}%, "
            f"고위험 커버리지 {coverage_delta:+.1f}%p 입니다."
        )

    # Baseline=회색(중립)/최적화=강조색으로 "어느 쪽이 우리 방식인지"를 구분한다.
    # 이 매핑은 방식(누구)에 대한 색이지 값의 좋고 나쁨에 대한 색이 아니므로, 두
    # 차트 모두 동일하게 적용한다 — 커버리지가 baseline보다 낮게 나와도(정책상
    # 자연스러운 트레이드오프이지 오류가 아님) 최적화 막대를 부정적 색으로
    # 바꾸지 않는다.
    METHOD_COLORS = {"Baseline": "#9CA3AF", "최적화": "#2563EB"}

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        chart_df = pd.DataFrame(
            {"방식": ["Baseline", "최적화"], "순회수액": [baseline_net, optimized_net]}
        )
        fig_net = px.bar(
            chart_df,
            x="방식",
            y="순회수액",
            color="방식",
            color_discrete_map=METHOD_COLORS,
            text="순회수액",
            title="순회수액 비교",
        )
        fig_net.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_net.update_layout(showlegend=False, yaxis_title="순회수액 (원)")
        st.plotly_chart(fig_net, use_container_width=True)

    with chart_col2:
        coverage_chart_df = pd.DataFrame(
            {"방식": ["Baseline", "최적화"], "고위험 커버리지": [baseline_coverage, optimized_coverage]}
        )
        fig_cov = px.bar(
            coverage_chart_df,
            x="방식",
            y="고위험 커버리지",
            color="방식",
            color_discrete_map=METHOD_COLORS,
            text="고위험 커버리지",
            title="고위험 커버리지 비교",
        )
        fig_cov.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_cov.update_layout(showlegend=False, yaxis_title="고위험 커버리지 (%)", yaxis_range=[0, 100])
        st.plotly_chart(fig_cov, use_container_width=True)

    st.subheader("조사관별 담당 사건")
    for inv in sorted(optimized["assigned_investigator"].dropna().unique()):
        cases_for_inv = optimized[optimized["assigned_investigator"] == inv]
        with st.expander(f"조사관 {int(inv) + 1} — {len(cases_for_inv)}건"):
            st.dataframe(
                cases_for_inv[["case_id", "combined_score", "expected_hours", "expected_recovery"]],
                use_container_width=True,
            )

st.header("5. 사건 상세 & AI 조사 가이드")
if st.session_state.optimized_result is None:
    st.info("먼저 배정을 실행하세요.")
else:
    assigned = st.session_state.optimized_result.dropna(subset=["assigned_investigator"])
    if assigned.empty:
        st.info("배정된 사건이 없습니다.")
    else:
        selected_case = st.selectbox("사건 선택", assigned["case_id"].tolist())
        row = assigned[assigned["case_id"] == selected_case].iloc[0]

        with st.expander("📊 스코어 근거", expanded=True):
            ml_score = row.get("ml_score")
            llm_adj = row.get("llm_suspicion_adjustment")
            if pd.notna(ml_score) and pd.notna(llm_adj):
                st.markdown(
                    f"**결합 스코어 {row['combined_score']:.2f}** = "
                    f"ML 스코어 {ml_score:.2f} × {SCORE_WEIGHT_ML} + "
                    f"LLM 보정 {llm_adj:+.2f} × {SCORE_WEIGHT_LLM}"
                )
                st.caption(f"LLM 가중치 {SCORE_WEIGHT_LLM}은 MVP 시연용 설계값이며, 탐지 성능 최적화로 검증된 값이 아닙니다.")
            else:
                st.markdown(f"**결합 스코어 {row['combined_score']:.2f}**")
            if st.session_state.model is not None:
                st.caption("이 사건 값 vs 전체 평균 (수치형 피처)")
                st.dataframe(
                    _feature_evidence_table(row, st.session_state.scored_df, CORE_NUMERIC_FEATURE_COLS),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("모델 상위 피처 중요도 (범주형은 One-Hot 항목별)")
                st.dataframe(
                    _top_feature_importance_table(st.session_state.model),
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("📝 LLM 요약", expanded=True):
            st.write(row.get("narrative_text", "") or "(사고경위서 없음)")
            if row.get("llm_status") == "스킵(필터링)":
                st.caption(
                    "🔕 계층적 필터링으로 이 사건은 LLM 분석이 스킵되었습니다 (ML 점수 기준 하위 "
                    "우선순위) — 아래 키워드/의심 사유는 산출되지 않았고, 스코어는 ML 점수만 반영합니다."
                )
            else:
                keywords = row.get("llm_keywords") or []
                st.markdown(f"**사고 정황 키워드:** {', '.join(keywords) if keywords else '없음'}")
                explanation = row.get("llm_explanation") or ""
                if explanation:
                    st.markdown(f"**의심 사유:** {explanation}")

        st.subheader("✅ 조사 체크리스트")
        cached_checklist = st.session_state.guide_checklists.get(selected_case)
        button_label = "조사 가이드 다시 생성" if cached_checklist else "조사 가이드 생성"
        if st.button(button_label):
            keywords = row.get("llm_keywords") or []
            try:
                with st.spinner("조사 체크리스트 생성 중..."):
                    st.session_state.guide_checklists[selected_case] = generate_investigation_checklist(
                        row.get("narrative_text", ""), keywords
                    )
                cached_checklist = st.session_state.guide_checklists[selected_case]
            except Exception as e:
                st.error(f"조사 가이드 생성 중 오류가 발생했습니다: {e}")

        if cached_checklist:
            for i, item in enumerate(cached_checklist):
                st.checkbox(item, key=f"guide_{selected_case}_{i}")
        else:
            st.caption("이 사건에 대한 조사 가이드가 아직 생성되지 않았습니다.")

        st.divider()
        st.subheader("AI 판단 피드백 (Human-in-the-loop)")
        st.caption(
            "AI 판단(스코어링·조사 가이드)에 오류(오탐)가 있다고 판단되면 피드백을 남겨주세요. "
            "최종 조사 여부 판단 권한은 조사관에게 있으며, AI 판단은 참고용입니다."
        )
        fb_col1, fb_col2 = st.columns(2)
        if fb_col1.button("동의 — 조사 필요성이 인정됨", key=f"fb_agree_{selected_case}"):
            st.session_state.guide_feedback_log.append({"case_id": selected_case, "decision": "agree"})
            st.toast(f"'{selected_case}' 사건 피드백(동의)이 접수되었습니다.")
        if fb_col2.button("비동의 — 오탐으로 판단됨", key=f"fb_disagree_{selected_case}"):
            st.session_state.guide_feedback_log.append({"case_id": selected_case, "decision": "disagree"})
            st.toast(f"'{selected_case}' 사건 피드백(비동의)이 접수되었습니다.")

        agree_count = sum(1 for f in st.session_state.guide_feedback_log if f["decision"] == "agree")
        disagree_count = sum(1 for f in st.session_state.guide_feedback_log if f["decision"] == "disagree")
        st.markdown(f"**이번 세션 피드백: 동의 {agree_count}건 / 비동의 {disagree_count}건**")
        st.caption(
            "MVP에서는 세션 단위로만 집계되며, 실 도입 시 조사 이력 DB에 축적되어 모델 재학습에 활용됩니다."
        )
