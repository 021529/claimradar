import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import SCORE_WEIGHT_LLM, SCORE_WEIGHT_ML
from src.data.loader import load_sample_claims, load_uploaded_claims, normalize_columns
from src.guide.guide_generator import generate_investigation_checklist
from src.optimization.assignment import optimize_assignment
from src.optimization.baseline import baseline_assignment
from src.scoring.combine import combine_scores
from src.scoring.llm_analysis import analyze_narrative
from src.scoring.ml_model import predict_fraud_score, train_fraud_model

CORE_NUMERIC_FEATURE_COLS = [
    "driver_age",
    "vehicle_price",
    "deductible",
    "driver_rating",
    "past_number_of_claims",
]
# 범주형 피처 보강(scripts/ml_model_comparison_results.md에서 AUC-ROC 0.53→0.80,
# PR-AUC 0.07→0.17로 검증됨). scripts/prepare_app_dataset.py가 만드는 컬럼과
# 정확히 일치해야 한다. CSV 업로드 등 옛 5피처 스키마와도 호환되도록 "존재하는
# 것만" 쓰는 선택적 목록으로 둔다 — 없어도 스코어링은 CORE_NUMERIC_FEATURE_COLS만으로 동작한다.
OPTIONAL_CATEGORICAL_FEATURE_COLS = [
    "age_of_vehicle_rank",
    "age_of_policyholder_rank",
    "days_policy_accident_rank",
    "days_policy_claim_rank",
    "number_of_suppliments_rank",
    "address_change_claim_rank",
    "number_of_cars_rank",
    "Make",
    "Sex",
    "MaritalStatus",
    "Fault",
    "PolicyType",
    "VehicleCategory",
    "AccidentArea",
    "PoliceReportFiled",
    "WitnessPresent",
    "AgentType",
    "BasePolicy",
]
LABEL_COL = "FraudFound_P"
REQUIRED_COLUMNS = CORE_NUMERIC_FEATURE_COLS + [
    LABEL_COL,
    "case_id",
    "narrative_text",
    "expected_hours",
    "expected_recovery",
    "investigation_cost",
]


def _feature_cols_for(df: pd.DataFrame) -> list[str]:
    """업로드된 데이터에 실제로 있는 컬럼만 골라 사용 (옛 5피처 스키마와 호환)."""
    return CORE_NUMERIC_FEATURE_COLS + [c for c in OPTIONAL_CATEGORICAL_FEATURE_COLS if c in df.columns]

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


# 24건 샘플 데이터(조사관 3명x10시간) 기준 lambda_sweep_24case.py 재현 결과로 정한 대표값.
# 0~90,000 사이는 완만한 전환 구간, 90,000~450,000은 평평, 480,000부터 baseline과
# 동일(고위험 커버리지 100%)로 수렴해 90,000을 "균형", 480,000을 "고위험 차단 우선"의
# 대표값으로 삼았다. 실제 카드에 표시되는 숫자는 이 값이 아니라 현재 업로드된 데이터로
# 매번 다시 계산한 값이다 (아래 _compute_policy_previews).
POLICY_PRESETS = {
    "회수액 우선": 0,
    "균형": 90_000,
    "고위험 차단 우선": 480_000,
}
POLICY_DESCRIPTIONS = {
    "회수액 우선": "순회수액을 최대화합니다. 고위험 건이라도 기대회수액이 낮으면 배정에서 밀릴 수 있습니다.",
    "균형": "순회수액과 고위험 커버리지를 절충합니다.",
    "고위험 차단 우선": "고위험 건을 최대한 빠짐없이 조사합니다. 순회수액은 가장 낮아질 수 있습니다.",
}


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


def _lambda_regime(risk_weight: int) -> str:
    if risk_weight <= 0:
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
            st.session_state.claims_df = normalize_columns(load_uploaded_claims(uploaded))
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
elif st.button("스코어링 실행"):
    missing = _validate_columns(st.session_state.claims_df)
    if missing:
        st.error(f"필수 컬럼이 없어 스코어링을 실행할 수 없습니다: {', '.join(missing)}")
    else:
        df = st.session_state.claims_df.copy()
        try:
            feature_cols = _feature_cols_for(df)
            with st.spinner("ML 모델 학습 중..."):
                model, _ = train_fraud_model(df[feature_cols + [LABEL_COL]], class_weight="balanced")
                ml_score = predict_fraud_score(model, df[feature_cols])

            if "llm_suspicion_adjustment" in df.columns:
                st.info("사전 계산된 LLM 분석 결과를 재사용합니다 (추가 API 호출 없음).")
                llm_adjustment = df["llm_suspicion_adjustment"]
                llm_keywords = df.get("llm_keywords", pd.Series("", index=df.index))
                llm_explanation = df.get("llm_explanation", pd.Series("", index=df.index))
            else:
                llm_adjustment = pd.Series(0.0, index=df.index)
                llm_keywords = pd.Series([[]] * len(df), index=df.index)
                llm_explanation = pd.Series("", index=df.index)
                with st.spinner("사고경위서 이상징후 분석 중..."):
                    progress = st.progress(0.0)
                    for i, (idx, row) in enumerate(df.iterrows()):
                        result = analyze_narrative(row["narrative_text"])
                        llm_adjustment.loc[idx] = result.suspicion_adjustment
                        llm_keywords.loc[idx] = result.keywords
                        llm_explanation.loc[idx] = result.explanation
                        progress.progress((i + 1) / len(df))
                    progress.empty()

            df["ml_score"] = ml_score
            df["llm_keywords"] = llm_keywords
            df["llm_explanation"] = llm_explanation
            df["llm_suspicion_adjustment"] = llm_adjustment
            df["combined_score"] = combine_scores(ml_score, llm_adjustment)
            st.session_state.model = model
            st.session_state.feature_cols = feature_cols
            st.session_state.scored_df = df.sort_values("combined_score", ascending=False)
        except Exception as e:
            st.error(f"스코어링 중 오류가 발생했습니다: {e}")

if st.session_state.scored_df is not None:
    st.dataframe(
        st.session_state.scored_df[
            ["case_id", "combined_score", "llm_keywords", "llm_explanation", "expected_hours", "expected_recovery"]
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
        max_value=500_000,
        step=1000,
        key="risk_weight_slider",
        help="목적함수 = 기대회수액 - 조사비용 + λ × 위험도스코어 × 배정여부. "
        "λ=0이면 기존 방식(회수액 극대화)과 동일합니다. 데모 데이터 기준 λ가 "
        "충분히 크면(수십만 단위) 회수액을 포기하는 대신 고위험 건 커버리지가 "
        "baseline 수준까지 올라갑니다 — 낮은 λ(수만 단위 이하)에서는 거의 "
        "변화가 없을 수 있습니다. 위 프리셋 값과 다르게 조정하면 세 정책 카드 "
        "중 어느 것도 선택됨(✅) 표시가 없는 상태가 됩니다.",
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

    chart_df = pd.DataFrame(
        {"방식": ["Baseline", "최적화"], "순회수액": [baseline_net, optimized_net]}
    )
    st.plotly_chart(px.bar(chart_df, x="방식", y="순회수액"), use_container_width=True)

    coverage_chart_df = pd.DataFrame(
        {"방식": ["Baseline", "최적화"], "고위험 커버리지(%)": [baseline_coverage, optimized_coverage]}
    )
    st.plotly_chart(
        px.bar(coverage_chart_df, x="방식", y="고위험 커버리지(%)"), use_container_width=True
    )

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
            keywords = row.get("llm_keywords") or []
            st.markdown(f"**이상징후 키워드:** {', '.join(keywords) if keywords else '없음'}")
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
