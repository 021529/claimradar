"""라벨 누출 없는 격리 재검증.

scripts/llm_contribution_test.py가 발견한 문제: 기존 사고경위서 생성 로직
(src/scoring/fraud_patterns.select_patterns())은 `if not _is_fraud(row):
return []`로 시작해, 사기 특유 정황 문구 주입 여부가 FraudFound_P 정답
라벨과 두 데모 데이터셋(24건/300건) 모두에서 100% 완전히 일치했다. 그래서
"LLM only가 ML only를 능가한다"는 관측이 LLM의 서사 이해 때문인지 데이터
생성 과정의 라벨 누출 때문인지 구분할 수 없었다.

이 스크립트는 그 라벨 누출을 제거한 별도의 격리된 데이터로 같은 ablation을
독립 재실행한다:
  - src/scoring/fraud_patterns.select_patterns_structural(): select_patterns()와
    똑같은 4개 패턴을 판단하되 `_is_fraud` 게이트와 사기 전용 랜덤 폴백을
    제거해, 오직 관측 가능한 구조적 조건(경찰 미신고+목격자 부재, 지연 접수,
    과거 청구 이력 등)만으로 트리거한다. FraudFound_P는 보지 않는다.
  - src/scoring/llm_analysis.generate_synthetic_narrative()의 새
    patterns_override 파라미터로 위 구조적 패턴을 주입해 새 사고경위서를 생성.

**기존 자산 보호**: data/sample/sample_claims.csv, data/processed/
app_demo_sample.csv와 그걸 쓰는 8개 스크립트(hierarchical_filtering.py,
lambda_sweep_24case.py, lambda_sweep_300case.py, sensitivity_analysis.py,
sensitivity_analysis_24case.py, greedy_vs_ortools_comparison.py,
noise_robustness_test.py, small_sample_cv_diagnosis.py)는 전혀 건드리지
않는다. select_patterns()/generate_synthetic_narrative()의 새 파라미터는
기본값이 기존 동작을 그대로 보존해, 위 스크립트를 재실행해도 결과가 바뀌지
않는다.

데이터: data/raw/fraud_oracle.csv(Kaggle 원본, .gitignore 대상 — 직접
다운로드 필요)에서, 기존 24/300건 데모에 이미 쓰인 case_id를 제외하고
새 시드(2026)로 사기/정상 반반씩 새로 표본 추출한다. 파일럿은 10건(5+5),
--full은 250건(125+125).

**첫 번째 검증 관문**: 새 방식으로도 패턴 주입 여부가 FraudFound_P와 여전히
높게 겹치면(예: 90%+) 이 재검증 자체가 무의미하므로, 그 경우 ablation을
계속하지 않고 일치율만 보고하고 중단한다.

비용: 건당 최대 2회 호출(사고경위서 생성 1회 + 분석 1회). 파일럿 10건=최대
20건, --full 250건=최대 500건. case_id별 증분 캐시라 파일럿 이후 --full로
확장해도 이미 처리한 케이스는 재호출하지 않는다.

사용법:
    python scripts/leak_free_contribution_test.py            # 파일럿 10건
    python scripts/leak_free_contribution_test.py --full      # 250건 전체
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from scripts.llm_contribution_test import (  # noqa: E402
    N_SHUFFLES,
    WEIGHT_SWEEP,
    classify_case,
    full_metric_suite,
    oof_ml_scores,
)
from scripts.prepare_app_dataset import transform  # noqa: E402
from src.config import RANDOM_SEED  # noqa: E402
from src.scoring.combine import combine_scores  # noqa: E402
from src.scoring.features import feature_cols_for  # noqa: E402
from src.scoring.fraud_patterns import select_patterns_structural  # noqa: E402
from src.scoring.llm_analysis import analyze_narrative, generate_synthetic_narrative  # noqa: E402

import numpy as np  # noqa: E402

LABEL_COL = "FraudFound_P"
RAW_PATH = Path("data/raw/fraud_oracle.csv")
EXISTING_SAMPLE_PATHS = [Path("data/sample/sample_claims.csv"), Path("data/processed/app_demo_sample.csv")]
NARRATIVES_PATH = Path("data/processed/leak_free_narratives.csv")
LLM_CACHE_PATH = Path("data/processed/leak_free_llm_cache.csv")
REPORT_PATH = Path("scripts/leak_free_contribution_results.md")

SEED = 2026
PILOT_SIZE = 10
FULL_SIZE = 250
CONCURRENCY = 8
LEAK_AGREEMENT_ABORT_THRESHOLD = 0.90


def sample_new_rows(n_total: int) -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH)
    excluded_ids = set()
    for p in EXISTING_SAMPLE_PATHS:
        if p.exists():
            excluded_ids |= set(pd.read_csv(p)["case_id"])
    pool = raw[~raw["PolicyNumber"].isin(excluded_ids)]

    n_fraud = n_total // 2
    n_normal = n_total - n_fraud
    fraud = pool[pool[LABEL_COL] == 1].sample(n=n_fraud, random_state=SEED)
    normal = pool[pool[LABEL_COL] == 0].sample(n=n_normal, random_state=SEED)
    combined = pd.concat([fraud, normal]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return combined


def _generate_one(row: pd.Series) -> tuple:
    """row는 raw fraud_oracle.csv 원본 행(Days_Policy_Claim 등 원본 컬럼명)이어야
    한다 — select_patterns_structural()과 generate_synthetic_narrative()의
    _row_context_lines가 원본 Kaggle 컬럼명을 보기 때문에(원래 파이프라인
    generate_synthetic_dataset.py도 transform() 이전의 raw row에 대해 이 함수들을
    호출한다). transform() 이후의 앱 스키마 행(days_policy_claim_rank 등)을 넘기면
    Days_Policy_Claim이 안 보여 지연신고 패턴이 영영 트리거되지 않는 버그가 생긴다
    (파일럿 10건에서 실제로 발견됨 — blackbox_mijangchak만 나오고 jiyeon_singo가
    한 번도 안 나왔음)."""
    patterns = select_patterns_structural(row)
    narrative = generate_synthetic_narrative(row, patterns_override=patterns)
    return row["PolicyNumber"], narrative, patterns


def get_or_build_narratives(raw_sample: pd.DataFrame, case_ids: list) -> pd.DataFrame:
    if NARRATIVES_PATH.exists():
        cache = pd.read_csv(NARRATIVES_PATH)
    else:
        cache = pd.DataFrame(columns=["case_id", "narrative_text", "injected_patterns"])
    existing = set(cache["case_id"])
    missing_ids = [cid for cid in case_ids if cid not in existing]

    if missing_ids:
        print(f"  [narratives] 신규 생성 {len(missing_ids)}건 (동시 {CONCURRENCY}개)")
        rows = [raw_sample.loc[raw_sample["PolicyNumber"] == cid].iloc[0] for cid in missing_ids]
        new_rows = []
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {executor.submit(_generate_one, row): row["PolicyNumber"] for row in rows}
            done = 0
            for future in as_completed(futures):
                cid, narrative, patterns = future.result()
                new_rows.append({"case_id": cid, "narrative_text": narrative, "injected_patterns": "|".join(patterns)})
                done += 1
                if done % 10 == 0 or done == len(missing_ids):
                    print(f"    {done}/{len(missing_ids)} 완료")
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        NARRATIVES_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(NARRATIVES_PATH, index=False)
    else:
        print("  [narratives] 캐시 재사용 — 신규 생성 0건")

    return cache[cache["case_id"].isin(case_ids)].copy()


def _analyze_one(case_id, narrative_text: str) -> tuple:
    result = analyze_narrative(narrative_text)
    return case_id, result


def get_or_build_llm_cache(narratives: pd.DataFrame, case_ids: list) -> pd.DataFrame:
    if LLM_CACHE_PATH.exists():
        cache = pd.read_csv(LLM_CACHE_PATH)
    else:
        cache = pd.DataFrame(columns=["case_id", "llm_keywords", "llm_explanation", "llm_suspicion_adjustment"])
    existing = set(cache["case_id"])
    missing_ids = [cid for cid in case_ids if cid not in existing]

    if missing_ids:
        print(f"  [llm_cache] 신규 분석 {len(missing_ids)}건 (동시 {CONCURRENCY}개)")
        text_by_id = dict(zip(narratives["case_id"], narratives["narrative_text"]))
        new_rows = []
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = {executor.submit(_analyze_one, cid, text_by_id[cid]): cid for cid in missing_ids}
            done = 0
            for future in as_completed(futures):
                cid, result = future.result()
                new_rows.append(
                    {
                        "case_id": cid,
                        "llm_keywords": "|".join(result.keywords),
                        "llm_explanation": result.explanation,
                        "llm_suspicion_adjustment": result.suspicion_adjustment,
                    }
                )
                done += 1
                if done % 10 == 0 or done == len(missing_ids):
                    print(f"    {done}/{len(missing_ids)} 완료")
        cache = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
        cache.to_csv(LLM_CACHE_PATH, index=False)
    else:
        print("  [llm_cache] 캐시 재사용 — 신규 분석 0건")

    return cache[cache["case_id"].isin(case_ids)].copy()


def check_leakage(df: pd.DataFrame, narratives: pd.DataFrame) -> dict:
    has_injection = dict(zip(narratives["case_id"], narratives["injected_patterns"].fillna("").str.len() > 0))
    injected = df["case_id"].map(has_injection)
    fraud = df[LABEL_COL].astype(bool)
    n = len(df)
    agree = int((injected == fraud).sum())
    return {
        "n": n,
        "agree_rate": agree / n,
        "n_fraud_with_injection": int((injected & fraud).sum()),
        "n_fraud_total": int(fraud.sum()),
        "n_normal_with_injection": int((injected & ~fraud).sum()),
        "n_normal_total": int((~fraud).sum()),
    }


def main() -> None:
    full_run = "--full" in sys.argv
    n_total = FULL_SIZE if full_run else PILOT_SIZE
    mode_label = "250건 전체" if full_run else f"파일럿 {n_total}건"
    print(f"=== 라벨 누출 없는 격리 재검증 — {mode_label} ===\n")

    raw_sample = sample_new_rows(n_total)
    case_ids = raw_sample["PolicyNumber"].tolist()
    print(f"신규 표본: {len(raw_sample)}건 (사기 {int(raw_sample[LABEL_COL].sum())}건)")

    narratives = get_or_build_narratives(raw_sample, case_ids)

    raw_with_narrative = raw_sample.copy()
    narrative_by_id = dict(zip(narratives["case_id"], narratives["narrative_text"]))
    raw_with_narrative["narrative_text"] = raw_with_narrative["PolicyNumber"].map(narrative_by_id)
    df = transform(raw_with_narrative)

    leakage = check_leakage(df, narratives)
    print(
        f"\n[라벨 일치율] 사기건 중 정황 있음 {leakage['n_fraud_with_injection']}/{leakage['n_fraud_total']}, "
        f"정상건 중 정황 있음 {leakage['n_normal_with_injection']}/{leakage['n_normal_total']}, "
        f"전체 일치율={leakage['agree_rate']:.1%}"
    )

    lines = [
        "# 라벨 누출 없는 격리 재검증 결과",
        "",
        f"- 모드: {mode_label}",
        f"- 데이터: `{RAW_PATH.as_posix()}`에서 기존 24/300건과 겹치지 않는 신규 표본, seed={SEED}",
        f"- 사고경위서: `select_patterns_structural()`(라벨 미참조) + `generate_synthetic_narrative(patterns_override=...)`로 신규 생성",
        "",
        "## 1차 검증 — 라벨 일치율 (100%가 아니어야 실험이 유효함)",
        "",
        f"- 사기건 중 정황 주입: {leakage['n_fraud_with_injection']}/{leakage['n_fraud_total']}",
        f"- 정상건 중 정황 주입: {leakage['n_normal_with_injection']}/{leakage['n_normal_total']}",
        f"- 전체 일치율: **{leakage['agree_rate']:.1%}**",
        "",
    ]

    if leakage["agree_rate"] >= LEAK_AGREEMENT_ABORT_THRESHOLD:
        lines.append(
            f"**⚠️ 일치율이 {LEAK_AGREEMENT_ABORT_THRESHOLD:.0%} 이상으로 여전히 높습니다 — 이 재검증은 "
            "무효화됩니다. select_patterns_structural()의 구조적 조건 자체가 이 표본에서 FraudFound_P와 "
            "우연히 강하게 겹칠 수 있으니 ablation을 진행하지 않고 여기서 중단합니다.**"
        )
        REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n일치율이 임계값 이상 — ablation 생략, 보고서만 저장.")
        return

    lines.append(
        "일치율이 100%보다 충분히 낮아 ablation을 계속 진행합니다."
    )
    lines.append("")

    llm_cache = get_or_build_llm_cache(narratives, case_ids)
    df = df.merge(llm_cache, on="case_id", how="left")

    feature_cols = feature_cols_for(df)
    ml_score = oof_ml_scores(df, feature_cols)
    llm_adj = df["llm_suspicion_adjustment"]

    ml_only = full_metric_suite(df, ml_score)
    llm_only = full_metric_suite(df, llm_adj)
    current = full_metric_suite(df, combine_scores(ml_score, llm_adj))

    rng = np.random.default_rng(RANDOM_SEED)
    shuffle_runs = []
    for _ in range(N_SHUFFLES):
        shuffled_adj = pd.Series(rng.permutation(llm_adj.to_numpy()), index=df.index)
        shuffle_runs.append(full_metric_suite(df, combine_scores(ml_score, shuffled_adj)))
    shuffle_mean = {k: float(np.mean([r[k] for r in shuffle_runs])) for k in shuffle_runs[0]}
    shuffle_std = {k: float(np.std([r[k] for r in shuffle_runs])) for k in shuffle_runs[0]}

    case, labels = classify_case(current, shuffle_mean, shuffle_std)
    print(f"\n판정: Case {case}")
    for r in labels:
        print(f"  {r['metric']}: 현재={r['actual']:.4f} 셔플평균={r['shuffle_mean']:.4f}±{r['shuffle_std']:.4f} z={r['z']:.2f} -> {r['label']}")

    sweep = {}
    for w_ml in WEIGHT_SWEEP:
        sweep[w_ml] = full_metric_suite(df, combine_scores(ml_score, llm_adj, w1=w_ml, w2=1 - w_ml))

    def fmt_row(name, m):
        return (
            f"| {name} | {m['auc_roc']:.3f} | {m['pr_auc']:.3f} | "
            f"{m['k_10pct_precision']:.3f} | {m['k_10pct_recall']:.3f} | {m['k_10pct_lift']:.2f} | "
            f"{m['capacity_precision']:.3f}/{m['capacity_recall']:.3f} |"
        )

    lines.extend(
        [
            "## 실험 1 — Ablation 비교",
            "",
            "| 변형 | AUC-ROC | PR-AUC | P@10% | R@10% | Lift@10% | 캐파(P/R) |",
            "|---|---|---|---|---|---|---|",
            fmt_row("ML only", ml_only),
            fmt_row("LLM only", llm_only),
            fmt_row("ML+LLM 현재(0.7/0.3)", current),
            fmt_row(f"ML+LLM 셔플 평균({N_SHUFFLES}회)", shuffle_mean),
            "",
            "### 셔플 대조군 대비 z-score 판정",
            "",
            "| 지표 | 현재 | 셔플 평균±표준편차 | z | 판정 |",
            "|---|---|---|---|---|",
        ]
    )
    for r in labels:
        z_str = f"{r['z']:.2f}" if abs(r["z"]) != float("inf") else ("+inf" if r["z"] > 0 else "-inf")
        lines.append(f"| {r['metric']} | {r['actual']:.4f} | {r['shuffle_mean']:.4f}±{r['shuffle_std']:.4f} | {z_str} | {r['label']} |")

    lines.extend(
        [
            "",
            f"**판정: Case {case}** "
            + {
                "A": "— 라벨 누출을 제거한 뒤에도 LLM이 셔플 대비 명확히 우수 → 탐지 기여 주장 가능.",
                "B": "— 개선은 있으나 크지 않음 → 탐지 기여자보다 설명/가이드 레이어 권장.",
                "C": "— 셔플과 구분 안 됨 → 탐지 기여를 주장할 근거 없음.",
            }[case],
            "",
            "## 실험 2 — LLM 가중치 스윕",
            "",
            "| ML가중치 | LLM가중치 | AUC-ROC | PR-AUC | P@10% | R@10% | Lift@10% |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for w_ml in WEIGHT_SWEEP:
        m = sweep[w_ml]
        lines.append(
            f"| {w_ml:.1f} | {1 - w_ml:.1f} | {m['auc_roc']:.3f} | {m['pr_auc']:.3f} | "
            f"{m['k_10pct_precision']:.3f} | {m['k_10pct_recall']:.3f} | {m['k_10pct_lift']:.2f} |"
        )

    lines.extend(["", "## 표본 샘플 (육안 검수용)", ""])
    sample_ids = case_ids[: min(3, len(case_ids))]
    for cid in sample_ids:
        row = df.loc[df["case_id"] == cid].iloc[0]
        lines.extend(
            [
                f"### case_id={cid} (FraudFound_P={row[LABEL_COL]})",
                "",
                f"- 주입된 패턴: {narratives.loc[narratives['case_id'] == cid, 'injected_patterns'].iloc[0] or '(없음)'}",
                f"- 경위서: {row['narrative_text']}",
                "",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n결과 저장 완료: {REPORT_PATH}")


if __name__ == "__main__":
    main()
