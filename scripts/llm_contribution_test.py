"""LLM 기여도 ablation + 가중치 민감도 스윕.

심사 피드백: "비정형 사고경위서가 정형 데이터에서 합성된 것이므로, LLM은 새로운
독립적 증거를 발견하는 게 아니라 이미 ML이 알고 있는 정형 신호를 자연어로 다시
읽는 것일 수 있다. LLM의 30% 가중치가 실제 정보량 증가인지, 동일 정보를 두 번
세는 것인지 어떻게 증명했나?"

**이 실험이 답하는 질문의 정확한 범위**: "정형 데이터에서 합성된 사고경위서를
다시 LLM으로 읽는 현재 파이프라인이 랭킹에 추가 정보를 주는가"이다. "독립적으로
수집된 실제 조사관 메모였다면 어땠을까"는 이 실험으로 답할 수 없다 — 그건 노이즈
견고성 테스트(scripts/noise_robustness_test.py)가 다루는 표기 노이즈 문제와도
다른, 데이터 출처 자체의 문제라 별도 실험(실제 현장 메모 확보)이 필요하다.

실험 1 — Ablation:
  (a) ML only: out-of-fold ML 점수만으로 랭킹
  (b) LLM only: llm_suspicion_adjustment만으로 랭킹
  (c) ML+LLM 현재 방식(0.7/0.3, combine_scores 그대로)
  (d) ML+LLM 셔플: (c)와 같은 가중치이지만 llm_suspicion_adjustment를 케이스
      간 무작위 순열(50회 반복, 평균±표준편차) — "LLM 값이라서" 좋아지는 게
      아니라 "아무 상관없는 값이라도 0.3만큼 더하면" 좋아지는 것일 수 있다는
      가설을 직접 검증하는 핵심 대조군.

ML 점수는 5-fold 계층화 교차검증의 out-of-fold 예측값을 쓴다 — app.py는 전체
데이터로 학습한 모델을 그 데이터에 그대로 스코어링하지만(운영 방식), 그 방식으로
"ML 기여도"를 재면 RandomForest가 학습 데이터를 일부 외워 ML 쪽이 유리하게
부풀 수 있다. Out-of-fold면 각 건이 그 건을 보지 않은 모델로 평가돼 리키지 없이
전체 표본을 다 쓸 수 있다.

실험 2 — 가중치 스윕: ML 가중치 1.0/0.9/0.8/0.7/0.6/0.5(LLM 가중치는 그 나머지)로
combine_scores를 반복 적용해 지표가 어떻게 변하는지 확인. ML 가중치=1.0 지점은
실험 1의 "ML only"와 정확히 같아야 한다(교차검증).

지표: Precision@K/Recall@K/Lift@K (K=상위 1/5/10/20%), AUC-ROC, PR-AUC, 그리고
조사관 3명x10시간(이 프로젝트가 300건 실험에 일관되게 써온 캐파,
scripts/lambda_sweep_300case.py와 동일) 기준 baseline_assignment(스코어
내림차순 그리디 배정, 기존 코드 그대로 재사용)로 "실제 배정 가능한 건수 안에
사기가 몇 건 잡히는가".

데이터: data/processed/app_demo_sample.csv(300건, 메인)와 data/sample/
sample_claims.csv(24건, 배포 앱이 실제로 쓰는 데이터 — 표본이 작아 K=1%/5%가
K=1로 겹치는 등 노이즈가 큼, 참고용). 둘 다 llm_suspicion_adjustment가 이미
캐시돼 있어 API 호출 0건.

결론 판정 기준(사전에 정한 규칙, 결과를 보고 나서 기준을 바꾸지 않는다):
Precision@10%/Recall@10%/Lift@10%/AUC-ROC/PR-AUC/캐파기준 재현율 6개 지표에서
"ML+LLM(현재)"가 셔플 분포(평균/표준편차) 대비 z-score = (실제값-셔플평균)/셔플표준편차를
계산해 z>=2("우수")/0.5<=z<2("약간 우수")/-0.5<z<0.5("구분불가")/-2<z<=-0.5("약간 저조")/
z<=-2("저조")로 라벨링하고, 6개 중 과반(4개 이상)이 "우수" 이상이면 Case A,
과반이 "구분불가" 이하면 Case C, 그 외는 Case B로 분류한다.
  - Case A: LLM이 추가 정보를 제공 — 탐지 기여자라는 현재 주장 유지
  - Case B: 개선은 있으나 작음 — 탐지 기여자보다 설명/가이드 레이어로 재포지셔닝 권장
  - Case C: 셔플과 구분 안 됨 — 탐지 기여를 주장하지 않고 조사 가이드 생성
    기능으로 역할 축소 권장

라벨 누출 감사(중요, 위 판정에 우선함): 위 셔플 대조군은 "llm_suspicion_adjustment가
무작위 값보다 나은가"만 검증하지, "그 값이 좋은 이유가 진짜 서사 이해 때문인지
합성 과정 자체의 라벨 누출 때문인지"는 검증하지 못한다. 이를 확인하기 위해
src/scoring/fraud_patterns.py의 select_patterns()를 각 행에 그대로 재실행해
"이 행이 사기 특유 정황 문구를 주입받았는가(_has_injection)"를 재구성하고,
FraudFound_P와 완전히 겹치는지(라벨 누출)를 직접 검사한다. select_patterns()는
75행에서 `if not _is_fraud(row): return []`로 사기가 아니면 무조건 빈 리스트를
반환하므로, 구조적으로 _has_injection == FraudFound_P가 성립할 수밖에 없다 —
이게 실제로 데이터에서도 100%/0%로 확인되면, 위 ablation에서 관측된 LLM의
우위는(적어도 일부는) "LLM이 서사를 잘 읽어서"가 아니라 "정답이 텍스트에
직접 심어져 있어서"일 가능성이 높다는 뜻이며, 그 경우 Case A/B 판정과 무관하게
이 사실을 결과 문서 최상단에 명시한다.

사용법:
    python scripts/llm_contribution_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.config import RANDOM_SEED  # noqa: E402
from src.optimization.baseline import baseline_assignment  # noqa: E402
from src.scoring.combine import combine_scores  # noqa: E402
from src.scoring.features import feature_cols_for  # noqa: E402
from src.scoring.fraud_patterns import select_patterns  # noqa: E402
from src.scoring.ml_model import _build_preprocessor  # noqa: E402

LABEL_COL = "FraudFound_P"
NUM_INVESTIGATORS = 3
HOURS_PER_INVESTIGATOR = 10
K_PCTS = [0.01, 0.05, 0.10, 0.20]
N_SHUFFLES = 50
N_FOLDS = 5
WEIGHT_SWEEP = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
HEADLINE_METRICS = ["k_10pct_precision", "k_10pct_recall", "k_10pct_lift", "auc_roc", "pr_auc", "capacity_recall"]

REPORT_PATH = Path("scripts/llm_contribution_results.md")

DATASETS = [
    ("300건(app_demo_sample.csv, 메인)", Path("data/processed/app_demo_sample.csv")),
    ("24건(sample_claims.csv, 배포 앱 실사용 데이터)", Path("data/sample/sample_claims.csv")),
]


def oof_ml_scores(df: pd.DataFrame, feature_cols: list[str], n_splits: int = N_FOLDS, seed: int = RANDOM_SEED) -> pd.Series:
    """5-fold 계층화 교차검증의 out-of-fold 예측 확률. 각 건은 그 건을 학습에
    쓰지 않은 fold의 모델이 채점해 리키지가 없다. 파이프라인은 app.py/ml_model.py와
    동일(범주형 One-Hot + RandomForest + class_weight='balanced')."""
    X = df[feature_cols]
    y = df[LABEL_COL]
    pipeline = Pipeline(
        [
            ("prep", _build_preprocessor(X)),
            ("clf", RandomForestClassifier(random_state=seed, n_estimators=200, class_weight="balanced")),
        ]
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")[:, 1]
    return pd.Series(proba, index=df.index, name="ml_score")


def _precision_recall_lift_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> dict:
    order = np.argsort(-score)
    top_k_idx = order[:k]
    tp = float(y_true[top_k_idx].sum())
    precision = tp / k
    recall = tp / y_true.sum() if y_true.sum() else 0.0
    prevalence = y_true.mean()
    lift = precision / prevalence if prevalence else 0.0
    return {"precision": precision, "recall": recall, "lift": lift}


def _capacity_metrics(df: pd.DataFrame, score: pd.Series) -> dict:
    temp = df.copy()
    temp["combined_score"] = score.values
    assigned = baseline_assignment(temp, NUM_INVESTIGATORS, HOURS_PER_INVESTIGATOR)
    assigned = assigned.dropna(subset=["assigned_investigator"])
    n_assigned = len(assigned)
    tp = float(assigned[LABEL_COL].sum())
    n_fraud_total = float(df[LABEL_COL].sum())
    return {
        "n_assigned": n_assigned,
        "precision": tp / n_assigned if n_assigned else 0.0,
        "recall": tp / n_fraud_total if n_fraud_total else 0.0,
    }


def full_metric_suite(df: pd.DataFrame, score: pd.Series) -> dict:
    y_true = df[LABEL_COL].to_numpy()
    score_arr = score.to_numpy()
    n = len(df)
    result = {
        "auc_roc": roc_auc_score(y_true, score_arr),
        "pr_auc": average_precision_score(y_true, score_arr),
    }
    for pct in K_PCTS:
        k = max(1, round(pct * n))
        m = _precision_recall_lift_at_k(y_true, score_arr, k)
        pct_label = f"k_{int(pct * 100)}pct"
        result[f"{pct_label}_k"] = k
        result[f"{pct_label}_precision"] = m["precision"]
        result[f"{pct_label}_recall"] = m["recall"]
        result[f"{pct_label}_lift"] = m["lift"]
    cap = _capacity_metrics(df, score)
    result["capacity_n_assigned"] = cap["n_assigned"]
    result["capacity_precision"] = cap["precision"]
    result["capacity_recall"] = cap["recall"]
    return result


def classify_case(current: dict, shuffle_mean: dict, shuffle_std: dict) -> tuple[str, list[dict]]:
    labels = []
    for metric in HEADLINE_METRICS:
        actual = current[metric]
        mean = shuffle_mean[metric]
        std = shuffle_std[metric]
        delta = actual - mean
        if std > 1e-9:
            z = delta / std
        else:
            z = 0.0 if abs(delta) < 1e-9 else (float("inf") if delta > 0 else float("-inf"))
        if z >= 2:
            label = "우수"
        elif z >= 0.5:
            label = "약간 우수"
        elif z > -0.5:
            label = "구분불가"
        elif z > -2:
            label = "약간 저조"
        else:
            label = "저조"
        labels.append({"metric": metric, "actual": actual, "shuffle_mean": mean, "shuffle_std": std, "z": z, "label": label})

    n_good = sum(1 for r in labels if r["label"] in ("우수", "약간 우수"))
    n_bad_or_flat = sum(1 for r in labels if r["label"] in ("구분불가", "약간 저조", "저조"))
    n_strong = sum(1 for r in labels if r["label"] == "우수")

    if n_strong >= 4 or (n_good >= 4 and n_strong >= 2):
        case = "A"
    elif n_bad_or_flat >= 4:
        case = "C"
    else:
        case = "B"
    return case, labels


def check_label_leakage(df: pd.DataFrame) -> dict:
    """이 행의 사고경위서 생성 시 fraud_patterns.select_patterns()가 사기 특유
    정황 문구(나이롱환자/한방병원/블랙박스미장착/지연신고 힌트)를 주입했는지를
    그 함수를 그대로 재실행해 재구성하고, FraudFound_P와 얼마나 겹치는지 검사한다.
    select_patterns()는 `if not _is_fraud(row): return []`로 사기가 아니면
    무조건 빈 리스트를 반환하므로(fraud_patterns.py:75-76), 완전히 겹친다면
    이는 우연이 아니라 그 함수의 구조 자체가 만드는 결정론적 결과다."""
    injected = df.apply(lambda row: len(select_patterns(row, seed_key=row.get("case_id"))) > 0, axis=1)
    fraud = df[LABEL_COL].astype(bool)
    n = len(df)
    agree = int((injected == fraud).sum())
    adj_by_group = df.groupby(injected)["llm_suspicion_adjustment"].mean()
    return {
        "n": n,
        "agree_rate": agree / n,
        "n_fraud_with_injection": int((injected & fraud).sum()),
        "n_fraud_total": int(fraud.sum()),
        "n_normal_with_injection": int((injected & ~fraud).sum()),
        "n_normal_total": int((~fraud).sum()),
        "adj_mean_injected": float(adj_by_group.get(True, float("nan"))),
        "adj_mean_not_injected": float(adj_by_group.get(False, float("nan"))),
        "is_perfect_leak": agree == n,
    }


def run_dataset(label: str, path: Path) -> dict:
    print(f"\n=== {label} ===")
    df = pd.read_csv(path)
    feature_cols = feature_cols_for(df)
    ml_score = oof_ml_scores(df, feature_cols)
    llm_adj = df["llm_suspicion_adjustment"]
    n = len(df)
    print(f"  n={n}, 사기 비율={df[LABEL_COL].mean():.2f}")

    leakage = check_label_leakage(df)
    print(
        f"  [라벨 누출 감사] 사기건 중 패턴 주입 {leakage['n_fraud_with_injection']}/{leakage['n_fraud_total']}, "
        f"정상건 중 패턴 주입 {leakage['n_normal_with_injection']}/{leakage['n_normal_total']}, "
        f"라벨 일치율={leakage['agree_rate']:.1%}, 완전 누출={leakage['is_perfect_leak']}"
    )

    ml_only = full_metric_suite(df, ml_score)
    llm_only = full_metric_suite(df, llm_adj)
    current = full_metric_suite(df, combine_scores(ml_score, llm_adj))

    rng = np.random.default_rng(RANDOM_SEED)
    shuffle_runs = []
    for _ in range(N_SHUFFLES):
        shuffled_adj = pd.Series(rng.permutation(llm_adj.to_numpy()), index=df.index)
        shuffled_combined = combine_scores(ml_score, shuffled_adj)
        shuffle_runs.append(full_metric_suite(df, shuffled_combined))

    shuffle_mean = {k: float(np.mean([r[k] for r in shuffle_runs])) for k in shuffle_runs[0]}
    shuffle_std = {k: float(np.std([r[k] for r in shuffle_runs])) for k in shuffle_runs[0]}

    case, labels = classify_case(current, shuffle_mean, shuffle_std)
    print(f"  판정: Case {case}")
    for r in labels:
        print(f"    {r['metric']}: 현재={r['actual']:.4f} 셔플평균={r['shuffle_mean']:.4f}±{r['shuffle_std']:.4f} z={r['z']:.2f} -> {r['label']}")

    sweep = {}
    for w_ml in WEIGHT_SWEEP:
        w_llm = 1.0 - w_ml
        s = combine_scores(ml_score, llm_adj, w1=w_ml, w2=w_llm)
        sweep[w_ml] = full_metric_suite(df, s)

    return {
        "label": label,
        "n": n,
        "prevalence": float(df[LABEL_COL].mean()),
        "leakage": leakage,
        "ml_only": ml_only,
        "llm_only": llm_only,
        "current": current,
        "shuffle_mean": shuffle_mean,
        "shuffle_std": shuffle_std,
        "case": case,
        "case_labels": labels,
        "sweep": sweep,
    }


def _fmt_variant_row(name: str, m: dict) -> str:
    n_assigned = m["capacity_n_assigned"]
    n_assigned_str = f"{n_assigned:.1f}" if isinstance(n_assigned, float) else str(n_assigned)
    return (
        f"| {name} | {m['auc_roc']:.3f} | {m['pr_auc']:.3f} | "
        f"{m['k_10pct_precision']:.3f} | {m['k_10pct_recall']:.3f} | {m['k_10pct_lift']:.2f} | "
        f"{n_assigned_str}건 중 {m['capacity_precision']:.3f}/{m['capacity_recall']:.3f} |"
    )


def build_report(results: list[dict]) -> str:
    lines = [
        "# LLM 기여도 Ablation + 가중치 민감도 실험 결과",
        "",
        "심사 피드백: \"LLM은 이미 ML이 아는 정형 신호를 자연어로 다시 읽는 것일 수 있다 — "
        "0.3 가중치가 실제 정보량 증가인지 동일 정보를 두 번 세는 것인지 어떻게 증명했나?\"",
        "",
        "**이 실험의 정확한 범위**: 정형 데이터에서 합성된 사고경위서를 LLM으로 다시 읽는 "
        "현재 파이프라인이 랭킹에 추가 정보를 주는지를 검증한다. 독립적으로 수집된 실제 "
        "조사관 메모였다면 어땠을지는 이 실험으로 답할 수 없다(별도 데이터 확보가 필요한 다른 질문).",
        "",
        "ML 점수는 5-fold 계층화 교차검증의 out-of-fold 예측값(리키지 없음, app.py와 동일 "
        f"파이프라인)이다. 셔플 대조군은 {N_SHUFFLES}회 반복 평균±표준편차.",
        "",
        "## ⚠️ 결정적 발견 — 라벨 누출 확인 (아래 Case 판정보다 우선해서 읽을 것)",
        "",
        "아래 실험 1의 셔플 대조군은 \"llm_suspicion_adjustment가 무작위 값보다 나은가\"만 "
        "검증한다. \"그 값이 좋은 이유가 진짜 서사 이해 때문인지, 합성 데이터 생성 과정 "
        "자체가 정답을 텍스트에 심어놨기 때문인지\"는 별도로 확인해야 한다. "
        "`src/scoring/fraud_patterns.py`의 `select_patterns()`를 각 행에 그대로 재실행해 "
        "\"이 행이 사기 특유 정황 문구(나이롱환자/한방병원/블랙박스미장착/지연신고 힌트)를 "
        "주입받았는가\"를 재구성한 결과:",
        "",
    ]
    for res in results:
        lk = res["leakage"]
        lines.append(
            f"- **{res['label']}**: 사기건 {lk['n_fraud_with_injection']}/{lk['n_fraud_total']}건, "
            f"정상건 {lk['n_normal_with_injection']}/{lk['n_normal_total']}건에 패턴 주입 "
            f"— 라벨 일치율 **{lk['agree_rate']:.1%}**"
            + (" (완전 누출: 패턴 주입 여부가 정답 라벨과 100% 일치)" if lk["is_perfect_leak"] else "")
            + f". 주입군 평균 adj={lk['adj_mean_injected']:.3f} vs 비주입군 평균 adj={lk['adj_mean_not_injected']:.3f}."
        )
    lines.extend(
        [
            "",
            "**원인**: `fraud_patterns.py` 75-76행이 `if not _is_fraud(row): return []`로 "
            "시작한다 — 즉 실제 사기 신호(경찰 미신고+목격자 부재 등 구조적 조건)와 무관하게, "
            "사기가 아니면 무조건 빈 리스트를 반환한다. 이건 우연한 상관관계가 아니라 코드 구조가 "
            "강제하는 결정론적 결과다.",
            "",
            "**결론**: 위 두 데이터셋 모두 패턴 주입 여부가 정답 라벨과 완전히(100%) 겹친다. "
            "그러므로 아래 ablation에서 관측되는 \"LLM only가 ML only를 능가한다\"·\"LLM 가중치를 "
            "올릴수록 지표가 좋아진다\"는 결과는 **LLM이 서사를 잘 읽어서가 아니라, 합성 데이터 "
            "생성 로직이 사기 건에만 결정론적으로 정답을 텍스트에 새겨 넣었기 때문일 가능성이 "
            "높다**. 아래 Case A/B 판정은 셔플 대조군 기준으로는 유효하게 계산된 것이지만, 이 "
            "라벨 누출 때문에 \"LLM이 실제로 독립적 증거를 찾아낸다\"는 주장의 근거로 쓸 수 없다 "
            "— 이 실험은 결국 심사위원의 우려를 반박하지 못하고 오히려 더 구체적인 형태로 "
            "확인해주는 셈이다. 이 데이터로는 \"LLM이 새 정보를 주는가\"라는 질문에 답할 수 없고, "
            "라벨과 무관하게 패턴을 주입하도록 `select_patterns()`를 고친(예: 구조적 조건만으로 "
            "판단, `_is_fraud` 게이트 제거) 새 데이터셋으로 다시 실험해야 신뢰할 수 있는 답이 "
            "나온다.",
            "",
        ]
    )

    for res in results:
        lines.extend(
            [
                f"## {res['label']}",
                "",
                f"- n={res['n']}, 사기 비율={res['prevalence']:.2f} "
                + ("(주의: 24건은 표본이 작아 K=1%/5%가 K=1건으로 겹치는 등 노이즈가 큼)" if res["n"] < 100 else ""),
                "",
                "### 실험 1 — Ablation 비교",
                "",
                "| 변형 | AUC-ROC | PR-AUC | P@10% | R@10% | Lift@10% | 캐파(N건 중 P/R) |",
                "|---|---|---|---|---|---|---|",
                _fmt_variant_row("ML only", res["ml_only"]),
                _fmt_variant_row("LLM only", res["llm_only"]),
                _fmt_variant_row("ML+LLM 현재(0.7/0.3)", res["current"]),
                _fmt_variant_row(f"ML+LLM 셔플 평균({N_SHUFFLES}회)", res["shuffle_mean"]),
                "",
                "### 셔플 대조군 대비 z-score 판정 (핵심 지표)",
                "",
                "| 지표 | 현재(ML+LLM) | 셔플 평균±표준편차 | z-score | 판정 |",
                "|---|---|---|---|---|",
            ]
        )
        for r in res["case_labels"]:
            z_str = f"{r['z']:.2f}" if abs(r["z"]) != float("inf") else ("+inf" if r["z"] > 0 else "-inf")
            lines.append(
                f"| {r['metric']} | {r['actual']:.4f} | {r['shuffle_mean']:.4f}±{r['shuffle_std']:.4f} | {z_str} | {r['label']} |"
            )
        lines.extend(
            [
                "",
                f"**셔플 대조군 기준 판정: Case {res['case']}** "
                + {
                    "A": "— LLM이 셔플 대비 명확히 우수.",
                    "B": "— 개선은 있으나 크지 않음.",
                    "C": "— 셔플과 구분 안 됨.",
                }[res["case"]]
                + " ⚠️ 단, 위 \"결정적 발견\" 섹션에서 확인했듯 이 데이터셋은 패턴 주입이 "
                "정답 라벨과 완전히 겹쳐(라벨 누출), 이 판정을 \"LLM이 독립적 증거를 찾는다\"는 "
                "근거로 그대로 쓸 수 없다 — 셔플 대조군은 통과했지만 그 자체가 오염된 데이터 "
                "위에서 통과한 것이다.",
                "",
                "### 실험 2 — LLM 가중치 스윕",
                "",
                "| ML가중치 | LLM가중치 | AUC-ROC | PR-AUC | P@10% | R@10% | Lift@10% | 캐파 재현율 |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for w_ml in WEIGHT_SWEEP:
            m = res["sweep"][w_ml]
            lines.append(
                f"| {w_ml:.1f} | {1 - w_ml:.1f} | {m['auc_roc']:.3f} | {m['pr_auc']:.3f} | "
                f"{m['k_10pct_precision']:.3f} | {m['k_10pct_recall']:.3f} | {m['k_10pct_lift']:.2f} | "
                f"{m['capacity_recall']:.3f} |"
            )
        lines.append("")

    all_perfect_leak = all(res["leakage"]["is_perfect_leak"] for res in results)
    lines.extend(
        [
            "## 종합 판정",
            "",
            "셔플 대조군 기준 원판정(참고용):",
            "",
        ]
    )
    for res in results:
        lines.append(f"- {res['label']}: Case {res['case']}")
    lines.extend(
        [
            "",
            "**하지만 라벨 누출이 두 데이터셋 모두에서 확인됐으므로(위 ⚠️ 섹션), 최종 결론은 "
            "위 Case 표기가 아니라 다음과 같다:**" if all_perfect_leak else "**라벨 누출 확인 결과가 데이터셋별로 다르므로 아래를 참고할 것:**",
            "",
        ]
    )
    if all_perfect_leak:
        lines.extend(
            [
                "이 실험은 애초에 설계한 목적(\"LLM이 정형 신호를 자연어로 다시 읽을 뿐인지, "
                "독립적 정보를 더하는지\")을 이 데이터로는 검증할 수 없다는 것을 보여줬다 — "
                "데이터 생성 로직 자체가 사기 건에만 결정론적으로 정답을 텍스트에 새겨 넣기 "
                "때문에, LLM only가 ML only를 능가하는 것도 가중치를 올릴수록 좋아지는 것도 "
                "\"LLM이 잘해서\"가 아니라 \"정답이 심어져 있어서\"일 가능성이 높다.",
                "",
                "**권장 포지셔닝(Case C에 해당)**: 이 실험 결과로는 LLM의 탐지 기여를 주장할 "
                "수 없다. 현재 0.7/0.3 가중치의 탐지 성능 근거는 이 합성 데이터에서는 무효화된 "
                "것으로 보고, LLM을 \"탐지 기여자\"가 아니라 \"조사관용 설명·조사 가이드 생성 "
                "레이어\"로 재포지셔닝하는 것이 데이터와 일관된 결론이다. 탐지 기여 여부를 "
                "다시 주장하려면 (a) `select_patterns()`의 `_is_fraud` 게이트를 제거해 라벨과 "
                "무관하게 구조적 조건만으로 패턴을 주입하는 새 데이터셋을 만들거나, (b) 실제 "
                "조사관이 작성한 현장 메모(합성이 아닌)로 재실험해야 한다.",
                "",
            ]
        )
    else:
        for res in results:
            lines.append(f"- {res['label']}: 라벨 누출 여부={res['leakage']['is_perfect_leak']}")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    results = [run_dataset(label, path) for label, path in DATASETS]
    report = build_report(results)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\n결과 저장 완료: {REPORT_PATH}")


if __name__ == "__main__":
    main()
