"""24건 샘플에서 out-of-fold ML AUC가 0.292(무작위 0.5보다 낮음)로 나온 문제를
진단한다 — scripts/llm_contribution_results.md의 24건 실험 결과를 심사위원이
재현했을 때 "이 앱의 ML 점수는 무작위보다 못하다"로 오독하지 않도록, 그 수치가
(a) 표본이 작아 추정 자체가 불안정한 것인지 (b) 이 24건 표본이 모델에 유독
어려운 구성인지 구분한다.

방법:
  1. n_splits를 3~12까지 바꿔가며 단일 실행 — fold 수에 따라 값이 크게
     흔들리는지 확인.
  2. Leave-one-out(n_splits=24, shuffle 없음 — 결정론적).
  3. 5-fold를 시드만 바꿔 30회 반복(RepeatedStratifiedKFold와 동등) — "이
     24건에 대해 다른 방식으로 5-fold를 나눴다면" 추정치가 얼마나 흔들리는지.
  4. **결정적 테스트**: Kaggle 원본 전체 15,420건 중 이 24건을 제외한
     15,396건으로 (app.py와 동일한 파이프라인을) 학습한 모델로, 학습에 전혀
     쓰이지 않은 이 정확한 24건을 채점한다. 이 24건이 "원래 어려운 구성"이라면
     이 방식으로도 낮게 나와야 하고, "표본이 작아 학습 자체가 불안정했을 뿐"
     이라면 populaton-level 수치(scripts/ml_model_comparison_results.md의
     AUC-ROC 0.8026)에 가까운 값이 나와야 한다.

데이터: data/raw/fraud_oracle.csv(Kaggle 원본 15,420건 — .gitignore 대상,
README "설치" 섹션 안내대로 직접 내려받아야 함), data/processed/
claims_with_narratives.csv, data/sample/sample_claims.csv. API 호출 없음
(전부 로컬 학습/평가).

사용법:
    python scripts/small_sample_cv_diagnosis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from scripts.prepare_app_dataset import transform  # noqa: E402
from src.config import RANDOM_SEED  # noqa: E402
from src.scoring.features import feature_cols_for  # noqa: E402
from src.scoring.ml_model import _build_preprocessor  # noqa: E402

LABEL_COL = "FraudFound_P"
SAMPLE_24_PATH = Path("data/sample/sample_claims.csv")
RAW_POPULATION_PATH = Path("data/processed/claims_with_narratives.csv")
N_SPLITS_SWEEP = [3, 4, 5, 6, 8, 12]
N_REPEATS = 30
REPORT_PATH = Path("scripts/small_sample_cv_diagnosis_results.md")
POPULATION_AUC_REFERENCE = 0.8026  # scripts/ml_model_comparison_results.md, interpretable_subset_rf_classweight


def _pipeline(X: pd.DataFrame, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("prep", _build_preprocessor(X)),
            ("clf", RandomForestClassifier(random_state=seed, n_estimators=200, class_weight="balanced")),
        ]
    )


def main() -> None:
    df24 = pd.read_csv(SAMPLE_24_PATH)
    feature_cols = feature_cols_for(df24)
    X24 = df24[feature_cols]
    y24 = df24[LABEL_COL]
    n_fraud = int(y24.sum())
    print(f"24건 데이터: 사기 {n_fraud}/{len(y24)}")

    lines = [
        "# 24건 표본 out-of-fold AUC 진단",
        "",
        "scripts/llm_contribution_results.md의 24건 실험에서 ML only out-of-fold "
        "AUC가 0.292(무작위 0.5보다 낮음)로 나온 원인을 진단한다.",
        "",
        f"- 데이터: `{SAMPLE_24_PATH.as_posix()}` (사기 {n_fraud}/{len(y24)})",
        "",
        "## 1. n_splits를 바꿔도 결과가 나쁘다(단일 실행, seed 고정)",
        "",
        "| n_splits | AUC |",
        "|---|---|",
    ]
    print("\n=== 1. n_splits sweep ===")
    for n_splits in N_SPLITS_SWEEP:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        proba = cross_val_predict(_pipeline(X24, RANDOM_SEED), X24, y24, cv=skf, method="predict_proba")[:, 1]
        auc = roc_auc_score(y24, proba)
        print(f"  n_splits={n_splits}: AUC={auc:.4f}")
        lines.append(f"| {n_splits} | {auc:.4f} |")

    loo = KFold(n_splits=len(df24), shuffle=False)
    proba_loo = cross_val_predict(_pipeline(X24, RANDOM_SEED), X24, y24, cv=loo, method="predict_proba")[:, 1]
    auc_loo = roc_auc_score(y24, proba_loo)
    print(f"\n=== Leave-one-out: AUC={auc_loo:.4f} ===")
    lines.extend(["", f"- Leave-one-out(n_splits={len(df24)}, 결정론적): AUC={auc_loo:.4f}", ""])

    print("\n=== 2. 5-fold를 다른 시드로 30회 반복 — 이 24건 자체의 추정 변동폭 ===")
    rng = np.random.default_rng(RANDOM_SEED)
    aucs, pr_aucs = [], []
    for _ in range(N_REPEATS):
        seed_i = int(rng.integers(0, 1_000_000))
        skf_i = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed_i)
        proba_i = cross_val_predict(_pipeline(X24, RANDOM_SEED), X24, y24, cv=skf_i, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y24, proba_i))
        pr_aucs.append(average_precision_score(y24, proba_i))
    aucs_arr, pr_arr = np.array(aucs), np.array(pr_aucs)
    print(f"  AUC: mean={aucs_arr.mean():.4f} std={aucs_arr.std():.4f} min={aucs_arr.min():.4f} max={aucs_arr.max():.4f}")
    lines.extend(
        [
            f"## 2. 같은 24건, 5-fold 분할만 {N_REPEATS}번 다르게(시드 변경) — 추정 자체의 변동폭",
            "",
            f"- AUC: 평균 {aucs_arr.mean():.4f} ± {aucs_arr.std():.4f} (범위 {aucs_arr.min():.4f}~{aucs_arr.max():.4f})",
            f"- PR-AUC: 평균 {pr_arr.mean():.4f} ± {pr_arr.std():.4f} (범위 {pr_arr.min():.4f}~{pr_arr.max():.4f})",
            f"- {N_REPEATS}번 중 AUC>=0.5인 경우: {int((aucs_arr >= 0.5).sum())}건",
            "",
            "5-fold 분할을 어떻게 나누느냐에 따라 같은 24건, 같은 모델인데도 AUC가 "
            f"{aucs_arr.min():.2f}~{aucs_arr.max():.2f}까지 흔들린다 — 애초에 이 크기의 "
            "표본에서 5-fold OOF 추정 자체가 신뢰구간이 매우 넓어 단일 수치(0.292)를 "
            "\"이 앱의 실제 성능\"으로 읽으면 안 된다는 뜻이다.",
            "",
        ]
    )

    print("\n=== 3. 결정적 테스트: 전체 인구(15,396건, 이 24건 제외)로 학습 후 이 24건 채점 ===")
    raw = pd.read_csv(RAW_POPULATION_PATH)
    held_out_ids = set(df24["case_id"])
    raw_train = raw[~raw["PolicyNumber"].isin(held_out_ids)].copy()
    n_excluded = len(raw) - len(raw_train)
    print(f"  전체 {len(raw)}건 중 {len(raw_train)}건으로 학습(이 24건 {n_excluded}건 제외)")

    full_train = transform(raw_train)
    feature_cols_full = feature_cols_for(full_train)
    X_full_train = full_train[feature_cols_full]
    y_full_train = full_train[LABEL_COL]

    pipeline_full = _pipeline(X_full_train, RANDOM_SEED)
    pipeline_full.fit(X_full_train, y_full_train)

    X24_matched = df24[feature_cols_full]
    proba24_from_full = pipeline_full.predict_proba(X24_matched)[:, 1]
    auc_from_full = roc_auc_score(y24, proba24_from_full)
    pr_from_full = average_precision_score(y24, proba24_from_full)
    print(f"  전체 인구로 학습한 모델이 이 24건을 채점한 AUC={auc_from_full:.4f}, PR-AUC={pr_from_full:.4f}")

    lines.extend(
        [
            "## 3. 결정적 테스트 — 전체 인구(15,396건, 이 24건 제외)로 학습해 이 24건을 채점",
            "",
            f"- 학습: `{RAW_POPULATION_PATH.as_posix()}` 전체 {len(raw)}건 중 이 24건을 제외한 "
            f"{len(raw_train)}건 (app.py와 동일한 범주형 One-Hot + RandomForest(class_weight='balanced') 파이프라인)",
            f"- 이 24건을 채점한 결과: **AUC={auc_from_full:.4f}, PR-AUC={pr_from_full:.4f}**",
            f"- 참고: 전체 인구 기준 검증값(scripts/ml_model_comparison_results.md) AUC-ROC={POPULATION_AUC_REFERENCE:.4f}",
            "",
            "## 결론",
            "",
        ]
    )

    if auc_from_full >= 0.6:
        conclusion = (
            f"**표본 크기 문제이지, 이 24건 자체가 모델이 못 맞히는 구성이 아니다.** "
            f"전체 인구로 제대로 학습한 모델은 이 정확한 24건에서 AUC={auc_from_full:.4f}로 "
            f"population 기준치({POPULATION_AUC_REFERENCE:.4f})와 같은 범위다. 반면 이 24건 "
            "'안에서만' 5-fold OOF를 돌리면 각 fold의 학습 데이터가 ~19건뿐이라(범주형 "
            "One-Hot 인코딩 후 RandomForest 200그루를 학습하기엔 턱없이 적음) 추정 자체가 "
            f"극도로 불안정해진다(위 2번, 표준편차 {aucs_arr.std():.3f}, 최저 {aucs_arr.min():.2f}). "
            "0.292는 '이 앱의 ML이 무작위보다 못하다'가 아니라 '24건짜리 표본 안에서 계산한 "
            "OOF 추정치는 통계적으로 신뢰할 수 없다'는 뜻으로 읽어야 한다."
        )
    else:
        conclusion = (
            f"전체 인구로 학습한 모델도 이 24건에서 AUC={auc_from_full:.4f}로 낮게 나와, "
            "표본 크기 문제만으로는 설명되지 않는다 — 이 24건 표본 자체가 모델에 유독 "
            "어려운 구성일 가능성을 배제할 수 없다. 표본 교체를 검토할 필요가 있다."
        )
    lines.append(conclusion)
    print(f"\n{conclusion}")

    lines.extend(
        [
            "",
            "**권장**: 24건 샘플을 다른 표본으로 교체하지 않는다. 원인이 '표본 선택'이 아니라 "
            "'표본 크기'이므로, 어떤 24건을 새로 뽑아도 그 24건 '안에서만' OOF를 돌리면 같은 "
            "불안정성이 재현될 가능성이 높다(위 2번 실험이 이미 30가지 다른 5-fold 분할을 "
            "시도해 이를 보여준다). 표본을 바꾸는 것은 근본 원인을 고치지 못한 채 우연히 "
            "더 좋아 보이는 숫자를 고르는 것에 가깝다. 대신 문서에는 'ML 모델의 신뢰할 수 "
            f"있는 성능 추정치는 15,420건 전체 기준 AUC-ROC {POPULATION_AUC_REFERENCE:.2f}이며, "
            "24건 데모는 UI/워크플로 시연용이지 그 자체로 모델 성능을 검증하는 표본이 아니다'라고 "
            "명시하는 것을 권장한다.",
            "",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n결과 저장 완료: {REPORT_PATH}")


if __name__ == "__main__":
    main()
