"""ML 스코어링 모델 보강 전/후 성능 비교 (Kaggle 원본 fraud_oracle.csv 15,420건 전체).

심사 지적: "범주형 처리, 스케일링, 클래스 불균형 처리 등 기초 피처 엔지니어링이
전부 누락됐다"에 대한 정량 검증. app.py/ml_model.py가 실제 쓰는 5개 수치형
피처만의 baseline과, Kaggle 원본 33컬럼 중 ID성(PolicyNumber)만 제외하고
범주형까지 전부 인코딩한 enhanced 후보들을 StratifiedKFold(5)로 비교한다.

데이터 누수 방지: 인코딩(OneHot/Ordinal)과 SMOTE는 imblearn Pipeline 안에
넣어 매 fold의 train 쪽에서만 fit되게 한다 (test fold는 transform만).

사용법:
    python scripts/ml_model_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline as SkPipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler  # noqa: E402

from scripts.prepare_app_dataset import (  # noqa: E402
    NOMINAL_COLS as INTERPRETABLE_NOMINAL_COLS,
    ORDINAL_ORDER as _SHARED_ORDINAL_ORDER,
    _PAST_CLAIMS_MIDPOINT,
    _VEHICLE_PRICE_MIDPOINT,
)

RAW_PATH = Path("data/raw/fraud_oracle.csv")
OUTPUT_PATH = Path("scripts/ml_model_comparison_results.md")
LABEL_COL = "FraudFound_P"
RANDOM_SEED = 42
N_FOLDS = 5

# --- 현재 프로덕션(app.py/ml_model.py)이 쓰는 5개 수치형 피처 ---
BASELINE_NUMERIC_COLS = ["driver_age", "vehicle_price", "deductible", "driver_rating", "past_number_of_claims"]

# --- 보강 피처 그룹 ---
NUMERIC_COLS = ["WeekOfMonth", "WeekOfMonthClaimed", "Age", "Deductible", "DriverRating", "Year"]

# prepare_app_dataset.ORDINAL_ORDER(라이브 스키마와 공유) + 이 벤치마크에서만
# 추가로 서열형-인코딩 방식을 시험해보는 VehiclePrice/PastNumberOfClaims
# (라이브 스키마에서는 대신 중간값 방식(_VEHICLE_PRICE_MIDPOINT 등)을 씀)
ORDINAL_ORDER: dict[str, list[str]] = {
    "VehiclePrice": [
        "less than 20000", "20000 to 29000", "30000 to 39000",
        "40000 to 59000", "60000 to 69000", "more than 69000",
    ],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    **_SHARED_ORDINAL_ORDER,
}
ORDINAL_COLS = list(ORDINAL_ORDER.keys())

# 벤치마크 전체 피처셋 (라이브 스키마보다 넓음 — 시간성 컬럼/RepNumber 포함)
NOMINAL_COLS = [
    "Month", "DayOfWeek", "Make", "AccidentArea", "DayOfWeekClaimed", "MonthClaimed",
    "Sex", "MaritalStatus", "Fault", "PolicyType", "VehicleCategory",
    "PoliceReportFiled", "WitnessPresent", "AgentType", "BasePolicy", "RepNumber",
]
DROP_COLS = ["PolicyNumber"]  # 순차 발급 ID, 정보 없음/누수 위험

ENHANCED_FEATURE_COLS = NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS

# 라이브 앱(app.py)에 실제로 반영하는 "해석 가능 부분집합": 시간성 컬럼과
# RepNumber를 뺀, 조사관이 봤을 때 의미를 알 수 있는 범주형만 사용.
# prepare_app_dataset.py의 NOMINAL_COLS/ORDINAL_ORDER와 정확히 동일해야
# "벤치마크로 검증한 것 == 라이브에 반영한 것"이 성립한다.
INTERPRETABLE_ORDINAL_COLS = list(_SHARED_ORDINAL_ORDER.keys())
INTERPRETABLE_FEATURE_COLS = BASELINE_NUMERIC_COLS + INTERPRETABLE_ORDINAL_COLS + INTERPRETABLE_NOMINAL_COLS


def _build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """prepare_app_dataset.transform()과 동일한 방식으로 5개 수치형 피처 산출."""
    out = pd.DataFrame(index=df.index)
    out["driver_age"] = df["Age"]
    out["vehicle_price"] = df["VehiclePrice"].map(_VEHICLE_PRICE_MIDPOINT)
    out["deductible"] = df["Deductible"]
    out["driver_rating"] = df["DriverRating"]
    out["past_number_of_claims"] = df["PastNumberOfClaims"].map(_PAST_CLAIMS_MIDPOINT)
    return out


def _no_encoding_preprocessor() -> str:
    return "passthrough"


def _plain_preprocessor() -> ColumnTransformer:
    """RF용: 수치형 그대로 + 서열형 정수 인코딩 + 명목형 원-핫 (스케일링 없음)."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_COLS),
            ("ord", OrdinalEncoder(categories=[ORDINAL_ORDER[c] for c in ORDINAL_COLS]), ORDINAL_COLS),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_COLS),
        ]
    )


def _interpretable_preprocessor() -> ColumnTransformer:
    """라이브 앱(app.py/prepare_app_dataset.py)이 실제로 넘기는 것과 동일한 부분집합."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", BASELINE_NUMERIC_COLS),
            (
                "ord",
                OrdinalEncoder(categories=[_SHARED_ORDINAL_ORDER[c] for c in INTERPRETABLE_ORDINAL_COLS]),
                INTERPRETABLE_ORDINAL_COLS,
            ),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), INTERPRETABLE_NOMINAL_COLS),
        ]
    )


def _scaled_preprocessor() -> ColumnTransformer:
    """로지스틱회귀 비교용: 수치형/서열형에 StandardScaler 적용, 명목형은 원-핫 그대로."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            (
                "ord",
                SkPipeline([
                    ("enc", OrdinalEncoder(categories=[ORDINAL_ORDER[c] for c in ORDINAL_COLS])),
                    ("scale", StandardScaler()),
                ]),
                ORDINAL_COLS,
            ),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NOMINAL_COLS),
        ]
    )


def _make_pipeline(preprocessor, classifier, smote: bool) -> ImbPipeline:
    steps = [("prep", preprocessor)]
    if smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_SEED)))
    steps.append(("clf", classifier))
    return ImbPipeline(steps)


def evaluate(name: str, X: pd.DataFrame, y: pd.Series, pipeline: ImbPipeline) -> dict:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    cv_scores = cross_validate(
        pipeline, X, y, cv=skf, scoring=["roc_auc", "average_precision"], n_jobs=-1
    )
    oof_pred = cross_val_predict(pipeline, X, y, cv=skf, method="predict", n_jobs=-1)
    cm = confusion_matrix(y, oof_pred)

    return {
        "name": name,
        "auc_mean": cv_scores["test_roc_auc"].mean(),
        "auc_std": cv_scores["test_roc_auc"].std(),
        "pr_auc_mean": cv_scores["test_average_precision"].mean(),
        "pr_auc_std": cv_scores["test_average_precision"].std(),
        "precision": precision_score(y, oof_pred, zero_division=0),
        "recall": recall_score(y, oof_pred, zero_division=0),
        "f1": f1_score(y, oof_pred, zero_division=0),
        "cm": cm,
    }


def format_cm(cm: np.ndarray) -> str:
    tn, fp, fn, tp = cm.ravel()
    return f"TN={tn} FP={fp} / FN={fn} TP={tp}"


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    y = df[LABEL_COL]

    print(f"데이터: {RAW_PATH} ({len(df)}건), FraudFound_P=1 비율 {y.mean():.4f}")

    X_baseline = _build_baseline_features(df)
    X_enhanced = df[ENHANCED_FEATURE_COLS]
    X_interpretable = pd.concat(
        [_build_baseline_features(df), df[INTERPRETABLE_ORDINAL_COLS], df[INTERPRETABLE_NOMINAL_COLS]], axis=1
    )

    experiments = [
        ("baseline_5feat_rf", X_baseline, _make_pipeline(_no_encoding_preprocessor(), RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED), smote=False)),
        ("enhanced_categorical_rf", X_enhanced, _make_pipeline(_plain_preprocessor(), RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED), smote=False)),
        ("enhanced_categorical_rf_classweight", X_enhanced, _make_pipeline(_plain_preprocessor(), RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced"), smote=False)),
        ("enhanced_categorical_rf_smote", X_enhanced, _make_pipeline(_plain_preprocessor(), RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED), smote=True)),
        ("enhanced_categorical_rf_smote_classweight", X_enhanced, _make_pipeline(_plain_preprocessor(), RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced"), smote=True)),
        ("enhanced_categorical_logreg_scaled_classweight", X_enhanced, _make_pipeline(_scaled_preprocessor(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED), smote=False)),
        ("interpretable_subset_rf_classweight (=라이브 앱 반영안)", X_interpretable, _make_pipeline(_interpretable_preprocessor(), RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced"), smote=False)),
    ]

    results = []
    for name, X, pipeline in experiments:
        print(f"평가 중: {name} ...")
        result = evaluate(name, X, y, pipeline)
        results.append(result)
        print(
            f"  AUC={result['auc_mean']:.4f}±{result['auc_std']:.4f}  "
            f"PR-AUC={result['pr_auc_mean']:.4f}±{result['pr_auc_std']:.4f}  "
            f"P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f}  "
            f"{format_cm(result['cm'])}"
        )

    lines = [
        "# ML 스코어링 모델 보강 전/후 성능 비교",
        "",
        f"- 데이터: `{RAW_PATH.as_posix()}` (Kaggle 원본 전체 {len(df)}건, PolicyNumber 제외 33컬럼 중 32개 대상)",
        f"- 라벨 불균형: FraudFound_P=1 비율 {y.mean():.4f} ({int(y.sum())}건 / {len(y)}건)",
        f"- 검증 방법: StratifiedKFold({N_FOLDS}), 인코딩·SMOTE는 imblearn Pipeline으로 감싸 매 fold의 train 쪽에서만 fit (data leakage 방지)",
        "- baseline: 현재 app.py/ml_model.py가 실제로 쓰는 5개 수치형 피처(driver_age, vehicle_price, deductible, driver_rating, past_number_of_claims), RandomForest(n_estimators=200), 불균형 처리 없음",
        "- enhanced: 위 5개 + 나머지 범주형(명목형 One-Hot, 서열형 구간은 순서 있는 정수로 인코딩), ID성 컬럼(PolicyNumber)만 제외",
        "",
        "## 결과",
        "",
        "| 실험 | AUC-ROC | PR-AUC | Precision | Recall | F1 | Confusion Matrix (OOF) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['auc_mean']:.4f}±{r['auc_std']:.4f} | "
            f"{r['pr_auc_mean']:.4f}±{r['pr_auc_std']:.4f} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {format_cm(r['cm'])} |"
        )

    baseline = results[0]
    best_pr_auc = max(results, key=lambda r: r["pr_auc_mean"])
    lines += [
        "",
        "## 해석",
        "",
        f"- Baseline PR-AUC: {baseline['pr_auc_mean']:.4f}, 최고 PR-AUC: {best_pr_auc['name']} ({best_pr_auc['pr_auc_mean']:.4f}, "
        f"Δ{best_pr_auc['pr_auc_mean'] - baseline['pr_auc_mean']:+.4f})",
        f"- Baseline AUC-ROC: {baseline['auc_mean']:.4f}, 최고 AUC-ROC 대비 Δ{max(r['auc_mean'] for r in results) - baseline['auc_mean']:+.4f}",
        "- (아래 결론은 스크립트 실행 결과를 보고 채워 넣을 것 — 정량 수치 없이 서술하지 않는다)",
    ]

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
