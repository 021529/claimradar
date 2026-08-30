import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import RANDOM_SEED

LABEL_COL = "FraudFound_P"


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """컬럼 dtype으로 수치형/범주형을 자동 구분해 전처리기를 구성한다.

    RandomForest는 스케일 불변이라 수치형은 그대로 통과시키고 범주형만
    One-Hot 인코딩한다. 서열형 구간 문자열(VehiclePrice 등)은 호출 측
    (scripts/prepare_app_dataset.py)이 이미 순서 보존 숫자로 변환해 넘기므로
    여기서는 dtype만 보면 된다 — 이 함수는 호출 측이 수치형만 넘기든
    (기존 5피처 스키마) 범주형까지 포함해 넘기든(보강된 스키마) 동일하게
    동작해 인터페이스 호환이 유지된다.

    결측치: 별도 대치(imputation) 전략은 구현돼 있지 않다. 수치형은
    "passthrough"라 결측이 그대로 RandomForest에 전달되며, sklearn(1.4+)의
    트리 분할이 NaN을 자체적으로 처리해 학습은 되지만 이는 명시적으로
    설계한 처리가 아니라 sklearn 기본 동작에 기대는 것이다. 범주형은
    OneHotEncoder가 NaN을 별도 카테고리로 원-핫 인코딩한다. 번들 데이터셋
    (data/sample/sample_claims.csv, data/processed/app_demo_sample.csv)에는
    결측치가 전혀 없어 지금까지 문제가 된 적은 없지만, 결측이 있는 데이터를
    업로드하면 이 암묵적 동작에 그대로 의존하게 된다.
    """
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    return ColumnTransformer(transformers)


def train_fraud_model(df: pd.DataFrame, label_col: str = LABEL_COL, class_weight: str | None = None):
    """정형 피처로 사기 확률 예측 모델 학습 (범주형 One-Hot + RandomForest).

    class_weight는 기본값 None(과거와 동일한 동작)으로 두어, 기존 스크립트들이
    이 함수를 인자 없이 호출하면 예전과 완전히 동일한 결과를 재현한다(회귀 방지).
    class_weight='balanced'는 SMOTE와 비교한 결과(scripts/ml_model_comparison.py,
    scripts/ml_model_comparison_results.md) AUC-ROC·PR-AUC·F1 전 지표에서 SMOTE보다
    우세했음이 검증됐고, 범주형까지 포함한 보강 스키마에서 app.py가 명시적으로
    켜서 쓴다. 스케일링은 RandomForest에 불필요함을 같은 벤치마크에서 확인해
    넣지 않았다.
    """
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    model = Pipeline(
        [
            ("prep", _build_preprocessor(X_train)),
            (
                "clf",
                RandomForestClassifier(
                    random_state=RANDOM_SEED, n_estimators=200, class_weight=class_weight
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model, (X_test, y_test)


def predict_fraud_score(model, df: pd.DataFrame) -> pd.Series:
    """0~1 사기 확률 스코어 산출."""
    proba = model.predict_proba(df)[:, 1]
    return pd.Series(proba, index=df.index, name="ml_score")
