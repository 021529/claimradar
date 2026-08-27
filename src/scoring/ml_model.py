import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED

LABEL_COL = "FraudFound_P"


def train_fraud_model(df: pd.DataFrame, label_col: str = LABEL_COL):
    """정형 피처로 사기 확률 예측 모델 학습 (RandomForest baseline).

    TODO: 피처 엔지니어링, XGBoost 비교, 클래스 불균형 처리(SMOTE 등).
    """
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    model = RandomForestClassifier(random_state=RANDOM_SEED, n_estimators=200)
    model.fit(X_train, y_train)
    return model, (X_test, y_test)


def predict_fraud_score(model, df: pd.DataFrame) -> pd.Series:
    """0~1 사기 확률 스코어 산출."""
    proba = model.predict_proba(df)[:, 1]
    return pd.Series(proba, index=df.index, name="ml_score")
