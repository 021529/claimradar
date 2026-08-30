import pandas as pd

from src.config import SAMPLE_DATA_DIR

SAMPLE_CLAIMS_FILE = SAMPLE_DATA_DIR / "sample_claims.csv"


def load_sample_claims() -> pd.DataFrame:
    """샘플로 체험하기 버튼 - 번들된 샘플 청구 데이터 로드."""
    return pd.read_csv(SAMPLE_CLAIMS_FILE)


def load_uploaded_claims(uploaded_file) -> pd.DataFrame:
    """사용자가 업로드한 CSV 청구 데이터 로드."""
    return pd.read_csv(uploaded_file)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명 정규화 (공백/대소문자 등 통일)."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df
