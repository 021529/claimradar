import pandas as pd

from src.config import SCORE_WEIGHT_LLM, SCORE_WEIGHT_ML


def combine_scores(
    ml_score: pd.Series,
    llm_adjustment: pd.Series,
    w1: float = SCORE_WEIGHT_ML,
    w2: float = SCORE_WEIGHT_LLM,
) -> pd.Series:
    """결합 스코어 = ML 스코어 * w1 + LLM 보정 * w2, 0~1로 클리핑."""
    combined = ml_score * w1 + llm_adjustment * w2
    return combined.clip(lower=0, upper=1).rename("combined_score")
