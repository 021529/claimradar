import pandas as pd


def select_mask_top_pct(ml_score: pd.Series, pct: float) -> pd.Series:
    """ml_score 상위 pct 비율(0~1)에 해당하는 사건만 True (LLM 호출 대상)."""
    if pct >= 1.0:
        return pd.Series(True, index=ml_score.index)
    n_selected = max(1, int(len(ml_score) * pct))
    rank_cutoff = ml_score.sort_values(ascending=False).iloc[n_selected - 1]
    return ml_score >= rank_cutoff


def select_mask_threshold(ml_score: pd.Series, threshold: float) -> pd.Series:
    """ml_score >= threshold 인 사건만 True (LLM 호출 대상)."""
    return ml_score >= threshold
