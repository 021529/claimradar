import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = ROOT_DIR / "data" / "sample"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = "claude-sonnet-5"

# 결합 스코어 = ML_SCORE * SCORE_WEIGHT_ML + LLM_SCORE * SCORE_WEIGHT_LLM
SCORE_WEIGHT_ML = 0.7
SCORE_WEIGHT_LLM = 0.3

RANDOM_SEED = 42
