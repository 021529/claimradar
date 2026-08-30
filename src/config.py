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

# 배포 앱은 퍼블릭 URL이라 누구나 CSV를 업로드해 "스코어링 실행"을 누를 수 있다.
# 업로드 데이터에는 LLM 캐시가 없으므로 행 수만큼 실제 Claude API 호출이 발생해,
# 상한이 없으면 대용량 CSV 하나로 API 크레딧이 즉시 소진될 수 있다. 100건은
# 심사위원이 자체 데이터로 기능을 테스트하기엔 충분하면서(샘플 데모 24건보다 넉넉),
# 캐시 없이 전량 호출되어도 예상 비용이 1달러 미만(app.py 비용 추정 참고)으로
# 제한되는 선이다. 실 서비스 규모(수천~수만 건)는 계층적 필터링 + 배치 처리로
# 대응한다(scripts/hierarchical_filtering.py 검증 참고).
MAX_UPLOAD_ROWS = 100

# 결합 스코어 = ML_SCORE * SCORE_WEIGHT_ML + LLM_SCORE * SCORE_WEIGHT_LLM
SCORE_WEIGHT_ML = 0.7
SCORE_WEIGHT_LLM = 0.3

RANDOM_SEED = 42
