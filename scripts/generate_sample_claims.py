"""data/sample/sample_claims.csv ("샘플로 체험하기" 버튼이 로드하는 데모 파일) 재생성.

이 파일은 원래 생성 스크립트 없이 손으로 만들어져 있었고, expected_recovery/
expected_hours/investigation_cost 값이 prepare_app_dataset.py의 공식과 맞지
않는 버그가 있었다 (예: vehicle_price=18000인데 expected_recovery=12000으로
vehicle_price*0.4 공식과 불일치). 이 스크립트는 prepare_app_dataset.py의
transform()을 그대로 재사용해 Kaggle 원본 24건(사기 12 / 정상 12)을 공식에
맞게 통과시켜 sample_claims.csv를 재생성한다. API 호출 없음 (narrative_text는
data/processed/claims_with_narratives.csv에 이미 생성돼 있는 값을 그대로 사용).

사용법:
    python scripts/generate_sample_claims.py

재현하려면 위 명령을 그대로 실행하면 된다 (입력 파일 경로/건수/시드가 모두
고정값으로 스크립트에 박혀 있음).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prepare_app_dataset import _llm_generated_mask, transform  # noqa: E402

INPUT_PATH = Path("data/processed/claims_with_narratives.csv")
OUTPUT_PATH = Path("data/sample/sample_claims.csv")
FRAUD_COUNT = 12
NORMAL_COUNT = 12
SEED = 42


def main() -> None:
    import pandas as pd

    df = pd.read_csv(INPUT_PATH)
    df = df.loc[_llm_generated_mask(df)]

    fraud = df.loc[df["FraudFound_P"] == 1]
    normal = df.loc[df["FraudFound_P"] == 0]
    fraud_sample = fraud.sample(n=min(FRAUD_COUNT, len(fraud)), random_state=SEED)
    normal_sample = normal.sample(n=min(NORMAL_COUNT, len(normal)), random_state=SEED)

    combined = pd.concat([fraud_sample, normal_sample]).sample(frac=1, random_state=SEED)
    result = transform(combined)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(
        f"완료: {OUTPUT_PATH} ({len(result)}건, 사기 {len(fraud_sample)}건 / "
        f"정상 {len(normal_sample)}건)"
    )


if __name__ == "__main__":
    main()
