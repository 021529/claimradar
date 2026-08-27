"""Kaggle 정형 데이터에 한글 합성 사고경위서를 매핑해 학습용 데이터셋을 생성한다.

사전 준비:
    Kaggle "Vehicle Insurance Claim Fraud Detection" CSV를 내려받아
    data/raw/ 아래에 두어야 한다.
    https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection

사용법:
    python scripts/generate_synthetic_dataset.py \
        --input data/raw/fraud_oracle.csv \
        --output data/processed/claims_with_narratives.csv \
        [--limit 200] [--style-reference path/to/nhtsa_sample.txt] \
        [--normal-sample 1000] [--concurrency 8]

ANTHROPIC_API_KEY가 설정되어 있지 않으면 전 건이 결정론적 템플릿 폴백으로 생성된다
(src/scoring/llm_analysis.py의 generate_synthetic_narrative 참고).

--normal-sample N을 지정하면 사기(라벨=1) 건은 전부, 정상 건은 N건만 무작위 표본으로
LLM 생성하고 나머지 정상 건은 템플릿 폴백으로 채운다 — Kaggle 데이터처럼 정상 건이
압도적으로 많을 때 API 호출량/비용/시간을 줄이기 위한 옵션. 미지정 시 전 건을 LLM으로
생성 시도한다(대량 데이터에서는 매우 느리고 비용이 큼).
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.scoring.fraud_patterns import select_patterns  # noqa: E402
from src.scoring.llm_analysis import _template_narrative, generate_synthetic_narrative  # noqa: E402


def _generate_for_position(pos, row, use_llm: bool, style_reference_text: str) -> tuple[int, str]:
    if use_llm:
        return pos, generate_synthetic_narrative(row, style_reference_text)
    patterns = select_patterns(row)
    return pos, _template_narrative(row, patterns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None, help="상위 N건만 처리 (테스트용)")
    parser.add_argument("--style-reference", type=Path, default=None)
    parser.add_argument("--label-col", default="FraudFound_P")
    parser.add_argument(
        "--normal-sample",
        type=int,
        default=None,
        help="정상 건 중 LLM으로 생성할 표본 크기. 미지정 시 정상 건도 전부 LLM 생성 시도.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8, help="동시 LLM 호출 수")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit)
    df = df.reset_index(drop=True)

    style_reference_text = ""
    if args.style_reference and args.style_reference.exists():
        style_reference_text = args.style_reference.read_text(encoding="utf-8")

    fraud_mask = df[args.label_col] == 1
    if args.normal_sample is not None:
        normal_positions = df.index[~fraud_mask]
        sampled = pd.Series(normal_positions).sample(
            n=min(args.normal_sample, len(normal_positions)), random_state=args.seed
        )
        llm_positions = set(df.index[fraud_mask]) | set(sampled)
    else:
        llm_positions = set(df.index)

    total = len(df)
    narratives: list[str | None] = [None] * total

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                _generate_for_position, pos, df.loc[pos], pos in llm_positions, style_reference_text
            )
            for pos in range(total)
        ]
        completed = 0
        for future in as_completed(futures):
            pos, text = future.result()
            narratives[pos] = text
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"  {completed}/{total} 건 생성 완료 (LLM 대상 {len(llm_positions)}건)", file=sys.stderr)

    df["narrative_text"] = narratives
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(
        f"완료: {args.output} ({total}건, LLM 생성 {len(llm_positions)}건 / "
        f"템플릿 폴백 {total - len(llm_positions)}건)"
    )


if __name__ == "__main__":
    main()
