# 클레임레이더 (가제)

보험사기 조사 우선순위를 AI 스코어링 + 수리적 최적화로 배정하는 웹서비스 MVP.
스펙 원본: [`../DEV_SPEC.md`](../DEV_SPEC.md)

## 실행 방법

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # ANTHROPIC_API_KEY 채우기
streamlit run app.py
```

## 프로젝트 구조

```
claimradar/
├── app.py                     # Streamlit 대시보드 엔트리포인트
├── src/
│   ├── config.py               # 경로, 가중치, API 키 등 설정
│   ├── data/loader.py          # CSV 업로드/샘플 로딩, 정규화
│   ├── scoring/
│   │   ├── ml_model.py         # RandomForest 기반 정형 피처 스코어링
│   │   ├── fraud_patterns.py   # 국내 특유 사기패턴 정의 + 정형 레코드 기반 트리거
│   │   ├── llm_analysis.py     # 합성 사고경위서 생성 + 이상징후 키워드/의심 요약 분석
│   │   └── combine.py          # ML + LLM 결합 스코어 계산
│   ├── optimization/
│   │   ├── baseline.py         # 단순 스코어 내림차순 배정 (비교군)
│   │   └── assignment.py       # OR-Tools 기반 최적 배정
│   └── guide/guide_generator.py # LLM 조사 체크리스트 생성
├── scripts/generate_synthetic_dataset.py  # Kaggle 원본 CSV -> 한글 합성 사고경위서 매핑 배치 스크립트
├── data/
│   ├── sample/sample_claims.csv   # 데모용 소규모 합성 데이터 (24건, 한글 사고경위서 포함)
│   ├── raw/                       # (git 미포함) Kaggle 원본 CSV를 직접 내려받아 배치
│   └── processed/                 # (git 미포함) 배치 스크립트 산출물
└── tests/                      # 최적화/스코어링 모듈 단위 테스트
```

## 현재 상태

- 데이터 로딩, ML 스코어링, OR-Tools 최적화/baseline 비교, Streamlit UI: 동작하는 스캐폴드 구현 완료
- 한글 합성 사고경위서 생성(`src/scoring/llm_analysis.py::generate_synthetic_narrative`): 구현 완료.
  국내 특유 사기패턴(나이롱환자/한방병원 장기입원/지연신고/블랙박스 미장착, `src/scoring/fraud_patterns.py`)을
  정형 레코드 신호에 따라 확률적으로 반영. `ANTHROPIC_API_KEY` 설정 시 LLM 생성, 미설정 시 결정론적 템플릿 폴백.
- 이상징후 분석(`analyze_narrative`): 구현 완료. 사고경위서를 읽고 tool use로 이상징후 키워드 + 의심도
  보정치(-1.0~1.0) + 1~2문장 의심 요약문을 구조화 추출. `ANTHROPIC_API_KEY` 미설정 시 키워드 사전 기반
  휴리스틱으로 폴백(정확도는 LLM 경로보다 낮음). Streamlit UI 스코어링 단계에 키워드/요약 컬럼 반영 완료.
- LLM 조사 체크리스트 생성(`src/guide/guide_generator.py::generate_investigation_checklist`): 구현 완료.
  배정된 사건의 사고경위서 + 이상징후 키워드를 받아 실무자용 확인 포인트 체크리스트 3~5개를 생성
  (대화 스크립트가 아닌 "확인할 것" 위주의 한두 문장 항목). `ANTHROPIC_API_KEY` 설정 시 LLM 생성,
  미설정 시 키워드→확인 포인트 템플릿 매핑으로 폴백. 앱 5단계에 사건 선택 + 체크박스 UI로 반영 완료.
- `data/sample/sample_claims.csv`는 파이프라인 시연용 소규모 합성 데이터(한글 사고경위서 포함, 국내 사기패턴 4종 예시 반영).
  실제 Kaggle "Vehicle Insurance Claim Fraud Detection" 데이터는 `data/raw/`에 직접 내려받은 뒤
  `python scripts/generate_synthetic_dataset.py --input data/raw/<파일명>.csv --output data/processed/claims_with_narratives.csv` 로 합성 사고경위서를 매핑 (DEV_SPEC.md 데이터 섹션 참고)

## 다음 단계 (DEV_SPEC.md 스프린트 순서)

1. ~~데이터 로딩 + 기초 EDA~~
2. ~~ML 스코어링 모델~~
3. ~~LLM 비정형 텍스트 분석 모듈 (합성 데이터 생성 포함)~~
4. ~~OR-Tools 최적화 모듈 + baseline 비교~~
5. ~~LLM 조사가이드 생성 모듈~~
6. ~~Streamlit UI 통합~~ (기본 골격, 세부 UI/UX 다듬기 필요)
7. 배포 (Streamlit Community Cloud) + 버그 수정
