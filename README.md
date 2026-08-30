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

## 발표 심사 대비 (2026-08-27)

### 1. 반응속도 실측 + 개선
- 스코어링(ML): 학습 0.22s, 예측 0.02s. LLM 사고경위서 분석은 건당 평균 5s, 순차 호출이라
  캐시 없는 신규 CSV 업로드 시 300건 기준 약 25분 — 데모용 데이터는 사전 캐시되어 있어
  실제 시연 시엔 이 병목을 건너뜀.
- 최적배정(OR-Tools): 조사관 5명 이상에서 CBC가 10초 타임리밋까지 항상 소진되던 문제를
  진단해 `SetTimeLimit`을 10s → 1.5s로 낮춤(5명 이상에서도 10.1s → 1.5~1.6s로 개선).
  최적성 격차는 데이터 규모에 따라 다르다(2026-08-30 ML 백본 기준 재측정,
  `scripts/solver_gap_check.py`·결과 `scripts/solver_gap_check_results.md`): **24건
  앱 데모 규모는 조사관 수·λ와 무관하게 항상 진짜 최적해(gap 0%)**이고, **300건
  이상에서 조사관 수·λ가 커지면 1.5초로도 최적성이 증명되지 않은 FEASIBLE 해로
  끝나는 경우가 있으며, 그때 목적함수 값은 200ms 해 대비 최대 약 8% 더 크다**
  (즉 이 규모에서는 1.5초 해도 "진짜 최적"이라 단정할 수 없고, 시간을 더 주면
  더 나은 해를 찾을 여지가 있다는 뜻). 이전에 있던 "0.05%/1% 미만" 수치는
  2026-08-27(옛 5피처 ML 백본) 측정값으로, 위 재측정 결과로 대체함.
- 조사가이드 생성: 사건별 캐싱 도입(`app.py` `guide_checklists` dict) — 같은 사건을 다시
  선택해도 재호출 없이 즉시 표시.

### 2. AI 환각 방어 논리 정리
- `llm_analysis.py` / `guide_generator.py`의 실제 방어 장치를 Q&A 대비용으로 정리: tool use
  강제로 자유 서술 여지 차단, 프롬프트의 "본문에 없는 내용 추측 금지" 명시적 지시, 숫자 필드
  클램핑·빈 항목 필터링 등 출력 위생처리, API 미설정/실패 시 결정론적 규칙 기반 폴백.
- 두 파일 간 비대칭(가이드 생성 프롬프트에 grounding 지시가 없던 점)을 이번에 보완.

### 3. 목적함수 위험도 가중치(λ) + 조사가이드 SIU 프로세스 반영
- `optimize_assignment`에 `risk_weight`(λ) 파라미터 추가 — 가중합 스칼라화(weighted sum
  scalarization, 자체 설계·선행연구 수식 인용 아님) 적용, λ=0이면 기존과 완전 동일.
  실측(300건)상 λ가 어느 지점까지는 회수액 손실 없이 고위험 커버리지가 개선되고, 그 이상은
  회수액 대비 커버리지 트레이드오프가 뚜렷해짐 — 다만 이 패턴은 데이터/캐파에 따라 달라질 수
  있어(목적함수가 "상위 20% 건수"가 아닌 "점수 합"을 최대화하므로 표본이 작으면 역전 가능),
  UI 안내 문구는 정적 예측 대신 배정 실행 후 실제 결과를 그대로 요약하는 방식으로 구현.
  `app.py`에 λ 슬라이더 + 순회수액/고위험 커버리지(상위 20%) 대시보드 지표 추가.
- `guide_generator.py` 체크리스트를 실제 SIU(보험사기특별조사팀) 조사 절차 7단계 중, 사건
  배정 후 조사관이 실제 수행하는 조회→자료취합→분석 3단계에 대응하도록 재구성 — 항목마다
  `[조회]`/`[자료취합]`/`[분석]` 태그 부여. 근거: 송윤아(2011), 「사기성 클레임에 대한
  최적 조사방안」, 보험연구원 경영보고서 2011-5, `<표 Ⅱ-14>` (통계 자체는 2011년 기준이나
  조사 프로세스 기본 구조는 최신 업계 동향으로도 재확인).
- 배포본에서 위 개선이 반영되지 않고 3개짜리 원론적 체크리스트만 나오는 이슈 발생 —
  원인은 Streamlit Cloud Secrets의 `ANTHROPIC_API_KEY`가 비어 있었던 것(코드 문제 아님).
  Secrets 재설정 + reboot으로 해결 확인. 재발 방지로 `anthropic` SDK 버전을 `>=1.0.0`으로
  상향 고정하고, LLM 호출 실패 시 폴백 원인이 로그에 남도록 `logging.exception` 추가.

### 다음에 이어서 할 것
- 실제 Kaggle 원본 데이터(`data/raw/`)로 스코어링·최적화 파이프라인 재검증 (현재는 합성
  데이터 위주로 검증됨)
- LLM 폴백 발생 시 사용자에게도 화면에 알려주는 UI 표시 (현재는 로그에만 남음, 로그는
  Streamlit Cloud 대시보드에서만 확인 가능)
- λ 슬라이더의 실시간 조작 데모를 실제 브라우저로 한 번 더 리허설 (이번 세션은 브라우저
  자동화 연결이 안 돼 Streamlit `AppTest` 헤드리스 검증으로만 확인함)

## 다음 단계 (DEV_SPEC.md 스프린트 순서)

1. ~~데이터 로딩 + 기초 EDA~~
2. ~~ML 스코어링 모델~~
3. ~~LLM 비정형 텍스트 분석 모듈 (합성 데이터 생성 포함)~~
4. ~~OR-Tools 최적화 모듈 + baseline 비교~~
5. ~~LLM 조사가이드 생성 모듈~~
6. ~~Streamlit UI 통합~~ (에러 처리, 로딩 상태, 사건 상세 뷰(스코어 근거/LLM 요약/체크리스트 통합) 반영 완료)
7. ~~배포 (Streamlit Community Cloud)~~ — https://github.com/021529/claimradar 연결, 배포 완료
8. ~~발표 심사 대비 (반응속도/환각 방어/λ 트레이드오프+조사가이드 개선)~~ — 위 섹션 참고, Streamlit Cloud에서 실동작 확인 완료
