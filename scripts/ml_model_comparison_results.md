# ML 스코어링 모델 보강 전/후 성능 비교

- 데이터: `data/raw/fraud_oracle.csv` (Kaggle 원본 전체 15420건, PolicyNumber 제외 33컬럼 중 32개 대상)
- 라벨 불균형: FraudFound_P=1 비율 0.0599 (923건 / 15420건)
- 검증 방법: StratifiedKFold(5), 인코딩·SMOTE는 imblearn Pipeline으로 감싸 매 fold의 train 쪽에서만 fit (data leakage 방지)
- baseline: 현재 app.py/ml_model.py가 실제로 쓰는 5개 수치형 피처(driver_age, vehicle_price, deductible, driver_rating, past_number_of_claims), RandomForest(n_estimators=200), 불균형 처리 없음
- enhanced: 위 5개 + 나머지 범주형(명목형 One-Hot, 서열형 구간은 순서 있는 정수로 인코딩), ID성 컬럼(PolicyNumber)만 제외

## 결과

| 실험 | AUC-ROC | PR-AUC | Precision | Recall | F1 | Confusion Matrix (OOF) |
|---|---|---|---|---|---|---|
| baseline_5feat_rf | 0.5314±0.0169 | 0.0685±0.0035 | 0.076 | 0.015 | 0.025 | TN=14327 FP=170 / FN=909 TP=14 |
| enhanced_categorical_rf | 0.8182±0.0052 | 0.2175±0.0074 | 0.667 | 0.004 | 0.009 | TN=14495 FP=2 / FN=919 TP=4 |
| enhanced_categorical_rf_classweight | 0.8278±0.0075 | 0.2229±0.0112 | 0.361 | 0.062 | 0.105 | TN=14396 FP=101 / FN=866 TP=57 |
| enhanced_categorical_rf_smote | 0.8191±0.0083 | 0.2004±0.0127 | 0.625 | 0.011 | 0.021 | TN=14491 FP=6 / FN=913 TP=10 |
| enhanced_categorical_rf_smote_classweight | 0.8203±0.0060 | 0.1954±0.0146 | 0.500 | 0.005 | 0.011 | TN=14492 FP=5 / FN=918 TP=5 |
| enhanced_categorical_logreg_scaled_classweight | 0.7943±0.0134 | 0.1574±0.0121 | 0.130 | 0.840 | 0.225 | TN=9311 FP=5186 / FN=148 TP=775 |
| interpretable_subset_rf_classweight (=라이브 앱 반영안) | 0.8026±0.0075 | 0.1747±0.0073 | 0.221 | 0.131 | 0.165 | TN=14071 FP=426 / FN=802 TP=121 |

## 해석

**1. 범주형 피처 = 압도적 개선 (심사 지적이 정량적으로 확인됨)**

5개 수치형만 쓰는 baseline은 AUC-ROC 0.5314로 사실상 랜덤 수준이다(0.5=동전 던지기). 범주형(Make/Sex/Fault/PolicyType/VehicleCategory/AccidentArea/PoliceReportFiled/WitnessPresent/AgentType/BasePolicy 등)과 서열형(VehiclePrice/PastNumberOfClaims/AgeOfVehicle/AgeOfPolicyHolder 등 구간 문자열)을 인코딩해 추가한 것만으로 AUC-ROC가 0.8182로, PR-AUC가 0.0685 → 0.2175(약 3.2배)로 뛴다. **"자전거 바퀴" 지적은 사실이었고, 범주형 피처 누락이 스코어 품질의 핵심 병목이었다.**

**2. 불균형 처리: class_weight='balanced'가 SMOTE보다 전 지표에서 우세**

동일한 범주형 피처 기준으로 SMOTE를 추가하면 PR-AUC(0.2175→0.2004)와 F1(class_weight의 0.105에 못 미치는 0.021)이 오히려 나빠진다. class_weight+SMOTE를 동시에 적용해도 더 나빠진다(PR-AUC 0.1954, F1 0.011). **class_weight='balanced' 단독이 AUC-ROC(0.8278, 전체 최고)·PR-AUC(0.2229, 전체 최고)·F1(0.105) 모두에서 SMOTE 계열을 이겼다.** → **채택: class_weight='balanced', SMOTE는 미채택** (imblearn 의존성도 프로덕션에 추가할 필요가 없어짐).

**3. 스케일링: RandomForest엔 불필요함을 확인**

StandardScaler + LogisticRegression(class_weight='balanced')은 재현율은 가장 높지만(0.840) 정밀도가 0.130까지 떨어져 FP가 5,186건(전체 정상건의 55%)에 달하고, 랭킹 품질(PR-AUC 0.1574)도 RF 계열보다 낮다. RandomForest는 스케일 불변이므로 프로덕션에 스케일링을 추가하지 않기로 결정.

**4. 라이브 앱 반영안(interpretable_subset_rf_classweight) 검증**

벤치마크 전체 피처셋(32컬럼, Month/DayOfWeek/RepNumber 등 해석성 낮은 시간성 컬럼 포함)이 아니라, 조사관 UI에 노출할 실익이 있는 부분집합만(5개 수치형 + 서열형 7개 + 명목형 11개, `scripts/prepare_app_dataset.py`의 `ORDINAL_ORDER`/`NOMINAL_COLS`와 정확히 동일) 골라 별도로 검증했다. 결과: **AUC-ROC 0.8026(Δ+0.2712), PR-AUC 0.1747(Δ+0.1062, 약 2.55배), F1 0.165(RF 계열 중 최고, Recall 0.131로 class_weight 단독 벤치마크의 0.062보다 오히려 높음)**. 전체 32컬럼 벤치마크보다는 PR-AUC가 다소 낮지만(0.2229 vs 0.1747) baseline 대비 개선폭은 압도적으로 유지되고, F1은 오히려 전체 피처셋보다 높다 — 해석 가능성을 위해 컬럼을 줄인 대가가 크지 않다는 뜻이다. **이 조합(범주형 One-Hot + 서열형 순서 인코딩 + RandomForest(class_weight='balanced'), 스케일링·SMOTE 없음)을 `src/scoring/ml_model.py`와 라이브 앱 파이프라인에 그대로 반영한다.**

**한계**: 그럼에도 절대 수준(PR-AUC 0.17, F1 0.17 내외)은 여전히 낮다 — 6% 불균형 사기탐지 문제의 특성상 예견된 결과이며, 이 프로젝트의 핵심 가치제안은 "ML 단독으로 사기를 완벽히 판별"이 아니라 "ML 스코어 + LLM 사고경위서 분석을 결합해 조사 우선순위를 배정"하는 것이므로, ML 스코어의 역할은 완벽한 분류기가 아니라 LLM 신호와 결합될 신뢰할 만한 랭킹 신호를 제공하는 것이다. 이번 보강으로 그 랭킹 신호의 신뢰도가 크게 개선됐다.