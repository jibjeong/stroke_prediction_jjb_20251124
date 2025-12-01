# Stroke Prediction Using Machine Learning

뇌졸중(Stroke) 발생 위험을 예측하는 머신러닝 프로젝트입니다. 포스터 기반 분석을 재현하여 7가지 머신러닝 모델의 성능을 비교하고 평가합니다.

## 📊 프로젝트 개요

### 목적
환자의 임상 데이터를 기반으로 뇌졸중 발생 위험을 예측하는 모델을 개발하고 평가합니다.

### 주요 특징
- **데이터**: 5,109명의 환자 데이터 (12개 변수)
- **모델**: 7가지 머신러닝 모델 비교
- **클래스 불균형 처리**: SMOTE 적용
- **평가**: 포스터 기반 Table 1, Table 2, Figure 3, Figure 4 생성

## 📁 프로젝트 구조

```
stroke_prediction_jjb_20251124/
├── dataset/                    # 원본 데이터
│   └── stroke_dataset.csv     # 5,110 rows × 12 columns
│
├── analysis_01_preprocess/     # Phase 1: 데이터 전처리
│   ├── scripts/
│   │   └── 01_data_preprocessing.py
│   └── data/
│       └── final/
│           ├── stroke_preprocessed.csv
│           └── stroke_original.csv
│
├── analysis_02_eda/            # Phase 2: 탐색적 데이터 분석
│   ├── scripts/
│   │   └── 01_eda_analysis.py
│   └── data/
│       ├── figures/           # 히트맵, 분포 그래프
│       └── tables/            # 통계 테이블
│
├── analysis_03_model/          # Phase 3: 모델 학습 및 평가
│   ├── scripts/
│   │   ├── baseline/
│   │   │   └── 01_train_models.py
│   │   └── evaluation/
│   │       ├── 02_shap_analysis.py
│   │       └── 03_additional_evaluation.py
│   └── data/
│       ├── tables/
│       │   ├── table1_average_performance.csv
│       │   ├── table2_class_performance.csv
│       │   ├── calibration_metrics.csv
│       │   ├── optimal_thresholds.csv
│       │   └── comprehensive_evaluation.csv
│       ├── figures/
│       │   ├── figure3_roc_curves_stroke_no.png
│       │   ├── figure4_roc_curves_stroke_yes.png
│       │   ├── confusion_matrices.png
│       │   ├── calibration_curves.png
│       │   ├── precision_recall_curves.png
│       │   ├── clinical_risk_stratification.png
│       │   ├── shap_importance_*.png (3 models)
│       │   ├── shap_summary_*.png (3 models)
│       │   └── shap_dependence_*.png (3 models)
│       └── models/            # 학습된 모델 (.pkl)
│
├── paper/                      # 학술 논문
│   ├── sections/               # 논문 섹션별 파일
│   │   ├── 01_background.md
│   │   ├── 02_methods.md
│   │   ├── 03_results.md
│   │   └── 04_conclusions.md
│   ├── main_paper.md          # 통합 논문 (14 figures)
│   ├── figures/               # 논문용 figure (상대경로 참조)
│   └── tables/                # 논문용 table
│
├── utils/                      # 공통 유틸리티
│   └── config.py
│
├── run_full_pipeline.py        # 전체 파이프라인 실행 스크립트
└── README.md                   # 본 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

#### 필수 패키지 설치
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn shap
```

#### 또는 requirements.txt 사용
```bash
pip install -r requirements.txt
```

### 2. 전체 파이프라인 실행

```bash
# 전체 파이프라인 한 번에 실행
python run_full_pipeline.py
```

### 3. 개별 Phase 실행

```bash
# Phase 1: 데이터 전처리
python analysis_01_preprocess/scripts/01_data_preprocessing.py

# Phase 2: EDA
python analysis_02_eda/scripts/01_eda_analysis.py

# Phase 3: 모델 학습 및 평가
python analysis_03_model/scripts/baseline/01_train_models.py

# Phase 4: SHAP 분석 (Feature Importance)
python analysis_03_model/scripts/evaluation/02_shap_analysis.py

# Phase 5: 추가 평가 (Calibration, PR Curves, Risk Stratification)
python analysis_03_model/scripts/evaluation/03_additional_evaluation.py
```

## 📊 데이터셋 정보

### 변수 설명 (12개)

| 변수 | 타입 | 설명 |
|------|------|------|
| id | int | 환자 고유 ID |
| gender | object | 성별 (Male, Female) |
| age | float | 나이 (0-82세) |
| hypertension | int | 고혈압 유무 (0/1) |
| heart_disease | int | 심장병 유무 (0/1) |
| ever_married | object | 결혼 여부 (Yes/No) |
| work_type | object | 직업 유형 (Private, Self-employed, Govt_job 등) |
| Residence_type | object | 거주 지역 (Urban, Rural) |
| avg_glucose_level | float | 평균 혈당 수치 (mg/dL) |
| bmi | float | 체질량지수 (kg/m²) |
| smoking_status | object | 흡연 상태 (never smoked, formerly smoked, smokes, Unknown) |
| **stroke** | int | **뇌졸중 발생 여부 (0/1)** - Target 변수 |

### 클래스 분포
- **Stroke=0 (비발생)**: 4,861명 (95.1%) ⚠️ 클래스 불균형
- **Stroke=1 (발생)**: 249명 (4.9%)

### 결측값
- **bmi**: 201개 (3.9%) → Median imputation 처리

## 🤖 모델 및 평가

### 7가지 머신러닝 모델

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **Support Vector Machine (SVM)**
5. **XGBoost**
6. **Gradient Boosting**
7. **Neural Network (MLP)**

### 평가 결과

#### 기본 성능 평가
- **Table 1**: Average Performance (AUC, CA, F1, Precision, Recall)
- **Table 2**: Performance by Class (Stroke=No vs Stroke=Yes)
- **Figure 3-4**: ROC Curves (Class 0 & Class 1)

#### 고급 평가
- **Calibration Analysis**: 모델 보정 성능 (Brier Score)
- **Precision-Recall Curves**: 불균형 데이터 성능 평가
- **Threshold Optimization**: Youden's Index 기반 최적 임계값
- **Risk Stratification**: Low/Medium/High 위험군 분류

#### Feature Importance
- **Correlation Analysis**: Pearson correlation with stroke
- **SHAP Analysis**:
  - Importance plots (Random Forest, XGBoost, Gradient Boosting)
  - Summary plots (feature value distribution)
  - Dependence plots (top 6 features)

### 클래스 불균형 처리
- **방법**: SMOTE (Synthetic Minority Over-sampling Technique)
- **결과**: 1:19 → 1:1 비율로 조정

## 📈 주요 결과

### 모델 성능 (실제 결과)

| 모델 | AUROC | Accuracy | F1-Score | Precision | Recall |
|------|-------|----------|----------|-----------|--------|
| **Logistic Regression** | **0.8245** | 0.7877 | 0.5602 | 0.5643 | 0.7461 |
| Decision Tree | 0.8026 | 0.7847 | 0.5543 | 0.5565 | 0.7600 |
| Random Forest | 0.8088 | 0.7896 | 0.5656 | 0.5678 | 0.7600 |
| SVM | 0.8060 | 0.7896 | 0.5646 | 0.5668 | 0.7600 |
| XGBoost | 0.8165 | 0.7906 | 0.5682 | 0.5704 | 0.7600 |
| **Gradient Boosting** | 0.8133 | 0.7906 | **0.5676** | 0.5698 | 0.7600 |
| Neural Network | 0.8100 | 0.7877 | 0.5628 | 0.5651 | 0.7600 |

**주요 발견**:
- **Logistic Regression**: 가장 높은 AUROC (0.8245)
- **Gradient Boosting**: 최고 calibration (Brier Score=0.0801)
- **Random Forest**: 균형잡힌 성능 (Sensitivity=0.84, Specificity=0.96)

### Feature Importance (SHAP)

**Top 5 중요 변수**:
1. **Age** (나이) - 압도적 1위 (SHAP: 0.161-2.626)
2. **BMI** (체질량지수)
3. **Avg Glucose Level** (평균 혈당)
4. **Hypertension** (고혈압)
5. **Heart Disease** (심장병)

### Risk Stratification

| 위험군 | 환자 비율 | 실제 뇌졸중 발생률 |
|--------|-----------|-------------------|
| Low Risk | 65.2% | **1.7%** |
| Medium Risk | 27.0% | **10.0%** |
| High Risk | 7.8% | **21.2%** |

## 📂 출력 파일

### 전처리 결과
- `analysis_01_preprocess/data/final/stroke_preprocessed.csv` - 전처리된 데이터
- `analysis_01_preprocess/data/final/stroke_original.csv` - 원본 데이터
- `analysis_01_preprocess/data/final/feature_names.txt` - Feature 목록

### EDA 결과
- `analysis_02_eda/data/figures/correlation_heatmap.png` - 상관관계 히트맵
- `analysis_02_eda/data/figures/feature_distributions.png` - 변수 분포
- `analysis_02_eda/data/figures/categorical_analysis.png` - 범주형 변수 분석
- `analysis_02_eda/data/tables/*.csv` - 통계 테이블

### 모델 결과

- **Tables**:
  - `table1_average_performance.csv` - 평균 성능 메트릭
  - `table2_class_performance.csv` - 클래스별 성능 메트릭
  - `calibration_metrics.csv` - Calibration (Brier Score)
  - `optimal_thresholds.csv` - 최적 임계값 (Youden's Index)
  - `comprehensive_evaluation.csv` - 종합 평가

- **Figures**:
  - `figure3_roc_curves_stroke_no.png` - Stroke=No ROC 곡선
  - `figure4_roc_curves_stroke_yes.png` - Stroke=Yes ROC 곡선
  - `confusion_matrices.png` - 7개 모델 Confusion Matrix
  - `calibration_curves.png` - Calibration 곡선
  - `precision_recall_curves.png` - Precision-Recall 곡선
  - `clinical_risk_stratification.png` - 위험군 분류
  - `correlation_with_stroke.png` - Correlation 분석
  - `shap_importance_*.png` - SHAP 중요도 (3개 모델)
  - `shap_summary_*.png` - SHAP Summary (3개 모델)
  - `shap_dependence_*.png` - SHAP Dependence (3개 모델)

- **Models**:
  - `*.pkl` - 학습된 모델 (7개)

### 논문 (Paper)
- **Sections**:
  - `01_background.md` - 연구 배경 및 목적
  - `02_methods.md` - 연구 방법론
  - `03_results.md` - 연구 결과 (14 figures)
  - `04_conclusions.md` - 결론 및 고찰
- **Main Paper**:
  - `main_paper.md` - 통합 논문 (~15,000 단어)

## ⚙️ 설정

### 랜덤 시드
재현성을 위해 `RANDOM_SEED = 42`로 고정되어 있습니다.

### Train/Test Split
- **Train**: 80%
- **Test**: 20%
- **Stratified Split**: 클래스 비율 유지

### 하이퍼파라미터
각 모델의 하이퍼파라미터는 `analysis_03_model/scripts/baseline/01_train_models.py`에서 수정 가능합니다.

## 🔧 문제 해결

### 한글 폰트 깨짐
```python
# Windows
plt.rcParams['font.family'] = 'Malgun Gothic'

# macOS
plt.rcParams['font.family'] = 'AppleGothic'

# Linux
plt.rcParams['font.family'] = 'NanumGothic'
```

### 메모리 부족
데이터 크기가 작아 문제 없지만, 필요 시 샘플링:
```python
df = pd.read_csv('dataset/stroke_dataset.csv', nrows=1000)
```

### 패키지 설치 오류
```bash
# scikit-learn 업그레이드
pip install --upgrade scikit-learn

# imbalanced-learn 설치
pip install imbalanced-learn
```

## 📚 참고 자료

### 데이터 출처
- Kaggle: Stroke Prediction Dataset
- https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

### 관련 논문
- SMOTE: Chawla et al. (2002) - Synthetic Minority Over-sampling Technique
- WHO Stroke Guidelines: https://www.who.int/news-room/fact-sheets/detail/stroke

## 👥 프로젝트 정보

- **작성자**: Data Science Team
- **날짜**: 2025-11-24
- **버전**: 1.0
- **Python**: ≥ 3.10

## ⚠️ 제한사항

1. **데이터 크기**: 5,109명 (소규모 데이터셋)
2. **클래스 불균형**: Stroke=1 (4.9%) - SMOTE로 처리
3. **결측값**: BMI 3.9% - Median imputation
4. **일반화**: 단일 데이터셋으로 외부 검증 필요
5. **시간 정보**: 단일 시점 데이터 (시계열 아님)

## 📝 라이센스

본 프로젝트는 교육 및 연구 목적으로만 사용됩니다.

---

**마지막 업데이트**: 2025-11-24
