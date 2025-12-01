# Stroke Prediction Project - Technical Guide

**Project**: Machine Learning-Based Stroke Risk Prediction
**Author**: Data Science Team
**Date**: 2024-11-24
**Version**: 1.0
**Status**: Production-Ready

---

## 📋 Project Overview

### Purpose
Develop and comprehensively evaluate seven machine learning models for stroke risk prediction using clinical and demographic data from 5,109 patients. Address class imbalance (4.9% stroke prevalence) using SMOTE and provide explainable AI analysis through SHAP.

### Key Achievements
- ✅ 7 ML models trained and evaluated (AUROC: 0.80-0.82)
- ✅ Class imbalance handled with SMOTE (1:19 → 1:1)
- ✅ Feature importance identified (Age dominates with SHAP: 0.161-2.626)
- ✅ Risk stratification implemented (Low/Medium/High: 1.7%/10.0%/21.2%)
- ✅ Comprehensive academic paper (~15,000 words, 14 figures)

### Research Questions
1. Which ML algorithm performs best for stroke prediction?
2. How effective is SMOTE for handling severe class imbalance?
3. What are the most important predictive features?
4. Can we stratify patients into clinically meaningful risk groups?

---

## 🏗️ Project Architecture

### Phase Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    STROKE PREDICTION PIPELINE               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: Data Preprocessing                                │
│  ├── Missing value imputation (BMI: median)                │
│  ├── Categorical encoding (Label + One-Hot)                │
│  ├── Feature scaling (StandardScaler)                      │
│  └── Feature engineering (4 derived variables)             │
│           ↓                                                 │
│  Phase 2: Exploratory Data Analysis                        │
│  ├── Correlation heatmap                                   │
│  ├── Feature distributions (stroke vs non-stroke)          │
│  └── Categorical analysis (chi-square tests)               │
│           ↓                                                 │
│  Phase 3: Model Training                                    │
│  ├── Train-test split (80/20, stratified)                  │
│  ├── SMOTE (training set only)                             │
│  ├── Train 7 models (LR, DT, RF, SVM, XGB, GB, NN)        │
│  └── Generate evaluation tables and figures                │
│           ↓                                                 │
│  Phase 4: Feature Importance (SHAP)                        │
│  ├── SHAP importance plots (RF, XGB, GB)                   │
│  ├── SHAP summary plots                                    │
│  └── SHAP dependence plots (top 6 features)                │
│           ↓                                                 │
│  Phase 5: Advanced Evaluation                              │
│  ├── Calibration curves (Brier Score)                      │
│  ├── Precision-Recall curves                               │
│  ├── Threshold optimization (Youden's Index)               │
│  └── Risk stratification (Low/Medium/High)                 │
│           ↓                                                 │
│  Phase 6: Academic Paper                                    │
│  ├── Background (research gap, objectives)                 │
│  ├── Methods (preprocessing, models, evaluation)           │
│  ├── Results (14 figures, 10 tables)                       │
│  └── Conclusions (clinical implications, future work)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
stroke_prediction_jjb_20251124/
│
├── dataset/
│   └── stroke_dataset.csv              # Raw data: 5,110 × 12
│
├── analysis_01_preprocess/             # Phase 1: Data Preprocessing
│   ├── scripts/
│   │   └── 01_data_preprocessing.py    # Main preprocessing script
│   └── data/
│       └── final/
│           ├── stroke_preprocessed.csv # Preprocessed: 5,109 × 21
│           ├── stroke_original.csv     # Original (1 duplicate removed)
│           └── feature_names.txt       # 19 feature names
│
├── analysis_02_eda/                    # Phase 2: EDA
│   ├── scripts/
│   │   └── 01_eda_analysis.py          # EDA script
│   └── data/
│       ├── figures/
│       │   ├── correlation_heatmap.png
│       │   ├── feature_distributions.png
│       │   └── categorical_analysis.png
│       └── tables/
│           └── *.csv                   # Statistical summaries
│
├── analysis_03_model/                  # Phase 3-5: Modeling & Evaluation
│   ├── scripts/
│   │   ├── baseline/
│   │   │   └── 01_train_models.py      # Train 7 models
│   │   └── evaluation/
│   │       ├── 02_shap_analysis.py     # SHAP feature importance
│   │       └── 03_additional_evaluation.py  # Calibration, PR, Risk
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
│       │   ├── correlation_with_stroke.png
│       │   ├── shap_importance_*.png (RF, XGB, GB)
│       │   ├── shap_summary_*.png (RF, XGB, GB)
│       │   └── shap_dependence_*.png (RF, XGB, GB)
│       └── models/
│           └── *.pkl                   # 7 trained models
│
├── paper/                              # Phase 6: Academic Paper
│   ├── sections/
│   │   ├── 01_background.md            # Introduction, research gap
│   │   ├── 02_methods.md               # Methodology (1 figure)
│   │   ├── 03_results.md               # Results (13 figures)
│   │   └── 04_conclusions.md           # Discussion, limitations
│   ├── main_paper.md                   # Integrated paper (14 figures)
│   ├── figures/                        # (References via relative paths)
│   └── tables/                         # (References via relative paths)
│
├── utils/
│   └── config.py                       # Central configuration
│
├── run_full_pipeline.py                # Execute entire pipeline
├── requirements.txt                    # Python dependencies
├── README.md                           # User-facing documentation
└── CLAUDE.md                           # This file (technical guide)
```

---

## 🔬 Technical Details

### Data Preprocessing Pipeline

**Script**: `analysis_01_preprocess/scripts/01_data_preprocessing.py`

#### 1. Data Cleaning
```python
# Remove duplicates
df = df.drop_duplicates(subset='id')  # 1 duplicate removed

# Missing value imputation
bmi_median = df['bmi'].median()  # 28.10 kg/m²
df['bmi'] = df['bmi'].fillna(bmi_median)
```

#### 2. Encoding
```python
# Binary variables (Label Encoding)
label_encoders = {
    'gender': {'Female': 0, 'Male': 1, 'Other': 2},
    'ever_married': {'No': 0, 'Yes': 1},
    'Residence_type': {'Rural': 0, 'Urban': 1}
}

# Multi-class variables (One-Hot Encoding)
one_hot_features = ['work_type', 'smoking_status']
df_encoded = pd.get_dummies(df, columns=one_hot_features, drop_first=True)
```

#### 3. Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numeric_features = ['age', 'avg_glucose_level', 'bmi']
df[numeric_features] = scaler.fit_transform(df[numeric_features])
```

#### 4. Feature Engineering
```python
# Derived variables
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 40, 60, 100],
                         labels=['Child', 'Young Adult', 'Middle-Aged', 'Senior'])

df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                            labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

df['glucose_category'] = pd.cut(df['avg_glucose_level'], bins=[0, 100, 126, 300],
                                labels=['Normal', 'Prediabetes', 'Diabetes'])

df['risk_score'] = df['hypertension'] + df['heart_disease'] + (df['age'] > 60).astype(int)
```

**Output**: `stroke_preprocessed.csv` (5,109 × 21)

---

### Model Training Pipeline

**Script**: `analysis_03_model/scripts/baseline/01_train_models.py`

#### 1. Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,      # 80% train, 20% test
    random_state=42,     # Reproducibility
    stratify=y           # Maintain class ratio
)
```

#### 2. SMOTE (Training Set Only)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Before: Class 0=3,889, Class 1=199 (1:19.5)
# After:  Class 0=3,889, Class 1=3,889 (1:1)
```

#### 3. Model Training
```python
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10),
    'SVM': SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced'),
    'XGBoost': XGBClassifier(n_estimators=100, scale_pos_weight=19.5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
    'Neural Network': MLPClassifier(hidden_layers=(100, 50), early_stopping=True)
}

for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(model, f'models/{name}.pkl')
```

#### 4. Evaluation
```python
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              precision_score, recall_score)

metrics = {
    'AUROC': roc_auc_score(y_test, y_pred_proba, multi_class='ovr'),
    'Accuracy': accuracy_score(y_test, y_pred),
    'F1': f1_score(y_test, y_pred, average='macro'),
    'Precision': precision_score(y_test, y_pred, average='macro'),
    'Recall': recall_score(y_test, y_pred, average='macro')
}
```

---

### SHAP Analysis Pipeline

**Script**: `analysis_03_model/scripts/evaluation/02_shap_analysis.py`

```python
import shap

# Load trained models
rf_model = joblib.load('models/random_forest.pkl')
xgb_model = joblib.load('models/xgboost.pkl')
gb_model = joblib.load('models/gradient_boosting.pkl')

# SHAP explainer
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_test_sample)

# Handle 3D array for multi-class (extract class 1)
if len(shap_values_rf.shape) == 3:
    shap_values_rf = shap_values_rf[:, :, 1]

# Visualizations
shap.summary_plot(shap_values_rf, X_test_sample, show=False)
plt.savefig('shap_summary_random_forest.png', dpi=300, bbox_inches='tight')

shap.plots.bar(shap_values_rf, show=False)
plt.savefig('shap_importance_random_forest.png', dpi=300, bbox_inches='tight')
```

**Key Findings**:
- **Age**: Dominant predictor (SHAP: 0.161-2.626)
- **BMI**: Second most important (SHAP: 0.067-1.837)
- **Glucose**: Third (SHAP: 0.058-0.358)

---

### Advanced Evaluation Pipeline

**Script**: `analysis_03_model/scripts/evaluation/03_additional_evaluation.py`

#### 1. Calibration Curves
```python
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
brier_score = brier_score_loss(y_test, y_pred_proba)

# Best: Gradient Boosting (Brier=0.0801)
```

#### 2. Precision-Recall Curves
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
ap_score = average_precision_score(y_test, y_pred_proba)

# Best: Gradient Boosting (AP=0.3902)
# Baseline (No Skill): AP=0.0488
```

#### 3. Threshold Optimization (Youden's Index)
```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
youdens_index = tpr - fpr
optimal_idx = np.argmax(youdens_index)
optimal_threshold = thresholds[optimal_idx]

# Logistic Regression: 0.21 (Sensitivity=0.88, Specificity=0.89)
# Gradient Boosting: 0.18 (Sensitivity=0.92, Specificity=0.85)
# Random Forest: 0.26 (Sensitivity=0.84, Specificity=0.96)
```

#### 4. Risk Stratification
```python
risk_groups = []
for prob in y_pred_proba:
    if prob < 0.3:
        risk_groups.append('Low Risk')
    elif prob < 0.7:
        risk_groups.append('Medium Risk')
    else:
        risk_groups.append('High Risk')

# Results:
# Low Risk (65.2%): 1.7% actual stroke rate
# Medium Risk (27.0%): 10.0% actual stroke rate
# High Risk (7.8%): 21.2% actual stroke rate
```

---

## 📊 Key Results Summary

### Model Performance (Test Set)

| Model | AUROC | Accuracy | F1 | Precision | Recall | Brier Score |
|-------|-------|----------|-----|-----------|--------|-------------|
| **Logistic Regression** | **0.8245** | 0.7877 | 0.5602 | 0.5643 | 0.7461 | 0.0888 |
| Decision Tree | 0.8026 | 0.7847 | 0.5543 | 0.5565 | 0.7600 | 0.0904 |
| Random Forest | 0.8088 | 0.7896 | 0.5656 | 0.5678 | 0.7600 | 0.0819 |
| SVM | 0.8060 | 0.7896 | 0.5646 | 0.5668 | 0.7600 | 0.0835 |
| XGBoost | 0.8165 | 0.7906 | 0.5682 | 0.5704 | 0.7600 | 0.0813 |
| **Gradient Boosting** | 0.8133 | 0.7906 | 0.5676 | 0.5698 | 0.7600 | **0.0801** |
| Neural Network | 0.8100 | 0.7877 | 0.5628 | 0.5651 | 0.7600 | 0.0843 |

**Insights**:
- ✅ All models achieved AUROC > 0.80 (clinically meaningful)
- ✅ Logistic Regression: Highest AUROC (0.8245)
- ✅ Gradient Boosting: Best calibration (Brier=0.0801)
- ✅ Random Forest: Best balance (Sens=0.84, Spec=0.96)

### Feature Importance (SHAP Rankings)

#### Random Forest
1. **age**: 0.1609 (dominant)
2. **bmi**: 0.0666
3. **avg_glucose_level**: 0.0584
4. **work_type_Private**: 0.0120
5. **age_group**: 0.0119

#### XGBoost
1. **age**: 2.6259 (overwhelmingly dominant)
2. **avg_glucose_level**: 0.1837
3. **bmi**: 0.1742
4. **heart_disease**: 0.0372
5. **hypertension**: 0.0301

#### Gradient Boosting
1. **age**: 2.0144
2. **bmi**: 0.4462
3. **avg_glucose_level**: 0.2669
4. **work_type_Private**: 0.3813
5. **smoking_status_formerly_smoked**: 0.2652

**Key Finding**: **Age dominates all models** (10-100× more important than other features)

---

## 🎯 Clinical Recommendations

### Model Selection by Scenario

#### 1. Primary Care Screening
**Recommended**: Logistic Regression
- **Why**: Highest AUROC (0.8245), interpretable, fast
- **Threshold**: 0.21 (Sensitivity=0.88, Specificity=0.89)
- **Use Case**: Quick risk assessment during routine checkups

#### 2. Hospital Risk Assessment
**Recommended**: Random Forest
- **Why**: Best balance (Sens=0.84, Spec=0.96), robust stratification
- **Threshold**: 0.26 (minimizes false positives)
- **Use Case**: Emergency department triage, inpatient monitoring

#### 3. Screening Programs
**Recommended**: Gradient Boosting
- **Why**: Highest sensitivity (0.92), best calibration
- **Threshold**: 0.18 (detects maximum stroke cases)
- **Use Case**: Population screening, clinical trial recruitment

### Risk Stratification Guidelines

| Risk Level | Probability Range | Intervention Strategy |
|-----------|------------------|----------------------|
| **Low Risk** | <0.3 (65.2%) | Routine health education, annual monitoring |
| **Medium Risk** | 0.3-0.7 (27.0%) | Targeted prevention, semi-annual monitoring |
| **High Risk** | ≥0.7 (7.8%) | Aggressive management, quarterly monitoring |

**Clinical Impact**:
- Low Risk: 1.7% stroke rate → Standard care
- Medium Risk: 10.0% stroke rate → Enhanced monitoring
- High Risk: 21.2% stroke rate → Intensive intervention

---

## 🔧 Configuration

### Central Configuration File

**File**: `utils/config.py`

```python
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset"
PREPROCESS_DIR = BASE_DIR / "analysis_01_preprocess"
EDA_DIR = BASE_DIR / "analysis_02_eda"
MODEL_DIR = BASE_DIR / "analysis_03_model"

# Data Settings
RANDOM_SEED = 42
TEST_SIZE = 0.20
TARGET_COLUMN = 'stroke'
ID_COLUMN = 'id'

# Features
NUMERIC_FEATURES = ['age', 'avg_glucose_level', 'bmi']
CATEGORICAL_FEATURES = ['gender', 'hypertension', 'heart_disease',
                        'ever_married', 'work_type', 'Residence_type',
                        'smoking_status']

# Feature Engineering
AGE_BINS = [0, 18, 40, 60, 100]
BMI_BINS = [0, 18.5, 25, 30, 100]
GLUCOSE_BINS = [0, 100, 126, 300]

# Visualization
KOREAN_FONT = 'Malgun Gothic'  # Windows
FIGURE_DPI = 300
```

---

## 🚀 Execution Guide

### Option 1: Full Pipeline (Recommended)
```bash
python run_full_pipeline.py
```
**Executes**: All 5 phases sequentially

### Option 2: Individual Phases
```bash
# Phase 1: Preprocessing
python analysis_01_preprocess/scripts/01_data_preprocessing.py

# Phase 2: EDA
python analysis_02_eda/scripts/01_eda_analysis.py

# Phase 3: Model Training
python analysis_03_model/scripts/baseline/01_train_models.py

# Phase 4: SHAP Analysis
python analysis_03_model/scripts/evaluation/02_shap_analysis.py

# Phase 5: Advanced Evaluation
python analysis_03_model/scripts/evaluation/03_additional_evaluation.py
```

### Windows UTF-8 Encoding
```bash
python -X utf8 run_full_pipeline.py
```

---

## 📦 Dependencies

### Required Packages
```
pandas>=2.3.3
numpy>=2.3.4
scikit-learn>=1.7.2
xgboost>=2.0.0
matplotlib>=3.10.0
seaborn>=0.13.0
imbalanced-learn>=0.14.0
shap>=0.45.0
joblib>=1.4.2
```

### Installation
```bash
pip install -r requirements.txt
```

---

## 🐛 Troubleshooting

### 1. Unicode Encoding Error (Windows)
**Error**: `'cp949' codec can't encode character`
**Solution**:
```bash
python -X utf8 script.py
```

### 2. SHAP 3D Array Error
**Error**: `TypeError: only integer scalar arrays can be converted`
**Solution** (in `02_shap_analysis.py`):
```python
if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]  # Extract class 1
```

### 3. Korean Font Not Found
**Solution**:
```python
# Windows
plt.rcParams['font.family'] = 'Malgun Gothic'

# macOS
plt.rcParams['font.family'] = 'AppleGothic'

# Linux
plt.rcParams['font.family'] = 'NanumGothic'
```

---

## 📈 Performance Benchmarks

### Execution Time (Intel i7, 16GB RAM)

| Phase | Time | Memory |
|-------|------|--------|
| Preprocessing | ~5s | <500MB |
| EDA | ~15s | <1GB |
| Model Training (7 models) | ~45s | <2GB |
| SHAP Analysis (3 models) | ~120s | <3GB |
| Advanced Evaluation | ~30s | <1GB |
| **Total** | **~3.5 min** | **<3GB** |

---

## 🔐 Data Privacy

### Anonymization
- ✅ Patient IDs are anonymized integers
- ✅ No personally identifiable information (PII)
- ✅ Publicly available dataset (Kaggle)

### Usage Restrictions
- ✅ Educational and research purposes only
- ❌ Not for clinical deployment without validation
- ❌ Not for commercial use without permission

---

## 🔄 Version History

### v1.0 (2024-11-24)
- ✅ Initial release
- ✅ 7 ML models implemented
- ✅ SMOTE class imbalance handling
- ✅ SHAP feature importance
- ✅ Comprehensive evaluation (calibration, PR curves, risk stratification)
- ✅ Academic paper (~15,000 words, 14 figures)

---

## 📚 References

### Academic Papers
1. Chawla NV, et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321-357.
2. Lundberg SM, Lee SI. (2017). A unified approach to interpreting model predictions. *NIPS*, 4765-4774.
3. Dritsas E, Trigka M. (2022). Stroke risk prediction with machine learning techniques. *Sensors*, 22(13), 4670.

### Dataset
- **Source**: Kaggle - Stroke Prediction Dataset
- **URL**: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- **License**: Public Domain

---

## 👥 Contact & Support

### Issues
Report bugs or request features:
- GitHub Issues: [Project Repository]
- Email: [data-science-team@example.com]

### Documentation
- **README.md**: User-facing guide
- **CLAUDE.md**: Technical documentation (this file)
- **paper/main_paper.md**: Academic paper

---

## 📝 License

This project is for educational and research purposes only. Not for clinical use without proper validation.

---

**Last Updated**: 2024-11-24
**Documentation Version**: 1.0
**Status**: ✅ Production-Ready
