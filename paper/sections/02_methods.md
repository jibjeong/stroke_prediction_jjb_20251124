# Methods

## Study Design and Dataset

### Data Source

This retrospective study utilized the Stroke Prediction Dataset obtained from Kaggle, consisting of 5,110 patient records collected in 2021. The dataset includes demographic information, clinical measurements, lifestyle factors, and stroke occurrence as the outcome variable.

### Study Population

**Inclusion Criteria**:
- Complete demographic and clinical data
- Age ≥ 18 years
- Available stroke outcome information

**Exclusion Criteria**:
- Duplicate patient records
- Missing target variable (stroke)

**Final Sample Size**: 5,109 patients (1 duplicate removed)
- Stroke cases (Class 1): 249 patients (4.9%)
- Non-stroke cases (Class 0): 4,861 patients (95.1%)

### Ethical Considerations

This study utilized publicly available, anonymized data. Patient identifiers were removed prior to analysis. The study was conducted in accordance with the principles of the Declaration of Helsinki.

---

## Variables and Definitions

### Outcome Variable

**Stroke** (Binary):
- 0: No stroke occurrence
- 1: Stroke occurrence
- **Definition**: Clinical diagnosis of stroke during the observation period

### Predictor Variables (n=11)

#### Demographic Variables
1. **Age** (years): Continuous variable, range 0-82 years
2. **Gender**: Male, Female, Other
3. **Ever Married**: Yes, No

#### Clinical Variables
4. **Hypertension**: Binary (0=No, 1=Yes)
   - Definition: Systolic BP ≥140 mmHg or Diastolic BP ≥90 mmHg, or current antihypertensive medication
5. **Heart Disease**: Binary (0=No, 1=Yes)
   - Definition: History of coronary artery disease, myocardial infarction, or heart failure
6. **Average Glucose Level** (mg/dL): Continuous variable
   - Categories: Normal (<100), Prediabetes (100-125), Diabetes (≥126)
7. **BMI** (kg/m²): Continuous variable
   - WHO Categories: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (≥30)

#### Lifestyle and Socioeconomic Variables
8. **Smoking Status**: Never smoked, Formerly smoked, Currently smokes, Unknown
9. **Work Type**: Private, Self-employed, Government job, Children, Never worked
10. **Residence Type**: Urban, Rural

### Missing Data

- **BMI**: 201 missing values (3.9%)
- **Other variables**: Complete data

---

## Data Preprocessing

### Phase 1: Data Cleaning

#### 1.1 Missing Value Imputation

**BMI Imputation** (Script: `analysis_01_preprocess/scripts/01_data_preprocessing.py:38-66`):
```
Method: Median imputation
Median BMI: 28.10 kg/m²
Rationale: Median is robust to outliers; maintains distribution shape
```

#### 1.2 Categorical Variable Encoding

**Binary Variables** (Label Encoding):
- gender: Male=1, Female=0, Other=2
- ever_married: Yes=1, No=0
- Residence_type: Urban=1, Rural=0

**Multi-class Variables** (One-Hot Encoding):
- work_type: 4 dummy variables (drop_first=True)
- smoking_status: 3 dummy variables (drop_first=True)

#### 1.3 Numerical Variable Scaling

**Method**: StandardScaler (z-score normalization)
- Formula: z = (x - μ) / σ
- Applied to: age, avg_glucose_level, bmi
- Result: Mean=0, SD=1 for all scaled features

### Phase 2: Feature Engineering

#### 2.1 Derived Variables

**Age Group** (Categorical):
- Child: age < 18
- Young Adult: 18 ≤ age < 40
- Middle-Aged: 40 ≤ age < 60
- Senior: age ≥ 60

**BMI Category** (WHO Classification):
- Underweight: BMI < 18.5
- Normal: 18.5 ≤ BMI < 25
- Overweight: 25 ≤ BMI < 30
- Obese: BMI ≥ 30

**Glucose Category** (ADA Criteria):
- Normal: glucose < 100 mg/dL
- Prediabetes: 100 ≤ glucose < 126 mg/dL
- Diabetes: glucose ≥ 126 mg/dL

**Risk Score** (Composite Variable):
- Formula: hypertension + heart_disease + (age > 60)
- Range: 0-3
- Higher score indicates higher cumulative risk

#### 2.2 Final Feature Set

- **Total Features**: 19 engineered features
- **Feature Types**:
  - Continuous: 3 (scaled)
  - Binary: 3
  - One-hot encoded: 7
  - Derived categorical: 4
  - Composite: 1
  - Patient ID: 1 (excluded from modeling)

---

## Model Development

### Train-Test Split

**Method**: Stratified random sampling
- Train set: 80% (n=4,088)
- Test set: 20% (n=1,022)
- Random seed: 42 (for reproducibility)
- Stratification: Maintain class ratio in both sets

### Handling Class Imbalance

**Problem**: Stroke cases represent only 4.9% of the dataset, leading to potential model bias toward the majority class.

**Solution**: SMOTE (Synthetic Minority Over-sampling Technique)
- Applied to training set only (avoid data leakage)
- Method: Generate synthetic samples using k-nearest neighbors (k=5)
- Result:
  - Before SMOTE: Class 0=3,889, Class 1=199 (1:19.5 ratio)
  - After SMOTE: Class 0=3,889, Class 1=3,889 (1:1 ratio)

### Machine Learning Algorithms

Seven ML algorithms were trained and compared:

#### 1. Logistic Regression (LR)
- **Type**: Linear classifier
- **Hyperparameters**:
  - max_iter=1000
  - class_weight='balanced'
- **Rationale**: Baseline model; high interpretability

#### 2. Decision Tree (DT)
- **Type**: Tree-based classifier
- **Hyperparameters**:
  - max_depth=10
  - min_samples_split=10
  - min_samples_leaf=5
  - class_weight='balanced'
- **Rationale**: Non-linear relationships; interpretable

#### 3. Random Forest (RF)
- **Type**: Ensemble (bagging)
- **Hyperparameters**:
  - n_estimators=100
  - max_depth=10
  - class_weight='balanced'
- **Rationale**: Robust to overfitting; feature importance

#### 4. Support Vector Machine (SVM)
- **Type**: Kernel-based classifier
- **Hyperparameters**:
  - kernel='rbf'
  - C=1.0
  - gamma='scale'
  - class_weight='balanced'
- **Rationale**: Effective in high-dimensional space

#### 5. XGBoost (XGB)
- **Type**: Gradient boosting ensemble
- **Hyperparameters**:
  - n_estimators=100
  - max_depth=5
  - learning_rate=0.1
  - scale_pos_weight=19.5 (original class ratio)
- **Rationale**: State-of-the-art performance

#### 6. Gradient Boosting (GB)
- **Type**: Gradient boosting ensemble
- **Hyperparameters**:
  - n_estimators=100
  - max_depth=5
  - learning_rate=0.1
- **Rationale**: Similar to XGBoost; alternative implementation

#### 7. Neural Network (NN)
- **Type**: Multi-layer Perceptron
- **Architecture**:
  - Hidden layers: (100, 50)
  - Activation: ReLU
  - Solver: Adam
  - max_iter=500
  - early_stopping=True
- **Rationale**: Deep learning approach; non-linear patterns

---

## Model Evaluation

### Performance Metrics

#### Primary Metrics

**1. AUROC (Area Under ROC Curve)**
- Range: 0.5 (random) to 1.0 (perfect)
- Interpretation: Overall discriminative ability
- **Clinical Relevance**: Preferred metric for imbalanced data

**2. Classification Accuracy (CA)**
- Formula: (TP + TN) / (TP + TN + FP + FN)
- **Limitation**: Misleading for imbalanced data

**3. F1-Score**
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- **Clinical Relevance**: Balances precision and recall

**4. Precision**
- Formula: TP / (TP + FP)
- **Clinical Relevance**: Proportion of predicted strokes that are true positives

**5. Recall (Sensitivity)**
- Formula: TP / (TP + FN)
- **Clinical Relevance**: Proportion of actual strokes correctly identified

#### Secondary Metrics

**6. Brier Score**
- Range: 0 (perfect) to 1 (worst)
- **Clinical Relevance**: Measures calibration (predicted probabilities vs. observed frequencies)

**7. Average Precision (AP)**
- Area under Precision-Recall curve
- **Clinical Relevance**: Emphasizes performance on minority class

**8. Specificity**
- Formula: TN / (TN + FP)
- **Clinical Relevance**: Proportion of non-strokes correctly identified

### Model Comparison

**Class-Averaged Metrics**:
- Macro-average: Unweighted mean of per-class metrics
- Used for Table 1: "Evaluation of Model Performance (Average)"

**Per-Class Metrics**:
- Separate metrics for Stroke=No (Class 0) and Stroke=Yes (Class 1)
- Used for Table 2: "Evaluation of Model Performance by Class"

### ROC Curve Analysis

**Overall ROC Curve**:
- Plots True Positive Rate vs. False Positive Rate
- Compares all seven models on a single plot

**Class-Specific ROC Curves**:
- Figure 3: ROC Curve for Stroke=No (Class 0)
- Figure 4: ROC Curve for Stroke=Yes (Class 1)

### Calibration Analysis

**Method**: Calibration curve (reliability diagram)
- Bins predicted probabilities into 10 groups
- Compares predicted probability vs. observed frequency
- Perfect calibration: predicted = observed (diagonal line)

**Metric**: Brier Score
- Lower score indicates better calibration

### Threshold Optimization

**Method**: Youden's Index
- Formula: J = Sensitivity + Specificity - 1
- Maximizes J to find optimal threshold
- Default threshold: 0.5

**Clinical Consideration**:
- High-sensitivity threshold (lower cutoff): Prioritize detecting strokes (fewer false negatives)
- High-specificity threshold (higher cutoff): Reduce false alarms (fewer false positives)

### Risk Stratification

**Method**: Tertile-based risk groups using Random Forest predicted probabilities
- **Low Risk**: Predicted probability < 0.3
- **Medium Risk**: 0.3 ≤ Predicted probability < 0.7
- **High Risk**: Predicted probability ≥ 0.7

**Evaluation**:
- Distribution of patients across risk groups
- Actual stroke rate within each group

---

## Feature Importance Analysis

### Correlation Analysis

**Method**: Pearson correlation coefficient
- Measures linear association between each predictor and stroke outcome
- Range: -1 (perfect negative) to +1 (perfect positive)
- Significance levels: * p<0.05, ** p<0.01, *** p<0.001

![Figure 1: Correlation Heatmap - Feature Correlations with Stroke](../../analysis_02_eda/data/figures/correlation_heatmap.png)

**Figure 1**: Correlation heatmap showing pairwise correlations among all features and the target variable (stroke). Age shows the strongest positive correlation with stroke occurrence (r=0.245***).

### SHAP (SHapley Additive exPlanations)

**Method**: Game-theoretic approach to explain model predictions
- Computes contribution of each feature to each prediction
- Provides both global (feature importance) and local (individual prediction) explanations

**Models Analyzed**: Random Forest, XGBoost, Gradient Boosting

**Visualizations**:
1. **SHAP Importance Plot**: Bar plot showing mean |SHAP value| for each feature
2. **SHAP Summary Plot**: Dot plot showing SHAP value distribution and feature values
3. **SHAP Dependence Plot**: Scatter plot showing relationship between feature value and SHAP value for top 6 features

**Sample Size**: 500 patients (computational efficiency)

---

## Statistical Analysis

### Software and Tools

- **Programming Language**: Python 3.14
- **Key Libraries**:
  - pandas (2.3.3): Data manipulation
  - numpy (2.3.4): Numerical computing
  - scikit-learn (1.7.2): Machine learning
  - xgboost (2.0+): Gradient boosting
  - imbalanced-learn (0.14.0): SMOTE
  - shap (0.50.0): Model interpretation
  - matplotlib (3.10.0), seaborn (0.13.0): Visualization

### Reproducibility

- **Random Seed**: 42 (set for all random processes)
- **Cross-Validation**: 5-fold stratified (for internal validation)
- **Code Availability**: Full analysis pipeline available in project repository

### Model Selection Criteria

**Primary Criterion**: AUROC
**Secondary Criteria**: F1-Score, Calibration (Brier Score), Clinical Utility (Risk Stratification)

**Clinical Scenario Considerations**:
1. **Primary Care**: Interpretability (Logistic Regression)
2. **Hospital Screening**: Balance (Random Forest)
3. **Research/Policy**: High Sensitivity (Gradient Boosting)

---

## Quality Control

### Data Validation

1. **Range Checks**: All continuous variables within physiologically plausible ranges
2. **Consistency Checks**: Cross-validation of related variables (e.g., age vs. work_type)
3. **Duplicate Detection**: Removed 1 duplicate patient ID

### Model Validation

1. **Train-Test Separation**: Strict separation to prevent data leakage
2. **SMOTE Applied to Training Only**: Avoid optimistic bias
3. **Stratified Sampling**: Maintain class ratio in train-test split

### Sensitivity Analysis

1. **Impact of Missing Data**: Compare models with/without BMI imputation
2. **Impact of SMOTE**: Compare models with/without resampling
3. **Threshold Variation**: Evaluate performance across different decision thresholds

---

## Limitations

### Methodological Limitations

1. **Retrospective Design**: Cannot establish causality
2. **Single Time Point**: No longitudinal data; cannot model temporal dynamics
3. **No External Validation**: Single dataset; generalizability uncertain
4. **Small Stroke Sample**: Only 249 stroke cases; limited statistical power

### Data Limitations

1. **Missing BMI Data**: 3.9% imputed; may introduce bias
2. **Class Imbalance**: Severe (4.9% stroke); requires SMOTE
3. **Unknown Smoking Status**: "Unknown" category in smoking_status variable
4. **No Genetic Data**: Lacks genetic risk factors

### Clinical Limitations

1. **Stroke Subtype**: Does not distinguish ischemic vs. hemorrhagic stroke
2. **Severity**: No information on stroke severity or functional outcome
3. **Treatment Information**: Lacks data on preventive medications (statins, anticoagulants)
4. **Follow-up Duration**: Unclear observation period

---

## Reporting Standards

This study adheres to the following reporting guidelines:
- **TRIPOD** (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis)
- **STROBE** (Strengthening the Reporting of Observational Studies in Epidemiology)
- **CONSORT-AI** (Consolidated Standards of Reporting Trials - Artificial Intelligence)
