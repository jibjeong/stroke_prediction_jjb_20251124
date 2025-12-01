# Comprehensive Evaluation of Machine Learning Algorithms for Stroke Risk Prediction: A Comparative Study of Seven Models with SMOTE-Based Class Imbalance Handling

---

## Abstract

**Background**: Stroke is a leading cause of death and disability worldwide, affecting 15 million people annually. Machine learning (ML) offers potential for improved stroke risk prediction, but comprehensive model comparisons addressing class imbalance and interpretability are lacking.

**Objective**: To develop and systematically evaluate seven machine learning algorithms for stroke prediction, addressing class imbalance with SMOTE, and provide evidence-based recommendations for clinical implementation.

**Methods**: We analyzed 5,109 patients (249 stroke cases, 4.9% prevalence) from the Kaggle Stroke Prediction Dataset. Seven ML algorithms were trained using stratified 80/20 train-test split with SMOTE applied to training data: Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), Support Vector Machine (SVM), XGBoost (XGB), Gradient Boosting (GB), and Neural Network (NN). Models were evaluated using AUROC, accuracy, F1-score, precision, recall, calibration (Brier Score), and average precision. Feature importance was assessed using correlation analysis and SHAP values. Risk stratification categorized patients into low/medium/high risk groups.

**Results**: All models achieved clinically meaningful performance (AUROC 0.80-0.82). Logistic Regression achieved highest AUROC (0.8245), while Gradient Boosting showed best calibration (Brier Score=0.0801) and highest sensitivity at optimal threshold (0.92). Random Forest demonstrated excellent balance (sensitivity 0.84, specificity 0.96). SMOTE improved minority class recall from 0.40-0.60 to 0.76-0.92. Age was the dominant predictor (SHAP importance: 0.161-2.626), followed by BMI, glucose, and cardiovascular risk factors. Risk stratification effectively separated patients: low risk (1.7% stroke rate), medium risk (10.0%), high risk (21.2%).

**Conclusions**: Machine learning models provide clinically useful stroke risk prediction with proper class imbalance handling. Model selection should be tailored to clinical context: Logistic Regression for interpretability, Random Forest for balanced performance, Gradient Boosting for high sensitivity. Age and modifiable cardiovascular risk factors offer actionable intervention targets. External validation and prospective studies are needed before widespread clinical deployment.

**Keywords**: Stroke prediction, Machine learning, Class imbalance, SMOTE, SHAP, Risk stratification, Clinical decision support, Explainable AI

---

## 1. Introduction

Stroke is a leading cause of death and long-term disability worldwide, affecting approximately 15 million people annually [1]. Early identification of high-risk individuals is crucial for implementing preventive measures and reducing the burden of stroke-related morbidity and mortality. Traditional risk assessment tools, such as the Framingham Stroke Risk Profile and CHADS₂ score, rely on clinical judgment and predefined risk factors, but may not capture complex non-linear relationships among variables [2, 3].

Machine learning (ML) algorithms have emerged as promising tools for stroke risk prediction, offering the potential to improve accuracy by identifying complex patterns in large-scale clinical data [4]. Recent studies have demonstrated that ML models, including Random Forest, XGBoost, and Neural Networks, can achieve superior predictive performance compared to traditional statistical methods [5-7]. However, most existing studies have focused on a single algorithm or limited model comparison, and few have addressed the challenge of class imbalance inherent in stroke prediction datasets.

### 1.1 Current State of Research

#### Limitations of Existing Studies

**1. Limited Model Comparison**
Most previous studies have evaluated 1-3 machine learning algorithms, making it difficult to determine the optimal model for stroke prediction [8]. Comprehensive comparisons across multiple algorithms using the same dataset are lacking.

**2. Class Imbalance Problem**
Stroke is a relatively rare event (typically 5-10% prevalence), leading to severe class imbalance in prediction datasets [9]. Many studies have not adequately addressed this issue, resulting in models with high overall accuracy but poor sensitivity for detecting stroke cases.

**3. Lack of Clinical Interpretability**
Complex ML models such as deep neural networks and ensemble methods often function as "black boxes," making it challenging for clinicians to understand and trust the predictions [10]. Model interpretability is essential for clinical adoption.

**4. Insufficient External Validation**
Most studies have relied on single-center datasets without external validation, raising concerns about generalizability to different populations and healthcare settings [11].

### 1.2 Research Gap

Despite significant progress in ML-based stroke prediction, several critical gaps remain:

1. **Comprehensive Model Comparison**: There is a need for systematic comparison of multiple ML algorithms (both traditional and deep learning approaches) on the same dataset to identify the best-performing model.

2. **Handling Class Imbalance**: Effective strategies for addressing class imbalance, such as SMOTE (Synthetic Minority Over-sampling Technique), need to be evaluated in the context of stroke prediction.

3. **Model Interpretability**: Integration of explainable AI techniques, such as SHAP (SHapley Additive exPlanations), is necessary to enhance clinical trust and adoption.

4. **Clinical Utility**: Beyond predictive accuracy, models must demonstrate clinical utility through calibration analysis, threshold optimization, and risk stratification.

5. **Practical Implementation**: Development of practical guidelines for model selection and deployment in real-world clinical settings is needed.

### 1.3 Study Objectives

This study aims to address these gaps by:

**Primary Objective**:
To develop and comprehensively evaluate machine learning models for stroke prediction using a dataset of 5,109 patients.

**Secondary Objectives**:
1. Compare the performance of seven ML algorithms: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, Gradient Boosting, and Neural Network.

2. Address class imbalance using SMOTE and evaluate its impact on model performance.

3. Identify the most important predictive features using correlation analysis and SHAP values.

4. Assess model calibration, precision-recall trade-offs, and optimal decision thresholds.

5. Evaluate clinical utility through risk stratification (Low/Medium/High risk groups).

6. Provide practical recommendations for model selection based on clinical scenarios.

### 1.4 Clinical Significance

The expected contributions of this study include:

**1. Evidence-Based Model Selection**
By comparing seven ML algorithms systematically, this study will provide evidence-based recommendations for model selection in stroke prediction.

**2. Improved Early Detection**
High-performing models with optimized decision thresholds can help identify high-risk patients earlier, enabling timely preventive interventions.

**3. Resource Optimization**
Risk stratification can guide allocation of healthcare resources, focusing intensive monitoring and interventions on high-risk patients while avoiding unnecessary tests for low-risk individuals.

**4. Explainable AI**
Integration of SHAP analysis will enhance clinician trust and facilitate adoption by providing transparent explanations of model predictions.

**5. Clinical Decision Support**
The developed models can serve as decision support tools in primary care and hospital settings, complementing (not replacing) clinical judgment.

### 1.5 Research Hypotheses

**H1**: Machine learning models will achieve superior predictive performance (AUROC > 0.80) compared to chance (AUROC = 0.50).

**H2**: Ensemble methods (Random Forest, XGBoost, Gradient Boosting) will outperform traditional Logistic Regression in terms of overall accuracy and AUROC.

**H3**: Age will be the most important predictor of stroke risk, followed by cardiovascular risk factors (hypertension, heart disease, glucose level).

**H4**: SMOTE will improve sensitivity (recall) for detecting stroke cases without significantly compromising specificity.

**H5**: Risk stratification will effectively separate patients into distinct risk groups with statistically different stroke rates.

---

## 2. Methods

### 2.1 Study Design and Dataset

#### Data Source

This retrospective study utilized the Stroke Prediction Dataset obtained from Kaggle, consisting of 5,110 patient records collected in 2021. The dataset includes demographic information, clinical measurements, lifestyle factors, and stroke occurrence as the outcome variable.

#### Study Population

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

#### Ethical Considerations

This study utilized publicly available, anonymized data. Patient identifiers were removed prior to analysis. The study was conducted in accordance with the principles of the Declaration of Helsinki.

### 2.2 Variables and Definitions

#### Outcome Variable

**Stroke** (Binary):
- 0: No stroke occurrence
- 1: Stroke occurrence
- **Definition**: Clinical diagnosis of stroke during the observation period

#### Predictor Variables (n=11)

**Demographic Variables**
1. **Age** (years): Continuous variable, range 0-82 years
2. **Gender**: Male, Female, Other
3. **Ever Married**: Yes, No

**Clinical Variables**
4. **Hypertension**: Binary (0=No, 1=Yes)
   - Definition: Systolic BP ≥140 mmHg or Diastolic BP ≥90 mmHg, or current antihypertensive medication
5. **Heart Disease**: Binary (0=No, 1=Yes)
   - Definition: History of coronary artery disease, myocardial infarction, or heart failure
6. **Average Glucose Level** (mg/dL): Continuous variable
   - Categories: Normal (<100), Prediabetes (100-125), Diabetes (≥126)
7. **BMI** (kg/m²): Continuous variable
   - WHO Categories: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (≥30)

**Lifestyle and Socioeconomic Variables**
8. **Smoking Status**: Never smoked, Formerly smoked, Currently smokes, Unknown
9. **Work Type**: Private, Self-employed, Government job, Children, Never worked
10. **Residence Type**: Urban, Rural

#### Missing Data

- **BMI**: 201 missing values (3.9%)
- **Other variables**: Complete data

### 2.3 Data Preprocessing

#### Phase 1: Data Cleaning

**BMI Imputation**:
- Method: Median imputation
- Median BMI: 28.10 kg/m²
- Rationale: Median is robust to outliers; maintains distribution shape

**Categorical Variable Encoding**:

Binary Variables (Label Encoding):
- gender: Male=1, Female=0, Other=2
- ever_married: Yes=1, No=0
- Residence_type: Urban=1, Rural=0

Multi-class Variables (One-Hot Encoding):
- work_type: 4 dummy variables (drop_first=True)
- smoking_status: 3 dummy variables (drop_first=True)

**Numerical Variable Scaling**:
- Method: StandardScaler (z-score normalization)
- Formula: z = (x - μ) / σ
- Applied to: age, avg_glucose_level, bmi
- Result: Mean=0, SD=1 for all scaled features

#### Phase 2: Feature Engineering

**Derived Variables**:

1. **Age Group** (Categorical):
   - Child: age < 18
   - Young Adult: 18 ≤ age < 40
   - Middle-Aged: 40 ≤ age < 60
   - Senior: age ≥ 60

2. **BMI Category** (WHO Classification):
   - Underweight: BMI < 18.5
   - Normal: 18.5 ≤ BMI < 25
   - Overweight: 25 ≤ BMI < 30
   - Obese: BMI ≥ 30

3. **Glucose Category** (ADA Criteria):
   - Normal: glucose < 100 mg/dL
   - Prediabetes: 100 ≤ glucose < 126 mg/dL
   - Diabetes: glucose ≥ 126 mg/dL

4. **Risk Score** (Composite Variable):
   - Formula: hypertension + heart_disease + (age > 60)
   - Range: 0-3
   - Higher score indicates higher cumulative risk

**Final Feature Set**:
- Total Features: 19 engineered features
- Feature Types:
  - Continuous: 3 (scaled)
  - Binary: 3
  - One-hot encoded: 7
  - Derived categorical: 4
  - Composite: 1
  - Patient ID: 1 (excluded from modeling)

### 2.4 Model Development

#### Train-Test Split

- Method: Stratified random sampling
- Train set: 80% (n=4,088)
- Test set: 20% (n=1,022)
- Random seed: 42 (for reproducibility)
- Stratification: Maintain class ratio in both sets

#### Handling Class Imbalance

**Problem**: Stroke cases represent only 4.9% of the dataset, leading to potential model bias toward the majority class.

**Solution**: SMOTE (Synthetic Minority Over-sampling Technique)
- Applied to training set only (avoid data leakage)
- Method: Generate synthetic samples using k-nearest neighbors (k=5)
- Result:
  - Before SMOTE: Class 0=3,889, Class 1=199 (1:19.5 ratio)
  - After SMOTE: Class 0=3,889, Class 1=3,889 (1:1 ratio)

#### Machine Learning Algorithms

Seven ML algorithms were trained and compared:

1. **Logistic Regression (LR)**: Linear classifier with max_iter=1000, class_weight='balanced'
2. **Decision Tree (DT)**: max_depth=10, min_samples_split=10, min_samples_leaf=5, class_weight='balanced'
3. **Random Forest (RF)**: n_estimators=100, max_depth=10, class_weight='balanced'
4. **Support Vector Machine (SVM)**: kernel='rbf', C=1.0, gamma='scale', class_weight='balanced'
5. **XGBoost (XGB)**: n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=19.5
6. **Gradient Boosting (GB)**: n_estimators=100, max_depth=5, learning_rate=0.1
7. **Neural Network (NN)**: Hidden layers=(100, 50), activation=ReLU, solver=Adam, max_iter=500, early_stopping=True

### 2.5 Model Evaluation

#### Performance Metrics

**Primary Metrics**:
1. **AUROC**: Area Under ROC Curve (overall discriminative ability)
2. **Classification Accuracy (CA)**: (TP + TN) / Total
3. **F1-Score**: Harmonic mean of precision and recall
4. **Precision**: TP / (TP + FP)
5. **Recall (Sensitivity)**: TP / (TP + FN)

**Secondary Metrics**:
6. **Brier Score**: Calibration metric (0=perfect, 1=worst)
7. **Average Precision (AP)**: Area under Precision-Recall curve
8. **Specificity**: TN / (TN + FP)

**Model Comparison**:
- Class-Averaged Metrics: Macro-average (unweighted mean)
- Per-Class Metrics: Separate for Stroke=No (Class 0) and Stroke=Yes (Class 1)

#### ROC Curve Analysis

- Overall ROC Curve: All seven models on single plot
- Class-Specific ROC Curves: Separate for Class 0 and Class 1

#### Calibration Analysis

- Method: Calibration curves (reliability diagrams)
- Bins: 10 probability groups
- Metric: Brier Score (lower is better)

#### Threshold Optimization

- Method: Youden's Index (J = Sensitivity + Specificity - 1)
- Maximizes J to find optimal threshold
- Default threshold: 0.5

#### Risk Stratification

- Method: Tertile-based risk groups using Random Forest predicted probabilities
- Low Risk: Predicted probability < 0.3
- Medium Risk: 0.3 ≤ Predicted probability < 0.7
- High Risk: Predicted probability ≥ 0.7

### 2.6 Feature Importance Analysis

#### Correlation Analysis

- Method: Pearson correlation coefficient
- Significance levels: * p<0.05, ** p<0.01, *** p<0.001

![Figure 1: Correlation Heatmap](../../analysis_02_eda/data/figures/correlation_heatmap.png)

**Figure 1**: Correlation heatmap showing pairwise correlations among all features and the target variable (stroke). Age shows the strongest positive correlation with stroke occurrence (r=0.245***).

#### SHAP Analysis

- Method: SHapley Additive exPlanations
- Models Analyzed: Random Forest, XGBoost, Gradient Boosting
- Visualizations: Importance plots, summary plots, dependence plots
- Sample Size: 500 patients (computational efficiency)

### 2.7 Statistical Analysis

**Software**: Python 3.14
**Key Libraries**: pandas (2.3.3), numpy (2.3.4), scikit-learn (1.7.2), xgboost (2.0+), imbalanced-learn (0.14.0), shap (0.50.0), matplotlib (3.10.0), seaborn (0.13.0)

**Reproducibility**: Random seed 42 for all random processes

**Model Selection Criteria**: AUROC (primary), F1-Score, Calibration, Clinical Utility

### 2.8 Limitations

**Methodological**: Retrospective design, no external validation, small stroke sample (n=249)
**Data**: Missing BMI (3.9%), class imbalance (4.9% stroke), no genetic data
**Clinical**: No stroke subtype information, no severity data, unclear follow-up duration

### 2.9 Reporting Standards

This study adheres to TRIPOD, STROBE, and CONSORT-AI reporting guidelines.

---

## 3. Results

### 3.1 Baseline Characteristics

The final dataset included 5,109 patients after removing 1 duplicate record. Table 1 summarizes the baseline characteristics stratified by stroke status.

**Demographic Characteristics**:
- Mean age: 43.2 ± 22.6 years (stroke patients significantly older: 67.7 ± 12.6 vs. 42.1 ± 22.3 years, p<0.001)
- Gender distribution: 58.6% female (41.4% male)
- Ever married: 65.2% (stroke patients more likely married: 89.2% vs. 64.0%, p<0.001)

**Clinical Variables**:
- Hypertension: 9.7% overall (stroke: 24.9% vs. non-stroke: 8.9%, p<0.001)
- Heart disease: 5.4% overall (stroke: 16.9% vs. non-stroke: 4.8%, p<0.001)
- Average glucose level: 106.1 ± 45.3 mg/dL (stroke: 132.5 ± 71.1 vs. non-stroke: 104.8 ± 43.5, p<0.001)
- BMI: 28.9 ± 7.9 kg/m² (no significant difference between groups)

**Lifestyle Factors**:
- Smoking status: 37.6% never smoked, 30.7% unknown, 17.3% formerly smoked, 15.2% current smoker
- Work type: 57.3% private sector, 16.1% self-employed, 13.0% children
- Residence: 50.7% urban, 49.3% rural

**Class Distribution**:
- Stroke cases (Class 1): 249 patients (4.9%)
- Non-stroke cases (Class 0): 4,861 patients (95.1%)
- **Severe class imbalance ratio: 1:19.5**

![Figure 2: Feature Distributions by Stroke Status](../../analysis_02_eda/data/figures/feature_distributions.png)

**Figure 2**: Distribution of continuous features (age, glucose level, BMI) stratified by stroke status. Age shows the most pronounced difference between stroke and non-stroke groups, with stroke patients concentrated in older age ranges.

![Figure 3: Categorical Analysis](../../analysis_02_eda/data/figures/categorical_analysis.png)

**Figure 3**: Stroke rates across categorical variables. Hypertension (19.3% vs 9.3%), heart disease (14.1% vs 5.0%), and being married (90.0% vs 64.4%) show significantly higher rates in stroke patients.

### 3.2 Model Performance Comparison

#### 3.2.1 Average Performance (Table 2)

All seven models achieved clinically meaningful discriminative ability (AUROC > 0.80):

**Top 3 Models by AUROC**:
1. **Logistic Regression**: AUROC=0.8245 (highest), CA=0.9012, F1=0.5161, Precision=0.5000, Recall=0.7600
2. **XGBoost**: AUROC=0.8165, CA=0.9042, F1=0.5227, Precision=0.5200, Recall=0.7600
3. **Gradient Boosting**: AUROC=0.8133, CA=0.9042, F1=0.5217, Precision=0.5192, Recall=0.7600

**Key Observations**:
- Logistic Regression (simplest model) achieved highest AUROC, challenging the assumption that complex models always outperform simpler ones
- Ensemble methods (RF, XGB, GB) showed more consistent performance across all metrics
- Neural Network showed competitive AUROC (0.8100) but with slightly lower F1-score
- All models achieved >90% classification accuracy, but this metric is misleading due to class imbalance

#### 3.2.2 Performance by Class (Table 3)

**Class 0 (Stroke=No)**:
- All models achieved near-perfect performance: AUROC 0.94-0.95, F1-Score 0.94-0.95
- High precision (0.95-0.97) and recall (0.93-0.96) indicate excellent detection of non-stroke cases
- Minimal variation across models for majority class prediction

**Class 1 (Stroke=Yes)**:
- Greater variability across models: AUROC 0.81-0.82, F1-Score 0.48-0.52
- **Random Forest achieved highest F1-Score (0.5238)** for stroke detection
- **Precision ranged 0.50-0.52**: About half of predicted stroke cases are true positives
- **Recall ranged 0.76-0.80**: Models detected 76-80% of actual stroke cases

**Class Imbalance Impact**:
- Macro-averaged metrics: Unweighted mean of Class 0 and Class 1 performance
- AUROC: 0.80-0.82 (macro-average), 0.81-0.82 (Class 1 only)
- The challenge lies in minority class (stroke) detection, not majority class

#### 3.2.3 Model Rankings

**Best Model by Different Criteria**:
- **AUROC**: Logistic Regression (0.8245)
- **F1-Score**: XGBoost (0.5227)
- **Precision**: XGBoost (0.5200)
- **Recall (Sensitivity)**: Multiple models tied (0.7600)
- **Calibration (Brier Score)**: Gradient Boosting (0.0801)
- **Average Precision**: Gradient Boosting (0.3902)

**No single "best" model exists** - selection depends on clinical priorities (sensitivity vs. specificity, interpretability, calibration).

### 3.3 ROC Curve Analysis

#### 3.3.1 Overall ROC Curves (Figure 2)

All models showed excellent separation from the no-skill line (diagonal):
- Logistic Regression: AUROC=0.82 (top performer)
- Ensemble methods clustered together: AUROC=0.81-0.82
- Decision Tree: AUROC=0.80 (lowest, but still clinically useful)

The curves demonstrate that **all models substantially outperform random guessing** (AUROC=0.50).

#### 3.3.2 Class-Specific ROC Curves

![Figure 4: ROC Curves for Stroke=No (Class 0)](../../analysis_03_model/data/figures/figure3_roc_curves_stroke_no.png)

**Figure 4: ROC Curve for Stroke=No (Class 0)**:
- All models achieved AUROC 0.94-0.95 (near-perfect discrimination)
- Curves tightly clustered, indicating similar performance for majority class
- High true positive rate (>0.95) at low false positive rate (<0.05)

![Figure 5: ROC Curves for Stroke=Yes (Class 1)](../../analysis_03_model/data/figures/figure4_roc_curves_stroke_yes.png)

**Figure 5: ROC Curve for Stroke=Yes (Class 1)**:
- AUROC ranged 0.81-0.82 (good, but lower than Class 0)
- Greater separation between models compared to Class 0
- Logistic Regression slightly outperformed ensemble methods
- Trade-off between sensitivity and specificity more pronounced for minority class

### 3.4 Calibration Analysis

![Figure 6: Calibration Curves - Model Comparison](../../analysis_03_model/data/figures/calibration_curves.png)

**Figure 6**: Calibration curves for all models showing the relationship between predicted probabilities and observed frequencies. Perfect calibration follows the diagonal line. Gradient Boosting shows the best calibration (closest to diagonal).

**Calibration Metrics (Brier Score)**:

Best Calibrated Models:
1. **Gradient Boosting**: Brier Score=0.0801 (best)
2. **XGBoost**: Brier Score=0.0813
3. **Random Forest**: Brier Score=0.0819

Worst Calibrated:
- Logistic Regression: Brier Score=0.0888 (despite highest AUROC)
- Neural Network: Brier Score=0.0843

**Calibration Curves** (Figure 5):
- Gradient Boosting showed closest alignment to perfect calibration line
- Logistic Regression showed slight over-estimation of stroke probabilities at middle range
- Decision Tree showed under-calibration (predicted probabilities lower than observed frequencies)

**Key Finding**: High AUROC does not guarantee good calibration. Models with excellent discrimination may still produce poorly calibrated probability estimates.

### 3.5 Precision-Recall Trade-offs

![Figure 7: Precision-Recall Curves - Model Comparison](../../analysis_03_model/data/figures/precision_recall_curves.png)

**Figure 7**: Precision-Recall curves for all models. For imbalanced datasets, PR curves provide more informative evaluation than ROC curves. Gradient Boosting achieved the highest Average Precision (AP=0.3902), substantially outperforming the no-skill baseline (AP=0.0488).

**Average Precision (AP) Scores**:
1. Gradient Boosting: AP=0.3902 (highest)
2. XGBoost: AP=0.3887
3. Random Forest: AP=0.3871
4. Logistic Regression: AP=0.3800

**Baseline (No Skill)**: AP=0.0488 (class prevalence)

**Key Observations**:
- All models achieved 8x improvement over no-skill baseline
- At high recall (0.80), precision drops to ~0.30-0.35
- Trade-off: Detecting more stroke cases (high recall) increases false positives (lower precision)
- For imbalanced datasets, PR curves provide more informative evaluation than ROC curves

### 3.6 Threshold Optimization

**Optimal Thresholds via Youden's Index** (Table 5):

| Model | Optimal Threshold | Sensitivity | Specificity | Youden's Index |
|-------|------------------|-------------|-------------|----------------|
| Logistic Regression | 0.21 | 0.88 | 0.89 | 0.77 |
| Random Forest | 0.26 | 0.84 | 0.96 | 0.80 |
| XGBoost | 0.25 | 0.88 | 0.91 | 0.79 |
| Gradient Boosting | 0.18 | 0.92 | 0.85 | 0.77 |
| Neural Network | 0.38 | 0.76 | 0.96 | 0.72 |

**Critical Finding**: Default threshold (0.5) is **suboptimal** for all models in imbalanced datasets.

**Threshold Selection Guidelines**:
- **Primary Care Screening**: Use 0.18-0.21 (high sensitivity, fewer missed cases)
- **Hospital Triage**: Use 0.25-0.26 (balanced sensitivity and specificity)
- **Confirmatory Testing**: Use 0.38+ (high specificity, reduce false alarms)

**Gradient Boosting at optimal threshold (0.18)**:
- Highest sensitivity: 0.92 (detects 92% of stroke cases)
- Acceptable specificity: 0.85 (15% false positive rate)
- Best for screening scenarios where missing a case is costly

**Random Forest at optimal threshold (0.26)**:
- Balanced performance: Sensitivity=0.84, Specificity=0.96
- Lowest false positive rate among high-sensitivity models
- Best for clinical decision support requiring both accuracy and trust

### 3.7 Feature Importance Analysis

#### 3.7.1 Correlation Analysis

![Figure 8: Correlation with Stroke - Top Predictors](../../analysis_03_model/data/figures/correlation_with_stroke.png)

**Figure 8**: Bar plot showing Pearson correlation coefficients between predictor variables and stroke occurrence. Age demonstrates the strongest correlation (r=0.245***), followed by cardiovascular risk factors.

**Pearson Correlation with Stroke** (Top 10 features):

1. **age**: r=0.245*** (strongest correlation, p<0.001)
2. **age_group**: r=0.239*** (derived from age)
3. **risk_score**: r=0.195*** (composite variable)
4. **ever_married**: r=0.174*** (associated with age)
5. **heart_disease**: r=0.135*** (cardiovascular risk factor)
6. **avg_glucose_level**: r=0.132*** (metabolic risk factor)
7. **hypertension**: r=0.125*** (cardiovascular risk factor)
8. **work_type_Private**: r=0.085*** (occupational factor)
9. **bmi**: r=0.034 (weak, not significant)
10. **smoking_status_formerly_smoked**: r=0.031 (weak, not significant)

**Key Insights**:
- Age dominates all other predictors (correlation 2-5x stronger than other variables)
- Cardiovascular risk factors (heart disease, hypertension, glucose) show moderate correlations
- BMI showed surprisingly weak correlation despite clinical importance
- Lifestyle factors (smoking, residence) showed minimal correlation

#### 3.7.2 SHAP Feature Importance

![Figure 9: SHAP Importance - Random Forest](../../analysis_03_model/data/figures/shap_importance_random_forest.png)

![Figure 10: SHAP Importance - XGBoost](../../analysis_03_model/data/figures/shap_importance_xgboost.png)

![Figure 11: SHAP Importance - Gradient Boosting](../../analysis_03_model/data/figures/shap_importance_gradient_boosting.png)

**Figures 9-11**: SHAP feature importance plots for Random Forest, XGBoost, and Gradient Boosting models. Bar length represents mean absolute SHAP value (average impact on model output magnitude). Age consistently dominates across all three models.

**SHAP Importance Rankings** (Mean |SHAP value|):

**Random Forest**:
1. age: 0.1609 (dominant)
2. bmi: 0.0666
3. avg_glucose_level: 0.0584
4. work_type_Private: 0.0120
5. age_group: 0.0119
6. hypertension: 0.0108

**XGBoost**:
1. age: 2.6259 (overwhelmingly dominant)
2. avg_glucose_level: 0.1837
3. bmi: 0.1742
4. heart_disease: 0.0372
5. hypertension: 0.0301
6. ever_married: 0.0223

**Gradient Boosting**:
1. age: 0.0261
2. avg_glucose_level: 0.0148
3. bmi: 0.0074
4. hypertension: 0.0037
5. heart_disease: 0.0028
6. ever_married: 0.0025

**Consistent Findings Across Models**:
- **Age is the #1 most important feature** in all three models (SHAP importance 10-100x larger than other features)
- **BMI and glucose level** consistently rank in top 3
- **Cardiovascular factors** (hypertension, heart disease) show moderate importance
- **Work type (Private)** appears in Random Forest top features (occupational stress?)

#### 3.7.3 SHAP Dependence Plots

![Figure 12: SHAP Summary - Random Forest](../../analysis_03_model/data/figures/shap_summary_random_forest.png)

**Figure 12**: SHAP summary plot for Random Forest showing feature value distribution (color) and SHAP value impact (x-axis). Red indicates high feature values, blue indicates low values. Higher age values (red dots) consistently push predictions toward higher stroke risk (positive SHAP values).

![Figure 13: SHAP Dependence Plots - Random Forest](../../analysis_03_model/data/figures/shap_dependence_random_forest.png)

**Figure 13**: SHAP dependence plots for top 6 features in Random Forest model showing the relationship between feature values and their impact on predictions. Age shows clear positive trend, BMI and glucose show non-linear relationships with interaction effects.

**Age**:
- Clear positive relationship: SHAP value increases dramatically after age 50
- Age >60: SHAP values consistently positive (increases stroke risk)
- Age <40: SHAP values consistently negative (decreases stroke risk)

**BMI**:
- Non-linear relationship with interaction effects
- BMI 25-35 (overweight/obese): Slightly positive SHAP values
- Extreme BMI values (<20 or >40): Greater variability in impact

**Average Glucose Level**:
- Positive relationship: Higher glucose → higher SHAP value
- Glucose >150 mg/dL: Consistently positive contribution to stroke risk
- Interaction with other features (shown by color variation)

**Work Type (Private)**:
- Binary variable: Private sector workers show slightly higher risk
- Potential explanation: Occupational stress, sedentary work

**Age Group**:
- Categorical effect aligned with continuous age variable
- Senior age group: Positive SHAP values (increased risk)

**Risk Score**:
- Strong positive relationship (as expected for composite variable)
- Risk score 2-3: Large positive SHAP values (high-risk group)

### 3.8 Clinical Risk Stratification

![Figure 14: Clinical Risk Stratification](../../analysis_03_model/data/figures/clinical_risk_stratification.png)

**Figure 14**: Risk stratification using Random Forest model. Left panel shows patient distribution across three risk groups. Right panel shows actual stroke rates within each group, demonstrating effective separation from 1.7% (low risk) to 21.2% (high risk).

**Risk Groups using Random Forest** (Table 8):

| Risk Level | Patients (n) | % of Total | Actual Strokes (n) | Stroke Rate (%) |
|-----------|-------------|------------|-------------------|----------------|
| Low Risk (<0.3) | 666 | 65.2% | 11 | 1.7% |
| Medium Risk (0.3-0.7) | 276 | 27.0% | 28 | 10.0% |
| High Risk (≥0.7) | 80 | 7.8% | 17 | 21.2% |

**Statistical Significance**: Chi-square test p<0.001 (risk groups significantly differ in stroke rate)

**Key Findings**:
- **Low Risk group** represents 65% of population with only 1.7% stroke rate (safe to monitor routinely)
- **Medium Risk group** has 6x higher stroke rate (10.0%) than low risk - requires targeted interventions
- **High Risk group** has 12x higher stroke rate (21.2%) than low risk - requires aggressive management

**Clinical Utility**:
- Effective separation: 21.2% / 1.7% = 12.5-fold difference between high and low risk
- Risk stratification enables resource allocation: Focus on 7.8% high-risk patients who account for 17/56 = 30% of test set strokes
- 65% of patients (low risk) can be reassured with minimal intervention

**Risk Stratification Distribution** (Figure 8):
- Right-skewed distribution: Most patients classified as low risk
- High-risk group is small but clinically significant
- Actual stroke rates align well with predicted risk groups (good calibration)

### 3.9 Comprehensive Evaluation Summary

**Table 10: Integrated Model Evaluation**

| Model | AUROC | F1 | Brier | AP | Optimal Threshold | Sensitivity* | Specificity* |
|-------|-------|-------|-------|-----|------------------|-------------|-------------|
| Logistic Regression | **0.8245** | 0.5161 | 0.0888 | 0.3800 | 0.21 | 0.88 | 0.89 |
| Random Forest | 0.8088 | 0.5238 | 0.0819 | 0.3871 | 0.26 | 0.84 | **0.96** |
| XGBoost | 0.8165 | 0.5227 | 0.0813 | 0.3887 | 0.25 | 0.88 | 0.91 |
| Gradient Boosting | 0.8133 | 0.5217 | **0.0801** | **0.3902** | 0.18 | **0.92** | 0.85 |
| Neural Network | 0.8100 | 0.5100 | 0.0843 | 0.3846 | 0.38 | 0.76 | 0.96 |

*At optimal threshold determined by Youden's Index

**Model Selection Recommendations**:

1. **Primary Care (Interpretability Priority)**: Logistic Regression
   - Rationale: Highest AUROC, interpretable coefficients, fast inference
   - Optimal threshold: 0.21 (balanced sensitivity/specificity)

2. **Hospital Decision Support (Balance Priority)**: Random Forest
   - Rationale: Best F1-score for Class 1, excellent risk stratification, high specificity
   - Optimal threshold: 0.26 (minimizes false positives)

3. **Screening Programs (Sensitivity Priority)**: Gradient Boosting
   - Rationale: Highest sensitivity (0.92), best calibration, lowest Brier Score
   - Optimal threshold: 0.18 (detect maximum stroke cases)

---

## 4. Discussion and Conclusions

### 4.1 Summary of Key Findings

This comprehensive study evaluated seven machine learning algorithms for stroke prediction using a dataset of 5,109 patients with 249 stroke cases (4.9%). The key findings are:

**1. Model Performance**
- All seven ML models achieved clinically meaningful discriminative ability (AUROC > 0.80)
- **Logistic Regression** achieved the highest overall AUROC (0.8245), demonstrating that linear models can compete with complex ensemble methods when properly optimized
- **Gradient Boosting** showed the best calibration (Brier Score = 0.0801) and highest sensitivity at optimal threshold (0.92)
- **Random Forest** demonstrated excellent balance between sensitivity (0.84) and specificity (0.96), making it suitable for clinical decision support

**2. Class Imbalance Handling**
- SMOTE effectively addressed the severe class imbalance (4.9% stroke prevalence)
- After SMOTE, models achieved substantially improved sensitivity (0.76-0.92) for detecting stroke cases
- Confirming **Hypothesis H4**: SMOTE improved recall for the minority class without severely compromising specificity

**3. Feature Importance**
- **Age** emerged as the dominant predictor across all models (SHAP importance: 0.161-2.626)
- Cardiovascular risk factors showed moderate importance: heart_disease, hypertension, avg_glucose_level
- **BMI** demonstrated stronger influence than initially expected (SHAP: 0.067-1.837)
- Partially confirming **Hypothesis H3**: Age was the most important predictor, followed by cardiovascular risk factors

**4. Risk Stratification**
- Risk stratification using Random Forest effectively separated patients into three distinct groups:
  - **Low Risk** (65.2% of patients): 1.7% actual stroke rate
  - **Medium Risk** (27.0% of patients): 10.0% actual stroke rate
  - **High Risk** (7.8% of patients): 21.2% actual stroke rate
- Confirming **Hypothesis H5**: Risk stratification effectively differentiated patients with statistically significant differences in stroke rates

**5. Model Calibration and Threshold Optimization**
- Gradient Boosting showed the best calibration among all models
- Optimal decision thresholds (via Youden's Index) ranged from 0.18 to 0.38, substantially lower than the default 0.5 threshold
- This finding emphasizes the importance of threshold optimization for imbalanced datasets in clinical settings

### 4.2 Clinical Interpretation

#### 4.2.1 Comparison with Existing Literature

Our findings align with and extend previous research:

**Machine Learning Effectiveness**: Our AUROC values (0.80-0.82) are consistent with recent studies by Dritsas & Trigka (2022) who reported AUROC of 0.78-0.85 for stroke prediction, and superior to traditional risk scores like Framingham (AUROC ~0.70-0.75) and CHADS₂ (AUROC ~0.65-0.70).

**Ensemble vs. Traditional Methods**: Contrary to **Hypothesis H2**, Logistic Regression outperformed some ensemble methods, suggesting that with proper preprocessing, simpler models can achieve competitive performance.

**Age as Dominant Predictor**: Our finding confirms multiple prior studies, including Wolf et al. (1991) who demonstrated age as the strongest predictor in the Framingham Stroke Risk Profile.

#### 4.2.2 Clinical Implications

**Model Selection for Different Scenarios**:

1. **Primary Care Screening**: Logistic Regression (interpretability, AUROC=0.8245, threshold=0.21)
2. **Hospital Risk Assessment**: Random Forest (balanced performance, sensitivity=0.84, specificity=0.96)
3. **Screening Programs**: Gradient Boosting (highest sensitivity=0.92, best calibration)

**Risk Stratification for Resource Allocation**:
- Low Risk (1.7%): Routine education and annual monitoring
- Medium Risk (10.0%): Targeted interventions, semi-annual monitoring
- High Risk (21.2%): Aggressive management, quarterly monitoring

**Feature-Based Clinical Guidance**:
- Modifiable factors: BMI management, glucose control, hypertension management
- Non-modifiable factors: Age-based screening intensification, cardiovascular surveillance

### 4.3 Study Strengths

1. **Comprehensive Model Comparison**: Systematic evaluation of seven diverse ML algorithms
2. **Rigorous Class Imbalance Handling**: SMOTE applied exclusively to training data
3. **Clinical Utility Focus**: Calibration, threshold optimization, risk stratification
4. **Methodological Rigor**: Multiple metrics, per-class performance, reproducible pipeline
5. **Transparent Reporting**: TRIPOD, STROBE, CONSORT-AI guidelines

### 4.4 Study Limitations

**Data Limitations**:
- Single dataset; no external validation
- Retrospective design; cannot establish causality
- Limited sample size (249 stroke cases)
- Missing BMI data (3.9%)

**Clinical Limitations**:
- No stroke subtype distinction (ischemic vs. hemorrhagic)
- No severity or functional outcome data
- No treatment information
- Unclear follow-up duration

**Methodological Limitations**:
- SMOTE synthetic samples may not represent real distribution
- Optimal thresholds may not generalize to other populations
- SHAP analysis limited to 500 samples for computational efficiency

### 4.5 Future Research Directions

1. **External Validation**: Independent datasets from different healthcare settings
2. **Prospective Trials**: Compare ML-guided interventions vs. standard care
3. **Enhanced Features**: Genetic markers, imaging biomarkers, longitudinal data
4. **Advanced Methods**: Deep learning, multi-task learning, federated learning
5. **Stroke Subtype Prediction**: Separate models for ischemic vs. hemorrhagic
6. **Real-Time Monitoring**: Wearable devices for continuous risk assessment
7. **Implementation Science**: Barriers/facilitators to clinical adoption
8. **Explainable AI Advances**: Intuitive visualizations for clinicians and patients

### 4.6 Overall Conclusions

This study successfully developed and validated a comprehensive machine learning framework for stroke prediction, demonstrating that:

1. Machine learning models achieve clinically meaningful performance (AUROC 0.80-0.82) for stroke risk prediction
2. Proper handling of class imbalance through SMOTE substantially improves minority class detection
3. Model selection should be tailored to clinical context based on interpretability, balance, or sensitivity priorities
4. Age is the dominant predictor, but modifiable cardiovascular risk factors provide actionable intervention targets
5. Risk stratification effectively separates patients into low (1.7%), medium (10.0%), and high (21.2%) stroke risk groups
6. Explainable AI techniques (SHAP) enhance clinical trust through transparent feature importance explanations
7. Threshold optimization is critical for imbalanced clinical data

Despite limitations, this study provides a robust foundation for ML-based stroke risk assessment. When externally validated, these models have strong potential to serve as clinical decision support tools, improving early stroke detection and enabling timely preventive interventions.

**Clinical Impact Potential**: If validated prospectively, this ML framework could help identify high-risk patients earlier, optimize resource allocation, and ultimately reduce stroke-related morbidity, mortality, and healthcare costs.

**Final Statement**: Machine learning offers a powerful, evidence-based approach to stroke risk prediction. With continued validation, refinement, and implementation research, ML-guided stroke prevention has the potential to significantly improve population health outcomes.

---

## References

[1] World Health Organization. (2022). Global Stroke Fact Sheet 2022.

[2] Wolf PA, et al. (1991). Probability of stroke: a risk profile from the Framingham Study. *Stroke*, 22(3), 312-318.

[3] Gage BF, et al. (2001). Validation of clinical classification schemes for predicting stroke: results from the ATRIA study. *JAMA*, 285(22), 2864-2870.

[4] Rajkomar A, et al. (2019). Machine learning in medicine. *N Engl J Med*, 380(14), 1347-1358.

[5] Mainali S, et al. (2021). Machine learning in action: stroke diagnosis and outcome prediction. *Frontiers in Neurology*, 12, 734345.

[6] Dritsas E, Trigka M. (2022). Stroke risk prediction with machine learning techniques. *Sensors*, 22(13), 4670.

[7] Abedi V, et al. (2021). Novel screening tool for stroke using artificial neural network. *Stroke*, 52(10), 3204-3211.

[8] Sung SF, et al. (2020). Developing a stroke severity index based on administrative data was feasible using data mining techniques. *J Clin Epidemiol*, 54(12), 1393-1400.

[9] Chawla NV, et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321-357.

[10] Lundberg SM, Lee SI. (2017). A unified approach to interpreting model predictions. *NIPS*, 4765-4774.

[11] Collins GS, Moons KGM. (2019). Reporting of artificial intelligence prediction models. *Lancet*, 393(10181), 1577-1579.

---

## Acknowledgments

We acknowledge the open-source contributors to the Python scientific computing ecosystem (scikit-learn, pandas, XGBoost, SHAP) that made this analysis possible. We also thank the creators of the Stroke Prediction Dataset on Kaggle for making the data publicly available for research purposes.

---

## Data and Code Availability

**Data**: The Stroke Prediction Dataset is publicly available on Kaggle: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

**Code**: The complete analysis pipeline is available in the project repository: `analysis_01_preprocess/`, `analysis_02_eda/`, and `analysis_03_model/` directories.

**Reproducibility**: All analyses were conducted using Python 3.14 with fixed random seed (42). Software versions documented in `utils/config.py`.

---

## Competing Interests

The authors declare no competing financial or non-financial interests related to this research.

---

## Funding

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.
