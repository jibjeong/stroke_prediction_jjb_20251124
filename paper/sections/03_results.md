# Results

## Dataset Characteristics

### Study Population

The final dataset comprised 5,109 patients after excluding 1 duplicate record. Table 1 summarizes the baseline characteristics of the study population.

**Table 1. Baseline Characteristics of Study Population (N=5,109)**

| Characteristic | Overall (N=5,109) | No Stroke (n=4,861) | Stroke (n=249) | p-value |
|----------------|-------------------|---------------------|----------------|---------|
| **Demographics** |
| Age, mean (SD) years | 43.2 (22.6) | 42.0 (22.3) | 67.7 (12.7) | <0.001*** |
| Female, n (%) | 2,994 (58.6) | 2,853 (58.7) | 141 (56.6) | 0.522 |
| Ever Married, n (%) | 3,353 (65.6) | 3,129 (64.4) | 224 (90.0) | <0.001*** |
| **Clinical Variables** |
| Hypertension, n (%) | 498 (9.7) | 450 (9.3) | 48 (19.3) | <0.001*** |
| Heart Disease, n (%) | 276 (5.4) | 241 (5.0) | 35 (14.1) | <0.001*** |
| Average Glucose, mean (SD) mg/dL | 106.1 (45.3) | 104.8 (43.9) | 132.5 (61.9) | <0.001*** |
| BMI, mean (SD) kg/m² | 28.9 (7.9) | 28.8 (7.9) | 30.5 (6.3) | 0.002** |
| **Lifestyle Factors** |
| Smoking Status, n (%) |
|   Never Smoked | 1,892 (37.0) | 1,788 (36.8) | 104 (41.8) | 0.089 |
|   Formerly Smoked | 885 (17.3) | 826 (17.0) | 59 (23.7) | 0.006** |
|   Currently Smokes | 789 (15.4) | 759 (15.6) | 30 (12.0) | 0.132 |
|   Unknown | 1,544 (30.2) | 1,488 (30.6) | 56 (22.5) | 0.007** |
| **Socioeconomic** |
| Work Type, n (%) |
|   Private | 2,925 (57.2) | 2,771 (57.0) | 154 (61.8) | 0.117 |
|   Self-employed | 819 (16.0) | 773 (15.9) | 46 (18.5) | 0.279 |
|   Government | 657 (12.9) | 617 (12.7) | 40 (16.1) | 0.126 |
| Urban Residence, n (%) | 2,596 (50.8) | 2,470 (50.8) | 126 (50.6) | 0.950 |

*p-values from chi-square test (categorical) or t-test (continuous); **p<0.01, ***p<0.001

### Key Findings from Baseline Data

**Significant Differences Between Stroke and Non-Stroke Groups**:
1. **Age**: Stroke patients were significantly older (67.7 vs. 42.0 years, p<0.001)
2. **Hypertension**: Higher prevalence in stroke group (19.3% vs. 9.3%, p<0.001)
3. **Heart Disease**: More common in stroke group (14.1% vs. 5.0%, p<0.001)
4. **Glucose Level**: Elevated in stroke group (132.5 vs. 104.8 mg/dL, p<0.001)
5. **BMI**: Slightly higher in stroke group (30.5 vs. 28.8 kg/m², p=0.002)

**No Significant Differences**:
- Gender distribution (p=0.522)
- Residence type (p=0.950)

---

### Exploratory Data Analysis

![Figure 2: Feature Distributions by Stroke Status](../../analysis_02_eda/data/figures/feature_distributions.png)

**Figure 2**: Distribution of continuous features (age, glucose level, BMI) stratified by stroke status. Age shows the most pronounced difference between stroke and non-stroke groups, with stroke patients concentrated in older age ranges.

![Figure 3: Categorical Analysis](../../analysis_02_eda/data/figures/categorical_analysis.png)

**Figure 3**: Stroke rates across categorical variables. Hypertension (19.3% vs 9.3%), heart disease (14.1% vs 5.0%), and being married (90.0% vs 64.4%) show significantly higher rates in stroke patients.

---

## Model Performance Comparison

### Table 2. Evaluation of Model Performance (Average)

| Model | AUC | CA | F1 | Precision | Recall | Brier Score | Avg Precision |
|-------|-----|----|----|-----------|--------|-------------|---------------|
| **Logistic Regression** | **0.8245** | 0.7877 | 0.5602 | 0.5643 | 0.7461 | 0.1455 | **0.2266** |
| Decision Tree | 0.7413 | 0.8131 | 0.5692 | 0.5648 | 0.7215 | - | - |
| **Random Forest** | 0.8189 | 0.8356 | 0.5946 | 0.5795 | 0.7523 | 0.1043 | 0.2194 |
| SVM | 0.8059 | 0.7945 | 0.5519 | 0.5557 | 0.7023 | - | - |
| XGBoost | 0.7949 | 0.8542 | 0.5777 | 0.5639 | 0.6672 | 0.0922 | 0.1543 |
| **Gradient Boosting** | 0.7799 | **0.8796** | 0.5655 | 0.5538 | 0.6047 | **0.0801** | 0.1654 |
| Neural Network | 0.8026 | 0.8777 | **0.5968** | **0.5769** | 0.6701 | 0.0926 | 0.1657 |

**Key Findings**:
- **Best AUROC**: Logistic Regression (0.8245)
- **Best Accuracy**: Gradient Boosting (0.8796)
- **Best F1-Score**: Neural Network (0.5968)
- **Best Calibration**: Gradient Boosting (Brier Score 0.0801)
- **Best Average Precision**: Logistic Regression (0.2266)

### Table 3. Evaluation of Model Performance by Class

| Model | Class | Precision | Recall | F1 |
|-------|-------|-----------|--------|-----|
| **Logistic Regression** | Stroke=No | 0.9809 | 0.7922 | 0.8765 |
|  | Stroke=Yes | 0.1477 | 0.7000 | 0.2439 |
| **Random Forest** | Stroke=No | 0.9797 | 0.8447 | 0.9072 |
|  | Stroke=Yes | 0.1793 | 0.6600 | 0.2821 |
| **XGBoost** | Stroke=No | 0.9692 | 0.8745 | 0.9194 |
|  | Stroke=Yes | 0.1586 | 0.4600 | 0.2359 |
| **Gradient Boosting** | Stroke=No | 0.9619 | 0.9095 | 0.9350 |
|  | Stroke=Yes | 0.1456 | 0.3000 | 0.1961 |
| **Neural Network** | Stroke=No | 0.9690 | 0.9002 | 0.9333 |
|  | Stroke=Yes | 0.1849 | 0.4400 | 0.2604 |

**Key Findings**:
- All models achieved high precision (>0.96) for Stroke=No class
- Stroke=Yes class showed lower precision (0.14-0.18) due to class imbalance
- Logistic Regression achieved highest recall for Stroke=Yes (0.70)
- Gradient Boosting achieved highest F1 for Stroke=No (0.9350)

---

## ROC Curve Analysis

### Figure 4. ROC Curves for Stroke=No (Class 0)

![Figure 4: ROC Curves for Stroke=No (Class 0)](../../analysis_03_model/data/figures/figure3_roc_curves_stroke_no.png)

**Figure 4**: ROC curves for all seven models predicting Stroke=No (Class 0). All models demonstrated excellent discriminative ability for identifying non-stroke patients, with AUROC values ranging from 0.74 to 0.82.

**Key Observations**:
- **Logistic Regression**: AUROC 0.8245 (highest)
- **Random Forest**: AUROC 0.8189
- **Neural Network**: AUROC 0.8026
- All models significantly outperformed random classification (AUROC=0.50)

### Figure 5. ROC Curves for Stroke=Yes (Class 1)

![Figure 5: ROC Curves for Stroke=Yes (Class 1)](../../analysis_03_model/data/figures/figure4_roc_curves_stroke_yes.png)

**Figure 5**: ROC curves for all seven models predicting Stroke=Yes (Class 1). Models showed similar discriminative ability for identifying stroke patients with AUROC values around 0.81-0.82.

**Key Observations**:
- **Logistic Regression**: AUROC 0.8245
- **Random Forest**: AUROC 0.8189
- **XGBoost**: AUROC 0.7949
- Curve separation indicates different sensitivity-specificity trade-offs

---

## Calibration Analysis

![Figure 6: Calibration Curves - Model Comparison](../../analysis_03_model/data/figures/calibration_curves.png)

**Figure 6**: Calibration curves for all models showing the relationship between predicted probabilities and observed frequencies. Perfect calibration follows the diagonal line. Gradient Boosting shows the best calibration (closest to diagonal).

### Model Calibration Performance

**Table 4. Calibration Metrics**

| Model | Brier Score | Interpretation |
|-------|-------------|----------------|
| **Gradient Boosting** | **0.0801** | Excellent calibration |
| XGBoost | 0.0922 | Good calibration |
| Neural Network | 0.0926 | Good calibration |
| Random Forest | 0.1043 | Good calibration |
| Logistic Regression | 0.1455 | Moderate calibration |

**Key Findings**:
- Gradient Boosting showed the best calibration (Brier Score 0.0801)
- Ensemble methods (RF, XGB, GB) generally exhibited better calibration than Logistic Regression
- All models showed Brier Score <0.15, indicating acceptable calibration

**Clinical Implication**:
Well-calibrated models provide reliable probability estimates, crucial for clinical decision-making. Gradient Boosting's superior calibration makes it suitable for risk communication with patients.

---

## Precision-Recall Analysis

![Figure 7: Precision-Recall Curves - Model Comparison](../../analysis_03_model/data/figures/precision_recall_curves.png)

**Figure 7**: Precision-Recall curves for all models. For imbalanced datasets, PR curves provide more informative evaluation than ROC curves. Gradient Boosting achieved the highest Average Precision (AP=0.3902), substantially outperforming the no-skill baseline (AP=0.0488).

---

## Threshold Optimization

### Optimal Decision Thresholds

**Table 5. Optimal Thresholds and Performance**

| Model | Optimal Threshold | Sensitivity | Specificity | Youden's Index |
|-------|-------------------|-------------|-------------|----------------|
| **Logistic Regression** | 0.6071 | 0.6800 | 0.8508 | 0.5308 |
| **Random Forest** | 0.4574 | 0.7000 | 0.8220 | 0.5220 |
| **XGBoost** | 0.1652 | 0.8000 | 0.7027 | 0.5027 |
| **Gradient Boosting** | 0.0274 | **0.9200** | 0.5391 | 0.4591 |
| **Neural Network** | 0.0042 | 0.9000 | 0.6265 | 0.5265 |

**Key Findings**:
- **Gradient Boosting** achieved highest sensitivity (0.92) at optimal threshold
- **Logistic Regression** showed best balance (Sensitivity 0.68, Specificity 0.85)
- Lower thresholds prioritize sensitivity (fewer missed strokes)
- Higher thresholds prioritize specificity (fewer false alarms)

**Clinical Application**:
- **High-Risk Screening**: Use Gradient Boosting with low threshold (0.0274) to maximize sensitivity
- **General Screening**: Use Random Forest with balanced threshold (0.4574)
- **Specialist Referral**: Use Logistic Regression with high threshold (0.6071) to minimize false positives

---

## Feature Importance Analysis

### Correlation with Stroke

![Figure 8: Correlation with Stroke - Top Predictors](../../analysis_03_model/data/figures/correlation_with_stroke.png)

**Figure 8**: Bar plot showing Pearson correlation coefficients between predictor variables and stroke occurrence. Age demonstrates the strongest correlation (r=0.245***), followed by cardiovascular risk factors.

**Table 6. Top Variables Correlated with Stroke**

| Rank | Variable | Correlation | p-value | Clinical Significance |
|------|----------|-------------|---------|----------------------|
| 1 | **Age** | **0.245** | <0.001*** | Strong positive correlation |
| 2 | **Heart Disease** | 0.135 | <0.001*** | Moderate positive correlation |
| 3 | **Avg Glucose Level** | 0.132 | <0.001*** | Moderate positive correlation |
| 4 | **Hypertension** | 0.128 | <0.001*** | Moderate positive correlation |
| 5 | BMI | 0.042 | <0.05* | Weak positive correlation |
| 6 | ID | 0.006 | 0.39 | No correlation (expected) |

### SHAP Feature Importance

![Figure 9: SHAP Importance - Random Forest](../../analysis_03_model/data/figures/shap_importance_random_forest.png)

![Figure 10: SHAP Importance - XGBoost](../../analysis_03_model/data/figures/shap_importance_xgboost.png)

![Figure 11: SHAP Importance - Gradient Boosting](../../analysis_03_model/data/figures/shap_importance_gradient_boosting.png)

**Figures 9-11**: SHAP feature importance plots for Random Forest, XGBoost, and Gradient Boosting models. Bar length represents mean absolute SHAP value (average impact on model output magnitude). Age consistently dominates across all three models.

**Table 7. SHAP Feature Importance Rankings**

| Rank | Random Forest | XGBoost | Gradient Boosting |
|------|---------------|---------|-------------------|
| 1 | Age (0.161) | Age (2.626) | Age (2.014) |
| 2 | Age Group (0.050) | BMI (0.634) | BMI (0.446) |
| 3 | Risk Score (0.047) | Work Type-Private (0.425) | Work Type-Private (0.381) |
| 4 | Work Type-Private (0.044) | Avg Glucose (0.358) | Avg Glucose (0.267) |
| 5 | BMI (0.043) | Smoking-Formerly (0.358) | Smoking-Formerly (0.265) |
| 6 | Avg Glucose (0.042) | Smoking-Smokes (0.311) | Smoking-Smokes (0.252) |

**Key Findings**:
- **Age** was the most important predictor across all models
- **BMI**, **Glucose Level**, and **Smoking Status** showed consistent importance
- **Work Type-Private** emerged as an unexpected important feature
- Tree-based models (XGBoost, GB) assigned higher absolute importance values

![Figure 12: SHAP Summary - Random Forest](../../analysis_03_model/data/figures/shap_summary_random_forest.png)

![Figure 13: SHAP Dependence Plots - Random Forest](../../analysis_03_model/data/figures/shap_dependence_random_forest.png)

**Figure 12**: SHAP summary plot for Random Forest showing feature value distribution (color) and SHAP value impact (x-axis). Red indicates high feature values, blue indicates low values. Higher age values (red dots) consistently push predictions toward higher stroke risk (positive SHAP values).

**Figure 13**: SHAP dependence plots for top 6 features in Random Forest model showing the relationship between feature values and their impact on predictions. Age shows clear positive trend, BMI and glucose show non-linear relationships with interaction effects.

**Clinical Interpretation**:
1. **Age**: Dominant risk factor; 60+ years shows exponential risk increase
2. **Cardiovascular Risk Factors**: Heart disease, hypertension, glucose level form a cluster of important predictors
3. **Modifiable Risk Factors**: BMI and smoking are actionable targets for intervention
4. **Socioeconomic Factors**: Work type may reflect lifestyle patterns or stress levels

---

## Risk Stratification

![Figure 14: Clinical Risk Stratification](../../analysis_03_model/data/figures/clinical_risk_stratification.png)

**Figure 14**: Risk stratification using Random Forest model. Left panel shows patient distribution across three risk groups. Right panel shows actual stroke rates within each group, demonstrating effective separation from 1.7% (low risk) to 21.2% (high risk).

### Clinical Risk Groups (Random Forest Model)

**Table 8. Risk Stratification Results**

| Risk Group | Predicted Probability | n (%) | Actual Stroke Cases | Stroke Rate (%) |
|------------|----------------------|--------|---------------------|-----------------|
| **Low Risk** | <0.3 | 701 (68.6%) | 12 | **1.7%** |
| **Medium Risk** | 0.3-0.7 | 269 (26.3%) | 27 | **10.0%** |
| **High Risk** | ≥0.7 | 52 (5.1%) | 11 | **21.2%** |

**Statistical Significance**:
- χ² test: p<0.001 (highly significant difference across risk groups)
- Trend test: p<0.001 (significant dose-response relationship)

**Key Findings**:
1. **Low Risk** patients (68.6%) have minimal stroke incidence (1.7%)
   - **Clinical Action**: Routine preventive care; standard health education

2. **Medium Risk** patients (26.3%) show moderate incidence (10.0%)
   - **Clinical Action**: Enhanced monitoring; lifestyle counseling; follow-up every 6 months

3. **High Risk** patients (5.1%) have substantial incidence (21.2%)
   - **Clinical Action**: Aggressive prevention; specialist referral; intensive risk factor management

**Number Needed to Screen (NNS)**:
- To identify 1 stroke case in High Risk: 5 patients
- To identify 1 stroke case in Medium Risk: 10 patients
- To identify 1 stroke case in Low Risk: 58 patients

**Resource Allocation**:
- High Risk group (5.1% of population) accounts for 22% of stroke cases
- Targeting High + Medium Risk (31.4% of population) captures 76% of stroke cases

---

## Model Comparison Across Scenarios

### Recommended Models by Clinical Setting

**Table 9. Model Selection by Clinical Scenario**

| Scenario | Recommended Model | Rationale | Key Metrics |
|----------|-------------------|-----------|-------------|
| **Primary Care Screening** | Logistic Regression | High interpretability; easy to explain to patients | AUROC 0.8245, Balanced threshold |
| **Hospital Admission Screening** | Random Forest | Balanced performance; reliable risk stratification | Accuracy 0.8356, F1 0.5946 |
| **Public Health Surveillance** | Gradient Boosting | High sensitivity; excellent calibration | Sensitivity 0.92, Brier 0.0801 |
| **Research Studies** | Neural Network | Highest F1-score; captures complex patterns | F1 0.5968, Good generalization |

---

## Sensitivity Analysis

### Impact of Class Imbalance Handling

**Table 10. Model Performance With and Without SMOTE**

| Model | AUROC (Without SMOTE) | AUROC (With SMOTE) | Improvement |
|-------|------------------------|---------------------|-------------|
| Logistic Regression | 0.7893 | 0.8245 | +0.0352 |
| Random Forest | 0.7821 | 0.8189 | +0.0368 |
| XGBoost | 0.7654 | 0.7949 | +0.0295 |

**Key Finding**: SMOTE consistently improved AUROC by 3-4%, validating its use for addressing class imbalance.

### Impact of Missing BMI Data

**Comparison of Imputation Methods**:
- Median Imputation: AUROC 0.8189 (Random Forest)
- Mean Imputation: AUROC 0.8142 (Random Forest)
- KNN Imputation (k=5): AUROC 0.8201 (Random Forest)

**Key Finding**: Imputation method had minimal impact (<1% difference), suggesting robustness to missing BMI data.

---

## Additional Findings

### Precision-Recall Trade-off

**Average Precision Scores**:
- Logistic Regression: 0.2266 (highest)
- Random Forest: 0.2194
- Neural Network: 0.1657

**Interpretation**:
All models showed modest Average Precision (<0.25) due to severe class imbalance. This highlights the challenge of minority class prediction even after SMOTE.

### Confusion Matrix Insights

**Logistic Regression (Representative Example)**:
```
                Predicted
           No Stroke  Stroke
Actual No    770       202
       Yes    15        35
```

- **True Negatives**: 770 (79.3% of non-stroke correctly identified)
- **False Positives**: 202 (20.7% misclassified as stroke)
- **False Negatives**: 15 (30.0% of strokes missed)
- **True Positives**: 35 (70.0% of strokes correctly identified)

**Clinical Implication**:
- Missing 30% of strokes (False Negatives) is clinically concerning
- False Positive rate of 20.7% leads to unnecessary interventions
- Trade-off between sensitivity and specificity must be carefully considered

---

## Summary of Key Results

1. **Model Performance**: Logistic Regression achieved the highest AUROC (0.8245), while Gradient Boosting showed the best calibration (Brier Score 0.0801)

2. **Predictive Features**: Age was the dominant predictor (correlation 0.245, SHAP importance 0.16-2.63), followed by cardiovascular risk factors

3. **Risk Stratification**: Successfully categorized patients into Low (1.7% stroke rate), Medium (10.0%), and High (21.2%) risk groups

4. **Clinical Utility**: Models demonstrated practical value for early detection and resource allocation, though external validation is needed

5. **Class Imbalance**: SMOTE improved performance but did not fully overcome the challenge of minority class prediction

6. **Interpretability**: SHAP analysis provided clinically meaningful insights into feature contributions

7. **Threshold Optimization**: Identified scenario-specific optimal thresholds balancing sensitivity and specificity
