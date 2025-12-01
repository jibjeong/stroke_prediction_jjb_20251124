# Background

## Introduction

Stroke is a leading cause of death and long-term disability worldwide, affecting approximately 15 million people annually [1]. Early identification of high-risk individuals is crucial for implementing preventive measures and reducing the burden of stroke-related morbidity and mortality. Traditional risk assessment tools, such as the Framingham Stroke Risk Profile and CHADS₂ score, rely on clinical judgment and predefined risk factors, but may not capture complex non-linear relationships among variables [2, 3].

Machine learning (ML) algorithms have emerged as promising tools for stroke risk prediction, offering the potential to improve accuracy by identifying complex patterns in large-scale clinical data [4]. Recent studies have demonstrated that ML models, including Random Forest, XGBoost, and Neural Networks, can achieve superior predictive performance compared to traditional statistical methods [5-7]. However, most existing studies have focused on a single algorithm or limited model comparison, and few have addressed the challenge of class imbalance inherent in stroke prediction datasets.

## Current State of Research

### Limitations of Existing Studies

**1. Limited Model Comparison**
Most previous studies have evaluated 1-3 machine learning algorithms, making it difficult to determine the optimal model for stroke prediction [8]. Comprehensive comparisons across multiple algorithms using the same dataset are lacking.

**2. Class Imbalance Problem**
Stroke is a relatively rare event (typically 5-10% prevalence), leading to severe class imbalance in prediction datasets [9]. Many studies have not adequately addressed this issue, resulting in models with high overall accuracy but poor sensitivity for detecting stroke cases.

**3. Lack of Clinical Interpretability**
Complex ML models such as deep neural networks and ensemble methods often function as "black boxes," making it challenging for clinicians to understand and trust the predictions [10]. Model interpretability is essential for clinical adoption.

**4. Insufficient External Validation**
Most studies have relied on single-center datasets without external validation, raising concerns about generalizability to different populations and healthcare settings [11].

## Research Gap

Despite significant progress in ML-based stroke prediction, several critical gaps remain:

1. **Comprehensive Model Comparison**: There is a need for systematic comparison of multiple ML algorithms (both traditional and deep learning approaches) on the same dataset to identify the best-performing model.

2. **Handling Class Imbalance**: Effective strategies for addressing class imbalance, such as SMOTE (Synthetic Minority Over-sampling Technique), need to be evaluated in the context of stroke prediction.

3. **Model Interpretability**: Integration of explainable AI techniques, such as SHAP (SHapley Additive exPlanations), is necessary to enhance clinical trust and adoption.

4. **Clinical Utility**: Beyond predictive accuracy, models must demonstrate clinical utility through calibration analysis, threshold optimization, and risk stratification.

5. **Practical Implementation**: Development of practical guidelines for model selection and deployment in real-world clinical settings is needed.

## Study Objectives

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

## Clinical Significance

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

## Research Hypotheses

**H1**: Machine learning models will achieve superior predictive performance (AUROC > 0.80) compared to chance (AUROC = 0.50).

**H2**: Ensemble methods (Random Forest, XGBoost, Gradient Boosting) will outperform traditional Logistic Regression in terms of overall accuracy and AUROC.

**H3**: Age will be the most important predictor of stroke risk, followed by cardiovascular risk factors (hypertension, heart disease, glucose level).

**H4**: SMOTE will improve sensitivity (recall) for detecting stroke cases without significantly compromising specificity.

**H5**: Risk stratification will effectively separate patients into distinct risk groups with statistically different stroke rates.

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
