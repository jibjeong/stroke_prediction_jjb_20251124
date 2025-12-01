"""
Additional Model Evaluation Metrics
- Calibration Plot (모델 보정 곡선)
- Precision-Recall Curve
- Threshold Optimization
- Clinical Impact Analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_curve, auc, brier_score_loss)
import warnings
warnings.filterwarnings('ignore')

from utils.config import *

# 한글 폰트 설정 (폰트가 없으면 기본 폰트 사용)
if KOREAN_FONT:
    plt.rcParams['font.family'] = KOREAN_FONT
plt.rcParams['axes.unicode_minus'] = False

def load_data_and_models():
    """데이터 및 모델 로딩"""
    print("=" * 60)
    print("Additional Model Evaluation")
    print("=" * 60)

    # 데이터 로딩
    data_file = PREPROCESS_DIR / "data" / "final" / "stroke_preprocessed.csv"
    df = pd.read_csv(data_file)

    X = df.drop([ID_COLUMN, TARGET_COLUMN], axis=1)
    y = df[TARGET_COLUMN]

    # Train-test split (동일한 random seed 사용)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\n> Data loaded: {X.shape[0]} samples")
    print(f"> Test set: {X_test.shape[0]} samples")

    # 모델 로딩
    model_dir = MODEL_DIR / "data" / "models"
    models = {
        'Logistic Regression': joblib.load(model_dir / 'logistic_regression.pkl'),
        'Random Forest': joblib.load(model_dir / 'random_forest.pkl'),
        'XGBoost': joblib.load(model_dir / 'xgboost.pkl'),
        'Gradient Boosting': joblib.load(model_dir / 'gradient_boosting.pkl'),
        'Neural Network': joblib.load(model_dir / 'neural_network.pkl')
    }

    print(f"> Models loaded: {list(models.keys())}")

    return X_test, y_test, models

def plot_calibration_curves(models, X_test, y_test):
    """Calibration Plot - 모델 보정 곡선"""
    print("\n" + "=" * 60)
    print("Creating Calibration Plots")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    calibration_metrics = []

    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]

        # 예측 확률
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

        # Brier score (보정 성능 지표)
        brier_score = brier_score_loss(y_test, y_pred_proba)

        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax.plot(prob_pred, prob_true, 's-', label=f'{name}\n(Brier={brier_score:.4f})')
        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Observed Frequency', fontsize=11)
        ax.set_title(f'Calibration Plot - {name}', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        calibration_metrics.append({
            'Model': name,
            'Brier_Score': brier_score
        })

        print(f"> {name}: Brier Score = {brier_score:.4f}")

    # 빈 subplot 제거
    if len(models) < len(axes):
        fig.delaxes(axes[-1])

    plt.suptitle('Calibration Curves - Model Comparison', fontsize=16, y=0.995)
    plt.tight_layout()

    # 저장
    output_dir = MODEL_DIR / "data" / "figures"
    output_file = output_dir / "calibration_curves.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"\n> Calibration curves saved to: {output_file}")
    plt.close()

    # Brier score 테이블 저장
    df_calibration = pd.DataFrame(calibration_metrics)
    output_file = MODEL_DIR / "data" / "tables" / "calibration_metrics.csv"
    df_calibration.to_csv(output_file, index=False)
    print(f"> Calibration metrics saved to: {output_file}")

    return df_calibration

def plot_precision_recall_curves(models, X_test, y_test):
    """Precision-Recall Curve"""
    print("\n" + "=" * 60)
    print("Creating Precision-Recall Curves")
    print("=" * 60)

    plt.figure(figsize=(12, 9))

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    pr_metrics = []

    for idx, (name, model) in enumerate(models.items()):
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ap_score = average_precision_score(y_test, y_pred_proba)

        plt.plot(recall, precision, color=colors[idx], lw=2,
                 label=f'{name} (AP = {ap_score:.3f})')

        pr_metrics.append({
            'Model': name,
            'Average_Precision': ap_score
        })

        print(f"> {name}: Average Precision = {ap_score:.4f}")

    # Baseline (No Skill)
    no_skill = (y_test == 1).sum() / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], 'k--', lw=2,
             label=f'No Skill (AP = {no_skill:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.title('Precision-Recall Curves - Model Comparison', fontsize=15, pad=20)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 저장
    output_file = MODEL_DIR / "data" / "figures" / "precision_recall_curves.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"\n> PR curves saved to: {output_file}")
    plt.close()

    # AP score 테이블 저장
    df_pr = pd.DataFrame(pr_metrics)
    output_file = MODEL_DIR / "data" / "tables" / "precision_recall_metrics.csv"
    df_pr.to_csv(output_file, index=False)

    return df_pr

def optimize_threshold(models, X_test, y_test):
    """Threshold Optimization - Youden's Index"""
    print("\n" + "=" * 60)
    print("Optimizing Decision Thresholds")
    print("=" * 60)

    threshold_results = []

    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        # Youden's Index = TPR - FPR
        youdens_index = tpr - fpr
        optimal_idx = np.argmax(youdens_index)
        optimal_threshold = thresholds[optimal_idx]

        # 최적 threshold에서의 성능
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

        # Sensitivity, Specificity
        tn = ((y_test == 0) & (y_pred_optimal == 0)).sum()
        fp = ((y_test == 0) & (y_pred_optimal == 1)).sum()
        fn = ((y_test == 1) & (y_pred_optimal == 0)).sum()
        tp = ((y_test == 1) & (y_pred_optimal == 1)).sum()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        threshold_results.append({
            'Model': name,
            'Optimal_Threshold': optimal_threshold,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Youdens_Index': youdens_index[optimal_idx]
        })

        print(f"\n> {name}:")
        print(f"  - Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  - Sensitivity (Recall): {sensitivity:.4f}")
        print(f"  - Specificity: {specificity:.4f}")
        print(f"  - Youden's Index: {youdens_index[optimal_idx]:.4f}")

    # 테이블 저장
    df_threshold = pd.DataFrame(threshold_results)
    output_file = MODEL_DIR / "data" / "tables" / "optimal_thresholds.csv"
    df_threshold.to_csv(output_file, index=False)
    print(f"\n> Optimal thresholds saved to: {output_file}")

    return df_threshold

def clinical_impact_analysis(models, X_test, y_test):
    """Clinical Impact Analysis"""
    print("\n" + "=" * 60)
    print("Clinical Impact Analysis")
    print("=" * 60)

    # Best model (Random Forest 또는 XGBoost)
    best_model_name = 'Random Forest'
    best_model = models[best_model_name]

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    # Risk stratification
    risk_groups = []
    for prob in y_pred_proba:
        if prob < 0.3:
            risk_groups.append('Low Risk')
        elif prob < 0.7:
            risk_groups.append('Medium Risk')
        else:
            risk_groups.append('High Risk')

    risk_groups = np.array(risk_groups)

    # 각 위험군별 실제 stroke 비율
    print(f"\n> Risk Stratification using {best_model_name}:")
    for risk_level in ['Low Risk', 'Medium Risk', 'High Risk']:
        mask = (risk_groups == risk_level)
        n_patients = mask.sum()
        n_stroke = y_test[mask].sum()
        stroke_rate = (n_stroke / n_patients * 100) if n_patients > 0 else 0

        print(f"\n  {risk_level}:")
        print(f"    - Patients: {n_patients} ({n_patients/len(y_test)*100:.1f}%)")
        print(f"    - Actual Strokes: {n_stroke} ({stroke_rate:.1f}%)")

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 위험군별 환자 수
    ax = axes[0]
    risk_counts = pd.Series(risk_groups).value_counts().reindex(['Low Risk', 'Medium Risk', 'High Risk'])
    colors = ['green', 'orange', 'red']
    bars = ax.bar(range(len(risk_counts)), risk_counts.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(risk_counts)))
    ax.set_xticklabels(risk_counts.index, fontsize=11)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Risk Stratification Distribution', fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(risk_counts.values):
        ax.text(i, v + 10, f'{v}\n({v/len(y_test)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    # 위험군별 실제 stroke 비율
    ax = axes[1]
    stroke_rates = []
    for risk_level in ['Low Risk', 'Medium Risk', 'High Risk']:
        mask = (risk_groups == risk_level)
        n_patients = mask.sum()
        n_stroke = y_test[mask].sum()
        stroke_rate = (n_stroke / n_patients * 100) if n_patients > 0 else 0
        stroke_rates.append(stroke_rate)

    bars = ax.bar(range(len(stroke_rates)), stroke_rates, color=colors, edgecolor='black')
    ax.set_xticks(range(len(stroke_rates)))
    ax.set_xticklabels(['Low Risk', 'Medium Risk', 'High Risk'], fontsize=11)
    ax.set_ylabel('Actual Stroke Rate (%)', fontsize=12)
    ax.set_title('Actual Stroke Rate by Risk Group', fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(stroke_rates):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.suptitle(f'Clinical Risk Stratification - {best_model_name}', fontsize=15, y=1.02)
    plt.tight_layout()

    # 저장
    output_file = MODEL_DIR / "data" / "figures" / "clinical_risk_stratification.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"\n> Risk stratification plot saved to: {output_file}")
    plt.close()

def create_summary_report(df_calibration, df_pr, df_threshold):
    """종합 평가 리포트"""
    print("\n" + "=" * 60)
    print("Creating Summary Report")
    print("=" * 60)

    # 기존 Table 1 로딩
    table1_file = MODEL_DIR / "data" / "tables" / "table1_average_performance.csv"
    df_table1 = pd.read_csv(table1_file)

    # 5개 모델만 선택 (추가 평가에서 사용한 모델)
    selected_models = ['Logistic Regression', 'Random Forest', 'XGBoost',
                      'Gradient Boosting', 'Neural Network']
    df_table1_filtered = df_table1[df_table1['Model'].isin(selected_models)].reset_index(drop=True)

    # 통합 테이블 생성
    df_summary = df_table1_filtered.copy()
    df_summary['Brier_Score'] = df_calibration['Brier_Score'].values
    df_summary['Average_Precision'] = df_pr['Average_Precision'].values
    df_summary['Optimal_Threshold'] = df_threshold['Optimal_Threshold'].values
    df_summary['Sensitivity_Optimal'] = df_threshold['Sensitivity'].values
    df_summary['Specificity_Optimal'] = df_threshold['Specificity'].values

    # 저장
    output_file = MODEL_DIR / "data" / "tables" / "comprehensive_evaluation.csv"
    df_summary.to_csv(output_file, index=False)
    print(f"\n> Comprehensive evaluation saved to: {output_file}")

    # 콘솔 출력
    print(f"\n{'=' * 120}")
    print("Comprehensive Model Evaluation Summary")
    print(f"{'=' * 120}")
    print(df_summary.to_string(index=False))

    return df_summary

def main():
    """메인 실행 함수"""
    # 1. 데이터 및 모델 로딩
    X_test, y_test, models = load_data_and_models()

    # 2. Calibration curves
    df_calibration = plot_calibration_curves(models, X_test, y_test)

    # 3. Precision-Recall curves
    df_pr = plot_precision_recall_curves(models, X_test, y_test)

    # 4. Threshold optimization
    df_threshold = optimize_threshold(models, X_test, y_test)

    # 5. Clinical impact analysis
    clinical_impact_analysis(models, X_test, y_test)

    # 6. Summary report
    df_summary = create_summary_report(df_calibration, df_pr, df_threshold)

    print("\n" + "=" * 60)
    print("> Additional evaluation completed successfully!")
    print("=" * 60)

    print(f"\n> Output files:")
    print(f"  - Figures:")
    print(f"    • calibration_curves.png")
    print(f"    • precision_recall_curves.png")
    print(f"    • clinical_risk_stratification.png")
    print(f"  - Tables:")
    print(f"    • calibration_metrics.csv")
    print(f"    • precision_recall_metrics.csv")
    print(f"    • optimal_thresholds.csv")
    print(f"    • comprehensive_evaluation.csv")

if __name__ == "__main__":
    main()
