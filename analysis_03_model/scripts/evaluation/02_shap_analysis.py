"""
SHAP Analysis for Stroke Prediction Models
- Calculate SHAP values for feature importance
- Create SHAP summary plots
- Analyze top features correlation with stroke
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
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
    print("SHAP Analysis for Stroke Prediction")
    print("=" * 60)

    # 데이터 로딩
    data_file = PREPROCESS_DIR / "data" / "final" / "stroke_preprocessed.csv"
    df = pd.read_csv(data_file)

    # Feature와 Target 분리
    X = df.drop([ID_COLUMN, TARGET_COLUMN], axis=1)
    y = df[TARGET_COLUMN]

    # Feature 이름 로딩
    feature_file = PREPROCESS_DIR / "data" / "final" / "feature_names.txt"
    with open(feature_file, 'r') as f:
        feature_names = [line.strip() for line in f]

    print(f"\n> Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # 원본 데이터 로딩 (상관관계 분석용)
    original_file = PREPROCESS_DIR / "data" / "final" / "stroke_original.csv"
    df_original = pd.read_csv(original_file)

    # 모델 로딩 (Random Forest, XGBoost, Gradient Boosting)
    model_dir = MODEL_DIR / "data" / "models"
    models = {
        'Random Forest': joblib.load(model_dir / 'random_forest.pkl'),
        'XGBoost': joblib.load(model_dir / 'xgboost.pkl'),
        'Gradient Boosting': joblib.load(model_dir / 'gradient_boosting.pkl')
    }

    print(f"> Models loaded: {list(models.keys())}")

    return X, y, feature_names, df_original, models

def analyze_correlation(df_original):
    """뇌졸중과 변수들의 상관관계 분석"""
    print("\n" + "=" * 60)
    print("Correlation Analysis with Stroke")
    print("=" * 60)

    # 수치형 변수만 선택
    numeric_df = df_original.select_dtypes(include=[np.number])

    # 상관관계 계산
    corr_with_stroke = numeric_df.corr()['stroke'].drop('stroke').sort_values(ascending=False)

    print(f"\n뇌졸중(Stroke)과 유의한 상관관계를 가지는 변수 (Top 10):\n")
    for idx, (feature, corr_value) in enumerate(corr_with_stroke.head(10).items(), 1):
        significance = "***" if abs(corr_value) > 0.2 else "**" if abs(corr_value) > 0.1 else "*"
        print(f"{idx:2d}. {feature:25s}: {corr_value:7.4f} {significance}")

    # 상관관계 시각화
    plt.figure(figsize=(10, 8))
    top_corr = corr_with_stroke.head(15)

    colors = ['red' if x > 0 else 'blue' for x in top_corr.values]
    bars = plt.barh(range(len(top_corr)), top_corr.values, color=colors, edgecolor='black')

    plt.yticks(range(len(top_corr)), top_corr.index, fontsize=10)
    plt.xlabel('Correlation with Stroke', fontsize=12)
    plt.title('Top 15 Features Correlated with Stroke', fontsize=14, pad=20)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(axis='x', alpha=0.3)

    # 값 표시
    for i, (bar, val) in enumerate(zip(bars, top_corr.values)):
        plt.text(val + 0.005 if val > 0 else val - 0.005, i,
                f'{val:.3f}', va='center',
                ha='left' if val > 0 else 'right', fontsize=9)

    plt.tight_layout()

    # 저장
    output_dir = MODEL_DIR / "data" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "correlation_with_stroke.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"\n> Correlation plot saved to: {output_file}")

    plt.close()

    return corr_with_stroke

def calculate_shap_values(models, X, feature_names):
    """SHAP 값 계산"""
    print("\n" + "=" * 60)
    print("Calculating SHAP Values")
    print("=" * 60)

    shap_results = {}

    # 샘플링 (계산 속도 향상)
    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=RANDOM_SEED)

    for model_name, model in models.items():
        print(f"\n> Calculating SHAP for {model_name}...")

        # TreeExplainer 사용
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # SHAP values 처리
        if isinstance(shap_values, list):
            # 리스트인 경우 class 1 선택
            shap_values = shap_values[1]
        elif len(shap_values.shape) == 3:
            # 3D array인 경우 class 1 선택 (Random Forest)
            shap_values = shap_values[:, :, 1]

        shap_results[model_name] = {
            'explainer': explainer,
            'shap_values': shap_values,
            'X_sample': X_sample
        }

        print(f"  - SHAP values shape: {shap_values.shape}")

    return shap_results

def plot_shap_summary(shap_results, feature_names):
    """SHAP Summary Plot (변수별 importance)"""
    print("\n" + "=" * 60)
    print("Creating SHAP Summary Plots")
    print("=" * 60)

    output_dir = MODEL_DIR / "data" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, result in shap_results.items():
        print(f"\n> Creating SHAP plot for {model_name}...")

        # Summary plot (bar)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            result['shap_values'],
            result['X_sample'],
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=20
        )
        plt.title(f'SHAP Feature Importance - {model_name}', fontsize=14, pad=20)
        plt.tight_layout()

        output_file = output_dir / f"shap_importance_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  - Saved to: {output_file}")
        plt.close()

        # Summary plot (dot) - 변수별 영향도
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            result['shap_values'],
            result['X_sample'],
            feature_names=feature_names,
            show=False,
            max_display=20
        )
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, pad=20)
        plt.tight_layout()

        output_file = output_dir / f"shap_summary_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  - Saved to: {output_file}")
        plt.close()

def plot_shap_feature_plots(shap_results, feature_names):
    """개별 변수별 SHAP 의존성 플롯"""
    print("\n" + "=" * 60)
    print("Creating SHAP Dependence Plots for Top Features")
    print("=" * 60)

    output_dir = MODEL_DIR / "data" / "figures"

    # Random Forest 기준으로 top 6 features 선택
    rf_result = shap_results['Random Forest']
    shap_values_abs = np.abs(rf_result['shap_values']).mean(axis=0)
    top_features_idx = np.argsort(shap_values_abs)[-6:][::-1]
    top_features = [feature_names[i] for i in top_features_idx]

    print(f"\n> Top 6 features: {top_features}")

    for model_name, result in shap_results.items():
        print(f"\n> Creating dependence plots for {model_name}...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()

        for idx, feature in enumerate(top_features):
            feature_idx = feature_names.index(feature)

            # SHAP dependence plot
            shap.dependence_plot(
                feature_idx,
                result['shap_values'],
                result['X_sample'],
                feature_names=feature_names,
                ax=axes[idx],
                show=False
            )
            axes[idx].set_title(f'{feature}', fontsize=12)

        plt.suptitle(f'SHAP Dependence Plots - {model_name}', fontsize=16, y=0.995)
        plt.tight_layout()

        output_file = output_dir / f"shap_dependence_{model_name.replace(' ', '_').lower()}.png"
        plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  - Saved to: {output_file}")
        plt.close()

def create_feature_importance_table(shap_results, feature_names):
    """Feature Importance 테이블 생성"""
    print("\n" + "=" * 60)
    print("Creating Feature Importance Table")
    print("=" * 60)

    importance_data = []

    for model_name, result in shap_results.items():
        # 평균 절대 SHAP 값
        shap_importance = np.abs(result['shap_values']).mean(axis=0)

        for feature, importance in zip(feature_names, shap_importance):
            importance_data.append({
                'Model': model_name,
                'Feature': feature,
                'SHAP_Importance': importance
            })

    df_importance = pd.DataFrame(importance_data)

    # 모델별로 정렬
    df_importance_sorted = df_importance.sort_values(
        ['Model', 'SHAP_Importance'],
        ascending=[True, False]
    )

    # 저장
    output_dir = MODEL_DIR / "data" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "shap_feature_importance.csv"
    df_importance_sorted.to_csv(output_file, index=False)
    print(f"\n> Feature importance table saved to: {output_file}")

    # 상위 10개 출력
    print(f"\n> Top 10 features by SHAP importance:")
    for model_name in shap_results.keys():
        print(f"\n{model_name}:")
        model_data = df_importance_sorted[df_importance_sorted['Model'] == model_name].head(10)
        for idx, row in enumerate(model_data.itertuples(), 1):
            print(f"  {idx:2d}. {row.Feature:25s}: {row.SHAP_Importance:.6f}")

    return df_importance_sorted

def main():
    """메인 실행 함수"""
    # 1. 데이터 및 모델 로딩
    X, y, feature_names, df_original, models = load_data_and_models()

    # 2. 상관관계 분석
    corr_with_stroke = analyze_correlation(df_original)

    # 3. SHAP 값 계산
    shap_results = calculate_shap_values(models, X, feature_names)

    # 4. SHAP Summary Plot
    plot_shap_summary(shap_results, feature_names)

    # 5. SHAP Dependence Plot (Top features)
    plot_shap_feature_plots(shap_results, feature_names)

    # 6. Feature Importance 테이블
    df_importance = create_feature_importance_table(shap_results, feature_names)

    print("\n" + "=" * 60)
    print("> SHAP analysis completed successfully!")
    print("=" * 60)

    print(f"\n> Output files:")
    print(f"  - Figures: {MODEL_DIR / 'data' / 'figures'}")
    print(f"    • correlation_with_stroke.png")
    print(f"    • shap_importance_*.png (3 models)")
    print(f"    • shap_summary_*.png (3 models)")
    print(f"    • shap_dependence_*.png (3 models)")
    print(f"  - Tables: {MODEL_DIR / 'data' / 'tables'}")
    print(f"    • shap_feature_importance.csv")

if __name__ == "__main__":
    main()
