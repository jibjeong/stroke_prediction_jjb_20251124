"""
Phase 2: Exploratory Data Analysis (EDA)
포스터 기반 분석:
- Correlation heatmap
- Feature distributions by stroke status
- Age, BMI, Glucose level analysis
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from utils.config import *

# 한글 폰트 설정 (폰트가 없으면 기본 폰트 사용)
if KOREAN_FONT:
    plt.rcParams['font.family'] = KOREAN_FONT
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """전처리된 데이터 로딩"""
    print("=" * 60)
    print("Phase 2: Exploratory Data Analysis (EDA)")
    print("=" * 60)

    # 원본 데이터 로드 (EDA는 원본 스케일로 수행)
    data_file = PREPROCESS_DIR / "data" / "final" / "stroke_original.csv"
    df = pd.read_csv(data_file)

    print(f"\n✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n Target distribution:")
    print(df[TARGET_COLUMN].value_counts())
    print(f"\n Class ratio: {df[TARGET_COLUMN].value_counts(normalize=True) * 100}")

    return df

def plot_correlation_heatmap(df):
    """상관관계 히트맵 (포스터 Figure 1)"""
    print("\n" + "=" * 60)
    print("Creating Correlation Heatmap")
    print("=" * 60)

    # 수치형 변수만 선택
    numeric_df = df.select_dtypes(include=[np.number])

    # 상관계수 계산
    corr = numeric_df.corr()

    # 히트맵 생성
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # 상삼각 마스킹

    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0,
                square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})

    plt.title('Correlation Heatmap of Numerical Features', fontsize=16, pad=20)
    plt.tight_layout()

    # 저장
    output_dir = EDA_DIR / "data" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "correlation_heatmap.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"\n✓ Heatmap saved to: {output_file}")

    plt.close()

    # 상관관계 요약 출력
    stroke_corr = corr[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)
    print(f"\n✓ Top 5 correlations with Stroke:")
    for feature, corr_value in stroke_corr.head(5).items():
        print(f"  - {feature}: {corr_value:.3f}")

def plot_feature_distributions(df):
    """주요 변수 분포 (포스터 Figure 2)"""
    print("\n" + "=" * 60)
    print("Creating Feature Distribution Plots")
    print("=" * 60)

    # 주요 수치형 변수
    numeric_features = ['age', 'avg_glucose_level', 'bmi']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    for idx, feature in enumerate(numeric_features):
        # Histogram
        ax = axes[idx]
        for stroke_val in [0, 1]:
            data = df[df[TARGET_COLUMN] == stroke_val][feature]
            label = 'Stroke' if stroke_val == 1 else 'No Stroke'
            color = 'red' if stroke_val == 1 else 'blue'
            ax.hist(data, bins=30, alpha=0.6, label=label, color=color, edgecolor='black')

        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of {feature}', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)

        # Box plot
        ax = axes[idx + 3]
        data_to_plot = [df[df[TARGET_COLUMN] == 0][feature].dropna(),
                        df[df[TARGET_COLUMN] == 1][feature].dropna()]
        bp = ax.boxplot(data_to_plot, labels=['No Stroke', 'Stroke'],
                        patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xlabel('Stroke', fontsize=12)
        ax.set_ylabel(feature, fontsize=12)
        ax.set_title(f'{feature} by Stroke Status', fontsize=14)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Feature Distributions by Stroke Status', fontsize=16, y=1.00)
    plt.tight_layout()

    # 저장
    output_file = EDA_DIR / "data" / "figures" / "feature_distributions.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"\n✓ Distribution plots saved to: {output_file}")

    plt.close()

    # 통계 요약
    print(f"\n✓ Statistical summary by stroke status:")
    for feature in numeric_features:
        print(f"\n  {feature}:")
        print(f"    No Stroke - Mean: {df[df[TARGET_COLUMN]==0][feature].mean():.2f}, "
              f"Std: {df[df[TARGET_COLUMN]==0][feature].std():.2f}")
        print(f"    Stroke    - Mean: {df[df[TARGET_COLUMN]==1][feature].mean():.2f}, "
              f"Std: {df[df[TARGET_COLUMN]==1][feature].std():.2f}")

def plot_categorical_analysis(df):
    """범주형 변수 분석"""
    print("\n" + "=" * 60)
    print("Creating Categorical Feature Analysis")
    print("=" * 60)

    categorical_features = ['gender', 'hypertension', 'heart_disease',
                            'ever_married', 'work_type', 'Residence_type',
                            'smoking_status']

    # 실제 존재하는 변수만 필터링
    categorical_features = [f for f in categorical_features if f in df.columns]

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()

    for idx, feature in enumerate(categorical_features):
        ax = axes[idx]

        # Stroke 비율 계산
        stroke_rate = df.groupby(feature)[TARGET_COLUMN].agg(['sum', 'count'])
        stroke_rate['rate'] = (stroke_rate['sum'] / stroke_rate['count']) * 100

        # 막대 그래프
        stroke_rate['rate'].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel('Stroke Rate (%)', fontsize=11)
        ax.set_title(f'Stroke Rate by {feature}', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # 비율 표시
        for i, v in enumerate(stroke_rate['rate']):
            ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    # 빈 subplot 제거
    for idx in range(len(categorical_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Stroke Rate by Categorical Features', fontsize=16, y=1.00)
    plt.tight_layout()

    # 저장
    output_file = EDA_DIR / "data" / "figures" / "categorical_analysis.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"\n✓ Categorical analysis saved to: {output_file}")

    plt.close()

def create_summary_statistics(df):
    """요약 통계 테이블 생성"""
    print("\n" + "=" * 60)
    print("Creating Summary Statistics Tables")
    print("=" * 60)

    output_dir = EDA_DIR / "data" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 전체 기술 통계
    desc = df.describe()
    desc_file = output_dir / "descriptive_statistics.csv"
    desc.to_csv(desc_file)
    print(f"\n✓ Descriptive statistics saved to: {desc_file}")

    # 2. Stroke별 통계
    stroke_stats = df.groupby(TARGET_COLUMN).describe()
    stroke_file = output_dir / "statistics_by_stroke.csv"
    stroke_stats.to_csv(stroke_file)
    print(f"✓ Statistics by stroke status saved to: {stroke_file}")

    # 3. 범주형 변수 빈도
    categorical_features = ['gender', 'hypertension', 'heart_disease',
                            'ever_married', 'work_type', 'Residence_type',
                            'smoking_status']
    categorical_features = [f for f in categorical_features if f in df.columns]

    freq_summary = []
    for feature in categorical_features:
        freq = df[feature].value_counts()
        for category, count in freq.items():
            freq_summary.append({
                'Feature': feature,
                'Category': category,
                'Count': count,
                'Percentage': (count / len(df)) * 100
            })

    freq_df = pd.DataFrame(freq_summary)
    freq_file = output_dir / "categorical_frequencies.csv"
    freq_df.to_csv(freq_file, index=False)
    print(f"✓ Categorical frequencies saved to: {freq_file}")

def main():
    """메인 실행 함수"""
    # Load data
    df = load_data()

    # 1. Correlation heatmap
    plot_correlation_heatmap(df)

    # 2. Feature distributions
    plot_feature_distributions(df)

    # 3. Categorical analysis
    plot_categorical_analysis(df)

    # 4. Summary statistics
    create_summary_statistics(df)

    print("\n" + "=" * 60)
    print("✓ EDA completed successfully!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Figures: {EDA_DIR / 'data' / 'figures'}")
    print(f"  - Tables: {EDA_DIR / 'data' / 'tables'}")

if __name__ == "__main__":
    main()
