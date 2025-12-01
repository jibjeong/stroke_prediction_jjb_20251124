"""
Phase 1: Data Preprocessing
- Load raw data
- Handle missing values
- Encode categorical variables
- Scale numerical features
- Save cleaned data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

from utils.config import *

def load_data():
    """원본 데이터 로딩"""
    print("=" * 60)
    print("Phase 1: Data Preprocessing")
    print("=" * 60)

    df = pd.read_csv(RAW_DATA_FILE)
    print(f"\n✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # 중복 제거
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        df = df.drop_duplicates()
        print(f"\n✓ Removed {n_duplicates} duplicate row(s)")
        print(f"✓ Data after removing duplicates: {df.shape[0]} rows × {df.shape[1]} columns")

    print(f"\nColumns: {list(df.columns)}")
    print(f"\nTarget distribution:\n{df[TARGET_COLUMN].value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    return df

def handle_missing_values(df):
    """결측값 처리"""
    print("\n" + "=" * 60)
    print("Step 1: Handling Missing Values")
    print("=" * 60)

    df = df.copy()  # 명시적 복사본 생성

    # BMI 결측값을 median으로 채우기
    if df['bmi'].isnull().sum() > 0:
        median_bmi = df['bmi'].median()
        n_missing = df['bmi'].isnull().sum()
        df.loc[:, 'bmi'] = df['bmi'].fillna(median_bmi)
        print(f"\n✓ BMI missing values filled with median ({median_bmi:.2f}): {n_missing} values")

    # 'N/A' 문자열을 실제 NaN으로 변환 (혹시 있을 경우)
    df = df.replace('N/A', np.nan)

    print(f"\n✓ Total missing values after imputation: {df.isnull().sum().sum()}")

    return df

def encode_categorical(df):
    """범주형 변수 인코딩"""
    print("\n" + "=" * 60)
    print("Step 2: Encoding Categorical Variables")
    print("=" * 60)

    df_encoded = df.copy()

    # Binary categorical variables (Label Encoding)
    binary_vars = ['gender', 'ever_married', 'Residence_type']
    for var in binary_vars:
        if var in df_encoded.columns:
            original_values = df[var].unique()
            le = LabelEncoder()
            df_encoded[var] = le.fit_transform(df_encoded[var])
            encoded_values = df_encoded[var].unique()
            print(f"\n✓ {var} encoded: {sorted(original_values)} → {sorted(encoded_values)}")

    # Multi-class categorical variables (One-Hot Encoding)
    multi_class_vars = ['work_type', 'smoking_status']
    for var in multi_class_vars:
        if var in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[var], prefix=var, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded.drop(var, axis=1, inplace=True)
            print(f"\n✓ {var} one-hot encoded: {len(dummies.columns)} new columns created")

    print(f"\n✓ Final shape after encoding: {df_encoded.shape}")

    return df_encoded

def scale_features(df):
    """수치형 변수 스케일링"""
    print("\n" + "=" * 60)
    print("Step 3: Scaling Numerical Features")
    print("=" * 60)

    df_scaled = df.copy()

    # Target과 ID 제외한 수치형 변수
    numeric_cols = [col for col in NUMERIC_FEATURES if col in df_scaled.columns]

    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    for col in numeric_cols:
        print(f"\n✓ {col} scaled - mean: {df_scaled[col].mean():.4f}, std: {df_scaled[col].std():.4f}")

    return df_scaled, scaler

def create_derived_features(df_original, df_scaled):
    """파생 변수 생성 (포스터 분석용)"""
    print("\n" + "=" * 60)
    print("Step 4: Creating Derived Features")
    print("=" * 60)

    df = df_scaled.copy()

    # 원본 데이터에서 파생 변수 생성 (스케일링 전 값 사용)
    # 1. Age group
    def get_age_group(age):
        if age < 18:
            return 0  # child
        elif age < 40:
            return 1  # young_adult
        elif age < 60:
            return 2  # middle_aged
        else:
            return 3  # senior

    df['age_group'] = df_original['age'].apply(get_age_group)

    # 2. BMI category
    def get_bmi_category(bmi):
        if bmi < 18.5:
            return 0  # underweight
        elif bmi < 25:
            return 1  # normal
        elif bmi < 30:
            return 2  # overweight
        else:
            return 3  # obese

    df['bmi_category'] = df_original['bmi'].apply(get_bmi_category)

    # 3. Glucose category
    def get_glucose_category(glucose):
        if glucose < 100:
            return 0  # normal
        elif glucose < 126:
            return 1  # prediabetes
        else:
            return 2  # diabetes

    df['glucose_category'] = df_original['avg_glucose_level'].apply(get_glucose_category)

    # 4. Risk score (합성 변수)
    df['risk_score'] = (df_original['hypertension'] +
                        df_original['heart_disease'] +
                        (df_original['age'] > 60).astype(int))

    print(f"\n✓ Derived features created:")
    print(f"  - age_group: {df['age_group'].unique()}")
    print(f"  - bmi_category: {df['bmi_category'].unique()}")
    print(f"  - glucose_category: {df['glucose_category'].unique()}")
    print(f"  - risk_score: {df['risk_score'].unique()}")

    return df

def save_preprocessed_data(df, df_original):
    """전처리된 데이터 저장"""
    print("\n" + "=" * 60)
    print("Step 5: Saving Preprocessed Data")
    print("=" * 60)

    # 최종 데이터 디렉토리 생성
    output_dir = PREPROCESS_DIR / "data" / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV 저장
    output_file = output_dir / "stroke_preprocessed.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ Preprocessed data saved to: {output_file}")
    print(f"  - Shape: {df.shape}")
    print(f"  - Features: {df.shape[1] - 2} (excluding ID and target)")

    # 원본 데이터도 함께 저장 (EDA용)
    original_output_file = output_dir / "stroke_original.csv"
    df_original.to_csv(original_output_file, index=False)
    print(f"\n✓ Original data saved to: {original_output_file}")

    # Feature 목록 저장
    feature_cols = [col for col in df.columns if col not in [ID_COLUMN, TARGET_COLUMN]]
    feature_file = output_dir / "feature_names.txt"
    with open(feature_file, 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    print(f"\n✓ Feature names saved to: {feature_file}")
    print(f"  - Total features: {len(feature_cols)}")

    return df

def main():
    """메인 실행 함수"""
    # Step 0: Load data
    df_original = load_data()

    # Step 1: Handle missing values
    df_clean = handle_missing_values(df_original.copy())

    # Step 2: Encode categorical variables
    df_encoded = encode_categorical(df_clean)

    # Step 3: Scale numerical features
    df_scaled, scaler = scale_features(df_encoded)

    # Step 4: Create derived features
    df_final = create_derived_features(df_original, df_scaled)

    # Step 5: Save preprocessed data
    df_preprocessed = save_preprocessed_data(df_final, df_original)

    print("\n" + "=" * 60)
    print("✓ Preprocessing completed successfully!")
    print("=" * 60)
    print(f"\nFinal dataset summary:")
    print(f"  - Total samples: {df_preprocessed.shape[0]}")
    print(f"  - Total features: {df_preprocessed.shape[1] - 2}")
    print(f"  - Stroke cases: {df_preprocessed[TARGET_COLUMN].sum()} ({df_preprocessed[TARGET_COLUMN].mean()*100:.1f}%)")
    print(f"  - Non-stroke cases: {(df_preprocessed[TARGET_COLUMN]==0).sum()} ({(1-df_preprocessed[TARGET_COLUMN].mean())*100:.1f}%)")

if __name__ == "__main__":
    main()
