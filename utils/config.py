"""
Configuration file for Stroke Prediction Project
"""

import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터 경로
DATA_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_FILE = DATA_DIR / "stroke_dataset.csv"

# 분석 경로
PREPROCESS_DIR = PROJECT_ROOT / "analysis_01_preprocess"
EDA_DIR = PROJECT_ROOT / "analysis_02_eda"
MODEL_DIR = PROJECT_ROOT / "analysis_03_model"

# 랜덤 시드 (재현성)
RANDOM_SEED = 42

# 데이터 설정
TARGET_COLUMN = 'stroke'
ID_COLUMN = 'id'

# 변수 타입
NUMERIC_FEATURES = ['age', 'avg_glucose_level', 'bmi']
CATEGORICAL_FEATURES = ['gender', 'hypertension', 'heart_disease',
                        'ever_married', 'work_type', 'Residence_type',
                        'smoking_status']

# BMI 카테고리 기준 (WHO)
BMI_CATEGORIES = {
    'underweight': (0, 18.5),
    'normal': (18.5, 25),
    'overweight': (25, 30),
    'obese': (30, float('inf'))
}

# 혈당 카테고리 기준 (ADA)
GLUCOSE_CATEGORIES = {
    'normal': (0, 100),
    'prediabetes': (100, 126),
    'diabetes': (126, float('inf'))
}

# 연령 그룹
AGE_GROUPS = {
    'child': (0, 18),
    'young_adult': (18, 40),
    'middle_aged': (40, 60),
    'senior': (60, float('inf'))
}

# 모델 설정
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
N_SPLITS = 5  # Cross-validation folds

# 평가 메트릭
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# 시각화 설정
FIGURE_DPI = 300
FIGURE_SIZE = (10, 6)

# 한글 폰트 설정 (폰트가 없을 경우 기본 폰트 사용)
import platform
import matplotlib.font_manager as fm

def get_korean_font():
    """시스템에 설치된 한글 폰트 찾기"""
    system = platform.system()

    # 우선순위 폰트 리스트
    if system == 'Darwin':  # macOS
        font_candidates = ['AppleGothic', 'AppleMyungjo']
    elif system == 'Windows':  # Windows
        font_candidates = ['Malgun Gothic', 'Gulim', 'Dotum']
    else:  # Linux
        font_candidates = ['NanumGothic', 'NanumBarunGothic', 'UnDotum']

    # 설치된 폰트 목록 가져오기
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 사용 가능한 첫 번째 한글 폰트 반환
    for font in font_candidates:
        if font in available_fonts:
            return font

    # 한글 폰트가 없으면 None 반환 (기본 폰트 사용)
    return None

KOREAN_FONT = get_korean_font()
