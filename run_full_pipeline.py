"""
Stroke Prediction Project - Full Pipeline Execution Script

이 스크립트는 전체 분석 파이프라인을 순차적으로 실행합니다:
1. Phase 1: Data Preprocessing
2. Phase 2: Exploratory Data Analysis (EDA)
3. Phase 3: Model Training and Evaluation

실행 방법:
    python run_full_pipeline.py
"""

import subprocess
import sys
import os
from pathlib import Path

# Windows UTF-8 인코딩 설정
if sys.platform == 'win32':
    # UTF-8 모드 환경 변수 설정
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    # stdout/stderr을 UTF-8로 재설정
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

def run_script(script_path, description):
    """스크립트 실행 함수"""
    print("\n" + "=" * 80)
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print("=" * 80 + "\n")

    try:
        # Windows에서 UTF-8 인코딩 사용
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        result = subprocess.run(
            [sys.executable, '-X', 'utf8', str(script_path)],
            check=True,
            capture_output=False,
            text=True,
            env=env,
            encoding='utf-8'
        )
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        print(f"Error: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("STROKE PREDICTION PROJECT - FULL PIPELINE")
    print("=" * 80)

    project_root = Path(__file__).parent

    # Phase 1: Data Preprocessing
    phase1_script = project_root / "analysis_01_preprocess" / "scripts" / "01_data_preprocessing.py"
    if not run_script(phase1_script, "Phase 1: Data Preprocessing"):
        print("\n❌ Pipeline stopped due to error in Phase 1")
        return False

    # Phase 2: Exploratory Data Analysis
    phase2_script = project_root / "analysis_02_eda" / "scripts" / "01_eda_analysis.py"
    if not run_script(phase2_script, "Phase 2: Exploratory Data Analysis"):
        print("\n❌ Pipeline stopped due to error in Phase 2")
        return False

    # Phase 3: Model Training and Evaluation
    phase3_script = project_root / "analysis_03_model" / "scripts" / "baseline" / "01_train_models.py"
    if not run_script(phase3_script, "Phase 3: Model Training and Evaluation"):
        print("\n❌ Pipeline stopped due to error in Phase 3")
        return False

    # 완료 메시지
    print("\n" + "=" * 80)
    print("✓ FULL PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    print("\n📊 Output Summary:")
    print(f"\n1. Preprocessed Data:")
    print(f"   - {project_root / 'analysis_01_preprocess' / 'data' / 'final' / 'stroke_preprocessed.csv'}")
    print(f"   - {project_root / 'analysis_01_preprocess' / 'data' / 'final' / 'stroke_original.csv'}")

    print(f"\n2. EDA Results:")
    print(f"   - Figures: {project_root / 'analysis_02_eda' / 'data' / 'figures'}")
    print(f"   - Tables: {project_root / 'analysis_02_eda' / 'data' / 'tables'}")

    print(f"\n3. Model Results:")
    print(f"   - Tables:")
    print(f"     • {project_root / 'analysis_03_model' / 'data' / 'tables' / 'table1_average_performance.csv'}")
    print(f"     • {project_root / 'analysis_03_model' / 'data' / 'tables' / 'table2_class_performance.csv'}")
    print(f"   - Figures:")
    print(f"     • {project_root / 'analysis_03_model' / 'data' / 'figures' / 'figure3_roc_curves_stroke_no.png'}")
    print(f"     • {project_root / 'analysis_03_model' / 'data' / 'figures' / 'figure4_roc_curves_stroke_yes.png'}")
    print(f"     • {project_root / 'analysis_03_model' / 'data' / 'figures' / 'confusion_matrices.png'}")
    print(f"   - Models: {project_root / 'analysis_03_model' / 'data' / 'models'}")

    print("\n" + "=" * 80)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
