"""
Phase 3: Model Training and Evaluation
포스터 기반 모델 (7개):
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- Gradient Boosting
- Neural Network (MLP)

평가 테이블:
- Table 1: Average performance (AUC, CA, F1, Precision, Recall)
- Table 2: Performance by class (Stroke=No, Stroke=Yes)

ROC Curves:
- Figure 3: ROC curve for each models (Stroke=No)
- Figure 4: ROC curve for each models (Stroke=Yes)

클래스 불균형 처리: SMOTE
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from utils.config import *

# 한글 폰트 설정 (폰트가 없으면 기본 폰트 사용)
if KOREAN_FONT:
    plt.rcParams['font.family'] = KOREAN_FONT
plt.rcParams['axes.unicode_minus'] = False

def load_preprocessed_data():
    """전처리된 데이터 로딩"""
    print("=" * 60)
    print("Phase 3: Model Training and Evaluation")
    print("=" * 60)

    data_file = PREPROCESS_DIR / "data" / "final" / "stroke_preprocessed.csv"
    df = pd.read_csv(data_file)

    print(f"\n✓ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")

    # Feature와 Target 분리
    X = df.drop([ID_COLUMN, TARGET_COLUMN], axis=1)
    y = df[TARGET_COLUMN]

    print(f"\n✓ Features: {X.shape[1]}")
    print(f"✓ Target distribution:")
    print(f"  - Class 0 (No Stroke): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"  - Class 1 (Stroke): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

    return X, y

def apply_smote(X_train, y_train):
    """SMOTE를 사용한 클래스 불균형 처리"""
    print("\n" + "=" * 60)
    print("Applying SMOTE for Class Imbalance")
    print("=" * 60)

    print(f"\n✓ Before SMOTE:")
    print(f"  - Class 0: {(y_train==0).sum()}")
    print(f"  - Class 1: {(y_train==1).sum()}")
    print(f"  - Ratio: 1:{(y_train==0).sum()/(y_train==1).sum():.1f}")

    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"\n✓ After SMOTE:")
    print(f"  - Class 0: {(y_train_resampled==0).sum()}")
    print(f"  - Class 1: {(y_train_resampled==1).sum()}")
    print(f"  - Ratio: 1:1")

    return X_train_resampled, y_train_resampled

def train_models(X_train, y_train, X_test, y_test):
    """여러 모델 학습 및 평가"""
    print("\n" + "=" * 60)
    print("Training Multiple Models (7 models)")
    print("=" * 60)

    # 모델 정의
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_SEED,
            class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=RANDOM_SEED,
            class_weight='balanced'
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=RANDOM_SEED,
            class_weight='balanced'
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
            scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
            eval_metric='logloss'
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=RANDOM_SEED
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=RANDOM_SEED,
            early_stopping=True
        )
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training {name}")
        print(f"{'=' * 60}")

        # 학습
        model.fit(X_train, y_train)
        trained_models[name] = model

        # 예측
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)  # [:, 0] for class 0, [:, 1] for class 1

        # Average metrics (macro average)
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

        # Per-class AUC
        roc_auc_class0 = roc_auc_score((y_test == 0).astype(int), y_pred_proba[:, 0])
        roc_auc_class1 = roc_auc_score((y_test == 1).astype(int), y_pred_proba[:, 1])

        results[name] = {
            # Average metrics (Table 1)
            'accuracy': accuracy,
            'precision_avg': precision_macro,
            'recall_avg': recall_macro,
            'f1_avg': f1_macro,
            'roc_auc': roc_auc,

            # Per-class metrics (Table 2)
            'precision_no_stroke': precision_per_class[0],  # Class 0
            'recall_no_stroke': recall_per_class[0],
            'f1_no_stroke': f1_per_class[0],
            'precision_stroke': precision_per_class[1],  # Class 1
            'recall_stroke': recall_per_class[1],
            'f1_stroke': f1_per_class[1],

            # Per-class AUC
            'roc_auc_no_stroke': roc_auc_class0,
            'roc_auc_stroke': roc_auc_class1,

            # For plotting
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        print(f"\n✓ {name} Results (Average):")
        print(f"  - Accuracy (CA):  {accuracy:.4f}")
        print(f"  - Precision:      {precision_macro:.4f}")
        print(f"  - Recall:         {recall_macro:.4f}")
        print(f"  - F1-score:       {f1_macro:.4f}")
        print(f"  - ROC AUC:        {roc_auc:.4f}")

        print(f"\n✓ {name} Results (Per-Class):")
        print(f"  - Stroke=No  -> Precision: {precision_per_class[0]:.4f}, Recall: {recall_per_class[0]:.4f}, F1: {f1_per_class[0]:.4f}, AUC: {roc_auc_class0:.4f}")
        print(f"  - Stroke=Yes -> Precision: {precision_per_class[1]:.4f}, Recall: {recall_per_class[1]:.4f}, F1: {f1_per_class[1]:.4f}, AUC: {roc_auc_class1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n✓ Confusion Matrix:")
        print(f"  [[TN={cm[0,0]}  FP={cm[0,1]}]")
        print(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")

    return results, trained_models

def create_table1_average_performance(results):
    """Table 1: Evaluation of model performance (average)
    컬럼: AUC, CA, F1, Precision, Recall
    """
    print("\n" + "=" * 60)
    print("Creating Table 1: Average Performance")
    print("=" * 60)

    table1_data = []
    for name, result in results.items():
        table1_data.append({
            'Model': name,
            'AUC': f"{result['roc_auc']:.4f}",
            'CA': f"{result['accuracy']:.4f}",  # CA = Classification Accuracy
            'F1': f"{result['f1_avg']:.4f}",
            'Precision': f"{result['precision_avg']:.4f}",
            'Recall': f"{result['recall_avg']:.4f}"
        })

    df_table1 = pd.DataFrame(table1_data)

    # 저장
    output_dir = MODEL_DIR / "data" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "table1_average_performance.csv"
    df_table1.to_csv(output_file, index=False)
    print(f"\n✓ Table 1 saved to: {output_file}")

    # 콘솔 출력
    print(f"\n{'=' * 100}")
    print("Table 1: Evaluation of Model Performance (Average)")
    print(f"{'=' * 100}")
    print(df_table1.to_string(index=False))

    return df_table1

def create_table2_class_performance(results):
    """Table 2: Evaluation of model performance (Stroke=No and Stroke=Yes)
    각 클래스별 Precision, Recall, F1
    """
    print("\n" + "=" * 60)
    print("Creating Table 2: Performance by Class")
    print("=" * 60)

    table2_data = []
    for name, result in results.items():
        # Stroke=No (Class 0)
        table2_data.append({
            'Model': name,
            'Class': 'Stroke=No',
            'Precision': f"{result['precision_no_stroke']:.4f}",
            'Recall': f"{result['recall_no_stroke']:.4f}",
            'F1': f"{result['f1_no_stroke']:.4f}"
        })

        # Stroke=Yes (Class 1)
        table2_data.append({
            'Model': name,
            'Class': 'Stroke=Yes',
            'Precision': f"{result['precision_stroke']:.4f}",
            'Recall': f"{result['recall_stroke']:.4f}",
            'F1': f"{result['f1_stroke']:.4f}"
        })

    df_table2 = pd.DataFrame(table2_data)

    # 저장
    output_file = MODEL_DIR / "data" / "tables" / "table2_class_performance.csv"
    df_table2.to_csv(output_file, index=False)
    print(f"\n✓ Table 2 saved to: {output_file}")

    # 콘솔 출력 (모델별로 그룹화하여 보기 좋게)
    print(f"\n{'=' * 100}")
    print("Table 2: Evaluation of Model Performance (Stroke=No and Stroke=Yes)")
    print(f"{'=' * 100}")

    for name in results.keys():
        model_data = df_table2[df_table2['Model'] == name]
        print(f"\n{name}:")
        print(model_data.to_string(index=False, header=True))

    return df_table2

def plot_roc_curve_stroke_no(results, y_test):
    """Figure 3: ROC curve for each models (Stroke=No)"""
    print("\n" + "=" * 60)
    print("Creating Figure 3: ROC Curves (Stroke=No)")
    print("=" * 60)

    plt.figure(figsize=(12, 9))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

    # Stroke=No를 positive class로 간주
    y_test_class0 = (y_test == 0).astype(int)

    for idx, (name, result) in enumerate(results.items()):
        # Class 0의 확률
        y_pred_proba_class0 = result['y_pred_proba'][:, 0]

        fpr, tpr, _ = roc_curve(y_test_class0, y_pred_proba_class0)
        roc_auc = result['roc_auc_no_stroke']

        plt.plot(fpr, tpr, color=colors[idx], lw=2,
                 label=f'{name} (AUC = {roc_auc:.3f})')

    # 대각선 (Random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('Figure 3: ROC Curves for Stroke=No (Class 0)', fontsize=15, pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 저장
    output_dir = MODEL_DIR / "data" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "figure3_roc_curves_stroke_no.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"\n✓ Figure 3 saved to: {output_file}")

    plt.close()

def plot_roc_curve_stroke_yes(results, y_test):
    """Figure 4: ROC curve for each models (Stroke=Yes)"""
    print("\n" + "=" * 60)
    print("Creating Figure 4: ROC Curves (Stroke=Yes)")
    print("=" * 60)

    plt.figure(figsize=(12, 9))

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

    # Stroke=Yes를 positive class로 간주
    y_test_class1 = (y_test == 1).astype(int)

    for idx, (name, result) in enumerate(results.items()):
        # Class 1의 확률
        y_pred_proba_class1 = result['y_pred_proba'][:, 1]

        fpr, tpr, _ = roc_curve(y_test_class1, y_pred_proba_class1)
        roc_auc = result['roc_auc_stroke']

        plt.plot(fpr, tpr, color=colors[idx], lw=2,
                 label=f'{name} (AUC = {roc_auc:.3f})')

    # 대각선 (Random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('Figure 4: ROC Curves for Stroke=Yes (Class 1)', fontsize=15, pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 저장
    output_file = MODEL_DIR / "data" / "figures" / "figure4_roc_curves_stroke_yes.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"\n✓ Figure 4 saved to: {output_file}")

    plt.close()

def plot_confusion_matrices(results, y_test):
    """Confusion Matrix 시각화"""
    print("\n" + "=" * 60)
    print("Creating Confusion Matrices")
    print("=" * 60)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.ravel()

    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(y_test, result['y_pred'])

        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    cbar_kws={'label': 'Count'},
                    annot_kws={'size': 12})
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}', fontsize=12)
        ax.set_xticklabels(['No Stroke', 'Stroke'], fontsize=10)
        ax.set_yticklabels(['No Stroke', 'Stroke'], fontsize=10)

    # 빈 subplot 제거
    for idx in range(len(results), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('Confusion Matrices - Model Comparison (7 Models)', fontsize=16, y=0.995)
    plt.tight_layout()

    # 저장
    output_file = MODEL_DIR / "data" / "figures" / "confusion_matrices.png"
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"\n✓ Confusion matrices saved to: {output_file}")

    plt.close()

def save_models(trained_models):
    """학습된 모델 저장"""
    print("\n" + "=" * 60)
    print("Saving Trained Models")
    print("=" * 60)

    output_dir = MODEL_DIR / "data" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in trained_models.items():
        model_name = name.replace(' ', '_').lower()
        output_file = output_dir / f"{model_name}.pkl"
        joblib.dump(model, output_file)
        print(f"✓ {name} saved to: {output_file}")

def main():
    """메인 실행 함수"""
    # 1. Load data
    X, y = load_preprocessed_data()

    # 2. Train-test split
    print("\n" + "=" * 60)
    print("Splitting Data")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\n✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")

    # 3. Apply SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    # 4. Train models
    results, trained_models = train_models(
        X_train_resampled, y_train_resampled, X_test, y_test
    )

    # 5. Create Table 1: Average performance
    df_table1 = create_table1_average_performance(results)

    # 6. Create Table 2: Performance by class
    df_table2 = create_table2_class_performance(results)

    # 7. Plot Figure 3: ROC curves (Stroke=No)
    plot_roc_curve_stroke_no(results, y_test)

    # 8. Plot Figure 4: ROC curves (Stroke=Yes)
    plot_roc_curve_stroke_yes(results, y_test)

    # 9. Plot confusion matrices
    plot_confusion_matrices(results, y_test)

    # 10. Save models
    save_models(trained_models)

    print("\n" + "=" * 60)
    print("✓ Model training and evaluation completed successfully!")
    print("=" * 60)

    # 최고 성능 모델 출력
    print(f"\n🏆 Best Models (by Average Performance):")
    print(f"  - Best by AUC: {df_table1.loc[df_table1['AUC'].astype(float).idxmax(), 'Model']} "
          f"({df_table1['AUC'].astype(float).max():.4f})")
    print(f"  - Best by F1: {df_table1.loc[df_table1['F1'].astype(float).idxmax(), 'Model']} "
          f"({df_table1['F1'].astype(float).max():.4f})")
    print(f"  - Best by CA: {df_table1.loc[df_table1['CA'].astype(float).idxmax(), 'Model']} "
          f"({df_table1['CA'].astype(float).max():.4f})")

    print(f"\n📊 Output Summary:")
    print(f"  - Tables: {MODEL_DIR / 'data' / 'tables'}")
    print(f"    • table1_average_performance.csv")
    print(f"    • table2_class_performance.csv")
    print(f"  - Figures: {MODEL_DIR / 'data' / 'figures'}")
    print(f"    • figure3_roc_curves_stroke_no.png")
    print(f"    • figure4_roc_curves_stroke_yes.png")
    print(f"    • confusion_matrices.png")
    print(f"  - Models: {MODEL_DIR / 'data' / 'models'}")

if __name__ == "__main__":
    main()
