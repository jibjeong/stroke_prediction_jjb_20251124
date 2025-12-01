# Stroke Prediction Project - Quick Start Guide

## Prerequisites

- Python 3.9-3.13 (Python 3.14 not supported due to numba compatibility)
- Windows/macOS/Linux

## Installation

### 1. Install uv (Package Manager)

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and Setup Project

```bash
cd stroke_prediction_jjb_20251124

# Create virtual environment with Python 3.12
uv venv --python 3.12

# Activate virtual environment
# Windows (Git Bash)
source .venv/Scripts/activate

# Windows (CMD)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# Install dependencies
uv sync
```

## Quick Run

### Option 1: Full Pipeline (Recommended)

```bash
# Windows
python -X utf8 run_full_pipeline.py

# macOS/Linux
python run_full_pipeline.py
```

This will execute all three phases:
1. Data Preprocessing
2. Exploratory Data Analysis (EDA)
3. Model Training (7 models)

**Execution Time:** ~3-5 minutes on modern hardware

### Option 2: Run Individual Phases

```bash
# Phase 1: Data Preprocessing
python -X utf8 analysis_01_preprocess/scripts/01_data_preprocessing.py

# Phase 2: EDA
python -X utf8 analysis_02_eda/scripts/01_eda_analysis.py

# Phase 3: Model Training
python -X utf8 analysis_03_model/scripts/baseline/01_train_models.py

# Optional: SHAP Analysis (requires trained models)
python -X utf8 analysis_03_model/scripts/evaluation/02_shap_analysis.py

# Optional: Additional Evaluation (requires trained models)
python -X utf8 analysis_03_model/scripts/evaluation/03_additional_evaluation.py
```

## Output Files

After running the pipeline, you'll find:

### 1. Preprocessed Data
```
analysis_01_preprocess/data/final/
├── stroke_preprocessed.csv    # Processed data (5,109 rows × 21 columns)
├── stroke_original.csv         # Original data (duplicates removed)
└── feature_names.txt           # List of 19 features
```

### 2. EDA Results
```
analysis_02_eda/data/
├── figures/
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   └── categorical_analysis.png
└── tables/
    ├── descriptive_statistics.csv
    ├── statistics_by_stroke.csv
    └── categorical_frequencies.csv
```

### 3. Model Results
```
analysis_03_model/data/
├── tables/
│   ├── table1_average_performance.csv
│   └── table2_class_performance.csv
├── figures/
│   ├── figure3_roc_curves_stroke_no.png
│   ├── figure4_roc_curves_stroke_yes.png
│   └── confusion_matrices.png
└── models/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── random_forest.pkl
    ├── svm.pkl
    ├── xgboost.pkl
    ├── gradient_boosting.pkl
    └── neural_network.pkl
```

## Model Performance Summary

| Model | AUROC | Accuracy | F1 | Best For |
|-------|-------|----------|-----|----------|
| **Logistic Regression** | **0.8245** | 0.7877 | 0.5602 | Primary care screening |
| Decision Tree | 0.7413 | 0.8131 | 0.5692 | Interpretability |
| Random Forest | 0.8189 | 0.8356 | 0.5946 | Balanced performance |
| SVM | 0.8059 | 0.7945 | 0.5519 | - |
| XGBoost | 0.7949 | 0.8542 | 0.5777 | - |
| **Gradient Boosting** | 0.7799 | **0.8796** | 0.5655 | High accuracy |
| Neural Network | 0.8026 | 0.8777 | **0.5968** | Complex patterns |

## Troubleshooting

### 1. UTF-8 Encoding Error (Windows)

**Problem:** `UnicodeEncodeError: 'cp949' codec can't encode character`

**Solution:** Always use `-X utf8` flag:
```bash
python -X utf8 script.py
```

### 2. Font Warning

**Problem:** Korean font not found

**Solution:** This is normal and won't affect results. The code automatically falls back to default fonts.

### 3. Python Version Error

**Problem:** `Cannot install on Python version 3.14.0`

**Solution:** Use Python 3.12:
```bash
uv venv --python 3.12
```

### 4. Module Not Found

**Problem:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:** Reinstall dependencies:
```bash
uv sync
```

## Next Steps

1. **View Results**: Check the output directories for figures and tables
2. **Modify Models**: Edit `analysis_03_model/scripts/baseline/01_train_models.py` to tune hyperparameters
3. **Add Features**: Modify `analysis_01_preprocess/scripts/01_data_preprocessing.py` to create new features
4. **Deploy Model**: Load any trained model from `analysis_03_model/data/models/` for predictions

## Additional Resources

- **Technical Documentation:** See `CLAUDE.md` for detailed architecture
- **Project Overview:** See `README.md` for research background
- **Academic Paper:** See `paper/main_paper.md` for full methodology

## Support

For issues or questions:
- Check `CLAUDE.md` for detailed troubleshooting
- Review error messages and logs
- Verify Python version compatibility

## License

Educational and research use only. Not for clinical deployment.

---

**Version:** 1.0
**Last Updated:** 2024-11-24
