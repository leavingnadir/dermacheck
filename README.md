# DermaCheck — Skin Lesion Diagnosis Triage (Classical ML)

**Module:** IT2011 – Artificial Intelligence and Machine Learning
**Group ID:** `2026-Y2S1-MLB-B3G2-03`
**Dataset:** HAM10000 metadata (CSV only — no lesion images used)

---

## 1. Overview

DermaCheck is a group project that builds a classical machine learning pipeline
to explore whether basic patient metadata — **age**, **sex**, and **lesion
location on the body** — carries any predictive signal for the diagnostic
class (`dx`) of a skin lesion, using the [HAM10000 dataset](https://doi.org/10.7910/DVN/DBW86T).

This is **not** an image-based classifier. No CNNs and no lesion images are
used — only the structured metadata CSV. The goal is to demonstrate a sound,
well-documented ML process (cleaning, EDA, feature engineering, model
comparison, honest evaluation) rather than to build a clinically accurate
diagnostic tool. Given the weak signal in metadata alone, **modest
performance is expected and is treated as an honest finding, not a failure.**

## 2. Problem Statement

Teledermatology triage systems often only have basic intake information
(age, sex, rough body location) before a clinician reviews an image. This
project asks: *how far can that limited metadata alone go in predicting a
lesion's diagnostic class, and where does it clearly fall short?* The answer
informs how much weight such metadata should carry in a real triage pipeline.

## 3. Dataset

| Detail | Description |
|---|---|
| Source | [HAM10000 "Skin Cancer MNIST" (Kaggle)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| File used | `HAM10000_metadata.csv` only |
| Rows | ~10,015 lesion records |
| Target | `dx` — 7 diagnostic classes (`nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`) |
| Features used | `age`, `sex`, `localization` |
| Not used | `image_id`, lesion images (`HAM10000_images_part_1/2`) |
| Citation | Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset*. Harvard Dataverse. doi:10.7910/DVN/DBW86T |

> The dataset is heavily imbalanced — roughly two-thirds of records are the
> benign `nv` class.

## 4. Repository Structure

```
Group_ID/
├── README.md
├── data/
│   ├── raw/
│   └── external/
├── notebooks/
│   ├── IT25101908_MissingValues.ipynb
│   ├── IT25102853_Encoding.ipynb
│   ├── IT25103722_OutlierRemoval.ipynb
│   ├── IT25101857_Scaling.ipynb
│   ├── IT25103708_FeatureEngineering.ipynb
│   └── IT25100928_ModelBuilding.ipynb
├── group_pipeline.ipynb
└── results/
    ├── eda_visualizations/
    ├── logs/
    └── outputs/
```

## 5. Setup & Installation

**Requirements:** Anaconda (or Miniconda) and Python 3.10.

```bash
# 1. Create and activate an isolated environment
conda create -n dermacheck python=3.10
conda activate dermacheck

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter

# 3. Launch Jupyter
jupyter notebook
```

## 6. Getting the Data

1. Download `HAM10000_metadata.csv` from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
2. Place it at `data/raw/HAM10000_metadata.csv`.

## 7. How to Run — Step-by-Step Guide

For anyone opening this repository for the first time, follow this order:

1. **Read this README fully** — it explains the goal, the data, and the folder layout before you touch any code.
2. **Set up the environment** 
3. **Open the individual preprocessing notebooks** in `notebooks/`, in this order, to see each technique in isolation:
   - `IT25101908_MissingValues.ipynb` — handling missing `age` values
   - `IT25102853_Encoding.ipynb` — encoding `sex` and `localization`
   - `IT25103722_OutlierRemoval.ipynb` — detecting/removing unrealistic `age` values
   - `IT25101857_Scaling.ipynb` — scaling `age` for distance-based models
   - `IT25103708_FeatureEngineering.ipynb` — assembling the final feature matrix
   - `IT25100928_ModelBuilding.ipynb` — baseline + stronger models, tuning, evaluation
   Each notebook is self-contained and shows one contributor's technique with a written interpretation.
4. **Open `group_pipeline.ipynb`** — this is the single integrated notebook that runs the *entire* pipeline end-to-end (cleaning → EDA → feature engineering → class-imbalance handling → model training → evaluation). Run this top-to-bottom to reproduce the final results.
5. **Check `results/eda_visualizations/`** for the saved charts referenced in the report, and `results/outputs/` for the final processed dataset / feature set.

## 8. How the Pipeline Is Assembled
 
The six notebooks in `notebooks/` are **not automatically linked** — Jupyter
notebooks can't import one another the way Python modules can. Each one is
self-contained on purpose, so each contributor's technique can be marked in
isolation. They connect in two ways:
 
1. **Every individual notebook starts with its own copy of the Step 1 data-loading
   code** (`pd.read_csv("data/raw/HAM10000_metadata.csv")` plus the initial
   `.shape` / `.info()` / `.head()` / `value_counts()` checks). Without this,
   a notebook has no `df` to work on, so it must load and inspect the raw
   CSV itself before applying its one technique.
2. **`group_pipeline.ipynb` is the single notebook where everything is
   chained together.** Once every member's notebook works standalone, the
   integrator (Member 6) copies each notebook's finished code cells — in
   the order below — into `group_pipeline.ipynb`, so the output dataframe of
   one step becomes the input to the next, forming one continuous run from
   raw CSV to final trained models.
**Assembly order in `group_pipeline.ipynb`:**
 
```
1. Load & inspect data        ← Step 1 (everyone's notebook starts here too)
2. Missing values             ← IT25101908_MissingValues.ipynb
3. Categorical encoding       ← IT25102853_Encoding.ipynb
4. Outlier removal            ← IT25103722_OutlierRemoval.ipynb
5. Scaling                    ← IT25101857_Scaling.ipynb
6. Feature engineering        ← IT25103708_FeatureEngineering.ipynb
7. Model building & evaluation← IT25100928_ModelBuilding.ipynb
```
 
**Two rules that make the chaining actually work:**
 
- Each step must run cleanly on the dataframe *as produced by the step
  before it* — so when copying your cell into the pipeline, remove any
  re-loading of the raw CSV (only the very first cell in the pipeline loads
  it) and make sure your code reads/modifies the same `df` the previous
  step left behind, not a fresh copy.
- Only the **final, working version** of each person's cell goes into the
  pipeline — exploratory/debugging cells stay in the individual notebook.
`group_pipeline.ipynb` is the notebook that should actually be *run*
top-to-bottom to reproduce the project's final results, charts, and
processed outputs. The individual notebooks are the record of who did what
and why, for the Progress Review I individual marks.
 
## 9. Preprocessing Techniques (by contributor)
 
| Notebook | Technique | Summary |
|---|---|---|
| `IT25101908_MissingValues.ipynb` | Missing value imputation | Median imputation for missing `age` values |
| `IT25102853_Encoding.ipynb` | Categorical encoding | One-hot encoding of `sex` and `localization` |
| `IT25103722_OutlierRemoval.ipynb` | Outlier handling | Detects/caps unrealistic `age` values via boxplot inspection |
| `IT25101857_Scaling.ipynb` | Feature scaling | Standardizes `age` with `StandardScaler` |
| `IT25103708_FeatureEngineering.ipynb` | Feature assembly | Builds the final `X` feature matrix and encodes the `dx` target |
| `IT25100928_ModelBuilding.ipynb` | Modeling | Baseline (Logistic Regression, Random Forest) + stronger models (SVM, Gradient Boosting), tuning, and evaluation |


## 10. Preprocessing Techniques (by contributor)

| Notebook | Technique | Summary |
|---|---|---|
| `IT25101908_MissingValues.ipynb` | Missing value imputation | Median imputation for missing `age` values |
| `IT25102853_Encoding.ipynb` | Categorical encoding | One-hot encoding of `sex` and `localization` |
| `IT25103722_OutlierRemoval.ipynb` | Outlier handling | Detects/caps unrealistic `age` values via boxplot inspection |
| `IT25101857_Scaling.ipynb` | Feature scaling | Standardizes `age` with `StandardScaler` |
| `IT25103708_FeatureEngineering.ipynb` | Feature assembly | Builds the final `X` feature matrix and encodes the `dx` target |
| `IT25100928_ModelBuilding.ipynb` | Modeling | Baseline (Logistic Regression, Random Forest) + stronger models (SVM, Gradient Boosting), tuning, and evaluation |

## 11. Modeling & Evaluation Summary

- **Class imbalance:** handled via `class_weight='balanced'` and/or SMOTE oversampling (training set only).
- **Models compared:** Logistic Regression and Random Forest (baseline); SVM and Gradient Boosting (stronger).
- **Tuning:** `GridSearchCV` on at least one model, optimizing macro-F1.
- **Metrics reported:** accuracy, macro-F1, per-class recall (with particular attention to malignant classes `mel`, `bcc`, `akiec`), 5-fold stratified cross-validation, and confusion matrices.
- **Baseline for comparison:** a model that always predicts `nv` scores ~67% accuracy but 0% recall on every malignant class — all trained models are judged against beating this on macro-F1 and malignant-class recall, not raw accuracy.


## 12. Team & Roles

| Member | Student ID | Focus |
|---|---|---|
| Member 1 | IT25101908 | Missing value handling + age distribution EDA |
| Member 2 | IT25102853 | Categorical encoding + class distribution EDA |
| Member 3 | IT25103722 | Outlier detection/removal + boxplot EDA |
| Member 4 | IT25101857 | Scaling/normalization + correlation heatmap EDA |
| Member 5 | IT25103708 | Feature engineering + baseline models |
| Member 6 | IT25100928 | Stronger models, hyperparameter tuning, evaluation, pipeline integration |

## 13. AI Tool Usage Declaration

| Tool & version | Aspect supported | Extent of use | How verified/owned |
|---|---|---|---|
| *Claude Sonnet* | *project scaffolding, code structure guidance* | *Moderate* | *All code run and checked against notebook output; results recalculated manually* |

## 14. References

- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Harvard Dataverse. https://doi.org/10.7910/DVN/DBW86T
- Kaggle dataset page: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
