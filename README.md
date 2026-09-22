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

> The dataset is heavily imbalanced.

## 4. Repository Structure

```
Group_ID/
├── README.md
├── group_pipeline.ipynb              # PR1 — integrated preprocessing pipeline
├── group_model_comparison.ipynb      # PR2 — combined model results comparison
├── data/
│   ├── raw/
│   └── external/
├── notebooks/
│   ├── IT25101908_MissingValues.ipynb           # PR1
│   ├── IT25102853_Encoding.ipynb                # PR1
│   ├── IT25103722_OutlierRemoval.ipynb          # PR1
│   ├── IT25101857_Scaling.ipynb                 # PR1
│   ├── IT25103708_FeatureEngineering.ipynb      # PR1
│   ├── IT25100928_FeatureSelection.ipynb        # PR1
│   ├── IT25101908_LogisticRegression_Model.ipynb # PR2
│   ├── IT25102853_SVM_Model.ipynb               # PR2
│   ├── IT25103722_RandomForest_Model.ipynb      # PR2
│   ├── IT25101857_KMeans_Model.ipynb            # PR2
│   ├── IT25103708_PCA_Model.ipynb               # PR2
│   └── IT25100928_MLP_Model.ipynb               # PR2
└── results/
    ├── eda_visualizations/       # PR1 charts (histograms, boxplots, heatmaps)
    ├── model_visualizations/     # PR2 charts (confusion matrices, ROC, cluster/PCA plots)
    ├── outputs/                  # PR1 final processed dataset / target
    ├── model_outputs/            # PR2 per-member metrics (CSV)
    └── logs/
```

## 5. Getting the Data & Running the Project

**Data setup:** Download `HAM10000_metadata.csv` from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) and place it at `data/raw/HAM10000_metadata.csv`.

**Setup environment:**

```bash
# 1. Create and activate an isolated environment
conda create -n dermacheck python=3.10
conda activate dermacheck

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter

# 3. Launch Jupyter
jupyter notebook
```

**Run order:**

1. **Read this README fully** before touching any code — it covers the goal, data, and folder layout.
2. **Set up the environment** (above).
3. **Preprocessing stage (PR1)** — open the six notebooks in `notebooks/`, in this order, to see each technique in isolation:
   - `IT25101908_MissingValues.ipynb`
   - `IT25102853_Encoding.ipynb`
   - `IT25103722_OutlierRemoval.ipynb`
   - `IT25101857_Scaling.ipynb`
   - `IT25103708_FeatureEngineering.ipynb`
   - `IT25100928_FeatureSelection.ipynb`

   Each is self-contained, loading the raw CSV independently and applying one technique with a written interpretation.
4. **Run `group_pipeline.ipynb`** (project root) — the single integrated notebook chaining all six PR1 steps end-to-end (cleaning → EDA → feature engineering → feature selection). Only its first cell loads the raw CSV; every step after reads/modifies the same evolving `df`, since notebooks can't import one another. Running this top-to-bottom reproduces `results/outputs/final_processed_dataset.csv` and `final_target_y.csv` — the shared inputs for every model notebook that follows. Only the finished, working cell from each member's notebook is copied in; exploratory/debugging cells stay in the individual notebooks, which remain the record of who did what for PR1 marks.
5. **Modeling stage (PR2)** — once the processed dataset exists, open the six model notebooks in `notebooks/`, each loading `final_processed_dataset.csv` + `final_target_y.csv` and training/tuning one model type independently:
   - `IT25101908_LogisticRegression_Model.ipynb`
   - `IT25102853_SVM_Model.ipynb`
   - `IT25103722_RandomForest_Model.ipynb`
   - `IT25101857_KMeans_Model.ipynb`
   - `IT25103708_PCA_Model.ipynb`
   - `IT25100928_MLP_Model.ipynb`

   Each saves its own chart(s) to `results/model_visualizations/` and its own metrics to `results/model_outputs/`.
6. **Run `group_model_comparison.ipynb`** (project root) — the PR2 counterpart to `group_pipeline.ipynb`. It reads every member's `results/model_outputs/*.csv`, builds the combined comparison of all 6 optimum results, and adds the group's written discussion of challenges and performance. This notebook depends on every individual model notebook having been run first.
7. **Check outputs** — `results/eda_visualizations/` for PR1 charts, `results/model_visualizations/` for PR2 charts, and `results/outputs/` / `results/model_outputs/` for processed data and metrics.

## 6. How the Pipeline Is Assembled

The notebooks in `notebooks/` are not automatically linked — Jupyter notebooks can't import one another the way Python modules can. Each one is self-contained on purpose, so each contributor's work can be marked in isolation. They connect in two ways, one for each stage:

**PR1 — Preprocessing:**
1. Every individual notebook starts with its own copy of the Step 1 data-loading code (`pd.read_csv("data/raw/HAM10000_metadata.csv")` plus the initial `.shape` / `.info()` / `.head()` / `value_counts()` checks). Without this, a notebook has no `df` to work on, so it must load and inspect the raw CSV itself before applying its one technique.
2. `group_pipeline.ipynb` is the single notebook where everything is chained together. Once every member's notebook works standalone, each finished code cell is copied — in the order below — into `group_pipeline.ipynb`, so the output dataframe of one step becomes the input to the next, forming one continuous run from raw CSV to final processed feature set.

Assembly order in `group_pipeline.ipynb`:
```
1. Load & inspect data        ← Step 1 (everyone's notebook starts here too)
2. Missing values             ← IT25101908_MissingValues.ipynb
3. Categorical encoding       ← IT25102853_Encoding.ipynb
4. Outlier removal            ← IT25103722_OutlierRemoval.ipynb
5. Scaling                    ← IT25101857_Scaling.ipynb
6. Feature engineering        ← IT25103708_FeatureEngineering.ipynb
7. Feature selection          ← IT25100928_FeatureSelection.ipynb
```

**PR2 — Modeling:**
1. Each individual model notebook starts by loading `results/outputs/final_processed_dataset.csv` and `final_target_y.csv` — the shared output of `group_pipeline.ipynb` — rather than the raw CSV. This is the one dependency between stages: PR2 notebooks cannot run until `group_pipeline.ipynb` has been run at least once.
2. Each member's notebook is independent of the others (no shared `df` to chain, unlike PR1), since each trains and tunes its own model type in isolation and writes its results out to `results/model_outputs/` and `results/model_visualizations/`.
3. `group_model_comparison.ipynb` is the integration point for PR2: it reads all six `results/model_outputs/*.csv` files, builds one combined comparison table/chart of the optimum result from each model, and adds the group's written discussion — the same collaborative role `group_pipeline.ipynb` plays for PR1.

Two rules that make both stages work:

- Each step must run cleanly on the input produced by the step before it — so when copying a preprocessing cell into `group_pipeline.ipynb`, remove any re-loading of the raw CSV (only the very first cell loads it), and make sure PR2 notebooks load the *processed* CSVs, not the raw one.
- Only the final, working version of each person's cell/notebook goes into the shared integration notebooks; exploratory/debugging cells stay in the individual notebooks, which remain the record of who did what for individual marks in both PR1 and PR2.

## 7. Preprocessing Techniques (by contributor)

| Notebook | Technique | Summary |
|---|---|---|
| `IT25101908_MissingValues.ipynb` | Missing value imputation | Median imputation for missing `age` values |
| `IT25102853_Encoding.ipynb` | Categorical encoding | One-hot encoding of `sex` and `localization` |
| `IT25103722_OutlierRemoval.ipynb` | Outlier handling | Detects/caps unrealistic `age` values via boxplot inspection |
| `IT25101857_Scaling.ipynb` | Feature scaling | Standardizes `age` with `StandardScaler` |
| `IT25103708_FeatureEngineering.ipynb` | Feature assembly | Builds the final `X` feature matrix and encodes the `dx` target |
| `IT25100928_FeatureSelection.ipynb` | Feature selection / dimensionality reduction | Removes low-variance and redundant (highly correlated) columns from the final feature matrix |

## 8. Modeling & Evaluation Summary (Progress Review II)

- **Class imbalance:** handled via `class_weight='balanced'` and/or SMOTE oversampling (training set only).
- **Models trained (one per member):** Logistic Regression, SVM, Random Forest, K-Means, PCA, MLP — see Section 9 for owners.
- **Tuning:** `GridSearchCV` or manual tuning per member, with at least two varieties compared per model (e.g. different hyperparameters, or with/without feature selection applied).
- **Metrics reported:** accuracy, macro-F1, per-class recall (with particular attention to malignant classes `mel`, `bcc`, `akiec`), cross-validation, and confusion matrices, saved per member to `results/model_outputs/`.
- **Baseline for comparison:** a model that always predicts `nv` scores ~67% accuracy but 0% recall on every malignant class — all trained models are judged against beating this on macro-F1 and malignant-class recall, not raw accuracy.
- **Group comparison:** `group_model_comparison.ipynb` combines the optimum result from each member's model and discusses challenges and expected behavior as a group.

## 9. Team & Roles

| Member | Student ID | PR1 Focus (Preprocessing) | PR2 Focus (Modeling) |
|---|---|---|---|
| Member 1 | IT25101908 | Missing value handling + age distribution EDA | Logistic Regression |
| Member 2 | IT25102853 | Categorical encoding + class distribution EDA | SVM |
| Member 3 | IT25103722 | Outlier detection/removal + boxplot EDA | Random Forest |
| Member 4 | IT25101857 | Scaling/normalization + correlation heatmap EDA | K-Means Clustering |
| Member 5 | IT25103708 | Feature engineering + final X/y assembly | PCA |
| Member 6 | IT25100928 | Feature selection/dimensionality reduction + correlation heatmap EDA | MLP (Deep Learning) |

## 10. AI Tool Usage Declaration

| Tool & version | Aspect supported | Extent of use | How verified/owned |
|---|---|---|---|
| *Claude Sonnet* | *project scaffolding, code structure guidance* | *Moderate* | *All code run and checked against notebook output; results recalculated manually* |

## 11. References

- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Harvard Dataverse. https://doi.org/10.7910/DVN/DBW86T
- Kaggle dataset page: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000