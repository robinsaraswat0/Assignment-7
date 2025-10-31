# DA5401 A7 — Multi-Class Model Selection using ROC and Precision–Recall Curves

##  Objective

This project implements **multi-class model selection** on the **UCI Landsat Satellite dataset** by comparing a diverse set of classifiers using **ROC** and **Precision–Recall (PRC)** analysis.  
The focus is on evaluating performance **beyond simple accuracy**, using threshold-independent metrics such as **AUC** and **Average Precision (AP)** to interpret class separability and ranking ability.

---

##   Dataset

**UCI Landsat Satellite Dataset**  
A classic multi-class dataset (6 land-cover classes) with high-dimensional features and class overlap.  
Source: [UCI ML Repository – Statlog (Landsat Satellite)](https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite))

---

##  Models Compared

| # | Model | Library | Expected Performance |
|:-:|:------|:---------|:--------------------|
| 1 | **K-Nearest Neighbors (KNN)** | `sklearn.neighbors.KNeighborsClassifier` | Moderate–Good |
| 2 | **Decision Tree** | `sklearn.tree.DecisionTreeClassifier` | Moderate |
| 3 | **Dummy (Prior)** | `sklearn.dummy.DummyClassifier` | Baseline (random-like) |
| 4 | **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | Linear baseline |
| 5 | **Gaussian Naive Bayes** | `sklearn.naive_bayes.GaussianNB` | Variable / Poor |
| 6 | **Support Vector Classifier (SVC)** | `sklearn.svm.SVC` | Good (nonlinear) |
| 7 | **Random Forest (Bonus)** | `sklearn.ensemble.RandomForestClassifier` | Strong Ensemble |
| 8 | **XGBoost (Bonus)** | `xgboost.XGBClassifier` | Strong Ensemble |

---

##  Part A — Baseline Evaluation

Each model was trained on standardized features and evaluated using **Test Accuracy** and **Weighted F1-Score**.

| Model | Test Accuracy | Weighted F1 |
|:------|:---------------:|:------------:|
| RandomForest | **0.9135** | **0.9113** |
| KNN | 0.9045 | 0.9037 |
| XGBoost | 0.9020 | 0.9004 |
| SVC | 0.8955 | 0.8925 |
| DecisionTree | 0.8510 | 0.8517 |
| LogisticRegression | 0.8395 | 0.8296 |
| GaussianNB | 0.7965 | 0.8036 |
| Dummy (Prior) | 0.2305 | 0.0864 |

**Observation:**  
RandomForest slightly outperformed KNN and XGBoost on both accuracy and F1, showing consistent threshold-level balance between precision and recall.

---

##  Part B — ROC Curve Analysis

### Macro/Micro-Averaged AUC

| Model | Macro AUC | Micro AUC |
|:------|:----------:|:----------:|
| **XGBoost** | **0.9898** | **0.9937** |
| RandomForest | 0.9896 | 0.9938 |
| SVC | 0.9852 | 0.9920 |
| KNN | 0.9786 | 0.9841 |
| LogisticRegression | 0.9757 | 0.9853 |
| GaussianNB | 0.9553 | 0.9611 |
| DecisionTree | 0.9003 | 0.9106 |
| Dummy (Prior) | 0.5000 | 0.6061 |

**Interpretation:**  
- **XGBoost** achieved the **highest Macro-AUC (0.9898)**, indicating near-perfect class separability.  
- Ensemble models (XGBoost, RandomForest) excelled due to their ability to model complex nonlinear boundaries.  
- Models like DecisionTree and GaussianNB underperformed, suggesting less stable ranking across thresholds.  
- Dummy classifier provided a near-random baseline (AUC ≈ 0.5).

---

##  Part C — Precision–Recall Curve (PRC) Analysis

| Model | Macro AP | Weighted AP |
|:------|:----------:|:-------------:|
| **XGBoost** | **0.949** | **0.960** |
| RandomForest | 0.949 | 0.959 |
| KNN | 0.922 | 0.934 |
| SVC | 0.918 | 0.934 |
| LogisticRegression | 0.871 | 0.899 |
| GaussianNB | 0.811 | 0.840 |
| DecisionTree | 0.737 | 0.765 |
| Dummy (Prior) | 0.167 | 0.185 |

**Interpretation:**  
Precision–Recall emphasizes minority class behavior. RandomForest and XGBoost maintained **high precision even at high recall**, showing excellent confidence calibration.

---

##  Part D — Synthesis and Recommendation

| Metric | Best Model(s) | Trend Summary | Takeaway |
|:--------|:--------------|:--------------|:----------|
| **Weighted F1** | RandomForest | RandomForest > KNN > XGBoost > SVC | Strong balance at single threshold |
| **ROC-AUC (Macro)** | XGBoost | XGBoost ≈ RandomForest > SVC | Best global ranking ability |
| **PRC-AP (Macro)** | RandomForest ≈ XGBoost | Ensembles > KNN > SVC | Best under class imbalance |

###  Interpretation
- **ROC-AUC** measures ranking ability across all thresholds.  
- **PRC-AP** focuses on handling imbalance and precision–recall trade-offs.  
- **F1-Score** captures single-threshold balance but not calibration.

Although **SVC** performs well in separability (AUC ≈ 0.985), ensemble models generalize better across thresholds and recall extremes.

###  **Final Recommendation**
> **Best Overall Model: RandomForest**  
> - Highest Weighted F1 (0.9113)  
> - Near-perfect AUC (0.9896)  
> - Top Precision–Recall performance (AP ≈ 0.949)  
> - Stable and interpretable with minimal overfitting  
> **Runner-up:** XGBoost (marginally better AUC but slightly lower F1)

---

##  Extra Experiment — Producing a Model with AUC < 0.5

A **controlled inversion experiment** was performed by flipping the SVC’s predicted scores.

| Model | Macro AUC | Micro AUC | Macro AP | Weighted AP |
|:------|:----------:|:----------:|:----------:|:-------------:|
| **SVC (Original)** | **0.9850** | **0.9920** | **0.9177** | **0.9336** |
| **SVC (Inverted)** | **0.0264** | **0.0191** | **0.0902** | **0.1008** |

### Interpretation
- **Inversion flips the ROC curve** below the diagonal:  
  \[
  \text{AUC}_{\text{inverted}} \approx 1 - \text{AUC}_{\text{original}}
  \]
- The inverted classifier ranks positives and negatives **in the opposite order**, producing “worse-than-random” performance.
- PRC performance collapses as precision drops to near zero at most recall levels.

**Takeaway:**  
An AUC < 0.5 usually signals **systematic score inversion** or **label mismatch**, not random noise. Fixing sign or label alignment often restores correct behavior.

---

##  Conceptual Takeaways

- **ROC-AUC**: Measures ranking quality; insensitive to class imbalance.  
- **PRC-AP**: Focuses on retrieval quality; more revealing under imbalance.  
- **AUC < 0.5**: Indicates reversed ranking — valuable diagnostic signal.  
- **Ensemble models (RF/XGB)** outperform others due to robustness and nonlinear feature handling.

---

##  References

- UCI ML Repository — [Statlog (Landsat Satellite)](https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite))  
- scikit-learn Documentation (ROC, PRC, Metrics)  
- XGBoost Documentation  
- Assignment Brief: *DA5401 A7 — Multi-Class Model Selection using ROC and PRC Curves*

---

##  Author

**Name:** *Robin*  
**Course:** *DA5401 - Machine Learning for Data Analytics*  

---
