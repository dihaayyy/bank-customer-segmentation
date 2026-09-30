# Bank Customer Segmentation & Classification

Segmenting bank customers from their transaction data with **K-Means clustering**, then training classifiers that assign new customers to a segment.

The project has two notebooks:

| Notebook | What it does |
|---|---|
| [`clustering_fraud_detection.ipynb`](clustering_fraud_detection.ipynb) | Cleans the data, builds K-Means segments, and interprets each cluster |
| [`classification_fraud_detection.ipynb`](classification_fraud_detection.ipynb) | Trains Decision Tree, Logistic Regression, and a tuned Random Forest on the cluster labels |

## Dataset

A bank transaction dataset provided in the Dicoding course *Belajar Machine Learning untuk Pemula* (BMLP). It covers transaction details, customer demographics, and account activity. The dataset was designed for exploring fraud detection and anomaly patterns.

- **Raw size:** 2,537 rows × 16 columns
- **After cleaning:** 1,945 rows × 8 features (+ 1 engineered feature)

| Group | Features |
|---|---|
| Transaction | `TransactionAmount`, `TransactionType`, `Channel`, `TransactionDuration` |
| Customer | `CustomerAge`, `CustomerOccupation`, `AccountBalance` |
| Activity | `LoginAttempts` |
| Dropped | IDs, `IP Address`, dates, `Location` |

## Workflow

### 1. Clustering
1. **Cleaning:** removed missing values and 21 duplicate rows.
2. **Feature selection:** dropped ID, IP, and date columns, plus `Location`. `Location` has 43 cities; label-encoding it created a fake alphabetical distance that dominated K-Means in an earlier version.
3. **Preprocessing:** label encoding, IQR outlier removal, standard scaling, and age binning (`Low` / `Medium` / `High`).
4. **Modeling:** K-Means with **k = 3**, which gives a **Silhouette Score of 0.263**. A PCA-based K-Means model was trained as a comparison.
5. **Interpretation:** inverse-transformed the data back to original values to describe each segment.

### 2. Classification
1. One-hot encoded categorical features and used an 80/20 stratified split.
2. Trained a **Decision Tree** (baseline) and **Logistic Regression** (in a `StandardScaler` pipeline).
3. Tuned a **Random Forest** with `GridSearchCV` (36 combinations, 5-fold CV).

## Results

### Customer segments

| Segment | Size | Avg. age | Avg. balance | Dominant occupation |
|---|---|---|---|---|
| **0 · Working-Age Professionals** | 797 | 42 | 6,575 | Engineer (51%), Doctor (39%) |
| **1 · Young Students** | 511 | 23 | 1,516 | Student (100%) |
| **2 · Senior Customers** | 637 | 65 | 6,131 | Retired (60%), Doctor (30%) |

The segments are mainly defined by **life stage and account balance**. Transaction amount, type, channel, and duration are similar across all three.

### Classification

| Model | Accuracy | Macro F1 |
|---|---|---|
| Decision Tree | 1.00 | 1.00 |
| Logistic Regression | 1.00 | 1.00 |
| Random Forest (tuned) | 1.00 | 1.00 |

The perfect scores are expected, not a sign of overfitting. The labels were produced by K-Means from the same features, so the classifiers only need to learn the clustering rules again. K-Means boundaries are linear, which is why even Logistic Regression separates the segments perfectly.

## Limitations

- **The labels come from clustering, not ground truth.** Both notebooks describe customer segments. Neither detects confirmed fraud.
- **Outlier removal removed the fraud signal.** After IQR filtering, every row has `LoginAttempts = 1`, so multiple-login transactions (the strongest anomaly signal) were lost. Capping outliers instead of dropping them would keep this information.
- **Moderate cluster separation.** A Silhouette Score of 0.263 means the segments overlap.

## How to Run

```bash
git clone https://github.com/dihaayyy/bank-customer-segmentation.git
cd bank-customer-segmentation
pip install -r requirements.txt
jupyter notebook
```

Run `clustering_fraud_detection.ipynb` first. It creates `data_clustering_inverse.csv`, which the classification notebook loads.

> **Note:** `yellowbrick 1.5` is not compatible with `scikit-learn >= 1.6`, and on Python 3.12 it needs `setuptools`. Both are pinned in `requirements.txt`.

## Tools

Python 3.12 · pandas · NumPy · scikit-learn · Yellowbrick · Matplotlib · Seaborn · joblib

## Author

**Muhammad Dihya Al Qalby**. Built as the final project for Dicoding's *Belajar Machine Learning untuk Pemula*, then extended for this portfolio.
