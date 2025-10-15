# **Imputation via Regression for Missing Data (DA5401 A6)**

**Student Name:** Shreehari Anbazhagan\
**Roll Number:** DA25C020

---

## **Project Overview**

This project tackles the common challenge of missing data in a credit risk assessment context. The goal is to compare different imputation strategies by applying them to the UCI Credit Card Default Clients dataset, which has been modified to include artificially introduced "Missing At Random" (MAR) values.

Four distinct methods for handling missing data are implemented and evaluated:
1.  **Simple Imputation** (Median)
2.  **Linear Regression Imputation** (Linear)
3.  **Non-Linear Regression Imputation** (K-Nearest Neighbors)
4.  **Listwise Deletion** (dropping rows with missing data)

The effectiveness of each method is measured by training a **Logistic Regression classifier** on the cleaned datasets and comparing their final classification performance, particularly the F1-score.

**Dataset:** [Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

---

## Folder Structure & Files

```
project-root/
│
├─ main.ipynb              # Core Jupyter Notebook with all code, visualizations, and explanations
├─ python-version          # Python version used for reproducibility
├─ pyproject.toml          # Project dependencies
├─ uv.lock                 # Locked dependency versions (for uv sync)
├─ .gitignore              # Excludes dataset files
├─ Instructions/           # Assignment PDF (problem statement & instructions)
    └─ DA5401 A6 Imputation via Regression.pdf
└─ datasets/               # Dataset folder
   └─ UCI_Credit_Card.csv  # Default of Credit Card Clients Dataset (from Kaggle)
```

### Notes:

* Run `uv sync` to install dependencies exactly as tested.
* The dataset is **not pushed to GitHub**. Please download it via Kaggle:
[Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
* Plots using **plotly are not visible in github** as they use JavaScript to render, Please run them locally or use nbviewer.
* The notebook is **self-contained**: all preprocessing, PCA, and classification steps are reproducible without manual intervention.

---

## Dependencies

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
```

---

## **Analysis Workflow**

The project follows a structured workflow designed to isolate the impact of each data-handling strategy:

1.  **Data Preparation**: The dataset is loaded, and 5-10% of values in 2-3 numerical columns are replaced with `NaN` to simulate a real-world missing data problem.
2.  **Imputation Strategies**:
    * **Dataset A (Baseline)**: Missing values are filled using the column's **median**.
    * **Dataset B (Linear)**: A **Linear Regression** model is used to predict and fill missing values based on other features.
    * **Dataset C (Non-Linear)**: A **K-Nearest Neighbors (KNN) Regressor** predicts and fills the same missing values.
3.  **Listwise Deletion**:
    * **Dataset D**: Created by dropping all rows containing any `NaN` values, serving as a comparison against imputation.
4.  **Model Training & Evaluation**: All four datasets are split into training and testing sets, standardized, and then used to train separate Logistic Regression models. Performance is evaluated using a full classification report.
5.  **Comparative Analysis**: The results from all four models are compiled into a summary table and visualized to compare their precision, recall, and F1-scores, leading to a final recommendation.

---

## **Key Findings**

The comparative analysis revealed clear performance differences between the four data-handling strategies:

* **Imputation Outperforms Deletion**: All three imputation methods (Models A, B, and C) resulted in better model performance than listwise deletion (Model D). Deleting rows, while simple, led to significant data loss, reducing the model's statistical power and resulting in the lowest accuracy and F1-score for the critical minority class (defaulters).
* **Linear Imputation Was Most Effective**: The **Linear Regression Imputation** (Model B) achieved the highest overall accuracy and the best F1-score for predicting defaults. This suggests that the relationships between the features in the dataset are predominantly linear, making the linear model a better fit for prediction than the non-linear KNN approach.
* **Simple Imputation Is a Strong Baseline**: Median imputation (Model A) performed nearly as well as the more complex regression methods, proving to be a robust and effective baseline.

| Model | Precision (0) | Recall (0) | F1-score (0) | Precision (1) | Recall (1) | F1-score (1) | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A: Median | 0.876 | 0.702 | 0.779 | 0.378 | 0.646 | 0.477 | 0.690 |
| B: Linear | 0.877 | 0.707 | 0.783 | 0.382 | 0.645 | 0.480 | 0.694 |
| C: KNN | 0.876 | 0.704 | 0.781 | 0.379 | 0.644 | 0.477 | 0.691 |
| D: Deletion | 0.875 | 0.698 | 0.777 | 0.366 | 0.637 | 0.465 | 0.685 |

---

## **Conclusion and Recommendations**

For this credit risk assessment scenario, the most effective strategy for handling missing data is **Linear Regression Imputation** (Model B).

This recommendation is based on the following:
1.  **Best Performance**: It delivered the highest F1-score for the minority class (defaulters), which is the most critical metric for a risk assessment model.
2.  **Data Preservation**: Unlike listwise deletion, it preserves the entire dataset, maximizing the information available for training.
3.  **Appropriate Complexity**: It leverages inter-feature relationships more effectively than simple median imputation without the unnecessary complexity of a non-linear model, which did not improve results.

Therefore, for this dataset and problem, using a linear regression model to impute missing values provides the best balance of performance, data integrity, and conceptual soundness.