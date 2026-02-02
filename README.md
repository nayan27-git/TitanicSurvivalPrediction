## Titanic Survival Prediction Pipeline

### Overview

This repository contains a robust, end-to-end Machine Learning pipeline designed to predict passenger survival on the Titanic. The solution features a modular object-oriented architecture using Scikit-learn's Pipeline and ColumnTransformer APIs. It implements custom transformers for feature engineering, handles advanced preprocessing (including log transformations and one-hot encoding), and supports multiple classification algorithms (Logistic Regression, Random Forest, and XGBoost) within a unified execution framework.

### Repository Structure
```bash
.
├── Dataset
│   ├── test.csv
│   └── train.csv
├── Notebooks
│   └── Logistic_regression_using_sklearn.ipynb
├── src
│   ├── __init__.py
│   ├── Logistic_regression.py
│   └── run_model.py
├── .gitignore
└── requirements.txt
```


### Feature Engineering

The pipeline utilizes a custom TitanicTransformer class that inherits from Scikit-learn's BaseEstimator and TransformerMixin. This ensures seamless integration into the Scikit-learn workflow. Key engineering steps include:

**Title Extraction:** Parses passenger names to extract titles (e.g., Mr, Mrs, Miss).

**Title Mapping:** Groups rare titles (Dr, Rev, Col, Major, Don, Lady) into a single "Rare" category and standardizes synonyms (e.g., Mlle to Miss, Mme to Mrs).

**Family Size Calculation:** Creates a new feature combining siblings/spouses and parents/children counts: Family_size = SibSp + Parch + 1.

**Smart Age Imputation:** Imputes missing Age values based on the mean age of the specific Title group. Remaining gaps are filled with the global mean.

**Embarked Imputation:** Fills missing embarkation ports with the mode (most frequent value).

**Feature Dropping:** Removes high-cardinality or redundant columns (PassengerId, Cabin, Ticket, Name) after information extraction.

### Data Preprocessing Pipeline

The ModelManager class orchestrates preprocessing using a ColumnTransformer to apply specific transformations to different data types:

| Feature Type | Columns | Transformation Applied |
| :--- | :--- | :--- |
| **Skewed Numerical** | `Fare` | Logarithmic transformation (`np.log1p`) to normalize distribution. |
| **Numerical** | `Pclass`, `Age`, `Family_size` | Standard Scaling (Zero mean, Unit variance). |
| **Categorical** | `Embarked`, `Title`, `Sex` | One-Hot Encoding (dropping the first category to avoid collinearity). |

### Model Pipeline Architecture

The solution employs a three-step sequential pipeline:

**Cleaner/Feature Engineer:** TitanicTransformer handles domain-specific logic and imputations.

**Preprocessor:** ColumnTransformer applies statistical scaling and encoding.

**Estimator:** The selected classification model (Logistic Regression, Random Forest, or XGBoost).

This architecture prevents data leakage by ensuring that calculation statistics (means, modes, scaling parameters) are derived solely from the training set during the fit phase and applied consistently during transform.

### Supported Models

The ModelManager class supports the following algorithms, selectable via the model_type argument:

**Logistic Regression (logistic):** Baseline linear classifier.

**Random Forest Classifier (rf):** Ensemble bagging method for capturing non-linear relationships.

**XGBoost Classifier (xgb):** Gradient boosting framework for optimized performance.

### Evaluation Metrics

The pipeline evaluates model performance on the test set using the following metrics:

**Accuracy:** Overall correctness of predictions.

**Precision:** Accuracy of positive predictions.

**Recall:** Ability to find all positive instances.

**Log Loss:** Performance of the classification model where the prediction input is a probability value between 0 and 1.

**Confusion Matrix:** A tabular summary of correct and incorrect predictions.

### Installation

#### Clone the repository:
```bash
git clone [https://github.com/your-username/titanic-prediction-pipeline.git](https://github.com/your-username/titanic-prediction-pipeline.git)
cd titanic-prediction-pipeline
```


#### Set up the environment:
It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```


### Install dependencies:

```bash
pip install -r requirements.txt
```


### Usage

To train and evaluate the models, execute the run_model.py script from the root directory. This script initializes the pipeline, loads the data, splits it into training and testing sets (stratified), and iterates through all supported models.

```bash
python src/run_model.py
```


### Expected Output:
The script will print initialization status and evaluation metrics for each model:

------Initializing LogisticRegression------
Model Initialized !
Accuracy : 0.81
Precision : 0.78
Recall : 0.72
...


### Jupyter Notebook

A Jupyter Notebook is provided in Notebooks/Logistic_regression_using_sklearn.ipynb for exploratory data analysis (EDA), visualization, and prototyping of the feature engineering logic used in the production pipeline.

Configuration & Extensibility

Adding Models: New models can be added by updating the models dictionary within the _get_model method of the ModelManager class in src/Logistic_regression.py.

Hyperparameters: Hyperparameters for specific models can be adjusted within the instantiation calls in the _get_model method.

Feature Engineering: Modifications to feature logic should be made within the TitanicTransformer class to ensure consistency across training and inference.

### Tech Stack

**Language:** Python 3.x

**Data Manipulation:** Pandas, NumPy

**Machine Learning:** Scikit-learn (Pipeline, Compose, Preprocessing, Linear Model, Ensemble, Metrics)

**Gradient Boosting:** XGBoost

**File Handling:** Pathlib

### Design Principles

**Modularity:** Separation of concerns between feature engineering, model management, and execution logic.

**Reproducibility:** Use of random_state in data splitting and pipeline components.

**Scalability:** The pipeline structure allows for easy swapping of estimators or addition of new preprocessing steps without refactoring the entire codebase.

### Future Improvements

Implementation of GridSearchCV or RandomizedSearchCV for automated hyperparameter tuning.

Integration of unit tests for the TitanicTransformer.

Model serialization using joblib to save trained pipelines for inference APIs.

Experimentation with additional feature interactions.
