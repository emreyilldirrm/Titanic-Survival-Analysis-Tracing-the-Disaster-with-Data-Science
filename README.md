# Titanic Survival Prediction Project

This project is an extensive exploration of the Titanic dataset, aimed at improving the predictive performance of various machine learning models. We progressively enhance the feature set and evaluate multiple models across various stages. Below is a detailed walkthrough of the steps taken and models implemented.

---

## Dataset Overview

The Titanic dataset contains information about passengers, such as:

- **PassengerId**: Unique ID for each passenger
- **Survived**: Survival status (0 = Not Survived, 1 = Survived)
- **Pclass**: Passenger class (1, 2, or 3)
- **Name**: Name of the passenger
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings or spouses aboard
- **Parch**: Number of parents or children aboard
- **Ticket**: Ticket number
- **Fare**: Ticket fare
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## Workflow

### 1. Base Model
- **Model Used**: `CatBoostClassifier`
- **Objective**: Establish a baseline model to measure improvement in subsequent steps.

### 2. Model Improvement and Feature Extraction Step (1)
- **Feature Engineering**: 
  - Handling missing values (e.g., `Age`, `Embarked`).
  - Encoding categorical variables (e.g., `Sex`, `Embarked`).
  - Creating new features such as `FamilySize` (SibSp + Parch + 1) and `IsAlone`.
- **Model Used**: `CatBoostClassifier`

### 3. Model Improvement and Feature Extraction Step (2)
- **Additional Steps**:
  - Feature scaling for numerical variables.
  - Additional feature extraction (e.g., `Title` extracted from names).
- **Models Used**:
  - `CatBoostClassifier`
  - `RandomForestClassifier`
  - `CART (Classification and Regression Tree)`

### 4. Model Improvement and Feature Extraction Step (3)
- **Further Enhancements**:
  - Interaction terms for key variables.
  - Binning of continuous variables like `Fare` and `Age`.
- **Models Used**:
  - `CatBoostClassifier`
  - `RandomForestClassifier`
  - `CART`

### 5. Model Improvement and Feature Extraction Step (4)
- **Incorporated Models**:
  - `CatBoostClassifier`
  - `RandomForestClassifier`
  - `CART`
  - `Linear Regression`
  - `LightGBM (LGBM)`

### 6. Model Improvement and Feature Extraction Step (5)
- **Final Iteration**:
  - Advanced hyperparameter tuning for all models.
  - Final selection of features with feature importance ranking.
- **Models Used**:
  - `CatBoostClassifier`
  - `RandomForestClassifier`
  - `CART`
  - `Linear Regression`
  - `LightGBM (LGBM)`

---

## Results

Each step includes model training and evaluation using metrics such as accuracy, precision, recall, and F1-score. The best-performing model was selected based on validation set performance.

| Model                | Accuracy  | Precision | Recall | F1-Score |
|----------------------|-----------|-----------|--------|----------|
| CatBoostClassifier   | X.XX%     | X.XX%     | X.XX%  | X.XX%    |
| RandomForest         | X.XX%     | X.XX%     | X.XX%  | X.XX%    |
| CART                 | X.XX%     | X.XX%     | X.XX%  | X.XX%    |
| Linear Regression    | X.XX%     | X.XX%     | X.XX%  | X.XX%    |
| LightGBM             | X.XX%     | X.XX%     | X.XX%  | X.XX%    |

---

## Conclusion

This project demonstrated the incremental improvement of predictive models for Titanic survival using feature engineering and multiple machine learning algorithms. The iterative process showcased the value of fine-tuning and feature extraction in achieving higher predictive performance.

The best model achieved an accuracy of **X.XX%**, showing significant improvement over the baseline.

---

## Files Included

- **train.csv**: Training dataset
- **test.csv**: Test dataset
- **PyCharm**: Python with code implementation

---

