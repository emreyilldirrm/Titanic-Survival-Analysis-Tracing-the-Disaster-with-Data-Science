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
- ![image](https://github.com/user-attachments/assets/531fb893-b1c4-4dc9-9bc0-14a4dabc1d40)
![image](https://github.com/user-attachments/assets/a04b935f-8233-4666-b24d-26de5a0f70f9)

- **Objective**: Establish a baseline model to measure improvement in subsequent steps.

### 2. Model Improvement and Feature Extraction Step (1)
- **Feature Engineering**: 
  - Handling missing values (e.g., `Age`, `Embarked`).
  - Encoding categorical variables (e.g., `Sex`, `Embarked`).
  - Creating new features such as `FamilySize` (SibSp + Parch + 1) and `IsAlone`.
- **Model Used**: `CatBoostClassifier`
- ![image](https://github.com/user-attachments/assets/ff9f9a29-8fd5-4c20-8624-9eaed98c4db0)
- ![image](https://github.com/user-attachments/assets/3ba3b52c-0f25-47dc-9255-d598e166970e)


### 3. Model Improvement and Feature Extraction Step (2)
- **Additional Steps**:
  - Feature scaling for numerical variables.
  - Additional feature extraction (e.g., `Title` extracted from names).
- **Models Used**:
  - `CatBoostClassifier`
  - ![image](https://github.com/user-attachments/assets/ba3f85c5-aff3-4c95-96ba-23309065c21e)
  - ![image](https://github.com/user-attachments/assets/c813940a-6e8d-4c6e-89d8-0b2a16ba6703)

  - `RandomForestClassifier`
  - ![image](https://github.com/user-attachments/assets/be7fdb63-1708-42e8-9de6-6a56611ddcb9)
  - ![image](https://github.com/user-attachments/assets/457bebcb-901f-4bc5-9716-575f71e470a3)

  - `CART (Classification and Regression Tree)`
  - ![image](https://github.com/user-attachments/assets/0861bbf2-d03d-4eab-a156-efb714eb4333)
  - ![image](https://github.com/user-attachments/assets/fc48c6d2-96c9-4fa0-964f-48464b33acb1)



### 4. Model Improvement and Feature Extraction Step (3)
- **Further Enhancements**:
  - Interaction terms for key variables.
  - Binning of continuous variables like `Fare` and `Age`.
- **Models Used**:
  - `CatBoostClassifier`
  - ![image](https://github.com/user-attachments/assets/28ef2125-b5b0-4178-97d1-bf79f4db2422)
  - ![image](https://github.com/user-attachments/assets/cf071360-351c-44e9-9247-ed87a669505c)

  - `RandomForestClassifier`
  - ![image](https://github.com/user-attachments/assets/16c25595-b636-415a-b2a3-348ffd31dd14)
  - ![image](https://github.com/user-attachments/assets/418ef019-c14a-48f4-a8e2-155d151384ba)

  - `CART`
  - ![image](https://github.com/user-attachments/assets/50538f3e-b0b7-4476-abe0-72d9b8123f9f)
  - ![image](https://github.com/user-attachments/assets/550182c6-bcf5-4649-993c-56649984883f)


### 5. Model Improvement and Feature Extraction Step (4)
- **Incorporated Models**:
  - `CatBoostClassifier`
  - ![image](https://github.com/user-attachments/assets/78b59ad4-3665-4ecb-8e1a-79da5c324339)
  - ![image](https://github.com/user-attachments/assets/2e128468-04e9-49b2-9428-fd7d74aa27e3)

  - `RandomForestClassifier`
  - ![image](https://github.com/user-attachments/assets/cf9b8019-4dad-4f16-834b-ad045a2c932d)
  - ![image](https://github.com/user-attachments/assets/f8a23796-e0a8-4e7d-b965-97339724d267)

  - `CART`
  - ![image](https://github.com/user-attachments/assets/77a69fab-46ef-4c90-a612-28c06b5bcefb)
  - ![image](https://github.com/user-attachments/assets/989b29d0-f52c-4cdc-8bce-40cf8b3829e9)


  - `Linear Regression`
  - ![image](https://github.com/user-attachments/assets/bc1c20f5-0bf0-457e-a0f2-94148260734f)
  - ![image](https://github.com/user-attachments/assets/119de102-e311-48e4-9565-074a5de86e6c)


  - `LightGBM (LGBM)`
  - ![image](https://github.com/user-attachments/assets/90061b3a-12b4-4569-8df0-ceb8b2ad6ad4)
  - ![image](https://github.com/user-attachments/assets/4e8de95d-a2c6-46ed-a028-def2f6d273ad)


### 6. Model Improvement and Feature Extraction Step (5)
- **Final Iteration**:
  - Final selection of features with feature importance ranking.
- **Models Used**:
  - `CatBoostClassifier`
  - ![image](https://github.com/user-attachments/assets/16474370-b3d2-4a77-8a58-44bc92e09d9b)
  - ![image](https://github.com/user-attachments/assets/648c5cc5-25d8-4744-9a8c-e4d88a95bbd5)

  - `RandomForestClassifier`
  - ![image](https://github.com/user-attachments/assets/4534e502-7c08-4ac5-b667-139c467ee3c1)
  - ![image](https://github.com/user-attachments/assets/e328c1fc-0f7c-47a0-a5c6-cd620cbe1525)

  - `CART`
  - ![image](https://github.com/user-attachments/assets/f5b71628-96b4-4da2-9d28-7cbd3a0804e1)
  - ![image](https://github.com/user-attachments/assets/446df8ac-7dfb-4f20-9dcd-405f66647357)

  - `Linear Regression`
  - ![image](https://github.com/user-attachments/assets/4876c077-1003-47b5-999d-d73a750c4242)
  - ![image](https://github.com/user-attachments/assets/d30e05ec-b835-4b05-81c6-13be4a7c6b93)

  - `LightGBM (LGBM)`
  - ![image](https://github.com/user-attachments/assets/c29ec7fd-870b-4caa-a260-ff1633048dcf)
  - ![image](https://github.com/user-attachments/assets/cae7ff25-b095-4731-b5e5-2510ee0a377b)


---

## Results

Each step includes model training and evaluation using metrics such as accuracy, precision, recall, and F1-score. The best-performing model was selected based on validation set performance.
  - ![image](https://github.com/user-attachments/assets/4534e502-7c08-4ac5-b667-139c467ee3c1)
  - ![image](https://github.com/user-attachments/assets/e328c1fc-0f7c-47a0-a5c6-cd620cbe1525)

- Advanced hyperparameter tuning for all models.
![image](https://github.com/user-attachments/assets/f28c2842-31cb-454e-bbf7-b660088d64c9)
![image](https://github.com/user-attachments/assets/944f087a-f4c6-4b22-9534-5537eccda7e6)


---

## Conclusion

This project demonstrated the incremental improvement of predictive models for Titanic survival using feature engineering and multiple machine learning algorithms. The iterative process showcased the value of fine-tuning and feature extraction in achieving higher predictive performance.

The best model achieved an accuracy of **0.9851**, showing significant improvement over the baseline.

---

## Files Included

- **train.csv**: Training dataset
- **test.csv**: Test dataset
- **PyCharm**: Python with code implementation

---

