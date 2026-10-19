# Mercedes-Benz Greener Manufacturing — Machine Learning Analysis

## Project Overview

The goal of this project is to predict the time a Mercedes-Benz car spends on the test bench while reducing the time and resources required for testing. Machine learning techniques are used to analyze vehicle configuration data, reduce the number of features, and build a predictive model.

The project uses the Mercedes-Benz Greener Manufacturing dataset and applies data preprocessing, feature encoding, dimensionality reduction, and XGBoost regression to predict test set values.

## Dataset Description

The dataset consists of two files: `train.csv` and `test.csv`.

- **Training set (`train.csv`):** 4,209 rows and 378 columns.
- **Test set (`test.csv`):** 4,209 rows and 377 columns.

### Key Columns

- **`ID`**: Unique identifier for each car configuration. It is not used as a predictive feature.
- **`y`**: Target variable representing the time, in seconds, that a car spends on the test bench. This column is available only in the training dataset.
- **`X0` to `X385`**: Anonymized features representing different car options and configurations.
- **Categorical features**: Eight categorical columns — `X0`, `X1`, `X2`, `X3`, `X4`, `X5`, `X6`, and `X8`.
- **Binary features**: The remaining feature columns are binary.

Both datasets are complete and contain no missing values.

## Project Objectives

The following tasks are performed as part of the machine learning workflow:

### 1. Remove Zero-Variance Features

- Calculate the variance of each feature.
- Identify columns with zero variance.
- Remove features that have the same value in every row, as they provide no predictive information.
- The training dataset contains 12 zero-variance columns.

### 2. Check Null and Unique Values

- Check for missing values in both training and test datasets.
- Count unique values in each column.
- Identify categorical and binary features.
- Compare feature structures between the training and test datasets.

### 3. Apply Label Encoding

- Apply label encoding to the eight categorical features.
- Fit the encoders using the combined categories from the training and test datasets.
- Transform categorical values into numerical representations.
- Handle categories that appear in the test set but are not present in the training set.

### 4. Perform Dimensionality Reduction

- Apply Principal Component Analysis (PCA) to the encoded features.
- Reduce the number of input features while preserving as much useful variance as possible.
- Select the number of principal components based on the desired explained variance.
- Analyze how dimensionality reduction affects the feature space and model performance.

### 5. Train the XGBoost Regressor

- Use the reduced training features and target variable `y` to train an XGBoost regression model.
- Learn patterns between car configurations and test bench time.
- Use appropriate validation methods to evaluate model performance.

### 6. Predict Test Set Values

- Apply the same preprocessing and dimensionality reduction steps to the test dataset.
- Generate predictions for the target variable `y` using the trained XGBoost regressor.
- Store the predicted values for further analysis or submission.

## Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

## Machine Learning Workflow

1. Load the training and test datasets.
2. Explore the data and inspect null and unique values.
3. Remove zero-variance features.
4. Encode categorical variables using label encoding.
5. Apply PCA for dimensionality reduction.
6. Train and validate an XGBoost regression model.
7. Predict the target values for the test dataset.
8. Save the predictions for further evaluation or submission.

## Expected Outcome

The expected outcome is a machine learning model that predicts Mercedes-Benz vehicle test bench time based on vehicle configuration features. The project also explores whether dimensionality reduction can simplify the feature space while retaining useful information for prediction.