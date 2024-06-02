# Big Mart Sales Prediction

This repository contains a project aimed at predicting sales for the Big Mart retail chain using machine learning techniques. By leveraging powerful models such as Random Forest Regressor and XGBRFRegressor, the project uncovers hidden patterns, optimizes inventory, and drives strategic decision-making.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Data Analysis and Preprocessing](#data-analysis-and-preprocessing)
  - [Null Values Handling](#null-values-handling)
  - [Feature Engineering](#feature-engineering)
- [Models and Training](#models-and-training)
  - [Random Forest Regressor](#random-forest-regressor)
  - [XGBRFRegressor](#xgbrfregressor)
- [Results](#results)
- [Conclusion and Recommendations](#conclusion-and-recommendations)
- [Acknowledgements](#acknowledgements)

## Introduction

The Big Mart Sales Prediction project explores the use of machine learning to forecast sales for Big Mart, a retail chain. This analysis aims to uncover hidden patterns, optimize inventory, and inform strategic decision-making.

## Dataset

The dataset for this project can be accessed from the following link:
[Big Mart Sales Dataset](https://drive.google.com/drive/folders/1DtEKw6KDTdI0IPCoqfX4X2Hsi4KimapT?usp=drive_link)

The dataset includes information about various products, sales, and store attributes. The primary target is to predict the sales of products in different stores.

## Project Overview

The project is divided into several key steps:

1. **Data Analysis and Preprocessing**: Understand the dataset, handle missing values, and perform feature engineering.
2. **Model Training**: Train predictive models using Random Forest Regressor and XGBRFRegressor.
3. **Evaluation**: Evaluate the performance of the models using appropriate metrics.
4. **Prediction**: Predict sales values using the trained models.

## Data Analysis and Preprocessing

### Null Values Handling

Handling missing values is crucial for the performance of machine learning models. Several techniques are used to handle missing values:

- **Data Imputation**: Replace missing values with the mean, median, or mode of the feature.
- **Interpolation**: Estimate missing values based on neighboring data points.
- **Advanced Imputation**: Use sophisticated methods like k-nearest neighbors (KNN) or multiple imputation by chained equations (MICE).

### Feature Engineering

Feature engineering involves creating new features and transforming existing ones to improve model performance:

- **Identify Relevant Features**: Determine which features are most relevant for predicting sales.
- **Create New Features**: Generate new features by combining or transforming existing ones.
- **Handle Categorical Variables**: Encode categorical variables using techniques like one-hot encoding or label encoding.

## Models and Training

### Random Forest Regressor

The Random Forest Regressor is a powerful machine learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(scores.mean())
```

### XGBRFRegressor

The XGBRFRegressor combines the strengths of XGBoost and Random Forest, making it robust to outliers and capable of handling non-linear relationships.

```python
from xgboost import XGBRFRegressor

xg = XGBRFRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(xg, X, y, cv=5, scoring='r2')
print(scores.mean())
```

## Results

- **Random Forest Regressor**: The model explains approximately 55.43% of the variance in the target variable.
- **XGBRFRegressor**: The model explains approximately 59.45% of the variance in the target variable.

## Conclusion and Recommendations

### Key Findings

The analysis of Big Mart sales data using Random Forest Regressor and XGBRFRegressor models uncovered valuable insights about the factors driving sales performance.

### Recommendations

Based on the insights gained, it is recommended to focus on optimizing product pricing, store location, and inventory management to boost overall sales revenue.

### Future Outlook

With continued refinement of the machine learning models and the incorporation of additional data sources, the accuracy of sales predictions can be further improved to support more informed business decisions.

## Acknowledgements

Thank you for reviewing this project. It was an insightful experience that enhanced my skills in data analysis and machine learning.

**Author**: Ayush Yadav

---

Feel free to reach out for any questions or further discussions on this project.
