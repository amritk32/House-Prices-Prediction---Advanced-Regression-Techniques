# House Price Prediction using Advanced Ensemble Regression

## Overview

This project is an end-to-end machine learning solution for predicting residential house prices using advanced feature engineering, preprocessing pipelines, and ensemble regression models.

The project was built on the famous Kaggle dataset:

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

The main objective was to build a highly optimized regression pipeline capable of learning complex real-estate pricing patterns while maintaining strong generalization performance.

---

# Problem Statement

Accurately estimating house prices is a challenging regression problem because property value depends on a large combination of:

- Structural features
- Neighborhood characteristics
- House quality
- Age of the property
- Garage and basement details
- Porch and pool features
- Renovation history
- Categorical property attributes

Traditional regression models often struggle to capture nonlinear interactions and feature relationships.

This project solves the problem using:

- Advanced feature engineering
- Log-transform based target stabilization
- Categorical encoding pipelines
- Ensemble learning
- Weighted voting regression

---

# Machine Learning Goal

Predict: SalePrice

using over 70+ numerical and categorical housing attributes

---
**Tech Stack**
Python
NumPy
Pandas
Scikit-Learn
XGBoost
LightGBM
Joblib


**Models Used**


1. Ridge Regression
Used as a strong linear baseline model.
Advantages:
Handles multicollinearity well
Excellent generalization
Performs strongly on engineered tabular datasets

---
2. XGBoost Regressor
Used for nonlinear boosting-based learning.
Advantages:
Captures complex feature interactions
Strong gradient boosting performance
Excellent bias reduction

---
3. LightGBM Regressor
Used for efficient gradient boosting.
Advantages:
Fast training
Leaf-wise boosting strategy
Handles high-dimensional encoded data efficiently

---
5. Voting Ensemble Regressor
Final prediction system combines:
Ridge Regression
XGBoost
LightGBM
using weighted ensemble voting.

---
This improved:
Stability
Generalization
Kaggle ranking performance

---
**Advanced Feature Engineering**


The project includes extensive handcrafted feature engineering.


Area-Based Features


TotalArea


TotalPorch


QualArea


Time-Based Features


GarageAge


Age


RemodAge


Bathroom Engineering


Bath = (
    FullBath
    + 0.5 * HalfBath
    + BsmtFullBath
    + 0.5 * BsmtHalfBath
)

---
**Boolean Features**


HasPool


Log Transformations


Applied on skewed features such as:


LotArea


MiscVal


Bath


MSSubClass


MasVnrArea


This significantly improved model stability and RMSLE score.

---
**Preprocessing Pipeline**


Implemented using:


Pipeline


ColumnTransformer


OneHotEncoder


StandardScaler


The preprocessing was fully automated and integrated into the training workflow.

---
**Evaluation Metrics**


The primary evaluation metric was:


**RMSLE (Root Mean Squared Log Error)**


This metric is ideal for house price prediction because:


Penalizes large percentage errors


Handles skewed target distributions


Reduces impact of extremely expensive houses


Additional metrics:


**Mean Squared Error (MSE)**


**R² Score**

---
**Final Results**
Model - XGBoost | Ridge Regression | LightGBM | Voting Ensemble


RMSLE - 0.1255 | 0.1160 | 0.1306 | 0.1129 


R² Score - 0.882 | 0.948 | 0.843 | 0.946

---
**Kaggle Performance**


Final Rank - 1972 / 5171

---
**Key Learning Outcomes**


This project demonstrates practical understanding of:


Ensemble Learning


Regression Modeling


Feature Engineering


Data Preprocessing


Handling Missing Values


Log Transformations


Production ML Pipelines


Scikit-Learn Pipelines


Model Evaluation


Kaggle-style Optimization

---
**Future Improvements**


Hyperparameter tuning with Optuna


CatBoost integration


Cross-validation ensembling


SHAP explainability


FastAPI deployment


Docker containerization


CI/CD based ML deployment

---
**Author**


Built by Amrit Gorai as part of hands-on Machine Learning and MLOps practice focused on production-ready AI systems.
