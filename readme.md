# Multi-Class Prediction of Obesity Risk

This project notebook leverages machine learning models to classify individuals based on obesity risk levels. The classification is built upon a Kaggle dataset containing various health and lifestyle attributes.

## Dataset

- Source: Obesity or CVD Risk - Classify/Regressor/Cluster Dataset: https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster
- Attributes:
    - Eating habits: Frequency of high-calorie food consumption (FAVC), vegetable consumption (FCVC), meal frequency (NCP), snacking habits (CAEC), water intake (CH20), alcohol consumption (CALC).
    - Physical condition: Calories monitoring (SCC), physical activity frequency (FAF), technology use time (TUE), transportation type (MTRANS).
    - Demographic data: Gender, age, height, and weight.
- Target Variable: Obesity levels defined as follows:
    -Underweight (<18.5 BMI)
    -Normal (18.5–24.9 BMI)
    -Overweight (25.0–29.9 BMI)
    -Obesity I (30.0–34.9 BMI)
    -Obesity II (35.0–39.9 BMI)
    -Obesity III (>40 BMI)

## Project Workflow
1. Data Exploration:

    - Basic statistics and data insights (e.g., mean, median).
    - Data quality checks, including handling missing values and data types.

2. Data Visualization:

    - Visualization of key features to reveal potential patterns, correlations, and class imbalances.
    - Use of histograms, scatter plots, and correlation heatmaps to illustrate relationships between health metrics and risk factors.

3. Data Manipulation:

    - Data distribution and outliers detection.
    - Data transformations manually to reduce ouliers and improve subsequent interpretation by the models.

4. Feature Engineering:
    - Encoding categorical features and scaling numerical features using scikit-learn
    - Feature importance analysis to understand the impact of each variable on the predictions (Importance permutation).

5. Modeling:
   
    - Various machine learning models are evaluated, including:
        - XGBoost
        - LightGBM
        - Deep Learning Model with TensorFlow and Keras
    - Hyperparameter tuning is performed with Optuna for optimization.
    - Probability thresholding is applied to optimize classification outcomes, adjusting the threshold for each class to improve sensitivity or specificity depending on the model's predictions.

6. Model Evaluation:

    - Evaluation metrics such as accuracy and AUC are used.
    - ROC curve analysis to assess the model performance.
    

5. Feature Importance:

    - 

## Requirements
Python (>= 3.7)
Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, lightgbm, tensorflow, optuna & keras_tuner
