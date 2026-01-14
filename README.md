# Stock Price Prediction using Machine Learning

This project predicts **next-day stock price movement** of Tesla using supervised machine learning techniques in Python. The task is formulated as a **binary classification problem**, where the model predicts whether the closing price will go up or down compared to the previous day.

## Dataset

* Historical Tesla stock price data (`Tesla.csv`)
* Features include Open, High, Low, Close, and Volume

## Exploratory Data Analysis (EDA)

* Line plots to visualize price trends
* Distribution and box plots to analyze feature spread and outliers
* Heatmaps to study feature correlations
* Year-wise and quarter-wise price behavior analysis

## Feature Engineering

* Date decomposition into Day, Month, and Year
* Derived features:

  * `open-close`
  * `low-high`
  * `isQuarterMonth`
* Binary target variable indicating next-day price increase

## Data Preprocessing

* Feature scaling using `StandardScaler`
* Train–validation split (90% / 10%) with reproducibility

## Models Used

* Logistic Regression
* Support Vector Classifier (Polynomial Kernel)
* XGBoost Classifier

## Evaluation Metric

* ROC-AUC score used to evaluate training and validation performance

## Technologies

* Python, NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost

## Conclusion

The project demonstrates how feature engineering and model selection impact the performance of machine learning models in financial time-series classification.
