# Stock-Prediction-Project
First ML Project attempting to predict next S&amp;P 500 index close

# Description
* This project uses yfinance library to grab historical S&P price and VIX data from 2000.
* The target variable is the "Next Close", which we try to predict next day's close from the latest close data

# Features Implemented
* Simple Moving Averages from 3,5,7 and 20 days
* Exponential Moving Averages from 9,12 and 26 days.
* Bollinger Upper and Lower bands
* Moving Average Convergence Divergence signals
* Relative Strength Index

# Time Series Splitting
Unlike a typical train test split with time-independent dataset, we have to implement time series splitting in order to test the model's performance against multiple folds. For the final model, we train it until Jan-01 2020 and test the model's performance every day afterward.

# Models Implemented
We test model performance by Mean Absolute Error scores

Linear Regression - 251

Random Forest - 20.46

XGBOOST - 11.49

Random Forest/XGBOOST performed significantly better than Linear Regression and was used as the final model

# Parameter Tuning
Tuned parameters: max_depth, learning_rate and min_child_weight and selected which combination had the lowest MAE



