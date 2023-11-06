# Stock-Prediction-Project
First ML Project attempting to predict next S&amp;P 500 index close

# Description
* This project uses yfinance library to grab historical S&P price and VIX data from 2000.
* The target variable is the "Next Close", which we try to predict next day's close from the latest close data

# Features Implemented
* Simple Moving Averages from 3,5,7 and 20 days. SMA gives equal weight to the rolling days and thus captures longer trends and eliminates short-term noise. 
* Exponential Moving Averages from 9,12 and 26 days. EMA gives more weight to recent data and is much quicker to react to short-term trends.
* Bollinger Upper and Lower bands. These are computed by finding the 2 Standard Deviation range of SMA 20 days. When the price crosses the upper band, it is considered to be overbought. When the price crosses the lower band, it is considered to be oversold.
* Moving Average Convergence Divergence signals. Divergence from the MACD can indicate potential reversals. MACD can also confirm volatility changes.
* Relative Strength Index. By calculating the average 1-day returns over a 14 day period, RSI can help identify overbought and oversold conditions.
* VIX. The Volatility Index is calculated by the implied volatility of S&P500 index options. VIX usually has an inverse relationship with the stock market and can help predict directional moves. 

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

# Using the Model
1) Use Pipenv to install the included Piplock file
2) Run train.py to generate the updated Model.bin trained on the latest data
3) Build using Dockerfile by running "docker build -t model ." in the folder directory
4) Run the model using "docker run -it --rm -p 8000:8000 model"
5) Connect via http://localhost:8000/predict


