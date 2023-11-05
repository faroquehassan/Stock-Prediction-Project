
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb

#prep

symbol = '^GSPC'
df = yf.download(symbol, start="2000-01-01")
df['Next_Close'] = df['Close'].shift(-1)

df_vix = yf.download('^VIX', start= "2000-01-01")
df_vix.rename(columns={'Close':'vix'},inplace=True)
df = pd.merge(df,df_vix['vix'],on='Date')

for i in (3, 5, 7):
    df[f'sma_{i}'] = df['Close'].rolling(i).mean()

for i in (9,12, 26):
    df[f'ema_{i}'] = df['Close'].ewm(span=i).mean()
    
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span = 9).mean()
df['macd_histogram'] = df['macd'] - df['macd_signal']

def calculate_rsi(df):
    close_diff = df['Close'].diff(1)
    gain = close_diff.where(close_diff > 0, 0)
    loss = -close_diff.where(close_diff < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['rsi'] = calculate_rsi(df)

features = ['sma_3', 'sma_5','sma_7',
    'ema_9', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_histogram',
    'rsi','Volume','vix']

target = 'Next_Close'

X_test = df.iloc[[-1]][features]

df.dropna(inplace=True)
X_train = df[features]
X_train.dropna(inplace=True)
y_train = df[target]
y_train.dropna(inplace=True)

model = xgb.XGBRegressor(booster='gbtree',
                        n_estimators=1000,
                        objective='reg:squarederror',
                        max_depth=3,
                        learning_rate=0.1,
                        min_child_weight = 5,
                        n_jobs = -1)
model.fit(X_train, y_train)

with open('model.bin', 'wb') as f_out:
   pickle.dump((model), f_out)
f_out.close()

with open('X_test.pkl', 'wb') as file:
    pickle.dump(X_test,file)
file.close()