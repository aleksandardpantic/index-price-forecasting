import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

seq_len = 12

data = pd.read_csv('data/processed/csv/input.csv')
data['Date'] = pd.to_datetime(data['Date'])
data['covid_flag'] = data['Date'].apply(lambda x: 1 if x<=pd.Timestamp(year=2020, month=6, day=1) and x>=pd.Timestamp(year=2020, month=2, day=1) else 0)
data.set_index('Date', inplace=True)

def get_log_returns(close: pd.Series) -> pd.Series:
    log_returns = np.empty(len(close),dtype=np.float32)
    log_returns[0] = np.NaN
    for i in range(1, len(close)):
        log_returns[i] = np.log(close[i]/close[i-1])
    
    assert len(log_returns) == len(close)
    return pd.Series(log_returns)

data.drop(columns=['Close', 'Open', 'EMA12', 'EMA3', 'STOCH'], inplace=True)
data['t'] = np.arange(len(data))

test_end_date = max(data.index)
test_start_date = test_end_date - timedelta(days = 730)

train_data: pd.DataFrame = data.loc[:test_start_date, ]
test_data: pd.DataFrame = data.loc[test_start_date:test_end_date,]

train_data.set_index('t', inplace=True)
test_data.set_index('t', inplace=True)

test_data.reset_index(drop=True,inplace=True)

train_data['Adj Close'] = get_log_returns(train_data['Adj Close'])
train_data['High'] = get_log_returns(train_data['High'])
train_data['Low'] = get_log_returns(train_data['Low'])

test_data['Adj Close'] = get_log_returns(test_data['Adj Close'])
test_data['High'] = get_log_returns(test_data['High'])
test_data['Low'] = get_log_returns(test_data['Low'])

robust_scaler = RobustScaler()
min_max_scaler = MinMaxScaler()

train_data[['Volume', 'VWMA']] = robust_scaler.fit_transform(train_data[['Volume', 'VWMA']])
train_data[['VIX', 'MACD']] = min_max_scaler.fit_transform(train_data[['VIX', 'MACD']])

test_data[['Volume', 'VWMA']] = robust_scaler.transform(test_data[['Volume', 'VWMA']])
test_data[['VIX', 'MACD']] = min_max_scaler.transform(test_data[['VIX', 'MACD']])

with open("data/processed/bin/scalers/robust_scaler.pkl", 'wb') as p:
    pickle.dump(robust_scaler, p)


with open("data/processed/bin/scalers/min_max_scaler.pkl", 'wb') as p:
    pickle.dump(min_max_scaler, p)

test_data.dropna(axis=0, inplace=True)
train_data.dropna(axis=0, inplace=True)

train_data.to_csv('data/processed/csv/train_data.csv')
test_data.to_csv('data/processed/csv/test_data.csv')

train_target = train_data['Adj Close']
train_data = train_data.copy()

test_target = test_data['Adj Close']
test_data = test_data.copy()

rows = len(train_data)
X_train = []
y_train = []

for i in range(seq_len, rows):
    X_train.append(train_data.iloc[i-seq_len:i, :])
    y_train.append(train_target.iloc[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

rows = len(test_data)
seq_len = 12
X_test = []
y_test = []

for i in range(seq_len, rows):
    X_test.append(test_data.iloc[i-seq_len:i, :])
    y_test.append(test_target.iloc[i])

X_test = np.array(X_test)
y_test = np.array(y_test)

with open("data/processed/bin/sequences/train/X_train.pkl", 'wb') as p:
    pickle.dump(X_train, p)

with open("data/processed/bin/sequences/train/y_train.pkl", 'wb') as p:
    pickle.dump(y_train, p)


with open("data/processed/bin/sequences/test/X_test.pkl", 'wb') as p:
    pickle.dump(X_test, p)

with open("data/processed/bin/sequences/test/y_test.pkl", 'wb') as p:
    pickle.dump(y_test, p)


