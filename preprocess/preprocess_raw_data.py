import pandas as pd
import pandas_ta as ta

spx = pd.read_csv('data/raw/csv/^SPX.csv')
csi = pd.read_csv('data/raw/csv/CSI.csv', sep=';')
fed_rate = pd.read_csv('data/raw/csv/FEDFUNDS.csv')
gdp = pd.read_csv('data/raw/csv/GDPC1.csv')
unemp = pd.read_csv('data/raw/csv/UNRATE.csv')
vix = pd.read_csv('data/raw/csv/VIX.csv', sep=';')

data = pd.merge(spx, unemp, how='inner', on = 'Date')
data['UNRATE'] = data['UNRATE'].apply(lambda x: x/100)

data = pd.merge(data, fed_rate, how='inner', on='Date')
data['FEDFUNDS'] = data['FEDFUNDS'].apply(lambda x:x/100)

start = min(gdp['Date'])
end = max(gdp['Date'])
date_list = pd.date_range(start,end, freq = 'MS', inclusive='both').to_numpy()
gdp['Date'] = gdp['Date'].astype('datetime64[s]')
fill = pd.DataFrame({'Date': date_list})
fill = pd.merge(fill, gdp, how='left')
fill['GDPC1'] = fill['GDPC1'].interpolate()
fill['GDP_GROWTH'] = (fill['GDPC1'] - fill['GDPC1'].shift(12)) / fill['GDPC1'].shift(12)
fill.drop(labels='GDPC1', axis='columns', inplace=True)
data['Date'] = data['Date'].astype('datetime64[s]')
data = pd.merge(data, fill, how='inner', on='Date')

csi['Date'] = csi['Date'].astype('datetime64[s]')
data = pd.merge(data, csi, how='inner', on='Date')
data['CSI'] = data['CSI'].apply(lambda x:x/100)

start = min(vix['Date'])
end = max(vix['Date'])
vix['Date'] = vix['Date'].astype('datetime64[s]')
data['Date'] = data['Date'].astype('datetime64[s]')
date_list = pd.date_range(start,end, freq = 'D', inclusive='both').to_numpy()
fill = pd.DataFrame({'Date': date_list})
fill = pd.merge(fill, vix, how='left', on='Date')

fill['VIX'] = fill['VIX'].bfill(axis=0)
data = pd.merge(data, fill, how='inner', on = 'Date')

data['EMA12'] = ta.ema(data['Adj Close'], 12)
data['EMA3'] = ta.ema(data['Adj Close'], 3)
data['MACD'] = data['EMA3'] - data['EMA12']
data['VWMA'] = ta.vwma(data['Adj Close'], data['Volume'], 12)
data['RSI'] = ta.rsi(data['Adj Close'], 12)
data['STOCH'] = ta.stoch(data['High'], data['Low'], data['Adj Close'], 9, 3)['STOCHd_9_3_3']
data['STOCH'] = data['STOCH'].apply(lambda x: x/100)
data['RSI'] = data['RSI'].apply(lambda x: x/100)

data = data.dropna(axis=0)
data = data.set_index('Date')

data.to_csv('data/processed/csv/input.csv')


