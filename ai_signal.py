import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier

def load_data(pair="EURUSD=X", interval="1h", period="60d"):
    df = yf.download(pair, interval=interval, period=period)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df

def add_features(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bollinger_h'] = bb.bollinger_hband()
    df['bollinger_l'] = bb.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['roc'] = ta.momentum.ROCIndicator(df['Close']).roc()
    df.dropna(inplace=True)
    return df

def add_target(df, future_period=5, pip_threshold=0.0015):
    df['future_return'] = df['Close'].shift(-future_period) - df['Close']
    df['target'] = (df['future_return'] > pip_threshold).astype(int)
    df.dropna(inplace=True)
    return df

def train_model(df):
    X = df[['rsi', 'macd', 'bollinger_h', 'bollinger_l', 'atr', 'roc']]
    y = df['target']
    model = RandomForestClassifier(n_estimators=200, max_depth=6)
    model.fit(X, y)
    return model

def get_latest_signal(model, df):
    X_latest = df[['rsi', 'macd', 'bollinger_h', 'bollinger_l', 'atr', 'roc']].iloc[-1:]
    prediction = model.predict(X_latest)[0]
    return "BUY" if prediction == 1 else "SELL"
