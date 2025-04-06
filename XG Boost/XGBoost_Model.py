import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import ta  # For technical indicators
from fredapi import Fred  # For macroeconomic data
import xgboost as xgb  # For XGBoost model

# Step 1: Fetch stock data using yfinance
tickers = ['NVDA', 'AVGO', 'META', 'GOOGL', 'AAPL', 'INTC']
data = yf.download(tickers, start="2018-01-01", end="2024-09-01")['Close']
print(data.head())

# Step 2: Compute stock returns and initialize stock_data
stock_data = pd.DataFrame()
returns = data.pct_change().dropna()
stock_data['returns'] = returns.mean(axis=1)

# Step 3: Add technical indicators
rsi = ta.momentum.RSIIndicator(stock_data['returns'], window=14)
stock_data['rsi'] = rsi.rsi()

bollinger = ta.volatility.BollingerBands(stock_data['returns'], window=20, window_dev=2)
stock_data['bb_upper'] = bollinger.bollinger_hband()
stock_data['bb_lower'] = bollinger.bollinger_lband()
stock_data['bb_middle'] = bollinger.bollinger_mavg()

macd = ta.trend.MACD(stock_data['returns'])
stock_data['macd'] = macd.macd()
stock_data['macd_signal'] = macd.macd_signal()

# Step 4: Fetch macroeconomic data using FRED API
fred = Fred(api_key="612d70d35cf1dbc9f784bd2826dde2d9")

def fetch_fred_series(series_id, start_date, end_date):
    """Fetch data from FRED API with error handling."""
    try:
        return fred.get_series(series_id, start_date, end_date)
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return pd.Series(np.nan, index=pd.date_range(start=start_date, end=end_date))

# Retrieve macroeconomic indicators
ten_year_yield = fetch_fred_series('DGS10', "2018-01-01", "2024-09-01")
stock_data['10yr_yield'] = ten_year_yield

cpi = fetch_fred_series('CPIAUCSL', "2018-01-01", "2024-09-01").pct_change().dropna()
stock_data['cpi'] = cpi.reindex(stock_data.index, method='ffill')

gdp = fetch_fred_series('GDP', "2018-01-01", "2024-09-01").pct_change().dropna()
stock_data['gdp'] = gdp.reindex(stock_data.index, method='ffill')

# Step 5: Generate target variable (1 for up, 0 for down)
stock_data['target'] = np.where(stock_data['returns'] > 0, 1, 0)

# Drop NaN values created by rolling and macro data
stock_data = stock_data.dropna()

# Step 6: Feature selection and data splitting
features = ['returns', 'rsi', 'bb_upper', 'bb_lower', 'bb_middle', 'macd', 'macd_signal', '10yr_yield', 'cpi', 'gdp']
X = stock_data[features]
y = stock_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 7: Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train XGBoost Model
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Step 9: Make predictions and evaluate the model
y_pred_xgb = xgb_model.predict(X_test_scaled)

accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Model Accuracy: {accuracy_xgb:.2f}")
print(classification_report(y_test, y_pred_xgb))

# Step 10: Visualize Feature Importance
xgb_importances = xgb_model.feature_importances_
indices = np.argsort(xgb_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("XGBoost Feature Importances")
plt.bar(range(X_train.shape[1]), xgb_importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

# Step 11: Plot Actual vs Predicted Stock Movement
results_xgb = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_xgb}, index=stock_data.index[-len(y_test):])

plt.figure(figsize=(12, 6))
plt.plot(results_xgb.index, results_xgb['Actual'], label='Actual', color='blue')
plt.plot(results_xgb.index, results_xgb['Predicted'], label='Predicted', color='red', linestyle='--')
plt.title('Actual vs Predicted Stock Movements (XGBoost)')
plt.xlabel('Date')
plt.ylabel('Movement (Up/Down)')
plt.legend()
plt.grid(True)
plt.show()
