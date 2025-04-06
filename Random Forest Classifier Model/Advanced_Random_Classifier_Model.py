#Ross E Cunningham
"""
Stock Market Prediction with Technical & Macroeconomic Indicators
------------------------------------------------------------------
This script uses a Random Forest model to predict stock market movements using 
technical indicators and macroeconomic data.

### Step-by-Step Breakdown:

1. **Import Required Libraries**  
   - `yfinance`: Fetches historical stock price data.  
   - `pandas`: Manages data processing.  
   - `numpy`: Supports numerical computations.  
   - `matplotlib.pyplot`: Plots results and feature importances.  
   - `sklearn.ensemble.RandomForestClassifier`: Implements a machine learning classification model.  
   - `sklearn.model_selection.train_test_split`: Splits data into training and test sets.  
   - `sklearn.metrics.classification_report`: Provides evaluation metrics.  
   - `sklearn.preprocessing.StandardScaler`: Standardizes data for better model performance.  
   - `ta`: Computes technical indicators (RSI, Bollinger Bands, MACD).  
   - `fredapi.Fred`: Retrieves macroeconomic data (interest rates, inflation, GDP).  

2. **Fetch Historical Stock Data (`yfinance.download`)**  
   - Retrieves **closing prices** of selected stocks from 2018 to 2023.  
   - Computes **daily percentage returns** using `pct_change()`, which measures daily price changes as a percentage.  
   - Stores the **average return of all stocks** as a feature to represent market trends.  

3. **Compute Technical Indicators (Using `ta` Library)**  
   - **Relative Strength Index (RSI)** (`window=14`): Measures market momentum, values above 70 indicate overbought conditions, below 30 indicate oversold.  
   - **Bollinger Bands (`window=20, window_dev=2`)**:  
     - `bb_upper`: Upper band, suggests potential overbought conditions.  
     - `bb_lower`: Lower band, suggests potential oversold conditions.  
     - `bb_middle`: Moving average of the returns.  
   - **Moving Average Convergence Divergence (MACD)**:  
     - `macd`: Difference between short-term (12-day) and long-term (26-day) EMAs.  
     - `macd_signal`: 9-day EMA of MACD, used as a signal line for trend reversals.  

4. **Fetch Macroeconomic Data (Using `fredapi.Fred`)**  
   - `DGS10`: **10-Year Treasury Yield**, affects interest rates and market sentiment.  
   - `CPIAUCSL`: **Consumer Price Index (CPI)**, measures inflation (computed as % change).  
   - `GDP`: **Gross Domestic Product (GDP) Growth**, economic growth indicator (computed as % change).  
   - **Data Alignment**:  
     - Uses `reindex(..., method='ffill')` to forward-fill missing macroeconomic values, ensuring proper date alignment.  

5. **Generate Target Variable (`stock_data['target']`)**  
   - Binary classification:  
     - `1` if market returns **> 0** (market up).  
     - `0` otherwise (market down).  
   - Drops NaN values created by rolling calculations and missing macroeconomic data.  

6. **Feature Selection & Data Splitting**  
   - Defines **input features (`X`)**:  
     - Technical indicators (`rsi`, `bb_upper`, `macd`, etc.).  
     - Macroeconomic indicators (`10yr_yield`, `cpi`, `gdp`).  
   - Defines **target variable (`y`)**: Market movement (up or down).  
   - Uses `train_test_split(test_size=0.2, shuffle=False)`:  
     - **Why not shuffle?** Market data follows a time sequence, so we must keep chronological order.  

7. **Data Normalization (`StandardScaler`)**  
   - Standardizes feature values for consistent scaling.  
   - Uses **separate fit/transform for training and test data** to avoid data leakage.  

8. **Train Random Forest Model (`RandomForestClassifier`)**  
   - `n_estimators=100`: Uses 100 decision trees for ensemble learning.  
   - `random_state=42`: Ensures reproducibility of results.  
   - **Why Random Forest?**  
     - Handles both numerical and categorical variables.  
     - Reduces overfitting by aggregating multiple trees.  
     - Works well with noisy financial data.  

9. **Make Predictions & Evaluate Performance**  
   - Uses `rf_model.predict(X_test_scaled)` to classify market movements.  
   - Evaluates performance using:  
     - **Accuracy Score (`accuracy_score`)**: Measures overall prediction correctness.  
     - **Classification Report (`classification_report`)**: Provides precision, recall, and F1-score.  

10. **Feature Importance Analysis (`rf_model.feature_importances_`)**  
    - Identifies which features contribute most to predictions.  
    - **Bar Chart Visualization**: Helps interpret which indicators (technical or macroeconomic) impact market movements the most.  

11. **Plot Actual vs. Predicted Market Movements**  
    - Compares real stock movements vs. model predictions over time.  
    - Uses `plt.plot()` to visualize the model's effectiveness in forecasting trends.  

### Key Considerations:
- **Stock markets are highly unpredictable**: Predictions should not be solely relied upon for trading.  
- **Feature selection is crucial**: Adding irrelevant indicators can reduce model accuracy.  
- **Macroeconomic indicators may have lag effects**: CPI and GDP updates happen less frequently than stock prices.  
- **Time-based splitting prevents look-ahead bias**: Always ensure test data is strictly after training data.  
"""


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import ta  # For technical indicators
from fredapi import Fred  # For macroeconomic data

# Step 1: Fetch stock data using yfinance
tickers = ['NVDA', 'AVGO', 'META', 'GOOGL', 'AAPL', 'INTC']
data = yf.download(tickers, start="2018-01-01", end="2023-01-01")['Close']
print(data.head())

# Step 2: Add technical indicators (RSI, Bollinger Bands, MACD)
# -----------------------------------
stock_data = pd.DataFrame()

# Calculate the mean return of all stocks (as a single "market" feature)
returns = data.pct_change().dropna()
stock_data['returns'] = returns.mean(axis=1)

# RSI (Relative Strength Index)
rsi = ta.momentum.RSIIndicator(stock_data['returns'], window=14)
stock_data['rsi'] = rsi.rsi()

# Bollinger Bands
bollinger = ta.volatility.BollingerBands(stock_data['returns'], window=20, window_dev=2)
stock_data['bb_upper'] = bollinger.bollinger_hband()  # Upper Band
stock_data['bb_lower'] = bollinger.bollinger_lband()  # Lower Band
stock_data['bb_middle'] = bollinger.bollinger_mavg()  # Moving Average Band

# MACD (Moving Average Convergence Divergence)
macd = ta.trend.MACD(stock_data['returns'])
stock_data['macd'] = macd.macd()  # MACD Line
stock_data['macd_signal'] = macd.macd_signal()  # Signal Line

# Step 3: Fetch macroeconomic data using FRED API
# -----------------------------------
fred = Fred(api_key="612d70d35cf1dbc9f784bd2826dde2d9")  # Insert your FRED API key here

# 10-Year Treasury Yield
ten_year_yield = fred.get_series('DGS10', start="2018-01-01", end="2023-01-01")
stock_data['10yr_yield'] = ten_year_yield

# Consumer Price Index (CPI) as a proxy for inflation
cpi = fred.get_series('CPIAUCSL', start="2018-01-01", end="2023-01-01").pct_change().dropna()
stock_data['cpi'] = cpi.reindex(stock_data.index, method='ffill')

# GDP Growth (Quarterly) - Adjust frequency to match stock data
gdp = fred.get_series('GDP', start="2018-01-01", end="2023-01-01").pct_change().dropna()
stock_data['gdp'] = gdp.reindex(stock_data.index, method='ffill')

# Step 4: Generate the target variable (1 for up, 0 for down)
# -----------------------------------
stock_data['target'] = np.where(stock_data['returns'] > 0, 1, 0)

# Drop NaN values created by rolling and macro data
stock_data = stock_data.dropna()

# Step 5: Feature selection and splitting data
# -----------------------------------
# Define features: including technical indicators and macroeconomic variables
features = ['returns', 'rsi', 'bb_upper', 'bb_lower', 'bb_middle', 'macd', 'macd_signal', '10yr_yield', 'cpi', 'gdp']

X = stock_data[features]  # Features (independent variables)
y = stock_data['target']  # Target (dependent variable)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 6: Data normalization
# -----------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Build Random Forest model
# -----------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Step 8: Make predictions and evaluate the model
# -----------------------------------
y_pred = rf_model.predict(X_test_scaled)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification report
print(classification_report(y_test, y_pred))

# Step 9: Feature importance visualization
# -----------------------------------
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

# Step 10: Actual vs Predicted stock movement
# -----------------------------------
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
}, index=stock_data.index[-len(y_test):])

# Plot actual vs predicted movements
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['Actual'], label='Actual', color='blue')
plt.plot(results.index, results['Predicted'], label='Predicted', color='red', linestyle='--')
plt.title('Actual vs Predicted Stock Movements')
plt.xlabel('Date')
plt.ylabel('Movement (Up/Down)')
plt.legend()
plt.grid(True)
plt.show()
