#Ross E Cunningham
"""
Stock Market Movement Prediction using Random Forest
-----------------------------------------------------
This script downloads historical stock data, calculates financial indicators, 
and trains a Random Forest model to predict whether the stock market will move 
up or down on a given day.

### Step-by-Step Breakdown:

1. **Import Required Libraries**  
   - `yfinance`: Fetches historical stock price data.  
   - `pandas`: Handles data processing and manipulation.  
   - `numpy`: Supports numerical computations.  
   - `matplotlib.pyplot`: Used for visualization.  
   - `sklearn.ensemble.RandomForestClassifier`: Machine learning model for classification.  
   - `sklearn.model_selection.TimeSeriesSplit`: Ensures correct time-based training/testing data split.  
   - `sklearn.metrics.classification_report`: Evaluates model performance.  
   - `sklearn.preprocessing.StandardScaler`: Standardizes features for better model performance.  

2. **Fetch Historical Stock Data (`yfinance.download`)**  
   - Downloads **closing prices** of selected stocks (NVDA, AAPL, etc.) from 2018 to 2023.  
   - **Why Closing Prices?** These are widely used in financial modeling as they reflect final market sentiment for the day.  
   - Ensures proper column structure to avoid MultiIndex issues (`data.columns.get_level_values(0)`).  
   - Checks if data is downloaded correctly (`if data.empty: raise ValueError(...)`).

3. **Calculate Returns and Financial Indicators**  
   - Computes **daily percentage returns** using `pct_change()`.  
   - Keeps **individual stock returns** as features instead of averaging all returns.  
   - Adds **technical indicators** to improve prediction:  
     - **20-day Moving Average (`rolling(window=10).mean()`)**: Smoothed trend indicator.  
     - **Volatility (`rolling(window=10).std()`)**: Measures risk based on past fluctuations.  
   - Defines **target variable (`target`)**:  
     - `1` if the average return of all stocks is **positive** (market went up).  
     - `0` otherwise (market went down).  
   - Ensures data integrity after processing (`if stock_data.empty: raise ValueError(...)`).

4. **Feature Selection and Data Splitting**  
   - Selects key input features:  
     - **Individual stock returns** (e.g., `return_NVDA`, `return_AAPL`, etc.).  
     - **Moving Average** and **Volatility**.  
   - Defines **X (input features)** and **y (target variable)**.  
   - Uses `TimeSeriesSplit(n_splits=5)`:  
     - Ensures past data is used to predict future data.  
     - **Why not `train_test_split`?** Traditional random splitting would leak future information into training data.  
   - Uses the **last split** for final training and testing.

5. **Data Normalization (`StandardScaler`)**  
   - Standardizes `X_train` and `X_test` using `StandardScaler()`:  
     - **Why?** Ensures numerical stability and improves model performance.  
     - Uses `fit_transform()` on training data and `transform()` on test data **to prevent data leakage**.

6. **Train Random Forest Model (`RandomForestClassifier`)**  
   - `n_estimators=100`: Uses 100 decision trees in the ensemble.  
   - `random_state=42`: Ensures reproducibility of results.  
   - `class_weight='balanced'`: Handles class imbalances by adjusting sample weights.  
   - **Why Random Forest?**  
     - Handles non-linear relationships in stock data.  
     - Reduces overfitting by averaging multiple decision trees.

7. **Make Predictions & Evaluate Performance**  
   - Predicts stock market movement (`rf_model.predict(X_test_scaled)`).  
   - Evaluates using:  
     - **Accuracy Score** (`accuracy_score(y_test, y_pred)`).  
     - **Classification Report** (precision, recall, F1-score for up/down movements).  

8. **Feature Importance Analysis (`rf_model.feature_importances_`)**  
   - Determines which features (returns, moving average, volatility) influence predictions the most.  
   - Normalizes importance values and plots a **bar chart** to visualize key predictors.  

9. **Plot Actual vs. Predicted Stock Movements**  
   - Creates a `DataFrame` comparing actual vs. predicted market movements.  
   - Uses `plt.plot()` to visualize how well predictions align with reality.  

### Key Considerations:
- **Market data is noisy**: Predictions will never be 100% accurate.  
- **Random Forest is not time-aware**: It treats each day as independent, unlike LSTMs.  
- **Avoid overfitting**: Too many features or too complex a model can reduce generalizability.  
- **Backtesting and validation** are essential before using predictions in real trading.  
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Step 1: Fetch historical stock data using yfinance
# -----------------------------------
tickers = ['NVDA', 'AVGO', 'META', 'GOOGL', 'AAPL', 'INTC']

data = yf.download(tickers, start="2018-01-01", end="2023-01-01")["Close"]

# Reset column structure to avoid multi-index issues
data.columns = data.columns.get_level_values(0)

# Check if data is downloaded properly
if data.empty:
    raise ValueError("Downloaded stock data is empty. Check the ticker symbols or internet connection.")
print(data.head())

# Step 2: Calculate returns and features
# -----------------------------------
stock_data = data.pct_change().dropna()

# Keep individual stock returns as features instead of averaging
for ticker in tickers:
    stock_data[f'return_{ticker}'] = stock_data[ticker]

# Add technical indicators with a reduced rolling window to avoid excessive NaNs
stock_data['20_day_ma'] = stock_data.mean(axis=1).rolling(window=10).mean()
stock_data['volatility'] = stock_data.mean(axis=1).rolling(window=10).std()

# Generate target variable: 1 if average return > 0, else 0
stock_data['target'] = np.where(stock_data.mean(axis=1) > 0, 1, 0)
stock_data = stock_data.dropna()

# Check if data is still available after processing
if stock_data.empty:
    raise ValueError("Stock data is empty after preprocessing. Check rolling window size and missing data handling.")
print(stock_data.shape)
print(stock_data.head())

# Step 3: Feature selection and data splitting
# -----------------------------------
features = [f'return_{ticker}' for ticker in tickers] + ['20_day_ma', 'volatility']
X = stock_data[features]
y = stock_data['target']

# Ensure sufficient data for TimeSeriesSplit
if len(stock_data) < 10:
    raise ValueError("Not enough data after preprocessing. Reduce rolling window size or check data availability.")

# Use TimeSeriesSplit instead of train_test_split
tscv = TimeSeriesSplit(n_splits=5)
train_indices, test_indices = list(tscv.split(X))[-1]  # Use last split for final training/testing
X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

# Step 4: Data normalization (Avoiding Data Leakage)
# -----------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Build Random Forest model
# -----------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

# Step 6: Make predictions
# -----------------------------------
y_pred = rf_model.predict(X_test_scaled)

# Step 7: Evaluate performance
# -----------------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Step 8: Visualize feature importance
# -----------------------------------
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
importances /= importances.max()  # Normalize importance values

plt.figure(figsize=(8, 6))
plt.title("Feature Importances")
plt.bar(range(len(features)), importances[indices], align="center")
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

# Step 9: Plot Actual vs Predicted stock movements
# -----------------------------------
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred}, index=X_test.index)
plt.figure(figsize=(12, 6))
plt.plot(results.index, results['Actual'], label='Actual', color='blue')
plt.plot(results.index, results['Predicted'], label='Predicted', color='red', linestyle='--')
plt.title('Actual vs Predicted Stock Movements')
plt.xlabel('Date')
plt.ylabel('Movement (Up/Down)')
plt.legend()
plt.grid(True)
plt.show()
