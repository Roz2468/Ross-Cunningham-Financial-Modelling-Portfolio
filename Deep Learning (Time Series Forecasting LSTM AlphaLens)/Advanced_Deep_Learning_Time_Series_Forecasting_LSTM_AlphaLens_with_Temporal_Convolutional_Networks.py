# Ross E Cunningham
# Advanced Deep Learning Time Series Forecasting LSTM AlphaLens with Temporal Convolutional Networks

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import alphalens as al

# Step 1: Download Financial Data
# We will use Apple (AAPL) stock data as an example
ticker = 'AAPL'
data = yf.download(ticker, start='2015-01-01', end='2023-01-01')[['Close']]

# Step 2: Data Preprocessing
# Normalize the data for better training stability
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Function to create time series sequences for LSTM
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Define time step window
time_steps = 60
X, y = create_sequences(data_scaled, time_steps)

# Split data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 3: Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_steps, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the LSTM Model
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Step 5: Evaluate the Model
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Convert back to original scale
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 6: Use AlphaLens for Factor Performance Evaluation
# Simulating a factor (e.g., momentum)
data['Returns'] = data['Close'].pct_change()
data['Momentum'] = data['Close'].pct_change(periods=5)

factor_data = data[['Momentum']].dropna()
pricing_data = data[['Close']]

# Prepare AlphaLens input
factor_data['factor'] = factor_data['Momentum']
factor_data = factor_data.drop(columns=['Momentum'])

# Ensure proper index formatting
factor_data.index = pd.MultiIndex.from_tuples(
    [(idx, ticker) for idx in factor_data.index], names=['date', 'asset']
)

# Convert pricing data to returns and rename column to match asset name
pricing_data = pricing_data.pct_change().dropna()
pricing_data.columns = [ticker]  # Ensure it matches the asset column in factor_data

# Ensure indices are properly formatted
pricing_data.index = pd.to_datetime(pricing_data.index)

# Fix multi-index levels properly
factor_data.index = factor_data.index.set_levels(
    [pd.to_datetime(factor_data.index.levels[0]), factor_data.index.levels[1]], level=[0, 1]
)

# Debugging: Print index structure to verify consistency
print("Factor Data Index:", factor_data.index.names)
print("Factor Data Sample:\n", factor_data.head())
print("Pricing Data Columns:", pricing_data.columns)
print("Pricing Data Sample:\n", pricing_data.head())

# Ensure no missing values
factor_data.dropna(inplace=True)
pricing_data.dropna(inplace=True)

# Ensure pricing data aligns with factor data
pricing_data = pricing_data.loc[factor_data.index.get_level_values('date')]

# Use only 1 forward return period
try:
    factor_data = al.utils.get_clean_factor_and_forward_returns(
        factor_data, pricing_data, periods=[1], filter_zscore=3
    )
    al.tears.create_full_tear_sheet(factor_data)
except Exception as e:
    print(f"Error in AlphaLens: {str(e)}. Check your pricing data range or forward return periods.")


