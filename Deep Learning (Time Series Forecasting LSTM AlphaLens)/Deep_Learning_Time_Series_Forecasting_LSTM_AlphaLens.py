#Ross E Cunningham
#Deep Learning Timeseries Forecasting Using LTSM and ALphaLens

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
data['Returns'] = data['Adj Close'].pct_change()
data['Momentum'] = data['Adj Close'].pct_change(periods=5)

factor_data = data[['Momentum']].dropna()
pricing_data = data[['Adj Close']]

# Prepare AlphaLens input
factor_data['factor'] = factor_data['Momentum']
factor_data['date'] = factor_data.index
factor_data['asset'] = ticker
factor_data = factor_data.drop(columns=['Momentum'])
factor_data.set_index(['date', 'asset'], inplace=True)

# Convert pricing data
tickers = [ticker]
pricing_data = pricing_data.pct_change().dropna()

# Run AlphaLens analysis
factor_data = al.utils.get_clean_factor_and_forward_returns(
    factor_data, pricing_data, periods=(1, 5, 10), filter_zscore=3
)
al.tears.create_full_tear_sheet(factor_data)