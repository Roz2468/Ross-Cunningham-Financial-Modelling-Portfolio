#Ross Ewan Cunningham
"""
Stock Market Prediction using Machine Learning
------------------------------------------------
This script uses historical stock data to train a machine learning model that predicts 
whether the stock market will move up or down on a given day. It includes both a 
Random Forest Classifier and a Long Short-Term Memory (LSTM) neural network for time-series forecasting.

### Step-by-Step Breakdown:

1. **Fetch Stock Data**  
   - Downloads historical stock prices for major tech companies (NVDA, AAPL, etc.).  
   - Extracts closing prices over a 5-year period.

2. **Calculate Returns & Indicators**  
   - Computes daily percentage returns for each stock.  
   - Adds technical indicators:  
     - **20-day moving average** (trend indicator).  
     - **Volatility** (measures stock fluctuations).

3. **Define Prediction Target**  
   - Labels each day as **1 (up) or 0 (down)** based on whether the average return of all stocks is positive.

4. **Prepare Data for Machine Learning**  
   - Splits data into training and testing sets (using time-based separation).  
   - Standardizes the features to ensure fair comparisons.  
   - **For LSTM models**, data is reshaped into sequences to capture temporal dependencies.

5. **Train Random Forest Model**  
   - Uses a **Random Forest Classifier**, a decision-tree-based algorithm.  
   - Learns patterns from past stock movements.

6. **Prepare Data for LSTM Model**  
   - The function `prepare_data(data, time_step=60)` transforms stock prices into sequences for LSTM training:  
     - `time_step=60`: Uses the past 60 days of stock prices to predict the next day's price.  
     - `MinMaxScaler(feature_range=(0,1))`: Normalizes data between 0 and 1 for better neural network performance.  
     - `X`: Contains sequences of stock prices over the last 60 days.  
     - `y`: Contains the corresponding next day's stock price.

7. **Build LSTM Model**  
   - The function `build_lstm_model(input_shape)` creates an LSTM neural network:  
     - **LSTM(50, return_sequences=True)**: First LSTM layer with 50 units, returning sequences to the next LSTM layer.  
     - **Dropout(0.2)**: Regularization to prevent overfitting.  
     - **LSTM(50, return_sequences=False)**: Second LSTM layer with 50 units, not returning sequences.  
     - **Dense(25)**: Fully connected layer with 25 neurons.  
     - **Dense(1)**: Output layer predicting a single stock price value.  
     - **Adam Optimizer & Mean Squared Error Loss**: Optimizes weights to minimize prediction errors.

8. **Make Predictions & Evaluate Models**  
   - The trained models predict stock movements or prices on unseen test data.  
   - Performance is measured using accuracy (Random Forest) and Mean Squared Error (LSTM).

9. **Analyze Feature Importance**  
   - Determines which factors (returns, moving average, volatility) influence predictions the most.

10. **Visualize Results**  
   - Plots actual vs. predicted stock movements to assess model performance.

### Key Considerations:
- Machine learning finds patterns but does not guarantee future accuracy.  
- Unexpected market events can disrupt predictions.  
- Feature selection and proper data scaling impact model quality.  
"""
 

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    return stock[['Close']]

# Prepare data for LSTM
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(X), np.array(y), scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main script
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-01-01'
data = get_stock_data(ticker, start_date, end_date)
X, y, scaler = prepare_data(data.values)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into training and testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the model
model = build_lstm_model((X_train.shape[1], 1))
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predict
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted.reshape(-1, 1))

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size+60:], data.values[train_size+60:], label='Actual Prices')
plt.plot(data.index[train_size+60:], predicted, label='Predicted Prices')
plt.legend()
plt.show()
