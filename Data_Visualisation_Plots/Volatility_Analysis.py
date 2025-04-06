import numpy as np
#Time Series Analysis 
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch historical stock data
stock = yf.Ticker("AAPL")  # Example: Apple Inc.
data = stock.history(period="1y")  # Last 1 year
# Calculate daily returns
data['Returns'] = data['Close'].pct_change()

# Calculate rolling volatility (standard deviation of returns)
data['Volatility'] = data['Returns'].rolling(window=30).std()  # 30-day window

# Plot volatility
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Volatility'], label='Rolling Volatility', color='red')
plt.title('Rolling Volatility of AAPL')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()
