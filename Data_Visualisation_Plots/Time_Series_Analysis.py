#Time Series Analysis 
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch historical stock data
stock = yf.Ticker("AAPL")  # Example: Apple Inc.
data = stock.history(period="1y")  # Last 1 year

# Plot the closing price
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Closing Price', color='blue')
plt.title('AAPL Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
