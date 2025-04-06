#Time Series Analysis 
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch historical stock data
stock = yf.Ticker("AAPL")  # Example: Apple Inc.
data = stock.history(period="1y")  # Last 1 year
# Calculate daily returns
data['Returns'] = data['Close'].pct_change()

# Plot histogram of returns
plt.figure(figsize=(12, 6))
plt.hist(data['Returns'].dropna(), bins=50, edgecolor='black', color='skyblue')
plt.title('Histogram of AAPL Returns')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
