import seaborn as sns
#Time Series Analysis 
import yfinance as yf
import matplotlib.pyplot as plt


# Fetch data for multiple stocks
stocks = ['AAPL', 'MSFT', 'GOOGL']
data = yf.download(stocks, period="1y")['Close']

# Calculate daily returns
returns = data.pct_change()

# Calculate correlation matrix
corr_matrix = returns.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Stock Returns')
plt.show()
