import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Define the ticker for the US 10-year Treasury yield
ticker = "^TNX"  # Yahoo Finance ticker for the 10-year US Treasury yield

# Fetch historical data (1 month of daily data for better visualization)
data = yf.download(ticker, period="1mo", interval="1d")

# Ensure we have valid data
if not data.empty:
    # Extract the 'Close' prices (which represent the yield values)
    data['Yield'] = data['Close'] / 100  # TNX is reported as percentage, so divide by 100

    # Plot the yield curve
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Yield'], label="US 10-Year Treasury Yield", color="blue", marker="o")

    # Customize the plot
    plt.title("US 10-Year Treasury Yield Curve (Last Month)", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Yield (%)", fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

else:
    print("Failed to retrieve data for the US 10-year Treasury yield.")
