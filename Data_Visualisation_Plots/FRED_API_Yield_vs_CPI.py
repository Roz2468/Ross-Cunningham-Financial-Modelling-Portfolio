from fredapi import Fred
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Set your FRED API key
fred = Fred(api_key="612d70d35cf1dbc9f784bd2826dde2d9")

# Fetch CPI data from FRED
cpi_data = fred.get_series('CPIAUCSL', start='2023-01-01')  # Example: retrieve from 2023 onwards

# Fetch historical data (last 1 year of daily data for yields)
tickers = {
    "3M Yield": "^IRX",         # US 3-month Treasury Yield
    "2Y Yield": "^FVX",         # US 2-year Treasury Yield
    "5Y Yield": "^FVX",         # US 5-year Treasury Yield
    "10Y Yield": "^TNX",        # US 10-year Treasury Yield
    "30Y Yield": "^TYX"         # US 30-year Treasury Yield
}

# Download Treasury yields data
yield_3m_data = yf.download(tickers["3M Yield"], period="1y", interval="1d")
yield_2y_data = yf.download(tickers["2Y Yield"], period="1y", interval="1d")
yield_5y_data = yf.download(tickers["5Y Yield"], period="1y", interval="1d")
yield_10y_data = yf.download(tickers["10Y Yield"], period="1y", interval="1d")
yield_30y_data = yf.download(tickers["30Y Yield"], period="1y", interval="1d")

# Combine the CPI data with the yields (resample CPI to match the frequency)
cpi_data = cpi_data.resample('D').ffill()  # Forward fill to daily frequency
cpi_data = pd.DataFrame(cpi_data, columns=['CPI'])  # Convert to DataFrame

# Ensure we have valid data
if not yield_3m_data.empty and not yield_2y_data.empty and not yield_5y_data.empty and not yield_10y_data.empty and not yield_30y_data.empty:
    
    # Process the data: convert yields to percentages
    yield_3m_data['3M Yield'] = yield_3m_data['Close'] / 100  # 3M in percentage
    yield_2y_data['2Y Yield'] = yield_2y_data['Close'] / 100  # 2Y in percentage
    yield_5y_data['5Y Yield'] = yield_5y_data['Close'] / 100  # 5Y in percentage
    yield_10y_data['10Y Yield'] = yield_10y_data['Close'] / 100  # 10Y in percentage
    yield_30y_data['30Y Yield'] = yield_30y_data['Close'] / 100  # 30Y in percentage
    
    # Align data by date using the join method (inner join for common dates)
    combined_data = yield_3m_data[['3M Yield']].join([
        yield_2y_data[['2Y Yield']],
        yield_5y_data[['5Y Yield']],
        yield_10y_data[['10Y Yield']],
        yield_30y_data[['30Y Yield']]
    ], how='inner').join(cpi_data, how='inner')

    # Plot the Treasury Yields and CPI
    plt.figure(figsize=(14, 8))
    
    # Plot individual Treasury yields
    plt.plot(combined_data.index, combined_data['3M Yield'], label="3M Treasury Yield", color="orange", linestyle="-", marker="o")
    plt.plot(combined_data.index, combined_data['2Y Yield'], label="2Y Treasury Yield", color="green", linestyle="--", marker="o")
    plt.plot(combined_data.index, combined_data['5Y Yield'], label="5Y Treasury Yield", color="blue", linestyle=":", marker="o")
    plt.plot(combined_data.index, combined_data['10Y Yield'], label="10Y Treasury Yield", color="red", linestyle="-.", marker="o")
    plt.plot(combined_data.index, combined_data['30Y Yield'], label="30Y Treasury Yield", color="purple", linestyle="-", marker="o")
    
    # Plot CPI on a secondary y-axis
    ax2 = plt.gca().twinx()
    ax2.plot(combined_data.index, combined_data['CPI'], label="CPI (Index)", color="black", linestyle=":", marker="x")
    ax2.set_ylabel("CPI (Index)", fontsize=12)
    
    # Customize the plot
    plt.title("Comparison of US Treasury Yields vs CPI", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Yield (%)", fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Legends
    plt.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

else:
    print("Failed to retrieve all necessary data.")
