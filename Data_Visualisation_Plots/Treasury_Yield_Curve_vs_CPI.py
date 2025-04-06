import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Define tickers for Treasury yields and CPI
tickers = {
    "3M Yield": "^IRX",         # US 3-month Treasury Yield
    "2Y Yield": "^FVX",         # US 2-year Treasury Yield
    "5Y Yield": "^FVX",         # US 5-year Treasury Yield
    "10Y Yield": "^TNX",        # US 10-year Treasury Yield
    "30Y Yield": "^TYX",        # US 30-year Treasury Yield
    "CPI": "CPIAUCSL"           # US Consumer Price Index (from St. Louis Fed)
}

# Fetch historical data (last 1 year of daily data for yields, monthly for CPI)
yield_3m_data = yf.download(tickers["3M Yield"], period="1y", interval="1d")
yield_2y_data = yf.download(tickers["2Y Yield"], period="1y", interval="1d")
yield_5y_data = yf.download(tickers["5Y Yield"], period="1y", interval="1d")
yield_10y_data = yf.download(tickers["10Y Yield"], period="1y", interval="1d")
yield_30y_data = yf.download(tickers["30Y Yield"], period="1y", interval="1d")
cpi_data = yf.download(tickers["CPI"], period="1y", interval="1mo")  # Monthly data for CPI

# Ensure we have valid data
if not yield_3m_data.empty and not yield_2y_data.empty and not yield_5y_data.empty and not yield_10y_data.empty and not yield_30y_data.empty and not cpi_data.empty:
    
    # Process the data: convert yields to percentages
    yield_3m_data['3M Yield'] = yield_3m_data['Close'] / 100  # 3M in percentage
    yield_2y_data['2Y Yield'] = yield_2y_data['Close'] / 100  # 2Y in percentage
    yield_5y_data['5Y Yield'] = yield_5y_data['Close'] / 100  # 5Y in percentage
    yield_10y_data['10Y Yield'] = yield_10y_data['Close'] / 100  # 10Y in percentage
    yield_30y_data['30Y Yield'] = yield_30y_data['Close'] / 100  # 30Y in percentage
    
    # CPI doesn't need conversion, it's already in index form
    cpi_data['CPI'] = cpi_data['Close']  # No conversion needed, using it directly
    
    # Align data by date using the join method (inner join for common dates)
    combined_data = yield_3m_data[['3M Yield']].join([
        yield_2y_data[['2Y Yield']],
        yield_5y_data[['5Y Yield']],
        yield_10y_data[['10Y Yield']],
        yield_30y_data[['30Y Yield']],
        cpi_data[['CPI']]
    ], how='inner')

    # Plot the Treasury Yields
    plt.figure(figsize=(14, 8))
    
    # Plot individual Treasury yields
    plt.plot(combined_data.index, combined_data['3M Yield'], label="3M Treasury Yield", color="orange", linestyle="-", marker="o")
    plt.plot(combined_data.index, combined_data['2Y Yield'], label="2Y Treasury Yield", color="green", linestyle="--", marker="o")
    plt.plot(combined_data.index, combined_data['5Y Yield'], label="5Y Treasury Yield", color="blue", linestyle=":", marker="o")
    plt.plot(combined_data.index, combined_data['10Y Yield'], label="10Y Treasury Yield", color="red", linestyle="-.", marker="o")
    plt.plot(combined_data.index, combined_data['30Y Yield'], label="30Y Treasury Yield", color="purple", linestyle="-", marker="o")
    
    # Plot the CPI (on a secondary y-axis since its scale is different)
    ax2 = plt.gca().twinx()  # Create a secondary y-axis for CPI
    ax2.plot(combined_data.index, combined_data['CPI'], label="CPI (Index)", color="black", linestyle=":", marker="x")
    ax2.set_ylabel("CPI (Index)", fontsize=12)
    
    # Customize the plot
    plt.title("Comparison of US Treasury Yields (3M, 2Y, 5Y, 10Y, 30Y) vs CPI", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Yield (%)", fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Legends for both y-axes
    plt.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

else:
    print("Failed to retrieve all necessary data.")
