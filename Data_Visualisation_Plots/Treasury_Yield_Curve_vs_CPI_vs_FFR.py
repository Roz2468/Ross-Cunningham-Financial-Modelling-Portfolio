#Treasury_Yield_Curve_vs_CPI_vs_FFR
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

# Define tickers for US 10-year Treasury yield, Federal Funds Rate, and CPI
tickers = {
    "10Y Yield": "^TNX",         # US 10-year Treasury Yield
    "Fed Funds Rate": "^IRX",    # Federal Funds Rate
    "CPI": "CPIAUCSL"            # US Consumer Price Index (from St. Louis Fed)
}

# Fetch historical data (last 1 year of daily data)
yield_data = yf.download(tickers["10Y Yield"], period="1y", interval="1d")
fed_data = yf.download(tickers["Fed Funds Rate"], period="1y", interval="1d")
cpi_data = yf.download(tickers["CPI"], period="1y", interval="1mo")  # Monthly data for CPI

# Ensure we have valid data
if not yield_data.empty and not fed_data.empty and not cpi_data.empty:
    # Process the data: convert TNX and Fed Funds Rate to percentages
    yield_data['10Y Yield'] = yield_data['Close'] / 100  # TNX is in percentage
    fed_data['Fed Funds Rate'] = fed_data['Close'] / 100  # IRX is in percentage
    
    # CPI doesn't need conversion, it's already in index form
    cpi_data['CPI'] = cpi_data['Close']  # No conversion needed, using it directly
    
    # Align data by date using the join method (inner join for common dates)
    combined_data = yield_data[['10Y Yield']].join([fed_data[['Fed Funds Rate']], cpi_data[['CPI']]], how='inner')
    
    # Plot the 10-Year Yield, Federal Funds Rate, and CPI
    plt.figure(figsize=(12, 8))
    
    # Plot the US 10-year Treasury Yield
    plt.plot(combined_data.index, combined_data['10Y Yield'], label="US 10-Year Treasury Yield", color="blue", marker="o")
    
    # Plot the Federal Funds Rate
    plt.plot(combined_data.index, combined_data['Fed Funds Rate'], label="Federal Funds Rate", color="green", linestyle="--")
    
    # Plot the CPI (on a secondary y-axis since its scale is different)
    ax2 = plt.gca().twinx()  # Create a secondary y-axis for CPI
    ax2.plot(combined_data.index, combined_data['CPI'], label="CPI (Index)", color="red", linestyle=":")
    ax2.set_ylabel("CPI (Index)", fontsize=12)
    
    # Customize the plot
    plt.title("US 10-Year Treasury Yield vs Federal Funds Rate & CPI", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Yield (%)", fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

else:
    print("Failed to retrieve all necessary data.")
