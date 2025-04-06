import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import requests

# Fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock = stock[['Close']]
    stock.index.name = 'Date'  # Ensure the index is named "Date"
    return stock

# Fetch macroeconomic data from Alpha Vantage
def fetch_series(function, api_key):
    url = f"https://www.alphavantage.co/query?function={function}&interval=annual&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if "data" in data:
        df = pd.DataFrame(data["data"])
    elif "Realtime Series" in data:
        df = pd.DataFrame(data["Realtime Series"])
    elif "annual" in data:
        df = pd.DataFrame(data["annual"])
    else:
        print(f"Error fetching {function}: {data}")
        return pd.DataFrame()
    
    df.set_index(df.columns[0], inplace=True)  # Set first column as index (dates)
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert values to numeric
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.resample('D').ffill()  # Forward-fill to match daily stock data
    df.index.name = 'Date'  # Ensure the index is named "Date"
    return df

# Prepare data for regression
import pandas as pd
from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_data(stock_data, macro_data):
    # Drop NaN values
    stock_data = stock_data.dropna()
    macro_data = macro_data.dropna()

    # Align macro data index with stock data BEFORE computing pct_change
    macro_data = macro_data.reindex(stock_data.index, method='ffill')

    # Compute percentage changes only for stock_data
    stock_returns = stock_data.pct_change().dropna()
    stock_returns.columns = ['Stock_Returns']  # Flatten MultiIndex if necessary

    # Forward fill stock returns and macro data to align frequencies
    stock_returns = stock_returns.reindex(macro_data.index, method='ffill')
    macro_data = macro_data.reindex(stock_returns.index, method='ffill')

    # Print stock_returns and macro_data for debugging
    print("Stock Returns:\n", stock_returns.head())
    print("Macro Data:\n", macro_data.head())

    # Reset index for merging
    stock_returns = stock_returns.reset_index()
    macro_data = macro_data.reset_index()

    # Merge datasets using outer join to retain all possible dates
    merged_data = pd.merge(stock_returns, macro_data, on='Date', how='outer')
    
    # Drop remaining NaN values post-merge
    merged_data = merged_data.dropna()

    # Set Date as index again after merging
    merged_data.set_index('Date', inplace=True)

    # Scale and prepare input for regression
    scaler = StandardScaler()
    X = scaler.fit_transform(merged_data.iloc[:, 1:])  # Ensure correct feature selection
    y = merged_data.iloc[:, 0].values  # Ensure correct target selection
    
    return X, y


# Build regression model
def build_factor_model(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

# Main script
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-01-01'
vantage_api_key = 'K0E5Z0WIEI8IRH9M'  # Replace with a valid key

stock_data = get_stock_data(ticker, start_date, end_date)
print(stock_data)
gdp_data = fetch_series("REAL_GDP", vantage_api_key)
print(gdp_data)
inflation_data = fetch_series("INFLATION", vantage_api_key)
print(inflation_data)
interest_data = fetch_series("FEDERAL_FUNDS_RATE", vantage_api_key)
print(interest_data)

# Combine macroeconomic factors
factor_data = pd.concat([gdp_data, inflation_data, interest_data], axis=1).dropna()
print(factor_data)
# Prepare data for model
X, y = prepare_data(stock_data, factor_data)

# Train model
model = build_factor_model(X, y)
print(model.summary())
