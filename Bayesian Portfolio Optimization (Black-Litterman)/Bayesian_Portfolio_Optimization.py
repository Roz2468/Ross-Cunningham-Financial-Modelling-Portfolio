#Ross E Cunningham
#Bayesian Portfolio Optimization (Black-Litterman)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.black_litterman import market_implied_prior_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# ========================
# STEP 1: Load Market Data
# ========================

# Define the stock tickers for portfolio
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# Download historical adjusted closing prices from Yahoo Finance
data = yf.download(tickers, start="2018-01-01", end="2023-01-01")["Close"]
print(data.head())

# ==============================
# STEP 2: Compute Market Estimates
# ==============================

# Estimate the expected returns using historical mean method
mu = mean_historical_return(data)

# Compute the covariance matrix using the Ledoit-Wolf shrinkage method (improves stability)
cov_matrix = CovarianceShrinkage(data).ledoit_wolf()
print("Covariance matrix shape:", cov_matrix.shape)  # Should be (N, N)
cov_matrix = cov_matrix.to_numpy()  # or cov_matrix.values
print(cov_matrix)
# Define risk aversion (lambda)
risk_aversion = 2.5  # Example value

# ==============================
# STEP 3: Compute Market-Implied Prior Returns
# ==============================

# Approximate market capitalization values in USD for each stock
market_caps = np.array([2.8e12, 1.6e12, 2.5e12, 1.4e12, 0.7e12])  # in trillions
print("Market caps shape:", market_caps.shape)

# Convert market caps to weights
mkt_weights = market_caps / market_caps.sum()
mkt_weights = mkt_weights.flatten()
print("Market weights shape:", mkt_weights.shape)
print(cov_matrix.dot(mkt_weights))  # Should return a (5,) shaped vector



# Compute market-implied prior returns based on market capitalization-weighted risk-adjusted returns
market_prior = market_implied_prior_returns(mkt_weights, risk_aversion, cov_matrix)

# ==============================
# STEP 4: Incorporate Investor Views (Bayesian Adjustment)
# ==============================

# Define investor views using the Black-Litterman methodology
# In this example, we believe that AAPL (Apple) will outperform TSLA (Tesla)
# We express this by creating a view matrix P and an expected return difference Q

P = np.array([[1, 0, 0, 0, -1]])  # View matrix (1 for AAPL, -1 for TSLA)
Q = np.array([0.02])  # Expected return difference: AAPL outperforms TSLA by 2%

# Tau parameter (scales the impact of investor views relative to market data)
tau = 0.05  # Lower values give more weight to prior market beliefs

# Create Black-Litterman model using prior market returns and investor views
bl = BlackLittermanModel(cov_matrix, pi=market_prior, P=P, Q=Q, tau=tau)

# Compute the posterior mean and covariance after incorporating investor views
posterior_mu = bl.bl_returns()  # Adjusted expected returns
posterior_S = bl.bl_cov()  # Adjusted covariance matrix

# ==============================
# STEP 5: Portfolio Optimization
# ==============================

# Use Efficient Frontier to optimize portfolio based on Bayesian-adjusted returns & risk
ef = EfficientFrontier(posterior_mu, posterior_S)

# Maximize the Sharpe Ratio (risk-adjusted returns)
weights = ef.max_sharpe()

# Clean the portfolio weights (remove small allocations)
cleaned_weights = ef.clean_weights()

# Print optimized portfolio weights
print("\n✅ Optimized Portfolio Weights (Black-Litterman):")
print(cleaned_weights)

# Display portfolio performance (expected return, volatility, Sharpe Ratio)
ef.portfolio_performance(verbose=True)

# ==============================
# STEP 6: Visualization
# ==============================

# Plot Market Prior vs Black-Litterman Posterior Expected Returns
plt.figure(figsize=(10, 5))
plt.bar(tickers, market_prior, label="Market Prior Returns", alpha=0.6)
plt.bar(tickers, posterior_mu, label="Black-Litterman Posterior Returns", alpha=0.6)
plt.xlabel("Stocks")
plt.ylabel("Expected Return")
plt.legend()
plt.title("Market Prior vs Black-Litterman Adjusted Returns")
plt.show()
