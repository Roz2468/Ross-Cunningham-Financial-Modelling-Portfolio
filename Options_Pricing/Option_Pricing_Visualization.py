#Ross E Cunningham
"""
Black-Scholes Option Pricing Model
------------------------------------
This script implements the **Black-Scholes formula**, a mathematical model used 
to price European call and put options. The model assumes that stock prices 
follow a **lognormal distribution** and do not pay dividends.

### Step-by-Step Breakdown:

1. **Import Required Libraries**  
   - `scipy.stats.norm`: Provides the cumulative distribution function (CDF) for the standard normal distribution, which is essential for the Black-Scholes formula.  
   - `numpy`: Supports numerical computations like logarithms and square roots.  
   - `yfinance`: Can be used for fetching live stock data (though not directly used in this script).  
   - `matplotlib.pyplot`: Used to visualize the relationship between strike price and option prices.  

2. **Define the Black-Scholes Function (`black_scholes`)**  
   - Inputs:  
     - `S`: Current stock price (spot price).  
     - `K`: Strike price (price at which the option can be exercised).  
     - `T`: Time to expiration (in years).  
     - `r`: Risk-free interest rate (assumed constant).  
     - `sigma`: Volatility of the underlying asset (annualized standard deviation).  
     - `option_type`: Determines whether the function calculates a **call** or **put** option price.  
   - Computes **two key variables (`d1` and `d2`)**:  
     - `d1 = (ln(S/K) + (r + 0.5 * sigma²) * T) / (sigma * sqrt(T))`  
     - `d2 = d1 - sigma * sqrt(T)`  
   - Uses the cumulative distribution function (`norm.cdf()`) to calculate the probability of being in-the-money:  
     - **Call option price formula**:  
       \[
       C = S \cdot N(d1) - K e^{-rT} \cdot N(d2)
       \]  
     - **Put option price formula**:  
       \[
       P = K e^{-rT} \cdot N(-d2) - S \cdot N(-d1)
       \]  
   - Returns the option price for the given parameters.  

3. **Define Input Parameters for Pricing Options**  
   - `S = 100`: Assumes the underlying stock price is $100.  
   - `K = np.linspace(80, 120, 40)`: Generates 40 strike prices from $80 to $120.  
   - `T = 0.5`: The option has **6 months (0.5 years) to expiration**.  
   - `r = 0.01`: Assumes a **1% risk-free interest rate**.  
   - `sigma = 0.2`: Assumes a **20% annual volatility** of the underlying stock.  

4. **Calculate Call Option Prices**  
   - Calls the `black_scholes` function with `option_type='call'` for the range of strike prices.  

5. **Visualize Call Option Prices vs. Strike Price**  
   - Uses `plt.plot()` to create a graph of **call option prices vs. strike prices**.  
   - The graph typically shows a **downward-sloping** relationship:  
     - **Lower strike prices** → Higher call option prices (since the option is more likely to be in the money).  
     - **Higher strike prices** → Lower call option prices.  
   - Includes proper labels, title, and a grid for clarity.  

### Key Considerations:
- **The Black-Scholes model assumes no dividends and constant volatility.**  
- **Only valid for European-style options**, which can only be exercised at expiration.  
- **Does not account for market frictions**, such as transaction costs and liquidity constraints.  
- **Volatility (`sigma`) is a crucial input**: Higher volatility leads to higher option prices.  

"""


from scipy.stats import norm
import numpy as np
#Time Series Analysis 
import yfinance as yf
import matplotlib.pyplot as plt

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes formula for option pricing."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

# Parameters
S = 100  # Underlying stock price
K = np.linspace(80, 120, 40)  # Range of strike prices
T = 0.5  # Time to maturity in years
r = 0.01  # Risk-free rate
sigma = 0.2  # Volatility

# Calculate call option prices
call_prices = black_scholes(S, K, T, r, sigma, option_type='call')

# Plot call option prices
plt.figure(figsize=(12, 6))
plt.plot(K, call_prices, label='Call Option Prices', color='green')
plt.title('Call Option Prices vs. Strike Price')
plt.xlabel('Strike Price')
plt.ylabel('Call Option Price')
plt.legend()
plt.grid(True)
plt.show()
