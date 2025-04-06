#Ross E Cunningham
"""
Monte Carlo Simulation for European Call Option Pricing
--------------------------------------------------------
This script implements a **Monte Carlo simulation** to estimate the price of a 
European **call option** using a stochastic model of stock price movements.

### Step-by-Step Breakdown:

1. **Import Required Library**  
   - `numpy`: Used for numerical computations, including **random number generation** 
     for simulating stock price movements.  

2. **Define the Monte Carlo Simulation Function (`monte_carlo_simulation`)**  
   - This function **simulates the stock price path** over a given time period 
     and calculates the expected option payoff.  
   - Inputs:  
     - `S0`: Initial stock price (spot price).  
     - `K`: Strike price (price at which the option can be exercised).  
     - `T`: Time to expiration (in years).  
     - `r`: Risk-free interest rate (assumed constant).  
     - `sigma`: Volatility of the underlying asset (annualized standard deviation).  
     - `num_simulations`: The number of Monte Carlo simulation paths.  
     - `num_steps`: The number of time steps in each simulation (e.g., daily steps).  
   - **Simulation Process:**  
     - `dt = T / num_steps`: Defines the time increment (e.g., daily if `num_steps=252`).  
     - **For each simulation**:  
       - Starts with `S = S0` (initial stock price).  
       - Iteratively updates `S` based on the **geometric Brownian motion** model:  
         \[
         S_{t+dt} = S_t \times e^{(r - 0.5\sigma^2)dt + \sigma\sqrt{dt} Z}
         \]
         where `Z` is a random sample from a **standard normal distribution** (`np.random.normal()`).  
       - At the end of the time period, the **call option payoff** is computed:  
         \[
         \max(S - K, 0)
         \]
     - Finally, the function **discounts the average payoff** to present value:  
       \[
       \text{Option Price} = e^{-rT} \times \text{Expected Payoff}
       \]

3. **Define Input Parameters for the Simulation**  
   - `S0 = 100`: The stock price starts at **$100**.  
   - `K = 105`: The option has a **strike price of $105**.  
   - `T = 0.5`: The option expires in **6 months (0.5 years)**.  
   - `r = 0.01`: A **1% risk-free interest rate** is assumed.  
   - `sigma = 0.2`: The stock has **20% annualized volatility**.  
   - `num_simulations = 10,000`: Simulates **10,000 possible stock price paths**.  
   - `num_steps = 252`: The stock price is simulated in **daily time steps**.  

4. **Run the Simulation and Estimate the Call Option Price**  
   - Calls `monte_carlo_simulation()` with the given parameters.  
   - **Prints the estimated option price** (rounded to two decimal places).  

### Key Considerations:
- **Monte Carlo simulations are computationally intensive** but provide **flexibility** 
  to model complex option pricing scenarios.  
- **The accuracy improves with more simulations (`num_simulations`)** but at the 
  cost of increased computation time.  
- **Stock prices follow a stochastic process**, making Monte Carlo methods a 
  powerful tool for estimating derivative prices.  
- **This method is useful for pricing exotic options** that do not have a closed-form solution 
  (e.g., path-dependent options like Asian options).  

"""
import numpy as np

def monte_carlo_simulation(S0, K, T, r, sigma, num_simulations, num_steps):
    dt = T / num_steps
    simulations = np.zeros(num_simulations)
    
    for i in range(num_simulations):
        S = S0
        for _ in range(num_steps):
            S = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
        simulations[i] = max(S - K, 0)  # Call option payoff
    
    return np.exp(-r * T) * np.mean(simulations)  # Discounted average payoff

# Parameters
S0 = 100  # Initial stock price
K = 105  # Strike price
T = 0.5  # Time to maturity in years
r = 0.01  # Risk-free rate
sigma = 0.2  # Volatility
num_simulations = 10000  # Number of simulations
num_steps = 252  # Number of steps (daily)

# Estimate call option price
call_price = monte_carlo_simulation(S0, K, T, r, sigma, num_simulations, num_steps)

print(f"Estimated Call Option Price: {call_price:.2f}")
