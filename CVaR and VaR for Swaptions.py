#!/usr/bin/env python
# coding: utf-8

# In[1]:


#this is simulated code for risk management. the content was inspired by the book of Lech Gerzelak on computational finance and tutorials from ASX Portfolio
import numpy as np
import pandas as pd
from scipy.stats import norm

# Define a sample yield curve manually
yield_curve = {
    '1Y': 0.015,
    '2Y': 0.017,
    '3Y': 0.018,
    '5Y': 0.02,
    '7Y': 0.022,
    '10Y': 0.025
}

# Define swaption positions with notional values, fixed rate, floating rate, and maturity
swaptions = [
    {'notional': 1_000_000, 'fixed_rate': 0.02, 'floating_rate': 0.015, 'maturity': 2},  # 2Y swaption
    {'notional': 500_000, 'fixed_rate': 0.025, 'floating_rate': 0.02, 'maturity': 5},    # 5Y swaption
    {'notional': 750_000, 'fixed_rate': 0.03, 'floating_rate': 0.025, 'maturity': 7}     # 7Y swaption
]

# Define a function to calculate swaption payoff
def calculate_swaption_value(notional, fixed_rate, floating_rate, maturity, yield_curve):
    discount_factor = 1 / (1 + yield_curve[f'{maturity}Y']) ** maturity
    fixed_leg_value = notional * fixed_rate * maturity * discount_factor
    floating_leg_value = notional * floating_rate * maturity * discount_factor
    return floating_leg_value - fixed_leg_value

# Calculate initial portfolio value for swaptions
portfolio_value = sum([calculate_swaption_value(swaption['notional'], swaption['fixed_rate'], swaption['floating_rate'], swaption['maturity'], yield_curve) for swaption in swaptions])

# Generate historical returns for each swaption using a sample historical data (or simulated for this example)
np.random.seed(42)
num_simulations = 1000
returns_historical = np.random.normal(0, 0.02, num_simulations)

# Historical VaR and CVaR
confidence_level = 0.95
VaR_historical = -np.percentile(returns_historical, (1 - confidence_level) * 100) * portfolio_value
CVaR_historical = -np.mean([ret for ret in returns_historical if ret <= -VaR_historical / portfolio_value]) * portfolio_value

# Parametric VaR and CVaR (assuming normal distribution)
mean_return = np.mean(returns_historical)
std_dev = np.std(returns_historical)
VaR_parametric = -norm.ppf(1 - confidence_level) * std_dev * portfolio_value
CVaR_parametric = -portfolio_value * (mean_return - (std_dev * norm.pdf(norm.ppf(confidence_level)) / (1 - confidence_level)))

# Monte Carlo VaR and CVaR
returns_monte_carlo = np.random.normal(mean_return, std_dev, num_simulations)
VaR_monte_carlo = -np.percentile(returns_monte_carlo, (1 - confidence_level) * 100) * portfolio_value
CVaR_monte_carlo = -np.mean([ret for ret in returns_monte_carlo if ret <= -VaR_monte_carlo / portfolio_value]) * portfolio_value

# Display the results
print(f"Portfolio Value: {portfolio_value}")
print(f"Historical VaR (95%): {VaR_historical}")
print(f"Historical CVaR (95%): {CVaR_historical}")
print(f"Parametric VaR (95%): {VaR_parametric}")
print(f"Parametric CVaR (95%): {CVaR_parametric}")
print(f"Monte Carlo VaR (95%): {VaR_monte_carlo}")
print(f"Monte Carlo CVaR (95%): {CVaR_monte_carlo}")


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure
plt.figure(figsize=(14, 10))

# Plotting Historical Returns
plt.subplot(3, 1, 1)
sns.histplot(returns_historical, kde=True, color='blue', bins=30)
plt.axvline(-VaR_historical / portfolio_value, color='red', linestyle='--', label=f'VaR (Historical) = {VaR_historical:.2f}')
plt.axvline(-CVaR_historical / portfolio_value, color='purple', linestyle='--', label=f'CVaR (Historical) = {CVaR_historical:.2f}')
plt.title('Historical Method: Distribution of Returns with VaR and CVaR')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.legend()

# Plotting Parametric Normal Distribution
plt.subplot(3, 1, 2)
x = np.linspace(mean_return - 4 * std_dev, mean_return + 4 * std_dev, 100)
pdf = norm.pdf(x, mean_return, std_dev)
plt.plot(x, pdf, label='Normal Distribution', color='blue')
plt.axvline(-VaR_parametric / portfolio_value, color='red', linestyle='--', label=f'VaR (Parametric) = {VaR_parametric:.2f}')
plt.axvline(-CVaR_parametric / portfolio_value, color='purple', linestyle='--', label=f'CVaR (Parametric) = {CVaR_parametric:.2f}')
plt.title('Parametric Method: Normal Distribution with VaR and CVaR')
plt.xlabel('Returns')
plt.ylabel('Probability Density')
plt.legend()

# Plotting Monte Carlo Simulated Returns
plt.subplot(3, 1, 3)
sns.histplot(returns_monte_carlo, kde=True, color='green', bins=30)
plt.axvline(-VaR_monte_carlo / portfolio_value, color='red', linestyle='--', label=f'VaR (Monte Carlo) = {VaR_monte_carlo:.2f}')
plt.axvline(-CVaR_monte_carlo / portfolio_value, color='purple', linestyle='--', label=f'CVaR (Monte Carlo) = {CVaR_monte_carlo:.2f}')
plt.title('Monte Carlo Method: Distribution of Simulated Returns with VaR and CVaR')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




