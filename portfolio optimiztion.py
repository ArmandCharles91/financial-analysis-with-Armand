#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from scipy.optimize import minimize

# Stock tickers
tickers = ['SPY', 'BND', 'QQQ', 'GLD', 'VTI']

# start and end time determination
end_date = dt.datetime.today()
start_date = end_date - dt.timedelta(days=5*365)  # 5 years back
print("Start Date:", start_date)

# adjusted close prices
adj_close_def = pd.DataFrame()
for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    adj_close_def[ticker] = data['Adj Close']

print("Adjusted Close Prices:")
print(adj_close_def.head())

# log norm returns
log_returns = np.log(adj_close_def / adj_close_def.shift(1)).dropna()

# cov matrix for log returns
cov_matrix = log_returns.cov() * 252  # Annualize the covariance matrix
print("Covariance Matrix:")
print(cov_matrix)

# standard deviation
def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252  # Annualized return

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

# optimization process
# Objective function: Maximize negative Sharpe ratio (since minimize is used)
def negative_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

# Constraints and bounds
risk_free_rate = 0.02
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}  # Weights must sum to 1
bounds = [(0, 0.5) for _ in range(len(tickers))]  # Each weight between 0 and 0.5
initial_weights = np.ones(len(tickers)) / len(tickers)  # Equal weight initialization

# method for result
result = minimize(
    negative_sharpe_ratio,
    initial_weights,
    args=(log_returns, cov_matrix, risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Optimal weights
optimal_weights = result.x
print("Optimal Weights:", optimal_weights)

# portfolio weights 
portfolio_return = expected_return(optimal_weights, log_returns)
portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
portfolio_sharpe = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print("Portfolio Annual Return:", portfolio_return)
print("Portfolio Volatility:", portfolio_volatility)
print("Portfolio Sharpe Ratio:", portfolio_sharpe)

# plotting
def portfolio_metrics(weights):
    return (
        standard_deviation(weights, cov_matrix),
        expected_return(weights, log_returns)
    )

# Generate random portfolios for visualization
random_portfolios = 5000
random_weights = [np.random.dirichlet(np.ones(len(tickers))) for _ in range(random_portfolios)]
portfolios = np.array([portfolio_metrics(w) for w in random_weights])

# transpose vol and returns
volatilities, returns = portfolios.T

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(volatilities, returns, c=returns / volatilities, cmap='viridis', marker='o', alpha=0.5)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(portfolio_volatility, portfolio_return, c='red', marker='*', s=200, label='Optimal Portfolio')
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.legend()
plt.grid()
plt.show()

