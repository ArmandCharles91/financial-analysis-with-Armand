#!/usr/bin/env python
# coding: utf-8

# In[1]:


#In this section I using the same path as asx portfolio of estimate VAR and CVAR for the stock tickers (apple, johnson and jonson, exon, procter and gambel, and jp morgan chase).
#we consider our initial portfolio of 10000 dollars 
#we calculate both parametric method, historical method and montecorlo method.
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# we use yfinance for adjusted stock prices
def getData(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)
    stockData = stockData['Adj Close']  # We consider adjusted prices here
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return returns, meanReturns, covMatrix

# Portfolio performance estimation
def portfolioPerformance(weights, meanReturns, covMatrix, Time):
    returns = np.sum(meanReturns * weights) * Time
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(Time)
    return returns, std

# Stock tickers and period(apple,jp morgan chase, johnson and johnson, procter and gamble,and exon)
stockList = ['AAPL', 'JNJ', 'JPM', 'PG', 'XOM']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=500)

# Data fetching and portfolio setup
returns, meanReturns, covMatrix = getData(stockList, start=startDate, end=endDate)
returns = returns.dropna()  # Drop missing values

# Portfolio weights
weights = np.random.random(len(returns.columns))
weights /= np.sum(weights)  # Normalize weights to sum to one

# Calculate expected portfolio return and plot cumulative returns
returns['portfolio'] = returns.dot(weights)

# Plot cumulative returns of the portfolio
plt.figure(figsize=(8, 6))
returns['portfolio'].cumsum().plot()
plt.title('Cumulative Portfolio Returns')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.show()

# Historical VaR and CVaR
def historicalVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha=alpha)
    else:
        raise TypeError("Expected returns to be dataframe or series")

def historicalCVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVaR, alpha=alpha)
    else:
        raise TypeError("Expected returns to be dataframe or series")

# Parametric VaR and CVaR method
def var_parametric(portfolioReturns, portfolioStd, distribution='normal', alpha=5, dof=6):
    if distribution == 'normal':
        VaR = norm.ppf(1-alpha/100)*portfolioStd - portfolioReturns
    elif distribution == 't-distribution':
        VaR = np.sqrt((dof-2)/dof) * t.ppf(1-alpha/100, dof) * portfolioStd - portfolioReturns
    else:
        raise TypeError("Expected distribution type 'normal'/'t-distribution'")
    return VaR

def cvar_parametric(portfolioReturns, portfolioStd, distribution='normal', alpha=5, dof=6):
    if distribution == 'normal':
        CVaR = (alpha/100)**-1 * norm.pdf(norm.ppf(alpha/100))*portfolioStd - portfolioReturns
    elif distribution == 't-distribution':
        xanu = t.ppf(alpha/100, dof)
        CVaR = -1/(alpha/100) * (1-dof)**(-1) * (dof-2+xanu**2) * t.pdf(xanu, dof) * portfolioStd - portfolioReturns
    else:
        raise TypeError("Expected distribution type 'normal'/'t-distribution'")
    return CVaR

# Monte Carlo Simulation for VaR and CVaR
def monteCarloSimulations(meanReturns, covMatrix, weights, initialPortfolio, mc_sims=500, T=100):
    meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns).T
    portfolio_sims = np.zeros((T, mc_sims))
    
    for m in range(mc_sims):
        Z = np.random.normal(size=(T, len(weights)))
        L = np.linalg.cholesky(covMatrix)
        dailyReturns = meanM + np.inner(L, Z)
        portfolio_sims[:, m] = np.cumprod(np.inner(weights, dailyReturns.T) + 1) * initialPortfolio
    
    return portfolio_sims

def mcVaR(returns, alpha=5):
    return np.percentile(returns, alpha)

def mcCVaR(returns, alpha=5):
    belowVaR = returns <= mcVaR(returns, alpha=alpha)
    return returns[belowVaR].mean()

# Running the portfolio performance calculations
Time = 100
InitialInvestment = 10000
pRet, pStd = portfolioPerformance(weights, meanReturns, covMatrix, Time)

# Historical and Parametric VaR/CVaR calculations
hVaR = -historicalVaR(returns['portfolio'], alpha=5) * np.sqrt(Time)
hCVaR = -historicalCVaR(returns['portfolio'], alpha=5) * np.sqrt(Time)
normVaR = var_parametric(pRet, pStd)
normCVaR = cvar_parametric(pRet, pStd)
tVaR = var_parametric(pRet, pStd, distribution='t-distribution')
tCVaR = cvar_parametric(pRet, pStd, distribution='t-distribution')

# Monte Carlo Simulation Results
initialPortfolio = 10000
portfolio_sims = monteCarloSimulations(meanReturns, covMatrix, weights, initialPortfolio)
portResults = pd.Series(portfolio_sims[-1, :])

mc_VaR = initialPortfolio - mcVaR(portResults, alpha=5)
mc_CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)

# Output results
print(f"Expected Portfolio Return:      ${round(InitialInvestment * pRet, 2)}")
print(f"Historical VaR 95th CI:         ${round(InitialInvestment * hVaR, 2)}")
print(f"Historical CVaR 95th CI:        ${round(InitialInvestment * hCVaR, 2)}")
print(f"Normal VaR 95th CI:             ${round(InitialInvestment * normVaR, 2)}")
print(f"Normal CVaR 95th CI:            ${round(InitialInvestment * normCVaR, 2)}")
print(f"t-distribution VaR 95th CI:     ${round(InitialInvestment * tVaR, 2)}")
print(f"t-distribution CVaR 95th CI:    ${round(InitialInvestment * tCVaR, 2)}")
print(f"Monte Carlo VaR 95th CI:        ${round(mc_VaR, 2)}")
print(f"Monte Carlo CVaR 95th CI:       ${round(mc_CVaR, 2)}")

# Plot Monte Carlo Simulation results
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC Simulation of Stock Portfolio')
plt.show()


# In[ ]:




