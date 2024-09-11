#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams ['figure.figsize']=8,6
import seaborn as ab
ab.set()
#defining the start and end dates for the data retrieval
start = datetime.datetime(2019,1,1)
end= datetime.datetime (2024,1,1)
#fetch data for blackrock(BLK)
blackrock= yf.download('BLK',start=start,end=end)
print(blackrock.head())
#close column for blackrock
blackrock_close=blackrock['Close']
blackrock_return= round(np.log(blackrock_close).diff()*100,2)
#displaying the first few rows of the log returns
print(blackrock_return.head())
blackrock_return.plot()
#defining statistic parameters
blackrock_return.dropna(inplace=True)
blackrock_return.describe()
#more slightly different table of descriptive statistics for normal distribution
from scipy import stats
n,minmax,mean,var,skew,kurt = stats.describe(blackrock_return)
mini,maxi=minmax
std=var**.5
plt.figure(figsize=(10, 6))
plt.hist(blackrock_return, bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title('Histogram of BlackRock Log Returns')
plt.xlabel('Log Return (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
from scipy.stats import norm
x = norm.rvs(mean,std,n)
plt.hist(x,bins=15)
x_test = stats.kurtosistest(x)
blackrock_test = stats.kurtosistest(blackrock_return)
print(f'{"Test statistic":20}{"p-value":>15}')
print(f'{" "*5}{"-"*30}')
print(f"x:{x_test[0]:>17.2f}{x_test[1]:16.4f}")
print(f'blackrock:{blackrock_test[0]:13.2f}{blackrock_test[1]:16.4}')
plt.hist(blackrock_return,bins = 25 , edgecolor = 'w', density = True)
overlay= np.linspace(mini,maxi,100)
plt.plot(overlay,norm.pdf(overlay,mean,std));
#testing if daily prices are different from 0
stats.ttest_1samp(blackrock_return.sample(252),0,alternative ='two-sided')
blackrock_close = pd.DataFrame(blackrock_return,columns =['Close'])
blackrock_close['lag_1'] = blackrock_close.Close.shift(1)
blackrock_close['lag_2'] = blackrock_close.Close.shift(2)
blackrock_close.dropna(inplace=True)
blackrock_close.head()
np.linalg.lstsq(blackrock_close[['lag_1','lag_2']],blackrock_close['Close'],rcond= None )[0]
blackrock_close['predict'] = np.dot(blackrock_close[['lag_1', 'lag_2']], np.array([1, 1]))
blackrock_close.head()
blackrock_close[['Close','predict']].plot()


# In[ ]:




